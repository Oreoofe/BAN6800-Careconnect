import json, warnings, joblib, numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, selection_rate
RS=42
tr=pd.read_parquet("artifacts/train_set.parquet"); te=pd.read_parquet("artifacts/test_set.parquet")
ytr,yte=tr.pop("noshow_flag"),te.pop("noshow_flag")
R=json.load(open("artifacts/model_results.json"))
out={"model_results":R}
def met(y,p,t):
    yh=(p>=t).astype(int)
    return dict(threshold=float(t),accuracy=accuracy_score(y,yh),precision=precision_score(y,yh,zero_division=0),recall=recall_score(y,yh),f1=f1_score(y,yh),roc_auc=roc_auc_score(y,p),flag_rate=float(yh.mean()))
# recall-targeted threshold from train OOF (no test data used)
op={}
for name in ["logistic_regression","xgboost","random_forest"]:
    m=joblib.load(f"artifacts/{name}.joblib")
    oof=cross_val_predict(clone(m),tr,ytr,cv=StratifiedKFold(3,shuffle=True,random_state=RS),method="predict_proba")[:,1]
    ts=np.linspace(0.05,0.6,221); rec=np.array([recall_score(ytr,(oof>=t).astype(int)) for t in ts])
    t=float(ts[rec>=0.65].max()) if (rec>=0.65).any() else 0.2
    p=m.predict_proba(te)[:,1]
    op[name]=dict(at_default_0_5=met(yte,p,0.5),at_recall_target=met(yte,p,t))
    print(name,op[name]["at_recall_target"],flush=True)
out["operating"]=op
m=joblib.load("artifacts/xgboost.joblib"); T=op["xgboost"]["at_recall_target"]["threshold"]
p=m.predict_proba(te)[:,1]; yh=(p>=T).astype(int); out["threshold"]=T
out["confusion"]=dict(zip(["tn","fp","fn","tp"],map(int,confusion_matrix(yte,yh).ravel())))
# deciles / lift
q=pd.qcut(pd.Series(p).rank(method="first"),10,labels=False)
dec=pd.DataFrame({"d":q,"y":yte.values,"p":p}).groupby("d").agg(actual=("y","mean"),pred=("p","mean"),n=("y","size")).reset_index()
out["deciles"]=dec.to_dict("records"); out["base_rate"]=float(yte.mean())
# fairness
def fair(attr, labels=None):
    sf=te[attr].astype(str); g={}
    for v in sorted(sf.unique()):
        k=(sf==v).values; yt=yte.values[k]; yp=yh[k]
        tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
        g[v]=dict(n=int(k.sum()),no_show_rate=float(yt.mean()),flag_rate=float(yp.mean()),tpr=float(tp/(tp+fn)) if tp+fn else None,fpr=float(fp/(fp+tn)) if fp+tn else None,precision=float(tp/(tp+fp)) if tp+fp else None)
    sel=[v["flag_rate"] for v in g.values()]
    r=dict(groups=g,dpd=float(demographic_parity_difference(yte,yh,sensitive_features=sf)),eod=float(equalized_odds_difference(yte,yh,sensitive_features=sf)),
           tpr_gap=float(max(v["tpr"] for v in g.values())-min(v["tpr"] for v in g.values())),fpr_gap=float(max(v["fpr"] for v in g.values())-min(v["fpr"] for v in g.values())),
           di_ratio=float(min(sel)/max(sel)),base_rate_gap=float(max(v["no_show_rate"] for v in g.values())-min(v["no_show_rate"] for v in g.values())))
    r["pass_dpd"]=abs(r["dpd"])<=0.05; r["pass_eod"]=abs(r["eod"])<=0.05; r["pass_di"]=r["di_ratio"]>=0.8
    return r
te2=te.copy(); te2["Scholarship_grp"]=te2["Scholarship"].map({0:"No scholarship",1:"Bolsa Familia scholarship"})
te=te2
out["fairness"]={"Gender":fair("Gender"),"age_band":fair("age_band"),"Scholarship_grp":fair("Scholarship_grp")}
for a,r in out["fairness"].items(): print(a,{k:r[k] for k in ["dpd","eod","tpr_gap","fpr_gap","di_ratio","pass_dpd","pass_eod","pass_di"]})
# SHAP (exact TreeSHAP via xgboost)
pre,clf=m.named_steps["pre"],m.named_steps["clf"]; names=list(pre.get_feature_names_out())
idx=np.random.RandomState(RS).choice(len(te),4000,replace=False)
Xs=pre.transform(te.drop(columns=["Scholarship_grp"]).iloc[idx]); C=clf.get_booster().predict(xgb.DMatrix(Xs,feature_names=names),pred_contribs=True)[:,:-1]
def grp(n):
    b=n.split("__",1)[1]
    for g in ["Neighbourhood","age_band","Gender"]:
        if b.startswith(g): return g
    return b
G=pd.Series([grp(n) for n in names]); imp={}; direction={}
raw=te.drop(columns=["Scholarship_grp"]).iloc[idx].reset_index(drop=True)
for g in G.unique():
    cols=np.where(G==g)[0]; s=C[:,cols].sum(1); imp[g]=float(np.abs(s).mean())
    if g in raw.columns and pd.api.types.is_numeric_dtype(raw[g]) and raw[g].nunique()>2:
        direction[g]=float(np.corrcoef(raw[g],s)[0,1])
    elif g in raw.columns and pd.api.types.is_numeric_dtype(raw[g]):
        direction[g]=float(s[raw[g].values==1].mean()-s[raw[g].values==0].mean())
tot=sum(imp.values()); out["shap"]={"importance":{k:v for k,v in sorted(imp.items(),key=lambda x:-x[1])},"share":{k:v/tot for k,v in sorted(imp.items(),key=lambda x:-x[1])},"direction":direction}
print({k:round(v,3) for k,v in out["shap"]["share"].items()})
# sample patients (low / medium / high by score), with local SHAP
pp=p[idx]; order=np.argsort(pp); picks={"high":order[-30],"medium":order[len(order)//2+400],"low":order[40]}
samples=[]
for tier,i in picks.items():
    row=raw.iloc[i].to_dict(); contrib={}
    for g in G.unique(): contrib[g]=float(C[i,np.where(G==g)[0]].sum())
    top=sorted(contrib.items(),key=lambda x:-abs(x[1]))[:4]
    samples.append(dict(tier=tier,prob=float(pp[i]),features={k:(v if not hasattr(v,"item") else v.item()) for k,v in row.items()},top=top,actual=int(yte.values[idx][i])))
out["samples"]=samples
# what-if (population average change in predicted probability, percentage points)
base=pd.Series(p); X0=te.drop(columns=["Scholarship_grp"]); wi={}
def scen(label,f):
    X1=f(X0.copy()); wi[label]=float((m.predict_proba(X1)[:,1]-p).mean()*100)
scen("Booked 7 days ahead at most (lead time capped at 7 days)",lambda X:X.assign(lead_time_days=X.lead_time_days.clip(upper=7)))
scen("Booked 14 days ahead at most (lead time capped at 14 days)",lambda X:X.assign(lead_time_days=X.lead_time_days.clip(upper=14)))
scen("Everyone marked as SMS received",lambda X:X.assign(SMS_received=1))
scen("Nobody marked as SMS received",lambda X:X.assign(SMS_received=0))
scen("No prior missed appointments on record",lambda X:X.assign(prior_noShows=0,prior_noShow_rate=0.0))
out["whatif_avg_pp"]=wi; print(wi)
# descriptive (full train+test)
al=pd.concat([tr.assign(y=ytr),te.drop(columns=["Scholarship_grp"]).assign(y=yte)])
lt=pd.cut(al.lead_time_days,[-1,0,3,7,14,30,400],labels=["Same day","1-3 days","4-7 days","8-14 days","15-30 days","31+ days"])
d={"lead_time":al.groupby(lt).y.agg(["mean","size"]).reset_index().rename(columns={"lead_time_days":"bucket"}).to_dict("records"),
   "sms":al.groupby("SMS_received").y.agg(["mean","size"]).reset_index().to_dict("records"),
   "age_band":al.groupby("age_band").y.agg(["mean","size"]).reset_index().to_dict("records"),
   "prior":al.groupby(al.prior_noShows.clip(upper=3)).y.agg(["mean","size"]).reset_index().to_dict("records"),
   "scholarship":al.groupby("Scholarship").y.agg(["mean","size"]).reset_index().to_dict("records"),
   "gender":al.groupby("Gender").y.agg(["mean","size"]).reset_index().to_dict("records"),
   "sms_by_lead":al.groupby([lt,"SMS_received"]).y.mean().unstack().reset_index().rename(columns={"lead_time_days":"bucket"}).to_dict("records"),
   "n":int(len(al)),"overall":float(al.y.mean())}
out["descriptive"]=json.loads(json.dumps(d,default=str)); print(d["sms"],d["lead_time"])
json.dump(out,open("artifacts/analysis.json","w"),indent=1,default=str)
