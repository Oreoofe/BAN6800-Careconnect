import json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
tr=pd.read_parquet("artifacts/train_set.parquet"); te=pd.read_parquet("artifacts/test_set.parquet")
ytr,yte=tr.pop("noshow_flag"),te.pop("noshow_flag"); m=joblib.load("artifacts/xgboost.joblib")
oof=cross_val_predict(clone(m),tr,ytr,cv=StratifiedKFold(3,shuffle=True,random_state=42),method="predict_proba")[:,1]
pte=m.predict_proba(te)[:,1]
A=json.load(open("artifacts/analysis.json")); T=A["threshold"]
def report(yh):
    d={"recall":recall_score(yte,yh),"precision":precision_score(yte,yh),"flag_rate":float(yh.mean())}
    for a in ["age_band","Gender"]:
        sf=te[a].astype(str); d[a]=dict(dpd=float(demographic_parity_difference(yte,yh,sensitive_features=sf)),eod=float(equalized_odds_difference(yte,yh,sensitive_features=sf)))
    sf=te["Scholarship"].astype(str); d["Scholarship"]=dict(dpd=float(demographic_parity_difference(yte,yh,sensitive_features=sf)),eod=float(equalized_odds_difference(yte,yh,sensitive_features=sf)))
    g={}
    for v in sorted(te.age_band.astype(str).unique()):
        k=(te.age_band.astype(str)==v).values; tn,fp,fn,tp=confusion_matrix(yte[k],yh[k],labels=[0,1]).ravel()
        g[v]=dict(flag_rate=float(yh[k].mean()),tpr=float(tp/(tp+fn)),fpr=float(fp/(fp+tn)))
    d["age_groups"]=g; return d
res={"before":report((pte>=T).astype(int))}
# A) per-band thresholds: equal TPR (0.65) on train OOF
th={}
for v in sorted(tr.age_band.astype(str).unique()):
    k=(tr.age_band.astype(str)==v).values; ts=np.linspace(0.02,0.9,300)
    r=np.array([recall_score(ytr[k],(oof[k]>=t).astype(int)) for t in ts]); th[v]=float(ts[r>=0.65].max())
yhA=np.array([pte[i]>=th[b] for i,b in enumerate(te.age_band.astype(str))]).astype(int)
res["A_group_thresholds"]=dict(thresholds=th,**report(yhA))
# B) Fairlearn ThresholdOptimizer
for name,c in [("B_equalized_odds","equalized_odds"),("C_demographic_parity","demographic_parity")]:
    class Est(BaseEstimator,ClassifierMixin):
        def fit(s,X,y): return s
        def predict_proba(s,X): return np.c_[1-X["p"].values,X["p"].values]
        def predict(s,X): return (X["p"].values>=T).astype(int)
        classes_=np.array([0,1])
        def __sklearn_is_fitted__(s): return True
    to=ThresholdOptimizer(estimator=Est(),constraints=c,objective="balanced_accuracy_score",predict_method="predict_proba",prefit=True)
    to.fit(pd.DataFrame({"p":oof.astype("float64")}),ytr,sensitive_features=tr.age_band.astype(str))
    yh=to.predict(pd.DataFrame({"p":pte.astype("float64")}),sensitive_features=te.age_band.astype(str),random_state=42).astype(int)
    res[name]=report(yh)
for k,v in res.items(): print(k,{x:(round(y,3) if not isinstance(y,dict) else {a:round(b,3) for a,b in y.items() if not isinstance(b,dict)}) for x,y in v.items() if x!="age_groups" and x!="thresholds"})
json.dump(res,open("artifacts/mitigation.json","w"),indent=1,default=float)
