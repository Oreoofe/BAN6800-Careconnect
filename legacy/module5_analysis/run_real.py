import json, time, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
RS=42
df=pd.read_parquet("data/processed/careconnect_modeling.parquet")
# LEAKAGE FIX: drop raw target text, raw timestamps, pseudonym id
df=df.drop(columns=["No-show","ScheduledDay","AppointmentDay","patient_id_hash"])
df["age_band"]=df["age_band"].astype(str); df["Gender"]=df["Gender"].astype(str); df["Neighbourhood"]=df["Neighbourhood"].astype(str)
y=df.pop("noshow_flag"); X=df
NUM=["Age","lead_time_days","prior_appointments","prior_noShows","prior_noShow_rate","Scholarship","Hipertension","Diabetes","Alcoholism","Handcap","SMS_received","appointment_dow","appointment_month"]
CAT=["Gender","Neighbourhood","age_band"]
def pre(): return ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
print(len(Xtr),len(Xte),y.mean(),flush=True)
def ev(m,X,y):
    p=m.predict_proba(X)[:,1]; yh=(p>=0.5).astype(int)
    return dict(accuracy=accuracy_score(y,yh),precision=precision_score(y,yh,zero_division=0),recall=recall_score(y,yh),f1=f1_score(y,yh),roc_auc=roc_auc_score(y,p))
res={"n_train":len(Xtr),"n_test":len(Xte),"pos_rate":float(y.mean())}
# baselines
bl={}
for s in ["most_frequent","stratified"]:
    d=DummyClassifier(strategy=s,random_state=RS).fit(np.zeros((len(ytr),1)),ytr)
    yh=d.predict(np.zeros((len(yte),1))); pp=d.predict_proba(np.zeros((len(yte),1)))[:,1]
    bl[s]=dict(accuracy=accuracy_score(yte,yh),precision=precision_score(yte,yh,zero_division=0),recall=recall_score(yte,yh,zero_division=0),f1=f1_score(yte,yh,zero_division=0),roc_auc=roc_auc_score(yte,pp))
res["baselines"]=bl; print(bl,flush=True)
sub_idx=Xtr.sample(25000,random_state=RS).index
Xs,ys=Xtr.loc[sub_idx],ytr.loc[sub_idx]
cv=StratifiedKFold(3,shuffle=True,random_state=RS)
spw=float((ytr==0).sum()/(ytr==1).sum())
specs={
 "logistic_regression":(LogisticRegression(max_iter=1000,random_state=RS),{"clf__C":[0.01,0.1,1.0,10.0],"clf__class_weight":[None,"balanced"]}),
 "random_forest":(RandomForestClassifier(random_state=RS,n_jobs=1),{"clf__n_estimators":[100,200],"clf__max_depth":[6,10,None],"clf__min_samples_leaf":[1,5],"clf__class_weight":[None,"balanced"]}),
 "xgboost":(xgb.XGBClassifier(eval_metric="logloss",random_state=RS,n_jobs=1,tree_method="hist"),{"clf__n_estimators":[100,300],"clf__max_depth":[4,6,8],"clf__learning_rate":[0.05,0.1],"clf__scale_pos_weight":[1,spw]}),
}
models={}; res["models"]={}
for name in ["logistic_regression","xgboost","random_forest"]:
    est,grid=specs[name]
    t=time.time()
    gs=GridSearchCV(Pipeline([("pre",pre()),("clf",est)]),grid,cv=cv,scoring="roc_auc",n_jobs=1).fit(Xs,ys)
    bp={k.replace("clf__",""):v for k,v in gs.best_params_.items()}
    final=Pipeline([("pre",pre()),("clf",est.set_params(**bp))]).fit(Xtr,ytr)
    m=ev(final,Xte,yte); models[name]=final
    res["models"][name]=dict(metrics=m,best_params=bp,cv_auc_subsample=float(gs.best_score_),secs=round(time.time()-t))
    print(name,m,bp,round(time.time()-t),"s",flush=True)
    joblib.dump(final,f"artifacts/{name}.joblib")
best=max(res["models"],key=lambda k:res["models"][k]["metrics"]["roc_auc"]); res["best_by_auc"]=best
Xte.assign(noshow_flag=yte.values).to_parquet("artifacts/test_set.parquet",index=False)
Xtr.assign(noshow_flag=ytr.values).to_parquet("artifacts/train_set.parquet",index=False)
json.dump(res,open("artifacts/model_results.json","w"),indent=2,default=float)
print("BEST",best,flush=True)
