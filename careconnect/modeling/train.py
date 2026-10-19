"""Module 4 (rebuilt): baselines, three tuned models, operating point, robustness checks, MLflow.

    python -m careconnect.modeling.train

* Same algorithms and hyper-parameter grids as the Module 4 report; GridSearchCV with
  CARECONNECT_CV_FOLDS-fold (default 5) stratified CV on the full training set, scored by ROC-AUC.
* The outreach cut-off is chosen on out-of-fold *training* predictions to reach recall >= 0.65.
  The test set is used once, after all choices are fixed.
* Every run is logged to MLflow; the selected XGBoost model is registered as
  `careconnect_noshow_xgb` with alias `candidate` (promotion to `production` needs the Ethics
  Committee approval recorded in docs/model_card.md, so it is never automatic).
* Robustness: a patient-grouped split (no patient in both train and test) and a
  data-minimisation ablation without protected/sensitive health inputs.
"""
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from careconnect import config
from careconnect.modeling.data import load_table, preprocessor, split, xy

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE


def metrics(y, p, t):
    yh = (p >= t).astype(int)
    return dict(threshold=float(t), accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                flag_rate=float(yh.mean()))


def recall_threshold(y, oof, target=config.RECALL_TARGET):
    ts = np.round(np.arange(0.05, 0.6001, 0.0025), 4)
    rec = np.array([recall_score(y, (oof >= t).astype(int)) for t in ts])
    return float(ts[rec >= target].max()) if (rec >= target).any() else 0.2


def specs(spw: float):
    import xgboost as xgb
    return {
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=RS),
                                {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}),
        "random_forest": (RandomForestClassifier(random_state=RS, n_jobs=1),
                          {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 10, None],
                           "clf__min_samples_leaf": [1, 5], "clf__class_weight": [None, "balanced"]}),
        "xgboost": (xgb.XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1, tree_method="hist"),
                    {"clf__n_estimators": [100, 300], "clf__max_depth": [4, 6, 8], "clf__learning_rate": [0.05, 0.1],
                     "clf__scale_pos_weight": [1, spw]}),
    }


class Tracker:
    """Thin MLflow wrapper; if MLflow is unavailable the run continues and says so in the results."""

    def __init__(self):
        try:
            import mlflow
            config.ROOT.joinpath("mlruns").mkdir(exist_ok=True)
            mlflow.set_tracking_uri(config.MLFLOW_URI)
            if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
                mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                         artifact_location=(config.ROOT / "mlruns" / "artifacts").as_uri())
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
            self.mlflow = mlflow
        except Exception as e:  # pragma: no cover
            print("MLflow unavailable:", e)
            self.mlflow = None

    def log(self, name, params=None, metrics=None, tags=None, model=None, register=False, X_example=None):
        if not self.mlflow:
            return None
        m = self.mlflow
        with m.start_run(run_name=name) as run:
            m.set_tags({"project": "CareConnect", "model_version": config.MODEL_VERSION, **(tags or {})})
            if params:
                m.log_params({k: str(v) for k, v in params.items()})
            if metrics:
                m.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))})
            info = None
            if model is not None:
                import mlflow.sklearn
                kw = {"input_example": X_example.head(3)} if X_example is not None else {}
                try:
                    info = mlflow.sklearn.log_model(model, name="model", **kw)
                except TypeError:   # MLflow 2.x signature
                    info = mlflow.sklearn.log_model(model, artifact_path="model", **kw)
            if register and info is not None:
                mv = m.register_model(info.model_uri, config.REGISTERED_MODEL)
                client = m.tracking.MlflowClient()
                client.set_registered_model_alias(config.REGISTERED_MODEL, "candidate", mv.version)
                client.set_model_version_tag(config.REGISTERED_MODEL, mv.version, "approval",
                                             "pending Ethics Committee review")
                client.set_model_version_tag(config.REGISTERED_MODEL, mv.version, "semver", config.MODEL_VERSION)
                return {"run_id": run.info.run_id, "registered_version": mv.version}
            return {"run_id": run.info.run_id}


def main():
    t0 = time.time()
    df = load_table()
    tr, te = split(df)
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    trk = Tracker()
    res = {"n_train": len(Xtr), "n_test": len(Xte), "pos_rate": float(df[config.TARGET].mean()),
           "cv_folds": config.CV_FOLDS, "tuning_rows": len(Xtr), "model_version": config.MODEL_VERSION,
           "mlflow": bool(trk.mlflow), "baselines": {}, "models": {}}

    for s in ["most_frequent", "stratified"]:
        d = DummyClassifier(strategy=s, random_state=RS).fit(Xtr, ytr)
        p = d.predict_proba(Xte)[:, 1]
        m = metrics(yte, p, 0.5)
        res["baselines"][s] = m
        trk.log(f"baseline_{s}", params={"strategy": s}, metrics=m, tags={"kind": "baseline"})

    models = {}
    for name, (est, grid) in specs(spw).items():
        t = time.time()
        gs = GridSearchCV(Pipeline([("pre", preprocessor()), ("clf", est)]), grid, cv=cv, scoring="roc_auc",
                          n_jobs=-1, refit=True).fit(Xtr, ytr)
        best = gs.best_estimator_
        oof = cross_val_predict(clone(best), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = recall_threshold(ytr, oof)
        p = best.predict_proba(Xte)[:, 1]
        bp = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        r = dict(best_params=bp, cv_auc=float(gs.best_score_), oof_auc=float(roc_auc_score(ytr, oof)),
                 at_default_0_5=metrics(yte, p, 0.5), at_recall_target=metrics(yte, p, thr),
                 secs=round(time.time() - t))
        r["metrics"] = r["at_default_0_5"]          # schema kept for the Module 5 dashboard
        logged = trk.log(name, params={**bp, "cv_folds": config.CV_FOLDS, "flag_threshold": thr},
                         metrics={"cv_auc": r["cv_auc"], **{f"test_{k}": v for k, v in r["at_recall_target"].items()}},
                         tags={"kind": "candidate"}, model=best, register=(name == "xgboost"), X_example=Xtr)
        r["mlflow"] = logged
        res["models"][name] = r
        models[name] = (best, oof, thr)
        print(name, {k: round(v, 4) for k, v in r["at_recall_target"].items()}, bp, r["secs"], "s", flush=True)

    res["best_by_auc"] = max(res["models"], key=lambda k: res["models"][k]["at_recall_target"]["roc_auc"])
    res["deployed"] = "xgboost"
    best, oof, thr = models["xgboost"]

    # Robustness 1: patient-grouped split (no patient appears in both train and test)
    gss = GroupShuffleSplit(1, test_size=config.TEST_SIZE, random_state=RS)
    gi, go = next(gss.split(df, groups=df["patient_id_hash"]))
    Xg, yg = xy(df.iloc[gi])
    Xh, yh = xy(df.iloc[go])
    mg = clone(best).fit(Xg, yg)
    res["robustness_patient_grouped"] = metrics(yh, mg.predict_proba(Xh)[:, 1], thr)
    trk.log("xgboost_patient_grouped_split", params={"split": "GroupShuffleSplit by patient"},
            metrics=res["robustness_patient_grouped"], tags={"kind": "robustness"})

    # Robustness 2: data minimisation - drop protected / sensitive health inputs
    feats = [f for f in config.FEATURES if f not in config.SENSITIVE_INPUTS]
    abl = Pipeline([("pre", preprocessor(feats)), ("clf", clone(best.named_steps["clf"]))])
    abl.fit(Xtr[feats], ytr)
    oof_a = cross_val_predict(clone(abl), Xtr[feats], ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr_a = recall_threshold(ytr, oof_a)
    pa = abl.predict_proba(Xte[feats])[:, 1]
    res["ablation_no_sensitive_inputs"] = {"features": feats, "removed": config.SENSITIVE_INPUTS,
                                           **metrics(yte, pa, thr_a)}
    trk.log("xgboost_without_sensitive_inputs", params={"removed": ",".join(config.SENSITIVE_INPUTS)},
            metrics=res["ablation_no_sensitive_inputs"], tags={"kind": "ablation"})

    config.ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(best, config.ARTIFACTS / "xgboost.joblib", compress=3)
    joblib.dump(abl, config.ARTIFACTS / "xgboost_minimal.joblib", compress=3)
    for n in ["logistic_regression", "random_forest"]:
        joblib.dump(models[n][0], config.ARTIFACTS / f"{n}.joblib", compress=3)   # git-ignored (size)
    # predictions needed downstream (no identifiers; row order = the frames saved below)
    np.save(config.ARTIFACTS / "oof_xgb.npy", oof)
    tr.to_pickle(config.ARTIFACTS / "train_frame.pkl")      # git-ignored (pseudonymised rows)
    te.to_pickle(config.ARTIFACTS / "test_frame.pkl")       # git-ignored
    json.dump({"flag_threshold": thr, "high_threshold": config.HIGH_TIER, "model_version": config.MODEL_VERSION,
               "note": "flag_threshold chosen on training-set out-of-fold predictions to reach recall >= 0.65; never tuned on the test set"},
              open(config.ARTIFACTS / "operating_point.json", "w"), indent=1)
    res["total_secs"] = round(time.time() - t0)
    json.dump(res, open(config.ARTIFACTS / "model_results.json", "w"), indent=2, default=float)
    print("done in", res["total_secs"], "s")


if __name__ == "__main__":
    main()
