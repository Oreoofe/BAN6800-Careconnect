"""Evaluation, fairness audit, global/local SHAP, what-if and descriptive statistics.

    python -m careconnect.modeling.evaluate

Writes artifacts/analysis.json (the schema read by the dashboard) and figures in artifacts/figures/.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from careconnect import config
from careconnect.fairness.metrics import fairness_report
from careconnect.modeling.data import xy

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
FIG = config.ARTIFACTS / "figures"


def group_of(name: str) -> str:
    b = name.split("__", 1)[1]
    for g in config.CATEGORICAL:
        if b.startswith(g + "_"):
            return g
    return b


def tree_contributions(model, X: pd.DataFrame):
    """Exact TreeSHAP values from XGBoost (pred_contribs), summed over one-hot columns per feature."""
    import xgboost as xgb
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    C = clf.get_booster().predict(xgb.DMatrix(pre.transform(X), feature_names=names), pred_contribs=True)
    base = C[:, -1]
    C = C[:, :-1]
    groups = pd.Series([group_of(n) for n in names])
    G = pd.DataFrame({g: C[:, np.where(groups == g)[0]].sum(1) for g in groups.unique()}, index=X.index)
    return G, base


def shap_figures(G: pd.DataFrame, X: pd.DataFrame):
    """Beeswarm summary (shap library) on grouped values; falls back to matplotlib bars if shap is absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    disp = X[G.columns].copy()
    for c in config.CATEGORICAL:
        if c in disp:
            disp[c] = pd.factorize(disp[c])[0]
    try:
        import shap
        plt.figure()
        shap.summary_plot(G.values, disp.astype(float).values, feature_names=list(G.columns), show=False, max_display=12)
        plt.title("SHAP summary: what pushes no-show risk up (right) or down (left)", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "shap_summary.png", dpi=160)
        plt.close("all")
    except Exception as e:  # pragma: no cover
        print("shap plot skipped:", e)
    imp = G.abs().mean().sort_values()
    plt.figure(figsize=(7, 4.5))
    plt.barh(imp.index[-10:], imp.values[-10:], color="#0F4C5C")
    plt.xlabel("Mean |SHAP value| (log-odds)")
    plt.title("Global feature importance (exact TreeSHAP)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_importance.png", dpi=160)
    plt.close("all")


def main():
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    R = json.load(open(config.ARTIFACTS / "model_results.json"))
    op = json.load(open(config.ARTIFACTS / "operating_point.json"))
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    Xte, yte = xy(te)
    T = op["flag_threshold"]
    p = m.predict_proba(Xte)[:, 1]
    yh = (p >= T).astype(int)
    out = {"model_results": R, "model_version": config.MODEL_VERSION, "threshold": T,
           "operating": {k: {"at_default_0_5": v["at_default_0_5"], "at_recall_target": v["at_recall_target"]}
                         for k, v in R["models"].items()}}
    out["confusion"] = dict(zip(["tn", "fp", "fn", "tp"], map(int, confusion_matrix(yte, yh).ravel())))
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    dec = pd.DataFrame({"d": q, "y": yte.values, "p": p}).groupby("d").agg(actual=("y", "mean"), pred=("p", "mean"),
                                                                           n=("y", "size")).reset_index()
    out["deciles"] = dec.to_dict("records")
    out["base_rate"] = float(yte.mean())
    tiers = pd.cut(p, [-1, T, op["high_threshold"], 2], labels=["low", "medium", "high"])
    out["tiers"] = {str(k): {"share": float((tiers == k).mean()), "no_show_rate": float(yte.values[tiers == k].mean())}
                    for k in ["low", "medium", "high"]}

    sens = te.assign(Scholarship_grp=te["Scholarship"].map({0: "No scholarship", 1: "Bolsa Familia scholarship"}))
    out["fairness"] = {a: fairness_report(yte.values, yh, sens[a].astype(str).values)
                       for a in ["Gender", "age_band", "Scholarship_grp"]}

    # SHAP on a fixed sample of 4,000 test appointments
    idx = np.random.RandomState(RS).choice(len(Xte), 4000, replace=False)
    Xs = Xte.iloc[idx]
    G, base = tree_contributions(m, Xs)
    imp = G.abs().mean().sort_values(ascending=False)
    direction = {}
    for g in G.columns:
        if pd.api.types.is_numeric_dtype(Xs[g]):
            direction[g] = (float(np.corrcoef(Xs[g], G[g])[0, 1]) if Xs[g].nunique() > 2
                            else float(G[g][Xs[g] == 1].mean() - G[g][Xs[g] == 0].mean()) if Xs[g].nunique() == 2 else 0.0)
    out["shap"] = {"importance": imp.to_dict(), "share": (imp / imp.sum()).to_dict(), "direction": direction,
                   "method": "exact TreeSHAP (XGBoost pred_contribs), one-hot columns summed per feature",
                   "n_sample": len(Xs)}
    shap_figures(G, Xs)

    def score_tier(s):
        return "high" if s >= op["high_threshold"] else ("medium" if s >= T else "low")

    def sample(pos, label):
        Xi = Xte.iloc[[pos]]
        Gi, _ = tree_contributions(m, Xi)
        top = sorted(Gi.iloc[0].to_dict().items(), key=lambda x: -abs(x[1]))[:4]
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in Xi.iloc[0].to_dict().items()}
        return dict(tier=label, score_tier=score_tier(float(p[pos])), prob=float(p[pos]), single_cutoff_flag=bool(p[pos] >= T),
                    features=row, top=top, actual=int(yte.values[pos]), test_row=int(pos))

    # Examples chosen so the label IS the score tier: the median-scored appointment within each tier
    # (cycle 1 picked by rank position, which labelled a 0.245 score - below the 0.25 cut-off - "medium").
    tiers = np.array([score_tier(s) for s in p])
    picks = {}
    for t in ["high", "medium", "low"]:
        pos_t = np.where(tiers == t)[0]
        picks[t] = int(pos_t[np.argsort(p[pos_t])[len(pos_t) // 2]])
    out["samples"] = [sample(pos, t) for t, pos in picks.items()]
    # The three examples documented in verification cycle 1 (same split, same test rows), kept for comparison
    out["samples_documented_cycle1"] = [sample(pos, f"cycle1_{lbl}") for lbl, pos in
                                        [("high", 14074), ("medium", 20762), ("low", 5547)]]

    wi = {}

    def scen(label, f):
        wi[label] = float((m.predict_proba(f(Xte.copy()))[:, 1] - p).mean() * 100)
    scen("Booked 7 days ahead at most (lead time capped at 7 days)", lambda X: X.assign(lead_time_days=X.lead_time_days.clip(upper=7)))
    scen("Booked 14 days ahead at most (lead time capped at 14 days)", lambda X: X.assign(lead_time_days=X.lead_time_days.clip(upper=14)))
    scen("Everyone marked as SMS received", lambda X: X.assign(SMS_received=1))
    scen("Nobody marked as SMS received", lambda X: X.assign(SMS_received=0))
    scen("No prior missed appointments on record", lambda X: X.assign(prior_noShows=0, prior_noShow_rate=0.0))
    out["whatif_avg_pp"] = wi

    al = pd.concat([tr, te])
    y = al[config.TARGET]
    lt = pd.cut(al.lead_time_days, [-1, 0, 3, 7, 14, 30, 400],
                labels=["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "31+ days"])
    agg = lambda by: y.groupby(by).agg(["mean", "size"]).reset_index()
    d = {"lead_time": agg(lt).rename(columns={"lead_time_days": "bucket"}).to_dict("records"),
         "sms": agg(al.SMS_received).to_dict("records"),
         "age_band": agg(al.age_band).to_dict("records"),
         "prior": agg(al.prior_noShows.clip(upper=3)).to_dict("records"),
         "scholarship": agg(al.Scholarship).to_dict("records"),
         "gender": agg(al.Gender).to_dict("records"),
         "sms_by_lead": y.groupby([lt, al.SMS_received]).mean().unstack().reset_index()
                         .rename(columns={"lead_time_days": "bucket"}).to_dict("records"),
         "n": int(len(al)), "overall": float(y.mean())}
    out["descriptive"] = json.loads(json.dumps(d, default=str))
    hoods = sorted(al["Neighbourhood"].unique().tolist())
    json.dump(hoods, open(config.ARTIFACTS / "neighbourhoods.json", "w"))
    json.dump(out, open(config.ARTIFACTS / "analysis.json", "w"), indent=1, default=str)
    print("analysis.json written; AUC", round(R["models"]["xgboost"]["at_recall_target"]["roc_auc"], 4),
          "| age EOD", round(out["fairness"]["age_band"]["eod"], 3))


if __name__ == "__main__":
    main()
