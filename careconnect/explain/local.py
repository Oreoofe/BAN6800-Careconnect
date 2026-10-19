"""Local explanations for the three example patients: LIME and actionable counterfactuals.

    python -m careconnect.explain.local

LIME (Ribeiro et al., 2016) fits a sparse linear surrogate around one prediction. age_band is
re-derived from Age inside the prediction function so perturbed rows stay internally consistent.

Counterfactuals (Wachter et al., 2017) are searched only over changes the hospital controls:
booking lead time (offering an earlier slot) and sending an SMS reminder. Age, gender, history,
neighbourhood and health flags are immutable here; recommending changes to them would be neither
feasible nor ethical. The search is exhaustive over a small grid and returns the lowest-cost change
that moves the score below the outreach cut-off, or reports that none exists.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from careconnect import config
from careconnect.fairness.policy import load_policy
from careconnect.pipeline.features import age_band

warnings.filterwarnings("ignore")
FIG = config.ARTIFACTS / "figures"
CAT = ["Gender", "Neighbourhood", "age_band"]


def _per_label(v, label=1):
    """LIME 0.2.0.1 (PyPI) stores score/local_pred as one value; LIME master stores a dict per label."""
    if isinstance(v, dict):
        v = v[label]
    return float(np.ravel(v)[0])


def lime_explanations(model, train: pd.DataFrame, samples: list, n_features=6):
    from lime.lime_tabular import LimeTabularExplainer
    feats = config.FEATURES
    enc = {c: sorted(train[c].astype(str).unique()) for c in CAT}
    def encode(df):
        a = df[feats].copy()
        for c in CAT:
            a[c] = a[c].astype(str).map({v: i for i, v in enumerate(enc[c])}).fillna(0).astype(int)
        return a.astype(float).values
    def predict_fn(arr):
        d = pd.DataFrame(arr, columns=feats)
        for c in CAT:
            d[c] = [enc[c][int(round(min(max(v, 0), len(enc[c]) - 1)))] for v in d[c]]
        d["Age"] = d["Age"].clip(0, 100).round().astype(int)
        d["age_band"] = d["Age"].map(age_band)
        for c in ["lead_time_days", "prior_appointments", "prior_noShows"]:
            d[c] = d[c].clip(lower=0).round().astype(int)
        d["prior_noShows"] = np.minimum(d["prior_noShows"], d["prior_appointments"])
        d["prior_noShow_rate"] = np.where(d["prior_appointments"] > 0, d["prior_noShows"] / d["prior_appointments"].clip(lower=1), 0.0)
        return model.predict_proba(d[feats])
    bg = train.sample(5000, random_state=config.RANDOM_STATE)
    expl = LimeTabularExplainer(encode(bg), feature_names=feats, class_names=["attend", "no-show"],
                                categorical_features=[feats.index(c) for c in CAT],
                                categorical_names={feats.index(c): enc[c] for c in CAT},
                                discretize_continuous=True, random_state=config.RANDOM_STATE)
    out = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for s in samples:
        row = pd.DataFrame([s["features"]])
        e = expl.explain_instance(encode(row)[0], predict_fn, num_features=n_features, num_samples=3000)
        weights = e.as_list()
        out.append({"tier": s["tier"], "model_score": float(model.predict_proba(row[feats])[0, 1]),
                    "lime_local_prediction": _per_label(e.local_pred), "lime_r2": _per_label(e.score),
                    "weights": weights})
        fig = e.as_pyplot_figure()
        fig.set_size_inches(7, 3.6)
        plt.title(f"LIME: {s['tier']}-risk example (score {out[-1]['model_score']:.0%})", fontsize=11)
        plt.tight_layout()
        fig.savefig(FIG / f"lime_{s['tier']}.png", dpi=150)
        plt.close("all")
    return out


def counterfactual(model, features: dict, threshold: float, policy: dict | None = None):
    feats = config.FEATURES
    base = pd.DataFrame([features])[feats]
    p0 = float(model.predict_proba(base)[0, 1])
    lead0, sms0 = int(features["lead_time_days"]), int(features["SMS_received"])
    options = []
    for lead in sorted({0, 1, 2, 3, 5, 7, 14, 21, lead0}):
        if lead > lead0:
            continue                                   # only earlier slots are an intervention
        for sms in {sms0, 1}:                          # a reminder can be added, never taken away
            row = base.assign(lead_time_days=lead, SMS_received=sms)
            p = float(model.predict_proba(row)[0, 1])
            cost = (lead0 - lead) / 7 + (sms != sms0)
            options.append({"lead_time_days": lead, "SMS_received": sms, "score": p, "cost": round(cost, 3)})
    hits = sorted([o for o in options if o["score"] < threshold and o["cost"] > 0], key=lambda o: (o["cost"], o["score"]))
    best_any = min(options, key=lambda o: o["score"])
    if policy:
        from careconnect.fairness.policy import flag_probability
        g = features["age_band"]
        for o in options:
            o["policy_flag_probability"] = flag_probability(o["score"], g, policy)
    return {"original_score": p0, "threshold": threshold, "flagged": p0 >= threshold,
            "policy_flag_probability": flag_probability(p0, features["age_band"], policy) if policy else None,
            "counterfactual": hits[0] if hits else None, "lowest_reachable": best_any,
            "searched": len(options),
            "immutable": ["Age", "Gender", "Neighbourhood", "prior history", "health flags", "Scholarship"]}


def main():
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    A = json.load(open(config.ARTIFACTS / "analysis.json"))
    T = json.load(open(config.ARTIFACTS / "operating_point.json"))["flag_threshold"]
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    out = {"model_version": config.MODEL_VERSION, "flag_threshold": T,
           "lime": lime_explanations(m, tr, A["samples"]),
           "counterfactuals": [{"tier": s["tier"], **counterfactual(m, s["features"], T, load_policy())}
                               for s in A["samples"]]}
    # population view: share of flagged test appointments that an earlier slot / SMS could un-flag
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    flagged = te[m.predict_proba(te[config.FEATURES])[:, 1] >= T].sample(1500, random_state=config.RANDOM_STATE)
    reach = [counterfactual(m, r, T)["counterfactual"] is not None for r in flagged[config.FEATURES].to_dict("records")]
    out["population"] = {"flagged_sampled": len(flagged), "share_unflaggable_by_actionable_change": float(np.mean(reach))}
    json.dump(out, open(config.ARTIFACTS / "explanations.json", "w"), indent=1, default=float)
    for c in out["counterfactuals"]:
        print(c["tier"], round(c["original_score"], 3), c["counterfactual"])
    print(out["population"])


if __name__ == "__main__":
    main()
