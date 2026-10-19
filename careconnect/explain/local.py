"""Local explanations: LIME (two sampling configurations) and policy-aware actionable counterfactuals.

    python -m careconnect.explain.local      -> artifacts/explanations.json, artifacts/figures/lime_*.png

LIME (Ribeiro et al., 2016) fits a weighted linear surrogate around one prediction.
* "cycle1_training_distribution" - the cycle-1 configuration: perturbations drawn from the training
  distribution with quartile discretisation. Kept unchanged so results stay comparable.
* "local_sampling" - the correction proposed after cycle 1: perturbations centred on the explained
  appointment (sample_around_instance=True) without discretisation.
Both run on the same examples (the three documented in cycle 1 plus three tier-correct examples).
Fidelity is reported, not tuned: LIME's weighted R^2 on its own perturbations, |surrogate prediction
- model score| at the instance, and overlap of LIME's top-3 features with exact TreeSHAP's top-3.
The bar was fixed BEFORE running: R^2 >= 0.5 and |error| <= 0.05 -> "adequate", otherwise "weak".
age_band is re-derived from Age inside the prediction function so perturbed rows stay consistent.

Counterfactuals (Wachter et al., 2017) target the ADOPTED decision policy (fairness_policy.json): the
question is "what change the hospital controls would stop this appointment being flagged?", i.e. drive
the policy's flag probability to 0. Only two levers are searched: an earlier slot (lead time can only
fall) and adding an SMS reminder (never removed). Age, gender, history, neighbourhood and health flags
are immutable. Each counterfactual is re-scored through the model and the policy.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from careconnect import config
from careconnect.fairness.policy import flag_probability, load_policy
from careconnect.pipeline.features import age_band

warnings.filterwarnings("ignore")
FIG = config.ARTIFACTS / "figures"
CAT = ["Gender", "Neighbourhood", "age_band"]
FIDELITY_BAR = {"min_r2": 0.5, "max_abs_error": 0.05}       # fixed before the run; not tuned afterwards
LIME_CONFIGS = {"cycle1_training_distribution": dict(discretize_continuous=True, sample_around_instance=False),
                "local_sampling": dict(discretize_continuous=False, sample_around_instance=True)}


def _per_label(v, label=1):
    """LIME 0.2.0.1 (PyPI) stores score/local_pred as one value; LIME master stores a dict per label."""
    if isinstance(v, dict):
        v = v[label]
    return float(np.ravel(v)[0])


def _feature_of(name: str) -> str:
    hits = [f for f in config.FEATURES if f in name]
    return max(hits, key=len) if hits else name


def lime_explanations(model, train: pd.DataFrame, samples: list, n_features=6):
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
        for c in ["lead_time_days", "prior_appointments", "prior_noShows", "Handcap", "appointment_dow"]:
            d[c] = d[c].clip(lower=0).round().astype(int)
        for c in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]:
            d[c] = d[c].round().clip(0, 1).astype(int)
        d["appointment_month"] = d["appointment_month"].round().clip(1, 12).astype(int)
        d["appointment_dow"] = d["appointment_dow"].clip(0, 6)
        d["prior_noShows"] = np.minimum(d["prior_noShows"], d["prior_appointments"])
        d["prior_noShow_rate"] = np.where(d["prior_appointments"] > 0, d["prior_noShows"] / d["prior_appointments"].clip(lower=1), 0.0)
        return model.predict_proba(d[feats])

    bg = train.sample(5000, random_state=config.RANDOM_STATE)
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    for cname, kw in LIME_CONFIGS.items():
        expl = LimeTabularExplainer(encode(bg), feature_names=feats, class_names=["attend", "no-show"],
                                    categorical_features=[feats.index(c) for c in CAT],
                                    categorical_names={feats.index(c): enc[c] for c in CAT},
                                    random_state=config.RANDOM_STATE, **kw)
        for s in samples:
            row = pd.DataFrame([s["features"]])
            score = float(model.predict_proba(row[feats])[0, 1])
            e = expl.explain_instance(encode(row)[0], predict_fn, num_features=n_features, num_samples=5000)
            weights = e.as_list()
            r2, local = _per_label(e.score), _per_label(e.local_pred)
            lime_top3 = []
            for n, _ in weights:
                f = _feature_of(n)
                if f not in lime_top3:
                    lime_top3.append(f)
            lime_top3 = lime_top3[:3]
            shap_top3 = [k for k, _ in s["top"][:3]]
            norm = lambda f: "Age" if f == "age_band" else f
            overlap = len({norm(f) for f in lime_top3} & {norm(f) for f in shap_top3})
            adequate = r2 >= FIDELITY_BAR["min_r2"] and abs(local - score) <= FIDELITY_BAR["max_abs_error"]
            out.append({"config": cname, "example": s["tier"], "score_tier": s.get("score_tier"), "test_row": s.get("test_row"),
                        "model_score": score, "lime_local_prediction": local, "abs_error": abs(local - score),
                        "lime_r2": r2, "lime_top3": lime_top3, "shap_top3": shap_top3, "top3_overlap": overlap,
                        "fidelity": "adequate" if adequate else "weak", "weights": weights})
            fig = e.as_pyplot_figure()
            fig.set_size_inches(7, 3.6)
            plt.title(f"LIME ({cname.replace('_', ' ')}): {s['tier']} example, score {score:.0%}, R²={r2:.2f}", fontsize=9)
            plt.tight_layout()
            fig.savefig(FIG / f"lime_{cname}_{s['tier']}.png", dpi=150)
            plt.close("all")
    return out


def counterfactual(model, features: dict, policy: dict, threshold: float):
    """Lowest-cost actionable change that drives the ADOPTED policy's flag probability to 0."""
    feats = config.FEATURES
    g = str(features["age_band"])
    base = pd.DataFrame([features])[feats]
    p0 = float(model.predict_proba(base)[0, 1])
    pf0 = flag_probability(p0, g, policy)
    lead0, sms0 = int(features["lead_time_days"]), int(features["SMS_received"])
    grid = sorted({x for x in [0, 1, 2, 3, 5, 7, 14, 21, 28] if x < lead0} | {lead0})
    rows, meta = [], []
    for lead in grid:
        for sms in sorted({sms0, 1}):
            rows.append(base.assign(lead_time_days=lead, SMS_received=sms))
            meta.append((lead, sms))
    scores = model.predict_proba(pd.concat(rows, ignore_index=True))[:, 1]
    options = []
    for (lead, sms), s in zip(meta, scores):
        options.append({"lead_time_days": lead, "SMS_received": sms, "score": float(s),
                        "policy_flag_probability": flag_probability(float(s), g, policy),
                        "single_cutoff_flag": bool(s >= threshold),
                        "cost": round((lead0 - lead) / 7 + (sms != sms0), 3)})
    changes = [o for o in options if o["cost"] > 0]
    hits = sorted([o for o in changes if o["policy_flag_probability"] == 0.0], key=lambda o: (o["cost"], o["score"]))
    lowest = min(options, key=lambda o: (o["policy_flag_probability"], o["score"]))
    return {"original_score": p0, "original_policy_flag_probability": pf0, "age_band": g,
            "single_cutoff_flag": p0 >= threshold, "applicable": pf0 > 0,
            "counterfactual": hits[0] if (hits and pf0 > 0) else None,
            "lowest_reachable": lowest, "n_options": len(options),
            "levers": "earlier slot (lead time can only fall); add SMS reminder",
            "immutable": ["Age", "Gender", "Neighbourhood", "prior history", "health flags", "Scholarship"]}


def main():
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    A = json.load(open(config.ARTIFACTS / "analysis.json"))
    T = json.load(open(config.ARTIFACTS / "operating_point.json"))["flag_threshold"]
    policy = load_policy()
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    examples = A["samples_documented_cycle1"] + A["samples"]
    out = {"model_version": config.MODEL_VERSION, "flag_threshold": T, "policy": policy["name"],
           "fidelity_bar": FIDELITY_BAR, "lime": lime_explanations(m, tr, examples),
           "counterfactuals": [{"example": s["tier"], "score_tier": s["score_tier"], "test_row": s["test_row"],
                                **counterfactual(m, s["features"], policy, T)} for s in A["samples"]]}
    # population view under the adopted policy, on a fixed sample of test appointments with any chance of a flag
    p = m.predict_proba(te[config.FEATURES])[:, 1]
    pf = np.array([flag_probability(s, g, policy) for s, g in zip(p, te["age_band"].astype(str))])
    cand = te[pf > 0].sample(1500, random_state=config.RANDOM_STATE)
    res = [counterfactual(m, r, policy, T) for r in cand[config.FEATURES].to_dict("records")]
    lowered = [r["lowest_reachable"]["policy_flag_probability"] < r["original_policy_flag_probability"] for r in res]
    out["population"] = {"sampled_with_flag_probability_above_0": len(res),
                         "share_with_any_flag_probability_above_0_in_test": float((pf > 0).mean()),
                         "share_unflaggable_by_actionable_change": float(np.mean([r["counterfactual"] is not None for r in res])),
                         "share_flag_probability_reducible": float(np.mean(lowered)),
                         "share_already_same_day_booking": float(np.mean([r["lowest_reachable"]["lead_time_days"] == 0 and
                                                                          int(c["lead_time_days"]) == 0 for r, c in
                                                                          zip(res, cand[config.FEATURES].to_dict("records"))]))}
    json.dump(out, open(config.ARTIFACTS / "explanations.json", "w"), indent=1, default=float)
    for l in out["lime"]:
        print("LIME", l["config"], l["example"], round(l["model_score"], 3), round(l["lime_local_prediction"], 3),
              round(l["lime_r2"], 3), l["lime_top3"], l["shap_top3"], l["fidelity"])
    for c in out["counterfactuals"]:
        print("CF", c["example"], round(c["original_score"], 3), round(c["original_policy_flag_probability"], 3), c["counterfactual"])
    print(out["population"])


if __name__ == "__main__":
    main()
