"""Group fairness metrics.

Definitions (Fairlearn conventions, Bird et al. 2020; Hardt et al. 2016):
* dpd  - demographic parity difference: max - min selection (flag) rate across groups
* eod  - equalized odds difference: max of the TPR gap and the FPR gap across groups
* di_ratio - disparate impact ratio: min / max selection rate (four-fifths rule: >= 0.80)

Implemented in numpy so the same code runs in the API, the dashboard and monitoring without
Fairlearn; `crosscheck_with_fairlearn` confirms equality with the library in CI.
"""
import numpy as np

from careconnect import config


def group_rates(y, yhat, groups):
    y, yhat, groups = np.asarray(y), np.asarray(yhat), np.asarray(groups)
    out = {}
    for g in sorted(np.unique(groups)):
        k = groups == g
        yt, yp = y[k], yhat[k]
        pos, neg = yt == 1, yt == 0
        tp, fp = int((yp[pos] == 1).sum()), int((yp[neg] == 1).sum())
        out[str(g)] = dict(n=int(k.sum()), no_show_rate=float(yt.mean()), flag_rate=float(yp.mean()),
                           tpr=float(tp / pos.sum()) if pos.sum() else None,
                           fpr=float(fp / neg.sum()) if neg.sum() else None,
                           precision=float(tp / (tp + fp)) if tp + fp else None)
    return out


def fairness_report(y, yhat, groups, limit=config.FAIRNESS_LIMIT):
    g = group_rates(y, yhat, groups)
    sel = [v["flag_rate"] for v in g.values()]
    tpr = [v["tpr"] for v in g.values() if v["tpr"] is not None]
    fpr = [v["fpr"] for v in g.values() if v["fpr"] is not None]
    r = dict(groups=g, dpd=float(max(sel) - min(sel)), tpr_gap=float(max(tpr) - min(tpr)),
             fpr_gap=float(max(fpr) - min(fpr)), di_ratio=float(min(sel) / max(sel)) if max(sel) else None,
             base_rate_gap=float(max(v["no_show_rate"] for v in g.values()) - min(v["no_show_rate"] for v in g.values())))
    r["eod"] = max(r["tpr_gap"], r["fpr_gap"])
    r["pass_dpd"] = r["dpd"] <= limit
    r["pass_eod"] = r["eod"] <= limit
    r["pass_di"] = r["di_ratio"] is not None and r["di_ratio"] >= config.DI_MIN
    return r


def crosscheck_with_fairlearn(y, yhat, groups, tol=1e-9) -> dict:
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
    ours = fairness_report(y, yhat, groups)
    lib = dict(dpd=demographic_parity_difference(y, yhat, sensitive_features=groups),
               eod=equalized_odds_difference(y, yhat, sensitive_features=groups),
               di_ratio=demographic_parity_ratio(y, yhat, sensitive_features=groups))
    diffs = {k: abs(float(lib[k]) - float(ours[k])) for k in lib}
    return {"fairlearn": {k: float(v) for k, v in lib.items()}, "max_abs_diff": max(diffs.values()),
            "match": max(diffs.values()) <= tol}
