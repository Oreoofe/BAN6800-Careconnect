"""Render docs/model_card.md from the artifacts of the latest run (Mitchell et al., 2019 format).

Every number in the card is read from artifacts/*.json, so the card cannot drift from the model.
"""
import json
from datetime import date

from careconnect import config


def pct(x):
    return f"{x:.1%}"


def main():
    A = config.ARTIFACTS
    R = json.load(open(A / "model_results.json"))
    M = json.load(open(A / "mitigation.json"))
    E = json.load(open(A / "analysis.json"))
    P = json.load(open(A / "fairness_policy.json"))
    G = json.load(open(A / "promotion_decision.json"))
    Q = json.load(open(A / "pipeline" / "data_quality_report.json"))
    x = R["models"]["xgboost"]
    op = x["at_recall_target"]
    b, s = M["B_equalized_odds"], M["before"]
    rob, abl = R["robustness_patient_grouped"], R["ablation_no_sensitive_inputs"]
    share = E["shap"]["share"]
    top = ", ".join(f"{k} ({v:.0%})" for k, v in list(share.items())[:4])
    rows = "\n".join(f"| {n.replace('_', ' ')} | {v['at_recall_target']['roc_auc']:.3f} | {v['at_recall_target']['recall']:.3f} | "
                     f"{v['at_recall_target']['precision']:.3f} | {v['at_recall_target']['f1']:.3f} | {v['at_recall_target']['threshold']:.4f} |"
                     for n, v in R["models"].items())
    fair = "\n".join(f"| {lbl} | {M[k]['recall']:.3f} | {M[k]['flag_rate']:.3f} | {M[k]['age_band']['eod']:.3f} | "
                     f"{M[k]['Gender']['eod']:.3f} | {M[k]['Scholarship']['eod']:.3f} |"
                     for k, lbl in [("before", "Single cut-off"), ("A_group_thresholds", "A: cut-off per age group"),
                                    ("B_equalized_odds", "**B: equalized odds (adopted)**"), ("C_demographic_parity", "C: demographic parity")])
    over = [n for n, k in [("age", "age_band"), ("gender", "Gender"), ("scholarship", "Scholarship")]
            if b[k]["eod"] > config.FAIRNESS_LIMIT]
    b_text = ("all three gaps are at or below the limit" if not over
              else f"the {', '.join(over)} gap(s) remain above the limit, so mitigation fell short there")
    gate = "\n".join(f"| {c['check']} | {c['value']} | {'pass' if c['passed'] else 'FAIL'} |" for c in G["checks"])
    md = f"""# Model Card - CareConnect no-show risk model v{config.MODEL_VERSION}

*Generated {date.today().isoformat()} from the build artifacts by `careconnect/governance/model_card.py`. Do not edit numbers by hand.*

## Model details
- **Owner:** CareConnect Data Science Team (project lead: Gladys Izuagbe). **Approver:** AI Ethics Committee (binding veto).
- **Type:** XGBoost gradient-boosted trees inside a scikit-learn pipeline (scaling + one-hot encoding), followed by an
  equalized-odds post-processing rule by age group (Fairlearn ThresholdOptimizer, exported to `artifacts/fairness_policy.json`).
- **Version:** {config.MODEL_VERSION} (SemVer). v1.x was the Module 4/5 model, which used prior-attendance features that leaked
  outcomes not yet known at booking time; v2.0.0 corrects this.
- **Registry:** MLflow model `{config.REGISTERED_MODEL}`, alias `candidate`, tag `approval = pending Ethics Committee review`.
- **Best hyper-parameters:** {x['best_params']} ({R['cv_folds']}-fold CV ROC-AUC {x['cv_auc']:.3f}).

## Intended use
Rank booked appointments by the chance of a missed visit so care coordinators can **offer support** (reminder call,
transport voucher, earlier slot). Decision support only; a coordinator decides every action.

## Out-of-scope uses (prohibited)
Denying, delaying or restricting care; overbooking a named patient's slot; penalties or charges; any automated
contact or scheduling; use on populations it has not been validated on (including CrescentCare, until local validation).

## Training and evaluation data
Kaggle "Medical Appointment No Shows" (public-health appointments, Vitória, Brazil, 29 Apr-8 Jun 2016).
{Q['cleaning']['rows_in']:,} records -> {Q['cleaning']['rows_out']:,} after cleaning ({Q['cleaning']['invalid_age_removed']} invalid ages,
{Q['cleaning']['scheduled_after_appointment_removed']} bookings after the visit). Stratified 80/20 split, seed 42:
{R['n_train']:,} training / {R['n_test']:,} test appointments. No-show rate {pct(R['pos_rate'])}.

## Performance (unseen test set, cut-off chosen on training data for recall >= 0.65)
| Model | ROC-AUC | Recall | Precision | F1 | Cut-off |
| --- | --- | --- | --- | --- | --- |
{rows}

Baselines: majority class AUC 0.500, recall 0; stratified random AUC {R['baselines']['stratified']['roc_auc']:.3f}.
XGBoost PR-AUC {op['pr_auc']:.3f}; Brier score {op['brier']:.4f}. At the default 0.5 cut-off recall would be only {x['at_default_0_5']['recall']:.1%}.
Robustness: patient-grouped split AUC {rob['roc_auc']:.3f}; without protected/sensitive inputs AUC {abl['roc_auc']:.3f}.

**Main drivers (SHAP share of influence):** {top}. These are associations, not causes.

## Fairness (unseen test set; limit 0.05 equalized-odds gap)
| Rule | Recall | Share flagged | Age gap | Gender gap | Scholarship gap |
| --- | --- | --- | --- | --- | --- |
{fair}

Under the single cut-off the model flags older patients far less often (age gap {s['age_band']['eod']:.3f}), breaching
Charter Principle 2. Option B is adopted: {b_text}, at the cost of flagging
{pct(b['flag_rate'])} of appointments instead of {pct(s['flag_rate'])}. {pct(P['verification']['share_randomised'])} of test
decisions are randomised within an age group's threshold band; the draw is reproducible from the request ID.
Age groups have different true no-show rates, so equal flag rates (demographic parity) are neither required nor expected.

## Promotion gate (automated; promotion itself is a human decision)
| Check | Value | Result |
| --- | --- | --- |
{gate}

**Decision:** {G['decision']}.

## Limitations
- Trained on Brazilian 2016 public-health data covering about six weeks; CrescentCare's rate (28%) differs from the data's ({pct(R['pos_rate'])}).
- No insurance field, so Module 1 goal G3 (Medicaid vs private gap) cannot be measured; scholarship is a rough stand-in.
- Many flags are false alarms (precision {op['precision']:.0%} at the single cut-off; lower under Option B).
- Transport, childcare, cost and health-literacy barriers are not in the data.
- Randomised decisions under Option B must be explained to patients and staff.

## Monitoring and retraining
Monthly PSI on inputs and scores (alert >= 0.20), weekly equalized-odds gap by age (warn > 0.04, halt > 0.05),
AUC alert if more than 0.03 below validation. Retraining runs `python -m careconnect.run_all` and must pass the gate
above and Ethics Committee review before the MLflow `production` alias moves. Previous version kept for rollback.
"""
    (config.DOCS).mkdir(exist_ok=True)
    (config.DOCS / "model_card.md").write_text(md)
    print("docs/model_card.md written")


if __name__ == "__main__":
    main()
