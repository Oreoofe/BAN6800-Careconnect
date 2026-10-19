# Model Card - CareConnect no-show risk model v2.0.0

*Generated 2026-10-19 from the build artifacts by `careconnect/governance/model_card.py`. Do not edit numbers by hand.*

## Model details
- **Owner:** CareConnect Data Science Team (project lead: Gladys Izuagbe). **Approver:** AI Ethics Committee (binding veto).
- **Type:** XGBoost gradient-boosted trees inside a scikit-learn pipeline (scaling + one-hot encoding), followed by an
  equalized-odds post-processing rule by age group (Fairlearn ThresholdOptimizer, exported to `artifacts/fairness_policy.json`).
- **Version:** 2.0.0 (SemVer). v1.x was the Module 4/5 model, which used prior-attendance features that leaked
  outcomes not yet known at booking time; v2.0.0 corrects this.
- **Registry:** MLflow model `careconnect_noshow_xgb`, alias `candidate`, tag `approval = pending Ethics Committee review`.
- **Best hyper-parameters:** {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 300, 'scale_pos_weight': 1} (5-fold CV ROC-AUC 0.743).

## Intended use
Rank booked appointments by the chance of a missed visit so care coordinators can **offer support** (reminder call,
transport voucher, earlier slot). Decision support only; a coordinator decides every action.

## Out-of-scope uses (prohibited)
Denying, delaying or restricting care; overbooking a named patient's slot; penalties or charges; any automated
contact or scheduling; use on populations it has not been validated on (including CrescentCare, until local validation).

## Training and evaluation data
Kaggle "Medical Appointment No Shows" (public-health appointments, Vitória, Brazil, 29 Apr-8 Jun 2016).
110,527 records -> 110,514 after cleaning (8 invalid ages,
5 bookings after the visit). Stratified 80/20 split, seed 42:
88,411 training / 22,103 test appointments. No-show rate 20.2%.

## Performance (unseen test set, cut-off chosen on training data for recall >= 0.65)
| Model | ROC-AUC | Recall | Precision | F1 | Cut-off |
| --- | --- | --- | --- | --- | --- |
| logistic regression | 0.680 | 0.652 | 0.305 | 0.416 | 0.4800 |
| random forest | 0.743 | 0.649 | 0.345 | 0.450 | 0.2550 |
| xgboost | 0.746 | 0.656 | 0.343 | 0.450 | 0.2500 |

Baselines: majority class AUC 0.500, recall 0; stratified random AUC 0.504.
XGBoost PR-AUC 0.399; Brier score 0.1412. At the default 0.5 cut-off recall would be only 6.5%.
Robustness: patient-grouped split AUC 0.746; without protected/sensitive inputs AUC 0.742.

**Main drivers (SHAP share of influence):** lead_time_days (56%), Age (13%), Neighbourhood (7%), SMS_received (4%). These are associations, not causes.

## Fairness (unseen test set; limit 0.05 equalized-odds gap)
| Rule | Recall | Share flagged | Age gap | Gender gap | Scholarship gap |
| --- | --- | --- | --- | --- | --- |
| Single cut-off | 0.656 | 0.386 | 0.629 | 0.017 | 0.209 |
| A: cut-off per age group | 0.670 | 0.414 | 0.077 | 0.041 | 0.146 |
| **B: equalized odds (adopted)** | 0.852 | 0.607 | 0.041 | 0.031 | 0.029 |
| C: demographic parity | 0.878 | 0.596 | 0.063 | 0.037 | 0.048 |

Under the single cut-off the model flags older patients far less often (age gap 0.629), breaching
Charter Principle 2. Option B is adopted: all three gaps are at or below the limit, at the cost of flagging
60.7% of appointments instead of 38.6%. 81.6% of test
decisions are randomised within an age group's threshold band; the draw is reproducible from the request ID.
Option B figures are **expected values** over that randomisation. Any one set of decisions scatters around them: over
200 simulated draws the realised age-group gap ranged 0.029-0.070 (5th-95th percentile) and exceeded
0.05 in 42.0% of draws, so monthly audits will sometimes show breaches that are chance, not drift.
Age groups have different true no-show rates, so equal flag rates (demographic parity) are neither required nor expected.

## Promotion gate (automated; promotion itself is a human decision)
| Check | Value | Result |
| --- | --- | --- |
| data contract passed | True | pass |
| representation-bias gate passed | True | pass |
| ROC-AUC >= 0.70 | 0.7455 | pass |
| recall >= 0.65 (single cut-off) | 0.6558 | pass |
| F1 >= 0.55 | 0.4502 | FAIL |
| adopted policy: EXPECTED age-group equalized-odds gap <= 0.05 | 0.0408 | pass |
| adopted policy: EXPECTED gender gap <= 0.05 | 0.0312 | pass |
| adopted policy: EXPECTED scholarship gap <= 0.05 | 0.0293 | pass |
| adopted policy: 95th percentile of realised age gap (200 draws) <= 0.05 | 0.0697 | FAIL |
| patient-grouped AUC within 0.02 of random-split AUC | 0.7463 | pass |

**Decision:** eligible for Ethics Committee review.

## Limitations
- Trained on Brazilian 2016 public-health data covering about six weeks; CrescentCare's rate (28%) differs from the data's (20.2%).
- No insurance field, so Module 1 goal G3 (Medicaid vs private gap) cannot be measured; scholarship is a rough stand-in.
- Many flags are false alarms (precision 34% at the single cut-off; lower under Option B).
- Transport, childcare, cost and health-literacy barriers are not in the data.
- Randomised decisions under Option B must be explained to patients and staff.

## Monitoring and retraining
Monthly PSI on inputs and scores (alert >= 0.20), weekly equalized-odds gap by age (warn > 0.04, halt > 0.05),
AUC alert if more than 0.03 below validation. Retraining runs `python -m careconnect.run_all` and must pass the gate
above and Ethics Committee review before the MLflow `production` alias moves. Previous version kept for rollback.
