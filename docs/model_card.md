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
Charter Principle 2. Option B is the configured policy, pending an Ethics Committee decision: all three EXPECTED gaps are at or below the limit, at the cost of flagging
60.7% of appointments instead of 38.6%. 81.6% of test
decisions are randomised within an age group's threshold band; the draw is reproducible from the request ID.
Option B figures are **expected values** over that randomisation. Any one set of decisions scatters around them: over
200 simulated draws the realised age-group gap ranged 0.029-0.070 (5th-95th percentile) and exceeded
0.05 in 42.0% of draws. Option B therefore meets the limit on average, **not reliably in any one set of
decisions**, and fairness is not "solved" by it.

### Decision policies compared (same test set; randomised rules shown as expected values)
| Policy | Randomised | Recall | Precision | F1 | Flagged | Age EOD (95% range) | Age TPR gap | Age FPR gap | Gender EOD | Scholarship EOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single cutoff | 0% | 0.656 | 0.343 | 0.450 | 38.6% | 0.629 (0.587-0.675) | 0.629 | 0.427 | 0.017 | 0.209 |
| equal opportunity tpr65 | 0% | 0.664 | 0.327 | 0.439 | 41.0% | 0.064 (0.059-0.110) | 0.040 | 0.064 | 0.040 | 0.152 |
| equal opportunity tpr75 | 0% | 0.768 | 0.314 | 0.446 | 49.4% | 0.083 (0.064-0.136) | 0.057 | 0.083 | 0.046 | 0.112 |
| equal opportunity tpr85 | 0% | 0.864 | 0.300 | 0.446 | 58.1% | 0.100 (0.076-0.138) | 0.060 | 0.100 | 0.043 | 0.054 |
| minimax per group | 0% | 0.782 | 0.313 | 0.447 | 50.5% | 0.089 (0.077-0.157) | 0.089 | 0.079 | 0.037 | 0.108 |
| tpr parity randomised | 2% | 0.900 | 0.294 | 0.444 | 61.7% | 0.091 (0.075-0.135) | 0.030 | 0.091 | 0.044 | 0.035 |
| option B equalized odds | 82% | 0.852 | 0.283 | 0.425 | 60.7% | 0.041 (0.037-0.115) | 0.041 | 0.025 | 0.031 | 0.029 |
| unaware model no age | 0% | 0.659 | 0.334 | 0.443 | 39.9% | 0.254 (0.216-0.331) | 0.254 | 0.174 | 0.022 | 0.196 |

Deterministic rules can equalise one error rate per group (TPR or FPR) but generally not both, because the groups' ROC curves differ; Fairlearn reaches equal TPR and FPR by randomising between two cut-offs. The bootstrap intervals show how precisely a 22,103-row test set can measure a 0.05 gap for groups with a few hundred no-shows.
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

## Explanations
- **SHAP (explanation of record):** exact TreeSHAP from XGBoost; contributions sum to the model's log-odds (checked in verification).
- **LIME (cross-check only):** fidelity bar fixed before the run (R² >= 0.5, |error| <= 0.05).
  Cycle-1 sampling: 2/6 adequate; local sampling: 0/6 adequate. Per example (local sampling): cycle1_high R² 0.41, error 0.296, top-3 overlap with SHAP 2/3; cycle1_medium R² 0.47, error 0.098, top-3 overlap with SHAP 2/3; cycle1_low R² 0.52, error 0.158, top-3 overlap with SHAP 3/3; high R² 0.58, error 0.157, top-3 overlap with SHAP 1/3; medium R² 0.22, error 0.027, top-3 overlap with SHAP 2/3; low R² 0.63, error 0.071, top-3 overlap with SHAP 2/3. Where LIME is weak it is reported as a limitation, not tuned further.
- **Counterfactuals** target the adopted policy (flag probability 0), using only an earlier slot or an added reminder:
  - high (score 0.452, policy flag probability 0.90): no allowed change removes the flag (lowest reachable probability 0.11)
  - medium (score 0.314, policy flag probability 0.92): no allowed change removes the flag (lowest reachable probability 0.10)
  - low (score 0.066, policy flag probability 0.10): no allowed change removes the flag (lowest reachable probability 0.10)

## Limitations
- Trained on Brazilian 2016 public-health data covering about six weeks; CrescentCare's rate (28%) differs from the data's (20.2%).
- No insurance field, so Module 1 goal G3 (Medicaid vs private gap) cannot be measured; scholarship is a rough stand-in.
- Many flags are false alarms (precision 34% at the single cut-off; lower under Option B).
- Transport, childcare, cost and health-literacy barriers are not in the data.
- Randomised decisions under Option B must be explained to patients and staff.

## Monitoring and retraining
PSI on inputs and scores (warn >= 0.10, alert >= 0.20); AUC alert if more than 0.03 below validation. Fairness uses
rolling 28 days, evaluated weekly: a group is evaluated only with >= 50 no-shows and >= 50 attended visits;
the age-group gap gets a bootstrap 95% interval; halt only if the whole interval is above 0.05, ok only if it is wholly at or below,
otherwise inconclusive; too few eligible groups = insufficient data (never read as fair or unfair).
Replay result: 6 rolling windows -> 3 halt, 3 inconclusive; stress tests -> S1 data drift: halt; S2 bias drift: halt; S3 small group: inconclusive; S4 tiny batch: insufficient data.
The cycle-1 weekly rule (halt if the point estimate > 0.05) halted in 7 of 7 weeks; it is kept in the
replay output for audit. Retraining runs `python -m careconnect.run_all` and must pass the gate
above and Ethics Committee review before the MLflow `production` alias moves. Previous version kept for rollback.
