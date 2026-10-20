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
                                    ("B_equalized_odds", "**B: equalized odds (configured, not approved; expected values)**"), ("C_demographic_parity", "C: demographic parity")])
    sp = b["realisation_spread"]["age_band"]
    over = [n for n, k in [("age", "age_band"), ("gender", "Gender"), ("scholarship", "Scholarship")]
            if b[k]["eod"] > config.FAIRNESS_LIMIT]
    b_text = ("all three EXPECTED gaps are at or below the limit" if not over
              else f"the {', '.join(over)} gap(s) remain above the limit, so mitigation fell short there")
    PC = json.load(open(A / "policy_comparison.json"))
    X = json.load(open(A / "explanations.json"))
    MON = json.load(open(A / "monitoring" / "monitoring.json"))
    FR = MON["fairness_rule"]
    pcomp = "\n".join(f"| {k.split('_', 1)[1].replace('_', ' ')} | {v['share_decisions_randomised']:.0%} | {v['recall']:.3f} | {v['precision']:.3f} | "
                      f"{v['f1']:.3f} | {v['flag_rate']:.1%} | {v['age_band']['eod']:.3f} ({v['age_band']['eod_bootstrap_95ci'][0]:.3f}-"
                      f"{v['age_band']['eod_bootstrap_95ci'][1]:.3f}) | {v['age_band']['tpr_gap']:.3f} | {v['age_band']['fpr_gap']:.3f} | "
                      f"{v['Gender']['eod']:.3f} | {v['Scholarship']['eod']:.3f} |" for k, v in PC["policies"].items())
    pnote = PC["note"]
    loc = [e for e in X["lime"] if e["config"] == "local_sampling"]
    old = [e for e in X["lime"] if e["config"] != "local_sampling"]
    lime_txt = (f"  Cycle-1 sampling: {sum(e['fidelity'] == 'adequate' for e in old)}/{len(old)} adequate; "
                f"local sampling: {sum(e['fidelity'] == 'adequate' for e in loc)}/{len(loc)} adequate. Per example (local sampling): "
                + "; ".join(f"{e['example']} R² {e['lime_r2']:.2f}, error {e['abs_error']:.3f}, top-3 overlap with SHAP {e['top3_overlap']}/3" for e in loc)
                + ". Where LIME is weak it is reported as a limitation, not tuned further.")
    cf_txt = "\n".join(f"  - {c['example']} (score {c['original_score']:.3f}, policy flag probability {c['original_policy_flag_probability']:.2f}): "
                       + (f"book {c['counterfactual']['lead_time_days']} days ahead" + (" + SMS" if c['counterfactual']['SMS_received'] else "")
                          + f" -> score {c['counterfactual']['score']:.3f}, flag probability 0" if c["counterfactual"] else
                          ("not flagged, nothing to change" if not c["applicable"] else
                           f"no allowed change removes the flag (lowest reachable probability {c['lowest_reachable']['policy_flag_probability']:.2f})"))
                       for c in X["counterfactuals"])
    mech = PC["option_B_mechanism"]
    mech_groups = ", ".join(f"{g} {r['risk_unrelated_flag_probability']:.3f}" for g, r in sorted(mech["groups"].items()))
    eo = PC["policies"]["P1_equal_opportunity_tpr85"]
    S = MON["summary"]
    from collections import Counter
    cnt = Counter(r["fairness"]["status"] for r in MON["rolling_28d"])
    mon_txt = (f"{len(MON['rolling_28d'])} rolling windows -> " + ", ".join(f"{v} {k}" for k, v in cnt.items())
               + "; stress tests -> " + "; ".join(f"{t['name'].split(':')[0]}: {t['fairness']['status']}" for t in MON["stress_tests"]) + ".")
    n_halt1 = sum(w["status"]["bias_drift"] == "halt" for w in MON["weeks_cycle1_rule"])
    R4, S4 = MON["module4_rule"], S["module4_s7_weekly"]
    m4_counts = ", ".join(f"{v} week(s) {k}" for k, v in S4["counts"].items())
    m4_weeks = "Weeks: " + "; ".join(
        f"{w['start']} to {w['end']} {w['module4_rule']['status']} (age "
        + ("n/a" if w["module4_rule"]["attributes"]["age_band"]["disparity"] is None else f"{w['module4_rule']['attributes']['age_band']['disparity']:.3f}")
        + ", gender " + ("n/a" if w["module4_rule"]["attributes"]["Gender"]["disparity"] is None else f"{w['module4_rule']['attributes']['Gender']['disparity']:.3f}")
        + ")" for w in MON["weeks_cycle2_rule"]) + "."
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
Charter Principle 2. Option B is the configured policy, pending an Ethics Committee decision: {b_text}, at the cost of flagging
{pct(b['flag_rate'])} of appointments instead of {pct(s['flag_rate'])}. {pct(P['verification']['share_randomised'])} of test
decisions are randomised within an age group's threshold band; the draw is reproducible from the request ID.
Option B figures are **expected values** over that randomisation. Any one set of decisions scatters around them: over
{sp['n_seeds']} simulated draws the realised age-group gap ranged {sp['p05']:.3f}-{sp['p95']:.3f} (5th-95th percentile) and exceeded
0.05 in {pct(sp['share_above_limit'])} of draws. Option B therefore meets the limit on average, **not reliably in any one set of
decisions**, and fairness is not "solved" by it.

### Decision policies compared (same test set; randomised rules shown as expected values)
| Policy | Randomised | Recall | Precision | F1 | Flagged | Age EOD (95% range) | Age TPR gap | Age FPR gap | Gender EOD | Scholarship EOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{pcomp}

{pnote}

**No evaluated policy reliably met the 0.05 equalized-odds limit across all required groups** (age band, gender, scholarship as
the G3 stand-in), judged against the original acceptance criteria C1-C8 (see `docs/committee_decision_pack.md`). Option B meets
C4-C7 in expectation but fails C8 (reliability) and C3 (F1, blocking).

**Option B creates flags unrelated to predicted risk.** To give every age group the same error rates, Fairlearn's equalized-odds
post-processing pulls groups the model predicts well down to the lowest group's ROC curve by ignoring the score with probability
p_ignore and flagging at the common false-positive rate ({mech['x_best_common_false_positive_rate']:.3f}). The resulting score-independent flag probability is
{mech_groups}. Overall {mech['share_of_flags_unrelated_to_risk']:.1%} of Option B's expected flags are unrelated to predicted risk and
{mech['share_of_appointments_with_nonzero_flag_probability']:.1%} of appointments have a non-zero flag probability.

**Equal opportunity is not equalized odds.** The deterministic alternative (equal opportunity at TPR 0.85) equalises only the
share of real no-shows flagged (age TPR gap {eo['age_band']['tpr_gap']:.3f}); its false-positive gap is {eo['age_band']['fpr_gap']:.3f}, so its equalized-odds
gaps are age {eo['age_band']['eod']:.3f} ({eo['age_band']['eod_bootstrap_95ci'][0]:.3f}-{eo['age_band']['eod_bootstrap_95ci'][1]:.3f}), gender {eo['Gender']['eod']:.3f}
({eo['Gender']['eod_bootstrap_95ci'][0]:.3f}-{eo['Gender']['eod_bootstrap_95ci'][1]:.3f}), scholarship {eo['Scholarship']['eod']:.3f} ({eo['Scholarship']['eod_bootstrap_95ci'][0]:.3f}-{eo['Scholarship']['eod_bootstrap_95ci'][1]:.3f}).
It is offered for committee consideration only and does not meet the full fairness requirement.
Age groups have different true no-show rates, so equal flag rates (demographic parity) are neither required nor expected.

## Promotion gate (automated; promotion itself is a human decision)
| Check | Value | Result |
| --- | --- | --- |
{gate}

**Decision:** {G['decision']}. Original criteria not met: {'; '.join(G['original_criteria_not_met'])}.

**Recommendation:** {G['recommendation']}. Option B is configured for evaluation only and is not approved for operational use;
deterministic equal opportunity at TPR 0.85 is an alternative assessed, not a compliant solution. C3 (F1 >= 0.55) is blocking by
reviewer decision: Module 4 s.1 calls it part of "the acceptance threshold" but the s.7 staging list omits it.
**Deployment status:** {G['deployment_status']}.

## Explanations
- **SHAP (explanation of record), with limits:** exact TreeSHAP for the XGBoost score (contributions sum to the log-odds; checked).
  It explains the score, not the policy decision (Option B's random and per-group components are outside it); contributions are
  associations, not causes; correlated inputs (lead time, SMS) can share credit misleadingly; the summary-plot colour scale is not
  meaningful for categorical inputs.
- **LIME (cross-check only):** fidelity bar fixed before the run (R² >= {X['fidelity_bar']['min_r2']}, |error| <= {X['fidelity_bar']['max_abs_error']}).
{lime_txt}
- **Counterfactuals** target the configured Option B policy (flag probability 0), using only an earlier slot or an added reminder:
{cf_txt}

## Limitations
- Trained on Brazilian 2016 public-health data covering about six weeks; CrescentCare's rate (28%) differs from the data's ({pct(R['pos_rate'])}).
- No insurance field, so Module 1 goal G3 (Medicaid vs private gap) cannot be measured; scholarship is a rough stand-in.
- Many flags are false alarms: precision is {op['precision']:.4f} for the XGBoost score at cut-off {op['threshold']:.2f} (computed from 0/1 decisions),
  and {b['precision']:.4f} expected under Option B (expected true positives / expected flags, from flag probabilities).
- Transport, childcare, cost and health-literacy barriers are not in the data.
- Randomised decisions under Option B must be explained to patients and staff.

## Monitoring and retraining
PSI on inputs and scores (warn >= 0.10, alert >= 0.20); AUC alert if more than 0.03 below validation.

**Fairness: Module 4 s.7 weekly rule (governing).** Each calendar week, the realised error-rate disparity (larger of the FNR and FPR
gaps) for Gender and age_band; alert above {R4['alert_above']:.2f}, halt above {R4['halt_above']:.2f}; no minimum group size; an attribute with fewer than
two comparable groups is 'not evaluable' (never a pass). Replay: **governing status {S4['governing_status'].upper()}** - {m4_counts}.
By attribute: {'; '.join(f"{a} - " + ', '.join(f"{k} {v}" for k, v in c.items() if v) for a, c in S4['attribute_status_counts'].items())}.
{len([c for c in S4['coverage'] if c['partial_week']])} of {len(S4['coverage'])} weeks are partial (fewer than 5 appointment days); the smallest weekly age group had
{min(c['smallest_age_group_no_shows'] for c in S4['coverage'])}-{max(c['smallest_age_group_no_shows'] for c in S4['coverage'])} no-shows.
{m4_weeks}

**Supplementary uncertainty analysis (does not replace the s.7 rule):** {FR['window']}: a group is evaluated only with >= {FR['min_no_shows_per_group']} no-shows and >= {FR['min_attended_per_group']} attended visits;
the age-group gap gets a bootstrap 95% interval; halt only if the whole interval is above 0.05, ok only if it is wholly at or below,
otherwise inconclusive; too few eligible groups = insufficient data (never read as fair or unfair).
Replay result: {mon_txt}
Under Option B the expected age-group gap exceeds 0.05 in every monitored time window used by the monitoring rule: all
{S['rolling_28d_all']['windows_with_expected_gap_evaluable']} rolling 28-day windows ({S['rolling_28d_all']['expected_gap_range'][0]:.3f}-{S['rolling_28d_all']['expected_gap_range'][1]:.3f}) and both evaluable
calendar months ({S['calendar_months']['expected_gap_range'][0]:.3f}-{S['calendar_months']['expected_gap_range'][1]:.3f}); in weekly windows it exceeds 0.05 in
{S['weeks_cycle2_rule']['windows_with_expected_gap_above_0.05']} of {S['weeks_cycle2_rule']['windows_with_expected_gap_evaluable']} evaluable weeks. Outcomes are reported separately - rolling windows:
{', '.join(f"{k} {v}" for k, v in S['rolling_28d_all']['counts'].items())}; months: {', '.join(f"{k} {v}" for k, v in S['calendar_months']['counts'].items())}.
Insufficient data and unevaluated groups are never counted as a pass.
The cycle-1 weekly rule (halt if the point estimate > 0.05) halted in {n_halt1} of {len(MON['weeks_cycle1_rule'])} weeks; it is kept in the
replay output for audit. Retraining runs `python -m careconnect.run_all` and must pass the gate
above and Ethics Committee review before the MLflow `production` alias moves. Previous version kept for rollback.
"""
    (config.DOCS).mkdir(exist_ok=True)
    (config.DOCS / "model_card.md").write_text(md)
    print("docs/model_card.md written")


if __name__ == "__main__":
    main()
