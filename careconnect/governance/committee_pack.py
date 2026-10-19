"""Render docs/committee_decision_pack.md: the evidence an AI Ethics Committee needs to decide on the outreach policy.

Every figure is read from the build artifacts (policy_comparison.json, monitoring.json, explanations.json,
promotion_decision.json, model_results.json), so the pack cannot drift from the evidence. It presents options and
trade-offs; it does not choose a policy.
"""
import hashlib
import json
import os
from datetime import date

from careconnect import config

A = config.ARTIFACTS
NAMES = {"P0_single_cutoff": "Single cut-off (no mitigation)", "P1_equal_opportunity_tpr65": "Equal opportunity, TPR 0.65",
         "P1_equal_opportunity_tpr75": "Equal opportunity, TPR 0.75", "P1_equal_opportunity_tpr85": "Equal opportunity, TPR 0.85",
         "P2_minimax_per_group": "Minimax per-group cut-offs", "P3_tpr_parity_randomised": "TPR parity (Fairlearn, randomised)",
         "P4_option_B_equalized_odds": "Option B: equalized odds (Fairlearn, randomised)",
         "P5_unaware_model_no_age": "Model without age inputs"}
CS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def rci(f):
    return "-" if not f["ci95"] else f"{f['ci95'][0]:.3f}-{f['ci95'][1]:.3f}"


def ci(v):
    return f"{v['eod_bootstrap_95ci'][0]:.3f}-{v['eod_bootstrap_95ci'][1]:.3f}"


def main():
    PC = json.load(open(A / "policy_comparison.json"))
    MON = json.load(open(A / "monitoring" / "monitoring.json"))
    X = json.load(open(A / "explanations.json"))
    G = json.load(open(A / "promotion_decision.json"))
    R = json.load(open(A / "model_results.json"))
    E = json.load(open(A / "analysis.json"))
    P = PC["policies"]
    mech = PC["option_B_mechanism"]
    B, EO = P["P4_option_B_equalized_odds"], P["P1_equal_opportunity_tpr85"]
    S = MON["summary"]
    xm = R["models"]["xgboost"]["at_recall_target"]

    crit = "\n".join(f"| {c['id']} | {c['criterion']} | {c['source']} |" for c in PC["criteria"])
    matrix = "\n".join(f"| {NAMES.get(k, k)} | " + " | ".join("pass" if v["acceptance"][c] else "**fail**" for c in CS) +
                       f" | {'yes' if v['acceptance']['meets_full_fairness_requirement'] else 'no'} |" for k, v in P.items())
    perf = "\n".join(f"| {NAMES.get(k, k)} | {v['share_decisions_randomised']:.0%} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} | "
                     f"{v['flag_rate']:.1%} | {v['age_band']['eod']:.3f} ({ci(v['age_band'])}) | {v['Gender']['eod']:.3f} ({ci(v['Gender'])}) | "
                     f"{v['Scholarship']['eod']:.3f} ({ci(v['Scholarship'])}) |" for k, v in P.items())
    mrows = "\n".join(f"| {g} | {r['n_test']:,} | {r['p_ignore']:.3f} | {r['risk_unrelated_flag_probability']:.3f} | "
                      f"{r['risk_unrelated_flag_probability'] * 1000:.0f} |" for g, r in sorted(mech["groups"].items()))

    def eo_rows(attr):
        return "\n".join(f"| {attr} | {g} | {r['n']:,} | {r['no_shows']:,} | {r['selection']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} |"
                         for g, r in EO[attr]["groups"].items())
    eo_groups = "\n".join(eo_rows(a) for a in ["age_band", "Gender", "Scholarship"])
    eo_gaps = "\n".join(f"| {a} | {EO[a]['tpr_gap']:.3f} | {EO[a]['fpr_gap']:.3f} | {EO[a]['eod']:.3f} ({ci(EO[a])}) | {EO[a]['dpd']:.3f} | {EO[a]['di_ratio']:.3f} |"
                        for a in ["age_band", "Gender", "Scholarship"])
    b_gaps = "\n".join(f"| {a} | {B[a]['eod']:.3f} | {ci(B[a])} | {B[a]['realised_eod']['p05']:.3f}-{B[a]['realised_eod']['p95']:.3f} | "
                       f"{B[a]['realised_eod']['share_above_0.05']:.1%} |" for a in ["age_band", "Gender", "Scholarship"])
    win = "\n".join(f"| {r['batch']} | {r['n']:,} | {f3(r['expected_age_eod_evaluated_groups'])} | {f3(r['fairness']['eod'])} | "
                    f"{rci(r['fairness'])} | "
                    f"{r['fairness']['status']} | {', '.join(r['fairness']['groups_not_evaluated']) or 'none'} |"
                    for r in MON["rolling_28d"] + MON["calendar_months"])
    week = "\n".join(f"| {a['batch']} | {a['age_eod']:.3f} / {a['status']['bias_drift']} | {f3(b['expected_age_eod_evaluated_groups'])} | "
                     f"{f3(b['fairness']['eod'])} | {b['fairness']['status']} | {', '.join(b['fairness']['groups_not_evaluated']) or 'none'} |"
                     for a, b in zip(MON["weeks_cycle1_rule"], MON["weeks_cycle2_rule"]))
    cnt = lambda key: ", ".join(f"{k}: {v}" for k, v in S[key]["counts"].items())
    loc = [e for e in X["lime"] if e["config"] == "local_sampling"]
    old = [e for e in X["lime"] if e["config"] != "local_sampling"]
    cfs = "\n".join(f"| {c['example']} ({c['age_band']}) | {c['original_score']:.3f} | {c['original_policy_flag_probability']:.3f} | "
                    + ("yes" if c["counterfactual"] else "no") + f" | {c['lowest_reachable']['policy_flag_probability']:.3f} |" for c in X["counterfactuals"])
    pop = X["population"]
    model_sha = hashlib.sha256((A / "xgboost.joblib").read_bytes()).hexdigest()
    policy_sha = hashlib.sha256((A / "fairness_policy.json").read_bytes()).hexdigest()

    md = f"""# Committee decision pack - CareConnect outreach policy

*Generated {date.today().isoformat()} by `careconnect/governance/committee_pack.py` from the build artifacts of commit
`{R.get('git_sha')}` (GitHub run {os.environ.get('GITHUB_RUN_ID', 'local')}). Model sha256 `{model_sha[:16]}...`, policy sha256
`{policy_sha[:16]}...`. Numbers are not edited by hand. This pack presents evidence and options; it does not select a policy.*

## 1. Decision required
Which outreach decision policy, if any, may be piloted with the no-show model (v{config.MODEL_VERSION}), given that **No evaluated
policy reliably met the 0.05 equalized-odds limit across all required groups** (age band, gender, and scholarship as the G3 stand-in).
The promotion gate result under the original criteria is: **{G['decision']}**.

## 2. Original acceptance criteria (as written before results; not changed)
| ID | Criterion | Source |
| --- | --- | --- |
{crit}

## 3. Every evaluated policy against those criteria (held-out test set, 22,103 appointments)
| Policy | {' | '.join(CS)} | Meets full fairness requirement (C4-C8) |
| --- | {' | '.join('---' for _ in CS)} | --- |
{matrix}

| Policy | Randomised | Recall | Precision | F1 | Flagged | Age EOD (95% CI) | Gender EOD (95% CI) | Scholarship EOD (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{perf}

Model: ROC-AUC {xm['roc_auc']:.4f} (C1 met). Randomised rules are reported as expected values. No policy meets C3 (F1 >= 0.55).

## 4. Option B (configured policy): what it achieves and what it costs
| Group | Expected EOD | 95% CI (test sample) | Realised EOD, 5th-95th pct of {B['age_band']['realised_eod']['n_draws']} draws | Draws above 0.05 |
| --- | --- | --- | --- | --- |
{b_gaps}

**Flags unrelated to predicted risk.** Option B gives some appointments a non-zero flag probability that does not depend on
their score. {mech['explanation']}

| Age group | Test appointments | p_ignore | Risk-unrelated flag probability (p_ignore x {mech['x_best_common_false_positive_rate']:.3f}) | Risk-unrelated flags per 1,000 |
| --- | --- | --- | --- | --- |
{mrows}

Consequences: {mech['share_of_flags_unrelated_to_risk']:.1%} of Option B's expected flags ({mech['expected_risk_unrelated_flags_per_1000']:.0f} of
{mech['expected_total_flags_per_1000']:.0f} per 1,000 appointments) are unrelated to predicted risk; {mech['share_of_appointments_with_nonzero_flag_probability']:.1%}
of appointments have some chance of being flagged; equal error rates are reached partly by making decisions *less* accurate for
the groups the model predicts best ("levelling down"); no permitted change (earlier slot, reminder) can remove a flag in those
groups, and SHAP reason codes do not explain flags that came from the random component.

Is the mechanism necessary? For **exact** equalized odds by post-processing a single score, yes: when the groups' ROC curves differ, a
common (FPR, TPR) point exists only on or below the lowest curve, and groups above it can only be moved there by randomisation
(Hardt, Price & Srebro, 2016). It is not necessary for **equal opportunity** (TPR parity only), which Fairlearn achieved with 2% of
decisions randomised, nor for deterministic per-group cut-offs. Relaxed (tolerance-based) equalized-odds variants were not evaluated.

## 5. Alternative for consideration: deterministic equal opportunity at TPR 0.85
*Equal opportunity* equalises only the true-positive rate (the share of real no-shows that are flagged) across groups.
*Equalized odds* - the Charter requirement - equalises both the true-positive rate and the false-positive rate. Equal opportunity
is therefore a weaker criterion and **does not meet the full fairness requirement** (it fails {', '.join(c for c in CS if not EO['acceptance'][c])}).

| Attribute | TPR gap | FPR gap | EOD (95% CI) | DPD | DI ratio |
| --- | --- | --- | --- | --- | --- |
{eo_gaps}

| Attribute | Group | Test appointments | No-shows | Flag rate | TPR | FPR |
| --- | --- | --- | --- | --- | --- | --- |
{eo_groups}

Against Option B: recall {EO['recall']:.3f} vs {B['recall']:.3f}, precision {EO['precision']:.3f} vs {B['precision']:.3f}, F1 {EO['f1']:.3f} vs
{B['f1']:.3f}, flagged {EO['flag_rate']:.1%} vs {B['flag_rate']:.1%}; no randomised decisions and no flags unrelated to risk; but its
age-group false-positive gap ({EO['age_band']['fpr_gap']:.3f}) and scholarship gap ({EO['Scholarship']['eod']:.3f}) exceed 0.05.

## 6. Monitoring evidence (replay of held-out appointments; not live data)
Rule: {MON['fairness_rule']['window']}; groups need >= {MON['fairness_rule']['min_no_shows_per_group']} no-shows and >= {MON['fairness_rule']['min_attended_per_group']}
attended visits; halt only if the whole 95% interval is above 0.05; ok only if wholly at or below; otherwise inconclusive;
fewer than two eligible groups = insufficient data. {S['reading']}

Outcomes, reported separately: rolling 28-day windows - {cnt('rolling_28d_all')} (full windows only - {cnt('rolling_28d_full_windows_only')});
calendar months - {cnt('calendar_months')}; weekly windows under the same rule - {cnt('weeks_cycle2_rule')}; the cycle-1 weekly rule halted in
{S['weeks_cycle1_rule_halts']} of {len(MON['weeks_cycle1_rule'])} weeks.

**Under Option B the expected age-group gap exceeds 0.05 in every monitored time window that could be evaluated**: all
{S['rolling_28d_all']['windows_with_expected_gap_evaluable']} rolling 28-day windows (range {S['rolling_28d_all']['expected_gap_range'][0]:.3f}-{S['rolling_28d_all']['expected_gap_range'][1]:.3f})
and both evaluable calendar months (range {S['calendar_months']['expected_gap_range'][0]:.3f}-{S['calendar_months']['expected_gap_range'][1]:.3f}); April 2016
had insufficient data. Option B was fitted on all training data pooled, and its expected-value guarantee does not carry over to
individual time periods. Weekly windows: expected gap above 0.05 in {S['weeks_cycle2_rule']['windows_with_expected_gap_above_0.05']} of
{S['weeks_cycle2_rule']['windows_with_expected_gap_evaluable']} evaluable weeks.

| Window | Appointments | Expected age EOD | Realised age EOD | 95% CI (realised) | Status | Groups not evaluated |
| --- | --- | --- | --- | --- | --- | --- |
{win}

| Week | Cycle-1 rule (point / status) | Expected EOD | Realised EOD | Current rule status | Not evaluated |
| --- | --- | --- | --- | --- | --- |
{week}

Stress tests (synthetic): {'; '.join(f"{t['name']} -> fairness {t['fairness']['status']}, data drift {t['status']['data_drift']}" for t in MON['stress_tests'])}.

## 7. Explanations
- **SHAP is the explanation of record**, with limits: it is exact for the XGBoost score (contributions sum to the log-odds, checked),
  but it explains the *score*, not the policy *decision* (randomised and per-group cut-off effects are outside it); contributions are
  associations, not causes; correlated inputs (e.g. lead time and SMS reminders) share credit in ways that can mislead; one-hot
  categories are summed per feature; the colour scale in the summary plot is not meaningful for categorical inputs.
- **LIME failed the fidelity bar fixed before the run** (R² >= {X['fidelity_bar']['min_r2']}, |error| <= {X['fidelity_bar']['max_abs_error']}):
  cycle-1 sampling {sum(e['fidelity'] == 'adequate' for e in old)}/{len(old)} adequate; local sampling {sum(e['fidelity'] == 'adequate' for e in loc)}/{len(loc)}. It is kept as a reported cross-check only.
- **Counterfactuals under Option B:** none of the three tier examples can be un-flagged by an earlier slot or a reminder.

| Example (age group) | Score | Option B flag probability | Un-flaggable? | Lowest reachable flag probability |
| --- | --- | --- | --- | --- |
{cfs}

Across {pop['sampled_with_flag_probability_above_0']:,} sampled test appointments with any flag chance: {pop['share_unflaggable_by_actionable_change']:.1%} can be fully
un-flagged, {pop['share_flag_probability_reducible']:.1%} can have their flag probability reduced.

## 8. Options for the committee (trade-offs only; no option is presented as meeting the full fairness requirement)
| Option | What it gives | What it fails or costs |
| --- | --- | --- |
| Do not pilot; obtain CrescentCare data first | No patient exposure to an unvalidated rule; local base rates and an insurance field (G3) | Delays benefit; the Brazilian 2016 data may not represent CrescentCare |
| Pilot Option B with disclosure | Expected gaps within 0.05 for all three attributes (C4-C7) | Fails C8 and C3; {mech['share_of_flags_unrelated_to_risk']:.0%} of flags unrelated to risk; expected gap > 0.05 in every evaluable window; randomised decisions to explain to patients |
| Pilot equal opportunity at TPR 0.85 with disclosure | Equal catch rates (TPR gap {EO['age_band']['tpr_gap']:.3f} age), deterministic, higher precision and F1 than Option B | Fails {', '.join(c for c in CS if not EO['acceptance'][c])}; unequal false-alarm rates across age groups |
| Amend the Charter fairness criterion | Possible only as a separate, documented ethics decision | Must not be done to fit these results; this pack does not propose it |

Original success criterion C3 (F1 >= 0.55) is not met by any policy; the best is {max(v['f1'] for v in P.values()):.3f}.
"""
    (config.DOCS).mkdir(exist_ok=True)
    (config.DOCS / "committee_decision_pack.md").write_text(md)
    print("docs/committee_decision_pack.md written")


if __name__ == "__main__":
    main()
