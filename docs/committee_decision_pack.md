# Committee decision pack - CareConnect outreach policy

*Generated 2026-10-20 by `careconnect/governance/committee_pack.py` from the build artifacts of commit
`ce819fe` (GitHub run local). Model sha256 `6d60bea7544ffac6...`, policy sha256
`e18da7cd12d13fc4...`. Numbers are not edited by hand. This pack presents evidence, options and a recommendation; it does not approve a policy.*

## 1. Decision required
Which outreach decision policy, if any, may be piloted with the no-show model (v2.0.0), given the finding below.

**No evaluated policy reliably met the 0.05 equalized-odds limit across all required groups** (age band, gender, and scholarship as the G3 stand-in).
The promotion gate result under the original criteria is: **NOT ELIGIBLE under the original acceptance criteria; evidence package prepared for an Ethics Committee decision**.

**Project team recommendation: Do not deploy until a better model or sufficient additional evidence is available.** Option B remains configured for evaluation only and is not approved
for operational use. Deterministic equal opportunity at TPR 0.85 is documented as an alternative assessed, not as a compliant
solution. The original fairness criteria are unchanged. Deployment status: not deployed (no public API; live dashboard serves v1 from main).

Blocking criteria not met: C8 Limit holds reliably: upper 95% bound of every gap <= 0.05 (and, for randomised rules, the 95th percentile of realised gaps <= 0.05) (configured policy: Option B); C3 F1 >= 0.55; Ethics Committee has reviewed and approved the Model Card (Module 4 s.7).

## 2. Original acceptance criteria (as written before results; not changed)
| ID | Criterion | Source |
| --- | --- | --- |
| C1 | ROC-AUC >= 0.70 | Module 4 s.7 |
| C2 | Recall >= 0.65 | Module 4 s.7 |
| C3 | F1 >= 0.55 | Module 4 s.1 ('the acceptance threshold'; omitted from the s.7 staging list) - blocking by reviewer decision |
| C4 | Equalized-odds gap <= 0.05 for age band AND gender | Module 4 s.7 / Charter P2 |
| C5 | Demographic-parity gap <= 0.05 for age band AND gender | Module 4 s.7 ('all fairness metrics') |
| C6 | Disparate-impact ratio >= 0.80 for age band AND gender | Module 4 s.7 |
| C7 | Equalized-odds gap <= 0.05 for scholarship (G3 stand-in) | Module 1 G3 via Module 5 stand-in |
| C8 | Limit holds reliably: upper 95% bound of every gap <= 0.05 (and, for randomised rules, the 95th percentile of realised gaps <= 0.05) | Charter P2 ('automatic halt if threshold breached') |

## 3. Every evaluated policy against those criteria (held-out test set, 22,103 appointments)
| Policy | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | Meets full fairness requirement (C4-C8) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single cut-off (no mitigation) | pass | pass | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | no |
| Equal opportunity, TPR 0.65 | pass | pass | **fail** | **fail** | **fail** | pass | **fail** | **fail** | no |
| Equal opportunity, TPR 0.75 | pass | pass | **fail** | **fail** | **fail** | pass | **fail** | **fail** | no |
| Equal opportunity, TPR 0.85 | pass | pass | **fail** | **fail** | **fail** | pass | **fail** | **fail** | no |
| Minimax per-group cut-offs | pass | pass | **fail** | **fail** | **fail** | pass | **fail** | **fail** | no |
| TPR parity (Fairlearn, randomised) | pass | pass | **fail** | **fail** | **fail** | pass | pass | **fail** | no |
| Option B: equalized odds (Fairlearn, randomised) | pass | pass | **fail** | pass | pass | pass | pass | **fail** | no |
| Model without age inputs | pass | pass | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | no |

| Policy | Randomised | Recall | Precision | F1 | Flagged | Age EOD (95% CI) | Gender EOD (95% CI) | Scholarship EOD (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single cut-off (no mitigation) | 0% | 0.656 | 0.343 | 0.450 | 38.6% | 0.629 (0.587-0.675) | 0.017 (0.008-0.040) | 0.209 (0.173-0.243) |
| Equal opportunity, TPR 0.65 | 0% | 0.664 | 0.327 | 0.439 | 41.0% | 0.064 (0.059-0.110) | 0.040 (0.028-0.057) | 0.152 (0.112-0.187) |
| Equal opportunity, TPR 0.75 | 0% | 0.768 | 0.314 | 0.446 | 49.4% | 0.083 (0.064-0.136) | 0.046 (0.034-0.062) | 0.112 (0.077-0.143) |
| Equal opportunity, TPR 0.85 | 0% | 0.864 | 0.300 | 0.446 | 58.1% | 0.100 (0.076-0.138) | 0.043 (0.027-0.057) | 0.054 (0.028-0.080) |
| Minimax per-group cut-offs | 0% | 0.782 | 0.313 | 0.447 | 50.5% | 0.089 (0.077-0.157) | 0.037 (0.025-0.052) | 0.108 (0.077-0.136) |
| TPR parity (Fairlearn, randomised) | 2% | 0.900 | 0.294 | 0.444 | 61.7% | 0.091 (0.073-0.130) | 0.044 (0.029-0.059) | 0.035 (0.016-0.059) |
| Option B: equalized odds (Fairlearn, randomised) | 82% | 0.852 | 0.283 | 0.425 | 60.7% | 0.041 (0.028-0.089) | 0.031 (0.017-0.043) | 0.029 (0.013-0.047) |
| Model without age inputs | 0% | 0.659 | 0.334 | 0.443 | 39.9% | 0.254 (0.216-0.331) | 0.022 (0.011-0.046) | 0.196 (0.156-0.231) |

Model: ROC-AUC 0.7455 (C1 met). Randomised rules are reported as expected values. No policy meets C3 (F1 >= 0.55).

**C3 source and ambiguity.** Module 4 s.1 (Success criteria) states "The acceptance threshold is: ROC-AUC >= 0.70, Recall >= 0.65,
F1 >= 0.55". Module 4 s.7 ("The model is accepted for staging if") lists ROC-AUC, recall, all fairness metrics PASS (<5% disparity,
DIR >= 0.80) and Ethics Committee approval of the Model Card, but not F1. The capstone brief sets no numeric threshold and no
rubric was supplied. By reviewer decision C3 is treated as **blocking**; the threshold itself is unchanged.

## 4. Option B (configured policy): what it achieves and what it costs
| Group | Expected EOD | 95% CI (bootstrap of expected values) | Realised EOD, 5th-95th pct of 1000 draws | Draws above 0.05 |
| --- | --- | --- | --- | --- |
| age_band | 0.041 | 0.028-0.089 | 0.029-0.070 | 41.5% |
| Gender | 0.031 | 0.017-0.043 | 0.025-0.037 | 0.0% |
| Scholarship | 0.029 | 0.013-0.047 | 0.013-0.049 | 4.1% |

**Method (corrected in cycle 6).** Expected values come from each appointment's flag probability; recall, precision and F1 are
ratios of expected counts. The 95% interval resamples test appointments (300 bootstrap resamples) and recomputes the
expected-value gap. Up to cycle 5 the interval bootstrapped one random realisation instead (that superseded interval, recomputed:
age 0.040-0.107). Draw-to-draw
variation uses 1000 realisations (numpy default_rng(20000) .. default_rng(20999)); the realised age gap exceeds
0.05 in 41.5% of them (Monte Carlo standard error 1.6%).
The cycle-5 figure used the first 200 of these draws (38.5%); the difference is
simulation error from the smaller sample. Criterion C8 is unchanged.

**Flags unrelated to predicted risk.** Option B gives some appointments a non-zero flag probability that does not depend on
their score. Fairlearn's equalized-odds post-processing (Hardt et al., 2016) picks one operating point (common false-positive rate x_best, true-positive rate y_best) that every group can reach: the best point on the LOWEST of the groups' ROC curves. Groups whose ROC curve lies above that point are pulled down to it by mixing their own threshold rule with the ROC diagonal: with probability p_ignore the score is ignored and the appointment is flagged with probability x_best. That product is a flag probability that is the same for every appointment in the group, whatever its score. The group with the lowest curve (here: senior) needs no mixing (p_ignore = 0).

| Age group | Test appointments | p_ignore | Risk-unrelated flag probability (p_ignore x 0.538) | Risk-unrelated flags per 1,000 |
| --- | --- | --- | --- | --- |
| adult | 6,021 | 0.108 | 0.058 | 58 |
| child | 4,247 | 0.211 | 0.113 | 113 |
| elderly | 1,074 | 0.047 | 0.026 | 26 |
| senior | 4,433 | 0.000 | 0.000 | 0 |
| teen | 1,584 | 0.148 | 0.079 | 79 |
| young_adult | 4,744 | 0.181 | 0.098 | 98 |

Consequences: 10.8% of Option B's expected flags (66 of
607 per 1,000 appointments) are unrelated to predicted risk; 91.8%
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
is therefore a weaker criterion and **does not meet the full fairness requirement** (it fails C3, C4, C5, C7, C8).

| Attribute | TPR gap | FPR gap | EOD (95% CI) | DPD | DI ratio |
| --- | --- | --- | --- | --- | --- |
| age_band | 0.060 | 0.100 | 0.100 (0.076-0.138) | 0.077 | 0.877 |
| Gender | 0.005 | 0.043 | 0.043 (0.027-0.057) | 0.034 | 0.943 |
| Scholarship | 0.054 | 0.000 | 0.054 (0.028-0.080) | 0.023 | 0.962 |

| Attribute | Group | Test appointments | No-shows | Flag rate | TPR | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| age_band | adult | 6,021 | 1,213 | 0.587 | 0.857 | 0.519 |
| age_band | child | 4,247 | 879 | 0.546 | 0.858 | 0.465 |
| age_band | elderly | 1,074 | 178 | 0.623 | 0.916 | 0.565 |
| age_band | senior | 4,433 | 702 | 0.594 | 0.856 | 0.545 |
| age_band | teen | 1,584 | 406 | 0.622 | 0.887 | 0.531 |
| age_band | young_adult | 4,744 | 1,084 | 0.568 | 0.863 | 0.481 |
| Gender | F | 14,270 | 2,862 | 0.593 | 0.865 | 0.524 |
| Gender | M | 7,833 | 1,600 | 0.559 | 0.860 | 0.481 |
| Scholarship | 0 | 19,942 | 3,964 | 0.578 | 0.857 | 0.509 |
| Scholarship | 1 | 2,161 | 498 | 0.602 | 0.912 | 0.509 |

Against Option B: recall 0.864 vs 0.852, precision 0.300 vs 0.283, F1 0.446 vs
0.425, flagged 58.1% vs 60.7%; no randomised decisions and no flags unrelated to risk; but its
age-group false-positive gap (0.100) and scholarship gap (0.054) exceed 0.05.

## 6. Monitoring evidence (replay of held-out appointments; not live data)

### 6.1 Governing rule: the original Module 4 s.7 weekly rule
Module 4 s.7: "subgroup error-rate disparity tracked weekly; alert at >4%, halt at >5%"; Module 4 s.1 names Gender and age_band.
Implemented as written: each calendar week, the realised disparity (larger of the FNR and FPR gaps) for Gender and for age_band;
**alert above 0.04, halt above 0.05**; no minimum group size (none is specified). A group with no
no-shows or no attended visits cannot be compared; an attribute with fewer than two comparable groups is 'not evaluable', never a pass.
Expected disparities come from the policy's flag probabilities; a breach is attributed to the random draw or sampling only when the
expected disparity is itself within the threshold.

**Governing status over the replay: HALT** (halt 7, alert 0, ok 0, not evaluable 0).
By attribute: age_band - halt 7; Gender - halt 2, alert 2, ok 3; Scholarship - halt 5, alert 1, ok 1
(scholarship is reported but is not part of the s.7 rule).

**Coverage and precision.** 3 of 7 weekly windows are partial
(2016-04-29 to 2016-04-29: 1 appointment day(s); 2016-05-24 to 2016-05-25: 2 appointment day(s); 2016-06-06 to 2016-06-08: 3 appointment day(s)). The smallest
age group in each week had between 6 and 36 no-shows
(elderly in most weeks), so weekly point estimates are imprecise. Sparse weeks are reported, never read as evidence of fairness.

| Week | Appointments | Age disparity (FNR / FPR gap) | Age expected | Gender disparity | Gender expected | Status | Reason | Attribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016-04-29 to 2016-04-29 | 669 | 0.286 (0.200 / 0.286) | 0.262 | 0.125 (0.078 / 0.125) | 0.108 | halt | age_band: disparity 0.286 (FNR gap 0.200, FPR gap 0.286; expected 0.262) -> halt; Gender: disparity 0.125 (FNR gap 0.078, FPR gap 0.125; expected 0.108) -> halt. | age_band: policy's expected disparity (expected 0.262 above 0.05); age_band: small subgroup(s) (child 17 no-shows, elderly 6 no-shows, senior 15 no-shows, teen 11 no-shows) - point estimate imprecise; Gender: policy's expected disparity (expected 0.108 above 0.05); concurrent data-drift alert (prior_appointments PSI 2.10) |
| 2016-05-02 to 2016-05-06 | 4175 | 0.153 (0.153 / 0.050) | 0.104 | 0.040 (0.012 / 0.040) | 0.043 | halt | age_band: disparity 0.153 (FNR gap 0.153, FPR gap 0.050; expected 0.104) -> halt; Gender: disparity 0.040 (FNR gap 0.012, FPR gap 0.040; expected 0.043) -> alert. | age_band: policy's expected disparity (expected 0.104 above 0.05); age_band: small subgroup(s) (elderly 29 no-shows) - point estimate imprecise; Gender: policy's expected disparity (expected 0.043 above 0.04); concurrent data-drift alert (prior_appointments PSI 0.34) |
| 2016-05-09 to 2016-05-14 | 4382 | 0.119 (0.119 / 0.068) | 0.103 | 0.053 (0.013 / 0.053) | 0.036 | halt | age_band: disparity 0.119 (FNR gap 0.119, FPR gap 0.068; expected 0.103) -> halt; Gender: disparity 0.053 (FNR gap 0.013, FPR gap 0.053; expected 0.036) -> halt. | age_band: policy's expected disparity (expected 0.103 above 0.05); Gender: random draw / sampling (expected 0.036 within 0.05) |
| 2016-05-16 to 2016-05-20 | 4278 | 0.095 (0.068 / 0.095) | 0.072 | 0.046 (0.031 / 0.046) | 0.046 | halt | age_band: disparity 0.095 (FNR gap 0.068, FPR gap 0.095; expected 0.072) -> halt; Gender: disparity 0.046 (FNR gap 0.031, FPR gap 0.046; expected 0.046) -> alert. | age_band: policy's expected disparity (expected 0.072 above 0.05); Gender: policy's expected disparity (expected 0.046 above 0.04); concurrent data-drift alert (SMS_received PSI 2.73) |
| 2016-05-24 to 2016-05-25 | 1606 | 0.108 (0.050 / 0.108) | 0.080 | 0.019 (0.019 / 0.008) | 0.025 | halt | age_band: disparity 0.108 (FNR gap 0.050, FPR gap 0.108; expected 0.080) -> halt; Gender: disparity 0.019 (FNR gap 0.019, FPR gap 0.008; expected 0.025) -> ok. | age_band: policy's expected disparity (expected 0.080 above 0.05); age_band: small subgroup(s) (elderly 23 no-shows, teen 24 no-shows) - point estimate imprecise; concurrent data-drift alert (lead_time_days PSI 0.66) |
| 2016-05-30 to 2016-06-03 | 4198 | 0.197 (0.197 / 0.089) | 0.140 | 0.028 (0.028 / 0.014) | 0.030 | halt | age_band: disparity 0.197 (FNR gap 0.197, FPR gap 0.089; expected 0.140) -> halt; Gender: disparity 0.028 (FNR gap 0.028, FPR gap 0.014; expected 0.030) -> ok. | age_band: policy's expected disparity (expected 0.140 above 0.05) |
| 2016-06-06 to 2016-06-08 | 2795 | 0.133 (0.133 / 0.068) | 0.096 | 0.027 (0.003 / 0.027) | 0.013 | halt | age_band: disparity 0.133 (FNR gap 0.133, FPR gap 0.068; expected 0.096) -> halt; Gender: disparity 0.027 (FNR gap 0.003, FPR gap 0.027; expected 0.013) -> ok. | age_band: policy's expected disparity (expected 0.096 above 0.05); age_band: small subgroup(s) (elderly 23 no-shows) - point estimate imprecise |

Scholarship (G3 stand-in) is computed for every week but is not part of the s.7 rule; see `monitoring.json`.
Under the S2 stress test (older patients' flags removed) the s.7 rule returns halt.

### 6.2 Supplementary: rolling-window uncertainty analysis (does not replace 6.1)
Rule: rolling 28 days, evaluated weekly; groups need >= 50 no-shows and >= 50
attended visits; halt only if the whole 95% interval is above 0.05; ok only if wholly at or below; otherwise inconclusive;
fewer than two eligible groups = insufficient data. 'insufficient data' and groups 'not evaluated' are reported separately and are never counted as a pass; 'inconclusive' means the 95% interval straddles 0.05 and supports no conclusion either way.

Outcomes, reported separately: rolling 28-day windows - halt: 3, inconclusive: 3, ok: 0, insufficient data: 0 (full windows only - halt: 1, inconclusive: 2, ok: 0, insufficient data: 0);
calendar months - halt: 1, inconclusive: 1, ok: 0, insufficient data: 1; weekly windows under the same rule - halt: 5, inconclusive: 1, ok: 0, insufficient data: 1; the cycle-1 weekly rule halted in
7 of 7 weeks.

**Under Option B the expected age-group gap exceeds 0.05 in every monitored time window that could be evaluated**: all
6 rolling 28-day windows (range 0.052-0.100)
and both evaluable calendar months (range 0.066-0.133); April 2016
had insufficient data. Option B was fitted on all training data pooled, and its expected-value guarantee does not carry over to
individual time periods. Weekly windows: expected gap above 0.05 in 5.0 of
6 evaluable weeks; the exception is the week of 2016-05-16 to 2016-05-20 (expected 0.047; random draw / sampling (expected gap within the limit); concurrent data-drift alert (SMS_received PSI 2.73)), so the
"every monitored time window" statement applies to the rolling and monthly windows used by the rule, not to every week.

**Relationship between the two methods.** The Module 4 s.7 weekly rule (6.1) is the governing monitoring rule; it is not replaced.
The rolling 28-day bootstrap analysis (6.2) is supplementary: it shows how much of a weekly breach could be sampling noise (small
weekly groups) and whether the gap persists over longer windows. It cannot clear a s.7 alert or halt; if the two disagree, the s.7
result stands and the disagreement is reported. The cycle-1 weekly rule (age only, groups under 30 no-shows skipped) is a
superseded approximation of s.7, kept for audit.

| Window | Appointments | Expected age EOD | Realised age EOD | 95% CI (realised) | Status | Groups not evaluated |
| --- | --- | --- | --- | --- | --- | --- |
| 28 days to 2016-05-05 (partial: 7 days) | 4,106 | 0.052 | 0.078 | 0.056-0.180 | halt | elderly |
| 28 days to 2016-05-12 (partial: 14 days) | 8,432 | 0.100 | 0.120 | 0.072-0.186 | halt | none |
| 28 days to 2016-05-19 (partial: 21 days) | 12,786 | 0.062 | 0.079 | 0.050-0.154 | inconclusive | none |
| 28 days to 2016-05-26 | 15,110 | 0.062 | 0.078 | 0.046-0.135 | inconclusive | none |
| 28 days to 2016-06-02 | 14,387 | 0.058 | 0.085 | 0.053-0.143 | halt | none |
| 28 days to 2016-06-08 | 14,573 | 0.056 | 0.047 | 0.038-0.116 | inconclusive | none |
| calendar month 2016-04 | 669 | n/a | n/a | - | insufficient data | adult, child, elderly, senior, teen, young_adult |
| calendar month 2016-05 | 16,155 | 0.066 | 0.056 | 0.039-0.118 | inconclusive | none |
| calendar month 2016-06 | 5,279 | 0.133 | 0.167 | 0.100-0.245 | halt | elderly |

| Week | Cycle-1 rule (point / status) | Expected EOD | Realised EOD | Bootstrap rule status | Not evaluated | Attribution |
| --- | --- | --- | --- | --- | --- | --- |
| week of 2016-04-25 | 0.131 / halt | n/a | n/a | insufficient data | adult, child, elderly, senior, teen, young_adult | insufficient sample |
| week of 2016-05-02 | 0.089 / halt | 0.071 | 0.080 | halt | elderly | policy's expected disparity (expected gap above the limit); concurrent data-drift alert (prior_appointments PSI 0.34) |
| week of 2016-05-09 | 0.128 / halt | 0.088 | 0.084 | halt | elderly | policy's expected disparity (expected gap above the limit) |
| week of 2016-05-16 | 0.070 / halt | 0.047 | 0.068 | halt | elderly | random draw / sampling (expected gap within the limit); concurrent data-drift alert (SMS_received PSI 2.73) |
| week of 2016-05-23 | 0.183 / halt | 0.075 | 0.072 | inconclusive | child, elderly, teen | policy's expected disparity (expected gap above the limit); concurrent data-drift alert (lead_time_days PSI 0.66) |
| week of 2016-05-30 | 0.165 / halt | 0.140 | 0.197 | halt | elderly | policy's expected disparity (expected gap above the limit) |
| week of 2016-06-06 | 0.106 / halt | 0.092 | 0.133 | halt | elderly | policy's expected disparity (expected gap above the limit) |

Stress tests (synthetic): S1 data drift: lead times x1.5 + 7 days -> fairness halt, data drift alert; S2 bias drift: 60% of senior/elderly flags removed (last 28 days) -> fairness halt, data drift ok; S3 small group: elderly reduced to 25 appointments (last 28 days) -> fairness inconclusive, data drift warn; S4 tiny batch: 300 random appointments -> fairness insufficient data, data drift alert.

## 7. Explanations
- **SHAP is the explanation of record**, with limits: it is exact for the XGBoost score (contributions sum to the log-odds, checked),
  but it explains the *score*, not the policy *decision* (randomised and per-group cut-off effects are outside it); contributions are
  associations, not causes; correlated inputs (e.g. lead time and SMS reminders) share credit in ways that can mislead; one-hot
  categories are summed per feature; the colour scale in the summary plot is not meaningful for categorical inputs.
- **LIME failed the fidelity bar fixed before the run** (R² >= 0.5, |error| <= 0.05):
  cycle-1 sampling 2/6 adequate; local sampling 0/6. It is kept as a reported cross-check only.
- **Counterfactuals under Option B:** none of the three tier examples can be un-flagged by an earlier slot or a reminder.

| Example (age group) | Score | Option B flag probability | Un-flaggable? | Lowest reachable flag probability |
| --- | --- | --- | --- | --- |
| high (child) | 0.452 | 0.903 | no | 0.113 |
| medium (young_adult) | 0.314 | 0.916 | no | 0.098 |
| low (young_adult) | 0.066 | 0.098 | no | 0.098 |

Across 1,500 sampled test appointments with any flag chance: 12.5% can be fully
un-flagged, 67.3% can have their flag probability reduced.

## 8. Options for the committee (trade-offs only; no option is presented as meeting the full fairness requirement)
| Option | What it gives | What it fails or costs |
| --- | --- | --- |
| **Recommended: do not deploy; obtain a better model or CrescentCare data first** | No patient exposure to an unvalidated rule; local base rates and an insurance field (G3) | Delays benefit; the Brazilian 2016 data may not represent CrescentCare |
| Pilot Option B with disclosure (not recommended) | Expected gaps within 0.05 for all three attributes (C4-C7) | Fails C8 and C3; 11% of flags unrelated to risk; expected gap > 0.05 in every evaluable window; randomised decisions to explain to patients |
| Pilot equal opportunity at TPR 0.85 with disclosure (alternative assessed; not compliant) | Equal catch rates (TPR gap 0.060 age), deterministic, higher precision and F1 than Option B | Fails C3, C4, C5, C7, C8; unequal false-alarm rates across age groups |
| Amend the Charter fairness criterion | Possible only as a separate, documented ethics decision | Must not be done to fit these results; this pack does not propose it |

Original criterion C3 (F1 >= 0.55, blocking by reviewer decision) is not met by any policy; the best is 0.450.
