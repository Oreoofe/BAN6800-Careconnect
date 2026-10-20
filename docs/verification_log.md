# Verification log

Every verification cycle runs the full real-data build on GitHub Actions and publishes its outputs to a
`build-outputs-<run>` branch, so earlier results stay available after the code changes.

## Earlier run - 39297015988, commit 3378772

Independent verification 34 PASS / 1 FAIL: Option B's age gap was evaluated on one random draw (0.031 in the pipeline, 0.0557 in the independent re-draw). Led to evaluating randomised rules by expected value plus the spread of realisations.

## Cycle 1 - run 39300208619, commit d282180, outputs on branch `build-outputs-5`

Independent verification: 38 PASS, 0 FAIL. pytest: 25 passed. MLflow experiment screenshot: failed (timeout).

| Finding | Evidence (cycle 1) | Action in cycle 2 |
| --- | --- | --- |
| Option B meets the 0.05 age limit only in expectation | expected age EOD 0.041; realised gap > 0.05 in 42% of 200 draws; p95 0.070 | Compare deterministic alternatives on the same test set (`policy_comparison.py`) |
| LIME low fidelity | high example: model 0.602, LIME 0.114, R² 0.648, misses past-miss driver; medium example R² 0.104 | Add local sampling; measure fidelity against a bar fixed before the run |
| Counterfactual target wrong | "medium" example scored 0.245 (below the 0.25 cut-off) yet labelled medium; search used the single cut-off, not the adopted policy | Examples chosen within score tiers; target = policy flag probability 0 |
| Weekly fairness rule halts every week | 7 of 7 weeks "halt", weekly age EOD 0.070-0.183, small groups skipped | Rolling 28-day windows, minimum group sizes, bootstrap interval; cycle-1 rule kept in the output |
| API tier vs decision confusing | patient A: score 0.243, tier "low", flagged with probability 0.916 | Separate `score_tier` from the policy decision, with an explanation string |
| Model card wording | "all three gaps are at or below the limit" (true only in expectation) | Expected vs realised stated explicitly |

## Cycle 2 - run 39316354066, commit fc9d176, outputs on branch `build-outputs-6`

Independent verification: 60 PASS, 0 FAIL. pytest (against v2 artifacts): 29 passed. MLflow experiment screenshot captured.

| Component | Result | Status |
| --- | --- | --- |
| Policy comparison (8 rules, same test set) | No rule meets the 0.05 equalized-odds limit for age, gender and scholarship reliably; best deterministic (equal opportunity, TPR 0.85): age EOD 0.100 (0.076-0.138) | Negative finding, kept |
| Option B mechanism | Score-independent flag probability (p_ignore x 0.538): child 0.113, young adult 0.098, teen 0.079, adult 0.058, elderly 0.026, senior 0; about 11% of its flags | Disclosed in cycle 3 |
| Option B over time | Expected age EOD > 0.05 in all 6 rolling windows (0.052-0.100) and both evaluable months (0.066-0.133); weekly 5 of 6 evaluable weeks | Disclosed in cycle 3 |
| LIME | Original sampling 2/6 adequate; local sampling 0/6 adequate (bar fixed before the run) | Failed; kept as a limitation |
| Counterfactuals (policy-aware) | None of three tier examples can be un-flagged under Option B (lowest reachable flag probability 0.10-0.11) | Negative finding, kept |
| Monitoring rule | Rolling windows: 3 halt, 3 inconclusive; months: 1 halt, 1 inconclusive, 1 insufficient data; stress tests behave as designed | Rule works; outcome adverse |

## Cycle 3 - documentation and acceptance criteria

- Acceptance matrix against the original criteria C1-C8 computed in `policy_comparison.py` (criteria quoted from Modules 1, 4, 5).
- Promotion gate realigned to the original Module 4 s.7 list plus the Charter halt rule. The cycle-2 gate had made the realised-gap
  check non-blocking after cycle-1 results were known; that was a relaxation and is reversed. Gate outcome is now expected to be
  "not eligible".
- Option B mechanism quantified and checked independently; committee decision pack generated from artifacts.

Cycle 3 result: build run 39326455618 on commit 7fb5e7d, independent verification 84 pass / 0 fail, 30 tests pass on v2
artifacts (outputs on `build-outputs-10`).

## Cycle 4 - reviewer decisions applied

- Recommendation recorded: **do not deploy until a better model or sufficient additional evidence is available**. Option B stays
  configured for evaluation only; equal opportunity at TPR 0.85 documented as an alternative assessed, not as compliant.
- C3 (F1 >= 0.55) made **blocking** by reviewer decision. Source of the ambiguity: Module 4 s.1 calls it part of "the acceptance
  threshold"; the Module 4 s.7 staging list omits it; the capstone brief sets no numeric threshold. No threshold changed.
- The Module 4 s.7 staging condition "Ethics Committee reviews and approves the Model Card" added to the gate as a blocking check
  (not recorded, so not passed).
- Monitoring attribution now names a concurrent data-drift alert next to the fairness explanation (week of 16-20 May: random
  draw / sampling plus SMS_received PSI 2.73). The departure of the replay rule from the Module 4 s.7 weekly rule (alert > 4%,
  halt > 5%) is disclosed and referred for approval.
- `docs/committee_decision_pack.md` added to the published build outputs; submission checklist rechecked against the brief, with
  user-supplied items marked AWAITING USER MATERIAL.
- Run identifiers and results for this cycle are recorded on the build-outputs branch for the commit and in the evidence package,
  not in this file, so the verified commit is the final commit.

## Cycle 5 - original Module 4 s.7 monitoring rule; Docker Compose end to end

- Reviewer decisions: Ethics Committee approval stays a blocking gate (not recorded, so NOT ELIGIBLE); C3 (F1 >= 0.55) stays
  blocking; no acceptance threshold changed.
- **Governing monitoring rule is the original Module 4 s.7 rule**, implemented as written in `monitor.module4_rule()`: each
  calendar week, subgroup error-rate disparity (larger of the FNR and FPR gaps) for Gender and age_band (Module 4 s.1), alert
  > 0.04, halt > 0.05, no minimum group size; groups without defined FNR/FPR are excluded and an attribute with fewer than two
  comparable groups is 'not evaluable' (never a pass). Expected disparities come from the policy's flag probabilities and a breach
  is attributed to chance only when the expected disparity is itself within the threshold. The rolling 28-day bootstrap rule is
  kept as supplementary evidence about uncertainty; it cannot clear a s.7 alert or halt. The earlier statement that the replay rule
  "departs from the original specification and needs committee approval" is withdrawn: the original rule now governs.
- The verifier recomputes every week's realised disparity with Fairlearn from re-applied policy decisions, the expected disparity
  from the policy probabilities, the status thresholds, and the attribution rule.
- Docker Compose: a `compose-e2e` job in `build.yml` runs the pipeline, MLflow, monitoring, API and dashboard services end to end
  on the CI runner and publishes its evidence to `compose-e2e-<run>`. Containers receive the commit SHA through
  `CARECONNECT_GIT_SHA` because `.git` is not copied into images. Nothing is deployed or publicly reachable.
- Run identifiers and results are recorded on the build-outputs and compose-e2e branches for the commit and in the evidence
  package, not in this file, so the verified commit is the final commit.

## Cycle 6 - audit remediation (independent audit of commit 27943a1, branches `audit-27943a1` / `audit-outputs-16`)

The independent audit matched 139 of 141 checks. Its two mismatches, and its report findings, are remediated here. The audit
branches are preserved unchanged as evidence. Change log:

| Finding (audit) | Change | Files |
| --- | --- | --- |
| Randomised-policy interval bootstrapped ONE random realisation, not the expected-value gap it was printed beside | Interval now bootstraps the expected-value gap (resampling test appointments, 300 resamples, seed 0); the superseded single-draw interval is still reported, labelled; `ci_method` recorded per attribute | `careconnect/fairness/policy_comparison.py`, `committee_pack.py`, `src/app.py`, `tests/test_fairness_monitoring.py` |
| 200-draw realised-gap figures were sensitive to the simulation (38.5% with the project seeds vs 46.2% with 1,000 independent draws) | 1,000 documented draws (numpy `default_rng(20000)`..`default_rng(20999)`), Monte Carlo standard error reported, first-200 figures kept for continuity; `mitigation.py` uses the same draws so every document reports one distribution | `policy_comparison.py`, `mitigation.py` |
| Gender halts under-reported in the report; partial weeks and small groups not disclosed | Monitoring records appointment days per week, the smallest age group's no-shows, and s.7 status counts by attribute; the committee pack and Model Card print them | `careconnect/monitoring/monitor.py`, `committee_pack.py`, `model_card.py` |
| Model Card precision ambiguous | Model Card states both precisions and how each is computed | `model_card.py` |
| Fairness measures and Charter P2 wording; monitoring cadence | Definitions and the cadence hierarchy recorded | `docs/governance.md` |
| Independent verification only outside the build; raw logs not retrievable; LR/RF, TPR-parity, no-age model and LIME not independently checked | `scripts/independent_audit.py` runs inside every build after the tests, compares its own recomputation with the build's artifacts (not with typed numbers), re-scores LR and RF, re-fits the TPR-parity rule, re-trains the no-age model, re-runs LIME for reproducibility; each step's raw output is saved to `build-logs/` and published with the audit output | `scripts/independent_audit.py`, `.github/workflows/build.yml` |
| Final report: word count, references, appendices | Report rewritten to 2,975 counted words (method recorded in its Appendix E); APA 7 reference list rebuilt from verified sources, legal edition years flagged for confirmation; acceptance-gate table, evidence index and outstanding-items appendices added | `docs/submission_checklist.md` (report row) |
| Checklist out of date | Rechecked: original code supplied in an archive (notebook malformed), two Module 2 versions, APA 7 confirmed by the student, AI Disclosure Form left for the student, board screenshot to retake | `docs/submission_checklist.md` |

Acceptance criteria C1-C8 and the Ethics Committee gate are unchanged. Run identifiers and results for this cycle are recorded
on the `build-outputs-<run>` branch of the final commit and in the remediation report, not in this file, so the verified commit is
the final commit.
