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

## Cycle 3 - documentation and acceptance criteria (this commit)

- Acceptance matrix against the original criteria C1-C8 computed in `policy_comparison.py` (criteria quoted from Modules 1, 4, 5).
- Promotion gate realigned to the original Module 4 s.7 list plus the Charter halt rule. The cycle-2 gate had made the realised-gap
  check non-blocking after cycle-1 results were known; that was a relaxation and is reversed. Gate outcome is now expected to be
  "not eligible".
- Option B mechanism quantified and checked independently; committee decision pack generated from artifacts.
