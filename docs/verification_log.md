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
