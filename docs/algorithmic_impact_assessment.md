# Algorithmic Impact Assessment and Risk Register

## System summary
- **Purpose:** prioritise supportive outreach (reminder calls, transport vouchers, earlier slots) for appointments at higher risk of a missed visit, to serve Module 1 goals G1 (no-shows < 15%), G2 (slot fill > 90%) and G3 (equity gap < 5%).
- **Decision affected:** who is offered extra support first. The model never decides whether a patient receives care.
- **Autonomy level:** decision support; a coordinator chooses every action. Some flags are randomised within an age group's threshold band (Option B), with the draw logged.
- **Affected people:** patients (flagged and not flagged), care coordinators and social workers (workload), clinic operations, the AI Ethics Committee.

## Impact analysis

| Stakeholder | Possible benefit | Possible harm | Likelihood | Severity | Mitigation | Residual risk |
| --- | --- | --- | --- | --- | --- | --- |
| Older patients | Under Option B their no-shows are caught at a similar rate to younger patients' **in expectation** | Under a single cut-off, most of their no-shows would be missed (support withheld); under Option B any one month's decisions can still leave a gap above 0.05 | High without mitigation | High | Policy choice pending committee (Option B or a deterministic alternative); rolling 28-day fairness rule | Medium: realised gaps vary and small groups are hard to measure |
| Younger patients | Support offered earlier | More false alarms: contacted when they would have attended; may feel targeted | High | Low-Medium | Support-only wording; opt-out; complaint route | Low |
| Low-income patients (scholarship proxy) | Barriers addressed earlier | Over-flagging read as stigma | Medium | Medium | Scholarship gap audited; support framed as offer | Low-Medium |
| Coordinators | Shorter, ranked lists | Option B flags about 61% of appointments (single cut-off: about 39%), raising workload; alert fatigue; a 'low' score tier can still be flagged, which needs explaining | High | Medium | Capacity check (Module 2 A-3) before go-live; tiered outreach; API and dashboard show score tier and policy decision separately | Medium |
| All patients | - | Privacy loss from health data use | Low | High | Pseudonymisation, minimisation, access control, no raw features in logs | Low |
| Hospital | Fewer empty slots | Misuse of scores for overbooking or penalties | Medium | High | Prohibited uses in Model Card and API; audit of use; training | Medium |

## Updated risk register (Module 1 ER-01 to ER-04 + risks surfaced in Modules 3-5 and the capstone)

| ID | Risk | Source | Severity | Status | Mitigation / evidence |
| --- | --- | --- | --- | --- | --- |
| ER-01 | Algorithmic bias against protected groups | Module 1 | Critical | **Materialised** (age gap 0.629 at single cut-off) | Option B meets the limit in expectation only; deterministic alternatives compared; committee decision pending; rolling-window monitoring |
| ER-02 | Privacy breach / unauthorised access | Module 1 | Critical | Open (design) | Pseudonymisation, no PII in Git (CI check), API key, non-root containers |
| ER-03 | Misuse for punitive actions | Module 1 | Critical | Open | Prohibited-use statements; support-only outputs; usage audit |
| ER-04 | Regulatory non-compliance | Module 1 | Critical | Open | Applicability assessment (`regulatory_assessment.md`); counsel review required |
| ER-05 | Target leakage via history features | Capstone audit | High | **Fixed** in v2.0.0 | Outcome-known rule; brute-force test in CI |
| ER-06 | Raw label column leaking into training | Module 5 | High | **Fixed** | Label text dropped at source; test `test_no_direct_identifiers` |
| ER-07 | Population mismatch (Brazil 2016 vs CrescentCare) | Module 2 A-1 | High | Open | Local validation on CrescentCare sample before any use |
| ER-08 | Sensitive health inputs (alcoholism, disability) used for prediction | Capstone audit | Medium | Open - committee decision | Removing all 7 protected/sensitive inputs costs 0.003 AUC (0.7455 -> 0.7421): little predictive reason to keep them |
| ER-09 | Randomised decisions hard to explain to patients | Option B | Medium | Open | Draw logged and reproducible; plain-language explanation in trust statement |
| ER-10 | Coordinator overload from higher flag rate | Option B | Medium | Open | Capacity sign-off; tiered outreach |
| ER-11 | Reminder process change (SMS) shifts inputs | Monitoring replay | Medium | Monitored | PSI on SMS_received; process changes logged |
| ER-12 | Equity goal G3 unmeasurable (no insurance field) | Module 5 | Medium | Open | Add insurance type in production data |
| ER-13 | Fairness measurement noise: small groups (e.g. 1,074 elderly test appointments, 178 no-shows) make a 0.05 gap hard to confirm or rule out | Verification cycle 1 | High | Open | Bootstrap intervals reported; minimum group sizes; 28-day windows; 'insufficient data' never read as fair |
| ER-14 | LIME explanations with low local fidelity could mislead staff | Verification cycles 1-2 | Medium | **Confirmed**: 0/6 adequate with local sampling, 2/6 with the original sampling | SHAP is the explanation of record; LIME reported as a failed cross-check |
| ER-15 | Option B issues flags unrelated to predicted risk (score-independent flag probability between 0 (senior) and about 0.11 (child) by age group; about 11% of its flags) | Verification cycle 2 | High | Open - committee decision | Disclosed in model card, committee pack and trust statement; deterministic equal opportunity offered as an alternative (fails the full criterion) |
| ER-16 | Fairness guarantee does not hold over time: expected Option B age gap > 0.05 in every monitored time window used by the rule (all rolling 28-day windows and evaluable months; 5 of 6 evaluable weeks - the week of 16-20 May has 0.047 and a concurrent SMS_received drift alert) | Verification cycle 2 | High | Open | Rolling 28-day monitoring with explicit halt / inconclusive / insufficient-data outcomes; committee review |

## Review
Reviewed at each model release and quarterly by the AI Ethics Committee; updated after every S1/S2 incident.

## Fairness options assessed (cycle 2)
No evaluated policy reliably met the 0.05 equalized-odds limit across all required groups. Option B meets it only in expectation
on pooled data, creates flags unrelated to predicted risk, and its expected age-group gap exceeds 0.05 in every monitored time
window used by the monitoring rule. Deterministic equal opportunity at TPR 0.85 equalises catch rates but not false-alarm rates,
so it does not satisfy equalized odds. Residual risk for older patients therefore remains open under every option; see
`docs/committee_decision_pack.md`.

**Recommendation (cycle 4): do not deploy until a better model or sufficient additional evidence is available.** Option B stays
configured for evaluation only and is not approved for operational use; equal opportunity at TPR 0.85 is an alternative assessed,
not a compliant solution. No policy meets F1 >= 0.55 (C3, blocking by reviewer decision). The original criteria are unchanged.
Nothing is deployed: there is no public API, and the hosted dashboard still serves v1 from `main`.
