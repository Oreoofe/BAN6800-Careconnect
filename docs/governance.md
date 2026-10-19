# CareConnect Ethical AI Governance

Status: capstone design document (October 2026). Controls marked **implemented** exist in this repository and are tested; controls marked **designed** describe how CrescentCare would operate them and have not been exercised with real patients.

## 1. Principles (Module 1 Ethical Charter, as refined by Modules 3-5)

| Principle | Charter wording (Module 1) | Refinement from later evidence | Control | Status |
| --- | --- | --- | --- | --- |
| P1 Do no harm | Never used to deny care, penalise or restrict access | Unchanged | API returns support-only flags; prohibited uses in Model Card and `/model_info` | Implemented |
| P2 Fairness by design | Error-rate disparity between groups must not exceed 5%; automatic halt if breached | The Charter text mixed demographic parity with error-rate parity. Age groups have different true no-show rates (15.5%-26.1%), so the limit is applied to the **equalized-odds gap** (largest TPR or FPR difference). Parity and disparate impact are reported, not enforced. | Fairness gate in `careconnect/run_all.py`; Option B post-processing configured **pending committee decision**: it meets the limit in expectation only (single sets of decisions exceed 0.05 in roughly 4 of 10 simulated draws); deterministic alternatives compared in `artifacts/policy_comparison.json`; rolling 28-day fairness rule with minimum group sizes and confidence intervals | Implemented (gate, policy, comparison, replay); designed (production halt) |
| P3 Transparency | Black-box models prohibited | XGBoost is retained as a **documented exception**: exact TreeSHAP reason codes per prediction, LIME second opinion, logistic regression kept as an interpretable fallback (AUC below target). Exception requires Ethics Committee sign-off. | Reason codes in every API response; XAI dashboard page | Implemented; sign-off pending |
| P4 Human accountability | No automated decisions | Under Option B some flags are randomised; the coordinator still decides every action, and each draw is logged | Explanation store with request id, draw and reasons | Implemented |
| P5 Continuous monitoring | Monthly bias audits; immediate suspension if harm detected | A weekly point-estimate check was tried and rejected: it halted in every replayed week because weekly age groups are too small to measure a 0.05 gap. The rule now uses rolling 28-day windows, minimum group sizes (50 no-shows, 50 attended) and a bootstrap 95% interval; 'insufficient data' and 'inconclusive' are explicit outcomes | `careconnect/monitoring/monitor.py`; monthly `build-and-retrain` workflow | Implemented (replay); designed (live) |

## 2. Governance structure

| Body / role | Members | Responsibilities | Cadence |
| --- | --- | --- | --- |
| AI Ethics Committee | Chief Medical Officer (chair), Privacy Officer, External Bioethicist, Patient Advocate | Approves every model version and policy change; binding veto; receives all fairness halts | Quarterly + within 24 h of a halt |
| Model owner | Data Science Lead | Accountable for performance, monitoring, retraining, documentation | Weekly review of monitoring |
| Product owner | Project Lead (Gladys Izuagbe) | Business outcomes G1-G4, coordinator workflow, stakeholder communication | Sprint cadence |
| Privacy & Security | Privacy Officer, Security Officer | Data access, retention, breach response, DPIA / risk analysis | Release sign-off; annual review |
| Operational users | Care Coordinators, Social Workers | Act on flags; record outcome and any override reason | Daily |
| Patient voice | Patient Advocate, patient advisory panel | Review public trust statement and complaints | Quarterly |

**Decision rights.** Only the Ethics Committee can move the MLflow alias `production`. A training job only creates a `candidate` version tagged `approval = pending Ethics Committee review`. Rollback to the previous `production` version can be ordered by the Model owner without committee approval; the committee is informed within 24 hours.

## 3. Retraining, approval and rollback

1. **Triggers:** monthly schedule (`.github/workflows/build.yml`), any input PSI >= 0.20 that is not a known artefact (e.g. the data-window cold start for prior visits), AUC more than 0.03 below validation, a fairness 'halt', or 'inconclusive' in two consecutive rolling windows.
2. **Run:** `python -m careconnect.run_all` - same pipeline, data contract, bias gate, training, fairness mitigation, explanations, monitoring and Model Card generation.
3. **Automated gate** (`artifacts/promotion_decision.json`): data contract and bias gate pass; patient-grouped AUC within 0.02; and the original criteria C1, C2 and C4-C8 for the configured policy (Module 4 s.7 staging list plus the Charter halt rule, including reliability C8). F1 >= 0.55 (C3) appears in Module 4 s.1 as a success criterion but not in the s.7 staging list; the gate reports it as an unmet original criterion and does not block on it. That reading is an ambiguity in the source documents and is referred to the committee for a decision, not settled here.
4. **Human approval:** Ethics Committee reviews the Model Card, the gate output, the committee decision pack and the policy comparison; approval recorded in the MLflow version tags (approver, date, decision).
5. **Release:** API image rebuilt with the approved artifacts; previous version kept for rollback.
6. **Post-release:** rolling 28-day fairness windows reviewed weekly; groups that are 'not evaluated' (too few cases) are reviewed by hand rather than assumed fair.

## 4. Ethical incident response

| Severity | Example | First response | Owner | Time limit |
| --- | --- | --- | --- | --- |
| S1 Critical | Fairness rule returns 'halt' (whole 95% interval above 0.05); flags used to deny or delay care; data breach | Suspend flagging (API returns scores to Data Science only); notify Ethics Committee and Privacy Officer | Model owner | Suspend within 4 h; committee within 24 h |
| S2 High | Fairness 'inconclusive' in two consecutive windows, or a group repeatedly 'not evaluated'; input PSI >= 0.20; AUC below alert line | Root-cause review; retraining review; increased human review of flags | Model owner | 5 working days |
| S3 Moderate | Patient complaint about outreach; unexplained reason codes; a single 'inconclusive' window | Log, respond to patient, check explanation store record | Product owner | 10 working days |
| S4 Low | Documentation or dashboard error | Correct and note in change log | Product owner | Next sprint |

Every incident is logged with the affected request ids (from the explanation store), actions taken, and the committee's decision. Post-incident review within 30 days updates the risk register.

## 5. Ethical decommissioning

Triggers: G3 failure metric worsens for two consecutive quarters (Module 1 termination rule); persistent fairness breach that mitigation cannot fix; a replacement model is approved; or the outreach programme ends.

1. **Stop predictions:** disable the `/predict` route and remove the `production` alias; coordinators revert to the pre-CareConnect outreach process (documented in advance so no patient loses support).
2. **Notify:** coordinators and social workers (operational change), Ethics Committee, patient advisory panel; publish an update to the public trust statement explaining why.
3. **Preserve records:** explanation store, audit log, Model Cards and approvals retained for 7 years (Module 1 retention rule) in read-only storage.
4. **Revoke access:** API keys, service accounts, dashboard access and data-zone permissions removed; raw-zone data deleted per the 30-day rule.
5. **Retire artifacts:** MLflow versions archived with tag `decommissioned` and reason; container images removed from the registry.
6. **Review impact:** compare attendance and equity metrics in the three months after withdrawal; report to the committee.

## 6. Fairness evidence for the committee decision (verification cycles 1-2)

Figures are generated in `docs/committee_decision_pack.md` and `docs/model_card.md` from the build artifacts; the points below
summarise them and must be read with those files.

- **No evaluated policy reliably met the 0.05 equalized-odds limit across all required groups** (age band, gender, scholarship as
  the G3 stand-in). Eight policies were tested against the original acceptance criteria C1-C8; the criteria were not changed after
  the results were seen. No policy meets F1 >= 0.55 either.
- **Option B (configured, not approved) creates flags unrelated to predicted risk.** Fairlearn's equalized-odds post-processing
  equalises error rates by ignoring the score for a share of appointments in every age group whose ROC curve lies above the lowest
  group's (senior), and flagging those at the common false-positive rate. In the verified build this is a score-independent flag
  probability between 0 (senior) and about 0.11 (child) depending on age group, roughly 11% of all Option B flags. It also lowers decision quality for
  the groups the model predicts best, prevents counterfactual "what would remove this flag" answers in those groups, and means
  SHAP reason codes do not explain those flags. For exact equalized odds by post-processing this randomisation is unavoidable when
  the groups' ROC curves differ (Hardt et al., 2016).
- **The expected Option B age-group gap exceeds 0.05 in every monitored time window** used by the monitoring rule (all rolling
  28-day windows and both evaluable calendar months; April 2016 had insufficient data), although it is within 0.05 on the pooled
  test set. Monitoring outcomes are recorded separately as halt, inconclusive and insufficient data; insufficient data and
  unevaluated groups are never treated as a pass.
- **Deterministic equal opportunity at TPR 0.85 is presented as an alternative, not as compliant.** Equal opportunity equalises
  only the true-positive rate; equalized odds (the Charter requirement) also requires equal false-positive rates. This policy
  equalises catch rates closely but leaves age-group false-positive and scholarship gaps above 0.05, so it fails C4, C5, C7 and C8.
- **Decision rights.** Choosing between not piloting, piloting either policy with disclosure, or formally amending the Charter
  criterion belongs to the AI Ethics Committee. Amending the criterion is a separate ethics decision and is not proposed here.
