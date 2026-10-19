# CareConnect Ethical AI Governance

Status: capstone design document (October 2026). Controls marked **implemented** exist in this repository and are tested; controls marked **designed** describe how CrescentCare would operate them and have not been exercised with real patients.

## 1. Principles (Module 1 Ethical Charter, as refined by Modules 3-5)

| Principle | Charter wording (Module 1) | Refinement from later evidence | Control | Status |
| --- | --- | --- | --- | --- |
| P1 Do no harm | Never used to deny care, penalise or restrict access | Unchanged | API returns support-only flags; prohibited uses in Model Card and `/model_info` | Implemented |
| P2 Fairness by design | Error-rate disparity between groups must not exceed 5%; automatic halt if breached | The Charter text mixed demographic parity with error-rate parity. Age groups have different true no-show rates (15.5%-26.1%), so the limit is applied to the **equalized-odds gap** (largest TPR or FPR difference). Parity and disparate impact are reported, not enforced. | Fairness gate in `careconnect/run_all.py`; Option B post-processing; weekly bias-drift check | Implemented (gate, policy, replay); designed (production halt) |
| P3 Transparency | Black-box models prohibited | XGBoost is retained as a **documented exception**: exact TreeSHAP reason codes per prediction, LIME second opinion, logistic regression kept as an interpretable fallback (AUC below target). Exception requires Ethics Committee sign-off. | Reason codes in every API response; XAI dashboard page | Implemented; sign-off pending |
| P4 Human accountability | No automated decisions | Under Option B some flags are randomised; the coordinator still decides every action, and each draw is logged | Explanation store with request id, draw and reasons | Implemented |
| P5 Continuous monitoring | Monthly bias audits; immediate suspension if harm detected | Weekly fairness check added because group gaps move faster than monthly audits catch | `careconnect/monitoring/monitor.py`; monthly `build-and-retrain` workflow | Implemented (replay); designed (live) |

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

1. **Triggers:** monthly schedule (`.github/workflows/build.yml`), any input PSI >= 0.20, AUC more than 0.03 below validation, or a fairness warning on two consecutive weeks.
2. **Run:** `python -m careconnect.run_all` - same pipeline, data contract, bias gate, training, fairness mitigation, explanations, monitoring and Model Card generation.
3. **Automated gate** (`artifacts/promotion_decision.json`): data contract and bias gate pass; ROC-AUC >= 0.70; recall >= 0.65; adopted-policy equalized-odds gap <= 0.05 for age, gender and scholarship; patient-grouped AUC within 0.02. F1 is reported but non-blocking (it is not in the Module 4 acceptance criteria).
4. **Human approval:** Ethics Committee reviews the Model Card and gate output; approval recorded in the MLflow version tags (approver, date, decision).
5. **Release:** API image rebuilt with the approved artifacts; previous version kept for rollback.
6. **Post-release:** first four weeks monitored weekly with a lowered warning threshold (gap > 0.03).

## 4. Ethical incident response

| Severity | Example | First response | Owner | Time limit |
| --- | --- | --- | --- | --- |
| S1 Critical | Fairness gap > 0.05 in live monitoring; flags used to deny or delay care; data breach | Suspend flagging (API returns scores to Data Science only); notify Ethics Committee and Privacy Officer | Model owner | Suspend within 4 h; committee within 24 h |
| S2 High | Gap 0.04-0.05 two weeks running; input PSI >= 0.20; AUC below alert line | Root-cause review; retraining review; increased human review of flags | Model owner | 5 working days |
| S3 Moderate | Patient complaint about outreach; unexplained reason codes; single-week warning | Log, respond to patient, check explanation store record | Product owner | 10 working days |
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
