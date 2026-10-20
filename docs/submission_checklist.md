# Submission checklist (BAN 6800 capstone)

Rechecked in cycle 4 against the capstone brief ("Your Task" sections 1-8, "Deliverable & Submission Criteria" and
"Technical Requirements") and against the repository on branch `capstone-verify`.

Status key:
- **COMPLETE** - evidence exists in the repository or build outputs and was verified.
- **INCOMPLETE** - evidence is missing or falls short of the requirement as written.
- **NOT VERIFIED** - evidence may exist but could not be checked from this environment.
- **AWAITING USER MATERIAL** - only the student can supply it; nothing is substituted or invented.

The final report must not be written until every INCOMPLETE and AWAITING USER MATERIAL item is supplied or explicitly recorded
as absent in the report. Headless dashboard tests are not evidence of a live v2 deployment.

## Report and user-supplied materials

| Requirement (source) | Evidence | Status | Next action |
| --- | --- | --- | --- |
| Report, 2,500-3,000 words, visuals embedded (Deliverables) | Not started, by instruction | INCOMPLETE | Start only after reviewer authorisation |
| 2-3 key visuals that support decisions (s.6) | Candidate figures exist (`artifacts/figures/`, dashboard pages); not yet selected or placed in a report | INCOMPLETE | Choose in the report stage |
| AI Disclosure Form, official NXU form (Deliverables) | Not supplied | **AWAITING USER MATERIAL** | Student supplies the form; substantial AI involvement must be disclosed |
| Project Board screenshots showing Module 2 sprint completion (s.2) | None found in any repository or upload | **AWAITING USER MATERIAL** | Student supplies them, or records them as unavailable |
| NXU-formatted references (Deliverables) | NXU referencing guide not supplied; APA 7 used in the modules | **AWAITING USER MATERIAL** | Student supplies the guide so the format can be checked |

## Sections 1-8 of the brief

| Requirement (source) | Evidence | Status | Next action |
| --- | --- | --- | --- |
| 1. Business problem from Module 1 Vision Document; Ethical AI Charter | Module 1 Vision Document (.docx upload) | COMPLETE (source material) | Synthesise in report |
| 2. Module 2 overview: objectives, sources, timeline, stakeholders, milestones | Module 2 .docx upload; contains unrelated ED wait-time / churn template text | INCOMPLETE | Exclude template text when synthesising |
| 2. GitHub repository link with all Modules 2-5 code | `Oreoofe/BAN6800-Careconnect`; Modules 3-4 code is a **rebuild** (original code was not available); Module 5 scripts in `legacy/`; no notebooks; v2 work only on `capstone-verify` | INCOMPLETE | Merge decision; disclose the rebuild in the report |
| 3. Pipeline: collection, cleaning, transformation, storage, quality, visuals | `careconnect/pipeline/`, `artifacts/pipeline/*.json` | COMPLETE | Pipeline diagram to be drawn for the report |
| 3. Data Governance & Privacy documentation; bias detection results | `docs/data_governance_privacy.md`; `artifacts/pipeline/bias_report.json` | COMPLETE | - |
| 4. Model development, algorithm choice, validation, metrics, limitations | `careconnect/modeling/`, `model_results.json`, `verification_report.json` | COMPLETE | Report must state that no policy meets F1 >= 0.55 (C3, blocking) |
| 4. MLflow experiment tracking and Model Registry screenshot | `docs/evidence/mlflow_*.png` on `build-outputs-<run>`; registry v1 alias `candidate` only | COMPLETE | - |
| 5. Global SHAP; local LIME/SHAP | `figures/shap_*.png`, reason codes, additivity check; LIME fidelity 0/6 (local sampling), 2/6 (original sampling) | COMPLETE (SHAP); LIME reported as a failed cross-check | State SHAP limits; keep LIME failure |
| 5. Counterfactual explanations | `explanations.json`; none of 3 examples can be un-flagged under Option B | COMPLETE as analysis; negative finding | Report as a limitation |
| 5. Fairness metrics (DP, EO, DI) and bias mitigation documentation | `policy_comparison.json`, `docs/committee_decision_pack.md` (build output) | COMPLETE as evidence; **fairness requirement not met by any policy** | Committee decision; recommendation is no deployment |
| 5. Model Card, limitations statement, explanation store | `docs/model_card.md` (build output); API `logs/explanations.jsonl` (tested) | COMPLETE | - |
| 6. Live Dashboard URL and explainability dashboard (SHAP/LIME) | https://ban6800-careconnect.streamlit.app/ deploys from `main` and shows **v1**; v2 dashboard tested headlessly only; URL not re-checked this cycle (automated fetch blocked) | INCOMPLETE (v2 not live); reachability NOT VERIFIED | Depends on merge decision; student checks the URL in a browser |
| 7. Implementation and monitoring plan; KPIs; fairness monitoring; drift; incident response; decommissioning | `docs/governance.md` s.3-5, `careconnect/monitoring/monitor.py`, `monitoring.json` (replay, not live) | COMPLETE as plan and replay | Monitoring rule departs from Module 4 s.7 (weekly, 4% alert) - committee approval needed |
| 8. Ethical AI Framework, AIA, regulatory report (GDPR, CCPA, HIPAA), public trust statement, governance structure | `docs/governance.md`, `docs/algorithmic_impact_assessment.md`, `docs/regulatory_assessment.md`, `docs/public_trust_statement.md` | COMPLETE (not legal advice) | - |

## Technical requirements

| Requirement (source) | Evidence | Status | Next action |
| --- | --- | --- | --- |
| Live API: publicly accessible `/predict` (FastAPI + cloud) | Runs locally and in Docker; **no public deployment exists** | INCOMPLETE - not deployed | Record as not deployed, or supply hosting |
| Monitoring Dashboard URL (Evidently / Grafana) | Dashboard page 7 (v1 on the live site); Evidently HTML generated by each build, not hosted | INCOMPLETE | Depends on merge and hosting decisions |
| CI/CD (GitHub Actions) | `ci.yml` (tests the committed **v1** artifacts), `build.yml` (full v2 build, verification, tests) | COMPLETE | - |
| Docker Compose: pipeline + API + monitoring | `docker-compose.yml` defines pipeline, api, dashboard, mlflow, monitoring; only api and dashboard are run in CI | INCOMPLETE (pipeline, mlflow, monitoring services NOT VERIFIED) | Run the remaining services, or state the limit |
| XAI dashboard (SHAP + Streamlit) | Dashboard page 3; v2 tested headlessly | COMPLETE in code; live site shows v1 | - |
| Fairness monitoring (Fairlearn + Evidently) | `monitor.py`, rolling-window rule, stress tests S1-S4 | COMPLETE (replay) | - |
| Model retraining (Python + MLflow) | `build.yml` monthly schedule; promotion gate never auto-promotes | COMPLETE | - |
| README: overview, setup, live demo links | `README.md` (states v1 live, API not hosted, recommendation) | COMPLETE | - |
