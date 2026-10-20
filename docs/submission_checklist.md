# Submission checklist (BAN 6800 capstone)

Rechecked in cycle 6 (20 Oct 2026) against the capstone brief ("Your Task" sections 1-8, "Deliverable & Submission Criteria" and
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
| Report, 2,500-3,000 words, visuals embedded (Deliverables) | Final report rewritten in cycle 6 (Claude Doc + Word export); body counted including in-body tables, excluding title block, figures and captions, references and appendices | COMPLETE subject to the counting rule | Student confirms Nexford's counting convention |
| 2-3 key visuals that support decisions (s.6) | Three decision figures (SHAP drivers, policy comparison, weekly s.7 monitoring) plus a pipeline diagram and MLflow screenshots from the final build | COMPLETE | - |
| AI Disclosure Form, official NXU form (Deliverables) | The student holds the official form and completes it personally; it was deliberately not filled in, drafted or replaced | **AWAITING STUDENT ACTION** | Student completes and attaches the form; the report lists the AI-assisted activities to help an accurate declaration |
| Project Board screenshots showing Module 2 sprint completion (s.2) | No Module 2-era screenshot exists. A board was reconstructed on 20 Oct 2026 (24 issues, CC-101..CC-604). The first board screenshot showed partial item CC-603 in Done (moved by GitHub automation) | **AWAITING STUDENT ACTION** | Move CC-501, CC-601, CC-603 to In Progress, show labels, retake; caption as reconstructed, never as historical |
| NXU-formatted references (Deliverables) | Student confirmed that Nexford requires APA 7th edition; references and citations reformatted and each source checked | COMPLETE | Student checks against any Nexford template if one exists |

## Sections 1-8 of the brief

| Requirement (source) | Evidence | Status | Next action |
| --- | --- | --- | --- |
| 1. Business problem from Module 1 Vision Document; Ethical AI Charter | Module 1 Vision Document (.docx upload) | COMPLETE (source material) | Synthesise in report |
| 2. Module 2 overview: objectives, sources, timeline, stakeholders, milestones | Two Module 2 report versions exist: the first upload contains ED wait-time / churn template text; the version in the 20 Oct archive does not. Which was submitted cannot be established from the evidence | PARTIAL | Student confirms which version was submitted; the report uses only no-show content common to both |
| 2. GitHub repository link with all Modules 2-5 code | `Oreoofe/BAN6800-Careconnect`; final results come from the AI-assisted rebuild in `careconnect/` (v2). The student's original Module 3-5 code was supplied on 20 Oct 2026 in an archive (not in the repository); its one notebook (`careconnect_mod4/notebooks/shap_analysis.ipynb`) is malformed JSON and does not open. Module 5 scripts in `legacy/`; v2 only on `capstone-verify` | PARTIAL | Student decides whether to add the original code to the repository as an archive; merge decision |
| 3. Pipeline: collection, cleaning, transformation, storage, quality, visuals | `careconnect/pipeline/`, `artifacts/pipeline/*.json` | COMPLETE | Pipeline diagram to be drawn for the report |
| 3. Data Governance & Privacy documentation; bias detection results | `docs/data_governance_privacy.md`; `artifacts/pipeline/bias_report.json` | COMPLETE | - |
| 4. Model development, algorithm choice, validation, metrics, limitations | `careconnect/modeling/`, `model_results.json`, `verification_report.json` | COMPLETE | Report must state that no policy meets F1 >= 0.55 (C3, blocking) |
| 4. MLflow experiment tracking and Model Registry screenshot | `docs/evidence/mlflow_*.png` on `build-outputs-<run>`; registry v1 alias `candidate` only | COMPLETE | - |
| 5. Global SHAP; local LIME/SHAP | `figures/shap_*.png`, reason codes, additivity check; LIME fidelity 0/6 (local sampling), 2/6 (original sampling) | COMPLETE (SHAP); LIME reported as a failed cross-check | State SHAP limits; keep LIME failure |
| 5. Counterfactual explanations | `explanations.json`; none of 3 examples can be un-flagged under Option B | COMPLETE as analysis; negative finding | Report as a limitation |
| 5. Fairness metrics (DP, EO, DI) and bias mitigation documentation | `policy_comparison.json`, `docs/committee_decision_pack.md` (build output) | COMPLETE as evidence; **fairness requirement not met by any policy** | Committee decision; recommendation is no deployment |
| 5. Model Card, limitations statement, explanation store | `docs/model_card.md` (build output); API `logs/explanations.jsonl` (tested) | COMPLETE | - |
| 6. Live Dashboard URL and explainability dashboard (SHAP/LIME) | https://ban6800-careconnect.streamlit.app/ deploys from `main` and shows **v1**; v2 dashboard tested headlessly only; URL not re-checked this cycle (automated fetch blocked) | INCOMPLETE (v2 not live); reachability NOT VERIFIED | Depends on merge decision; student checks the URL in a browser |
| 7. Implementation and monitoring plan; KPIs; fairness monitoring; drift; incident response; decommissioning | `docs/governance.md` s.3-5, `careconnect/monitoring/monitor.py`, `monitoring.json` (replay, not live). Governing rule: the original Module 4 s.7 weekly rule (Gender and age_band; alert > 0.04, halt > 0.05); rolling 28-day bootstrap analysis kept as supplementary evidence | COMPLETE as plan and replay (not live monitoring) | Report the s.7 weekly outcomes as the governing result |
| 8. Ethical AI Framework, AIA, regulatory report (GDPR, CCPA, HIPAA), public trust statement, governance structure | `docs/governance.md`, `docs/algorithmic_impact_assessment.md`, `docs/regulatory_assessment.md`, `docs/public_trust_statement.md` | COMPLETE (not legal advice) | - |

## Technical requirements

| Requirement (source) | Evidence | Status | Next action |
| --- | --- | --- | --- |
| Live API: publicly accessible `/predict` (FastAPI + cloud) | Runs locally and in Docker; **no public deployment exists** | INCOMPLETE - not deployed | Record as not deployed, or supply hosting |
| Monitoring Dashboard URL (Evidently / Grafana) | Dashboard page 7 (v1 on the live site); Evidently HTML generated by each build and served by the compose `monitoring` container during the CI end-to-end test only - **not hosted, no public URL** | INCOMPLETE - no hosted monitoring dashboard | Depends on merge and hosting decisions |
| CI/CD (GitHub Actions) | `ci.yml` (tests the committed **v1** artifacts), `build.yml` (full v2 build, verification, tests) | COMPLETE | - |
| Docker Compose: pipeline + API + monitoring | `docker-compose.yml` defines pipeline, api, dashboard, mlflow, monitoring. The `compose-e2e` job in `build.yml` runs all five end to end on a CI runner: pipeline container (full build + independent verification), MLflow server queried over REST, monitoring container serving `monitoring.json` and the Evidently report, API and dashboard rebuilt with the pipeline's artifacts and smoke-tested; evidence on `compose-e2e-<run>` | COMPLETE as a CI test of the multi-container setup (ephemeral containers on the CI runner; nothing deployed or publicly reachable) | - |
| XAI dashboard (SHAP + Streamlit) | Dashboard page 3; v2 tested headlessly | COMPLETE in code; live site shows v1 | - |
| Fairness monitoring (Fairlearn + Evidently) | `monitor.py`: governing Module 4 s.7 weekly rule, supplementary rolling-window rule, stress tests S1-S4, Evidently report | COMPLETE (replay, not live) | - |
| Model retraining (Python + MLflow) | `build.yml` monthly schedule; promotion gate never auto-promotes | COMPLETE | - |
| README: overview, setup, live demo links | `README.md` (states v1 live, API not hosted, recommendation) | COMPLETE | - |
