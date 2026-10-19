# Submission checklist (BAN 6800 capstone)

Status key: **Ready** = evidence exists and is verified; **Partial** = exists with a stated limitation; **Missing** = not available yet.
The final report must not be written until every Missing item is either supplied or explicitly recorded as absent in the report.

| Brief requirement | Evidence | Status |
| --- | --- | --- |
| Report, 2,500-3,000 words, visuals embedded | Not started (by instruction) | Missing |
| 1. Business problem + Ethical AI Charter (Module 1) | Module 1 Vision Document (supplied .docx) | Ready |
| 2. Project overview; objectives, sources, timeline, stakeholders, milestones | Module 2 .docx - contains ED wait-time / churn template text that must not be carried into the report | Partial |
| 2. Project Board screenshots (Module 2 sprints) | None found in any repository or upload | **Missing - student to supply or record as unavailable** |
| 2. GitHub repository link (Modules 2-5 code) | `Oreoofe/BAN6800-Careconnect`, branch `capstone-verify` (Modules 3-4 code rebuilt, see README) | Partial - not merged to `main` |
| 3. Data pipeline, quality, governance & privacy, bias detection | `careconnect/pipeline/`, `docs/data_governance_privacy.md`, `artifacts/pipeline/*.json` | Ready |
| 4. Model development & validation; MLflow tracking + registry screenshot | `model_results.json`, `verification_report.json`, `docs/evidence/mlflow_*.png` | Ready |
| 5. SHAP global/local | `figures/shap_summary.png`, reason codes, additivity check | Ready (limits documented) |
| 5. LIME | `figures/lime_*`, fidelity table | Partial - failed fidelity bar, reported as limitation |
| 5. Counterfactuals | `explanations.json` | Partial - negative finding under Option B |
| 5. Fairness metrics (DP, EO, DI) + mitigation documentation | `policy_comparison.json`, `committee_decision_pack.md` | Ready - fairness requirement NOT met by any policy |
| 5. Model card, limitations statement, explanation store | `docs/model_card.md`, API `logs/explanations.jsonl` | Ready |
| 6. 2-3 key visuals; live dashboard URL; explainability dashboard | Streamlit app (7 pages) tested headlessly on v2 | Partial - live URL still serves v1 until a merge is approved |
| 7. Implementation & monitoring plan; fairness monitoring; drift; incident response; decommissioning | `monitor.py`, `monitoring.json`, `docs/governance.md` | Ready (replay, not live) |
| 7/Tech. Monitoring dashboard URL | Dashboard page 7 + Evidently HTML | Partial - same v1/v2 caveat |
| 8. Ethical AI framework, AIA, regulatory report, public trust statement, governance structure | `docs/governance.md`, `docs/algorithmic_impact_assessment.md`, `docs/regulatory_assessment.md`, `docs/public_trust_statement.md` | Ready (not legal advice) |
| Tech. Live API endpoint (publicly accessible) | API runs locally / Docker; no hosting account | **Missing - record as not deployed or supply a host** |
| Tech. CI/CD (GitHub Actions) | `ci.yml`, `build.yml`; runs listed in `docs/verification_log.md` | Ready |
| Tech. Docker Compose | api + dashboard services tested in CI; pipeline, mlflow, monitoring services not run | Partial |
| Tech. Model retraining | `build.yml` monthly schedule + promotion gate | Ready (never auto-promotes) |
| AI Disclosure Form (official NXU form) | Not supplied | **Missing - student to supply the form; substantial AI involvement must be disclosed** |
| NXU-formatted references | NXU referencing guide not supplied; APA 7 used in modules | **Missing - guide needed to verify format** |
