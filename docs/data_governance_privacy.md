# Data Governance and Privacy

## Data inventory and lineage

| Zone | Content | Location | Identifiers | In Git? | Retention |
| --- | --- | --- | --- | --- | --- |
| Raw | Kaggle "Medical Appointment No Shows" CSV (110,527 rows, 14 columns), SHA-256 pinned | `data/raw/` | PatientId, AppointmentID | No (git-ignored, CI check) | Production: 30 days (Module 1) |
| Processed | Modelling table: 110,514 rows, 16 features + target + pseudonym + dates | `data/processed/` | Salted HMAC-SHA256 pseudonym only | No | Production: 7 years, de-identified |
| Artifacts | Aggregate reports, model, policy, figures | `artifacts/` | None (aggregates; example patients carry no identifiers) | Yes | Versioned with the model |
| Audit | Pipeline audit log; API explanation store | `logs/` | Request id, input hash; no raw features | No | 7 years |

Lineage is recorded by `careconnect/pipeline/run.py` in `artifacts/pipeline/lineage.json` and in the append-only audit log (stage, rows in/out, git SHA, config hash). Stages: EXTRACT (checksum) -> VALIDATE (data contract, blocking) -> CLEAN -> ANONYMIZE -> FEATURE_ENGINEERING -> BIAS_CHECK (blocking) -> PUBLISH.

## Data quality controls

| Control | Rule | Result on real data |
| --- | --- | --- |
| Source integrity | SHA-256 of the CSV must match the pinned value | Pass |
| Data contract | 23 named checks: schema, identifiers, domains, ranges, dates, completeness, label-rate plausibility; schema or domain failures halt the run | 21 pass; 2 non-blocking warnings (8 ages outside 0-100; 5 bookings after the visit) |
| Cleaning | Remove duplicate appointment ids, impossible ages, inverted dates; Handcap outside 0-4 set to 0 | 13 rows removed, 0 duplicates |
| Leakage | Raw label text dropped at source; prior-visit features use only visits that happened before the booking day | Verified against a brute-force recount (0 mismatches on 3,000 patients) |
| Representation | Five gating checks (gender, age bands, neighbourhood concentration, scholarship levels, gender label-rate ratio) | All pass |

## Privacy controls

- **Minimisation:** AppointmentID dropped; PatientId pseudonymised and used only to build history features and patient-grouped validation. A model without protected or sensitive health inputs is trained and reported as an ablation, so the committee can see what those inputs add.
- **Pseudonymisation:** HMAC-SHA256 with a secret salt (`CARECONNECT_SALT`, held in a secret store; the default value is for local runs only).
- **k-anonymity on neighbourhood:** neighbourhoods with fewer than 30 records grouped as OTHER (81 -> 78 values).
- **Access (designed for production):** role-based access as in Module 1 (coordinators see score and reasons for their patients; operations see aggregates only; data science sees de-identified data; PHI access audit-logged); MFA; least privilege.
- **Security (implemented in the prototype):** non-root containers; API key and rate limit; input validation; no raw features in logs; secrets via environment, never committed.
- **Patient rights (designed):** notice in the public trust statement; opt-out honoured upstream (opted-out patients are never scored); access and complaint route through the Patient Advocate.

## Bias detection results (pipeline)

The five gating representation checks pass. The informational check shows real base-rate differences: no-show rate by age band ranges from 15.5% (senior) to 26.1% (teen), and scholarship holders miss 23.7% of visits against 19.8% for non-holders. These differences are not data errors; they explain why the model's flag rates differ by age and why the fairness limit is defined on error rates.
