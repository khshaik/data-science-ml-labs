# Assignment Requirement Gap Analysis

**Assignment:** Design and Build a Mini Production ML System

**Project audited:** Enterprise MLOps Churn Prediction System

**Audit date:** 08 August 2026

**Repository state:** audited working tree based on local `main` at `dae0856`; hosted repository state must be re-verified after the approved changes are committed and pushed

**Purpose:** assess every stated deliverable and requirement against committed code, documentation, artifacts, and fresh execution evidence.

## 1. Assessment method

This review uses the assignment brief as the source of truth. Existing project summaries were treated as navigation aids, not proof. Evidence was checked in:

- committed source, tests, configuration, documentation, and GitHub workflow files;
- local model, evaluation, monitoring, benchmark, MLflow, and prediction artifacts;
- the Git index, to distinguish locally present evidence from files actually available through the repository link;
- a fresh unit-test and coverage run;
- all six rendered pages of the final PDF.

### Status definitions

| Status | Meaning |
|---|---|
| **Complete** | The requirement is implemented or documented at the depth requested, with credible evidence. |
| **Partial** | Material coverage exists, but proof, integration, accuracy, or documentation needs improvement. |
| **Not covered** | No meaningful implementation or documentation was found. |
| **Optional prototype** | Optional scope exists but is not fully operational or verified. It does not reduce core compliance. |

## 2. Executive assessment

The submission covers the complete core assignment workflow: data quality and ingestion, six engineered features, leakage-safe preprocessing, two-model training and comparison, validation-based promotion, online and batch inference, performance measurement, drift checks, retraining decisions, monitoring design, an incident scenario, tests, configuration, a six-page design PDF, and a separate architecture image. Saved models, fitted preprocessing, and compact evaluation reports are included in the Git handoff, and the quick-start now distinguishes verified core paths from optional prototypes.

No mandatory top-level component is absent. Representative ingestion and local monitoring-stack evidence are now retained, so the remaining work is optional production hardening rather than missing core design or evidence. The highest-value correction is:

1. run the implemented CI workflow on hosted infrastructure and retain its successful run URL;

### Coverage summary

| Area | Assessment | Main remaining risk |
|---|---|---|
| Deliverables | **Complete** | Runtime models, preprocessor, compact evaluation reports, and representative ingestion evidence are included. |
| A. Data & Features | **Complete** | Ingestion execution proof and exact dataset provenance/rights are retained. |
| B. Training & Evaluation | **Complete** | Test coverage of the orchestration script is relatively low. |
| C. Serving & Inference | **Complete** | Saved artifacts permit immediate champion loading after dependency setup. |
| D. Monitoring & Retraining | **Complete for required scope** | Local dashboard/rule wiring is verified; Alertmanager has internal audit delivery and a secret-managed external routing template. Real external delivery and delayed-label KPI collection remain environment-dependent. |
| Code quality & reproducibility | **Complete for core** | Hosted execution evidence for the implemented optional CI workflow remains. |
| Design & communication | **Complete** | A few claims in the long Markdown design document are older than the final PDF. |

### Rubric-weighted readiness estimate

This is not a prediction of the instructor's exact mark. Based only on observable alignment, the project appears ready for approximately **26.5-27.5 out of 28**, with the lower end reflecting repository reproducibility and evidence-retention concerns rather than missing core ML-system components.

## 3. Deliverables checklist

| ID | Expected deliverable | Status | Evidence | Gap / improvement |
|---|---|---|---|---|
| DEL-01 | Code repository, zipped or linked | **Complete locally; push verification pending** | The project resides in the linked `data-science-ml-labs` repository and the complete handoff is present in the audited working tree. | After commit and push, confirm the public link opens without authentication and displays the final commit and retained runtime artifacts. |
| DEL-02 | Training pipeline | **Complete** | `src/training/train.py` performs raw load, stratified split, feature creation, train-only preprocessing fit, training, validation/test evaluation, artifact saving, and MLflow logging. | Add a single end-to-end smoke test for the orchestration path. |
| DEL-03 | Inference service | **Complete** | `src/serving/api.py` provides FastAPI `/predict`, `/health`, `/metrics`, and optional `/explain`; required saved artifacts are included in the Git handoff. | None required. |
| DEL-04 | Batch or micro-batch ingestion | **Complete** | `src/data/ingestion.py` reads, validates, merges, deduplicates, and writes training data plus a timestamped JSON audit record. `artifacts/logs/ingestion_20260808_082509.json` retains a successful 7,043-row end-to-end reproducibility replay. | The replay uses the authoritative repository dataset and is explicitly not represented as newly arrived production data. |
| DEL-05 | Basic tests and configs | **Complete** | 100 unit tests, 17 integration tests (including five real saved-artifact serving checks), `config/config.yaml`, `config/feature_config.yaml`, and `pytest.ini`. | Retain a successful hosted CI run URL. |
| DEL-06 | Design document: 4-6 pages or about 1,500-2,000 words | **Complete** | `output/pdf/enterprise_mlops_churn_submission.pdf` is exactly 6 A4 pages and 1,523 extracted words. All pages were rendered and visually checked. | None required. |
| DEL-07 | Problem definition and metrics | **Complete** | PDF page 1 defines target, operational decision, recall objective, AUC guardrail, and latency goal. | Optional: add an explicit cost-based threshold-selection calculation. |
| DEL-08 | Data and feature design | **Complete** | PDF page 3, `config/feature_config.yaml`, and `docs/dataset_provenance_and_license.md` describe the exact IBM sample lineage, rights label, integrity digest, assumptions, six features, availability, and skew controls. | None required. |
| DEL-09 | Model choice and evaluation | **Complete** | PDF page 4 and committed compact `artifacts/eval/*` reports compare Logistic Regression and TensorFlow NN using validation and untouched test results. | None required. |
| DEL-10 | Serving and inference pattern | **Complete** | PDF pages 2 and 5 explain online plus batch paths, users, latency, and throughput. | None required. |
| DEL-11 | Data pipeline and retraining strategy | **Complete** | PDF pages 2, 3, and 6 plus ingestion and retraining modules. | Scheduling remains external, which is allowed by the brief and is correctly disclosed. |
| DEL-12 | Monitoring plan and basic alerts | **Complete; locally verified** | PDF page 6, API metrics, DQ/drift code, four mounted Prometheus rules, provisioned Grafana datasource/dashboard, Alertmanager internal audit routing, and a secret-managed external webhook/Slack/email template. | Retain real external delivery evidence after approved credentials are supplied. |
| DEL-13 | Trade-offs, limitations, future work | **Complete** | PDF page 6 and `docs/design_document.md` cover model complexity, recall/precision, hybrid inference, limitations, and next work. | Align older Markdown wording with final measured model results. |
| DEL-14 | Architecture diagram as one image | **Complete** | `docs/architecture_diagram.svg` and PDF page 2 show source, ingestion/DQ, features/prep, training, champion, online/batch serving, observation, and retraining loop. | Optional registry is represented by the champion manifest rather than a full registry, which is acceptable. |

## 4. Section A - Data & Features (7/28)

| ID | Requirement / sub-topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| A-01 | Choose a supported ML task | **Complete** | Binary churn classification is explicit throughout. | None. |
| A-02 | Describe source / dataset | **Complete** | The IBM sample name, original filename, IBM archived reference, Kaggle catalogue record, 7,043 rows, 21 columns, access date, local rename, file size, and SHA-256 are documented. | None. |
| A-03 | Define target label | **Complete** | `Churn` Yes/No and positive class are explicit. | None. |
| A-04 | State assumptions | **Complete** | Cross-sectional teaching data, no causal claim, point-in-time limitations, class imbalance, and request-time availability are described. | None. |
| A-05 | Cleaning steps | **Complete** | Numeric conversion/fill for `TotalCharges`, identifier removal for modelling, schema/range/missing/duplicate/consistency checks. | Reconsider whether all missing `TotalCharges` should always become zero; explicitly tie this rule to tenure-zero customers. |
| A-06 | At least five non-trivial features | **Complete** | Six features: average charge ratio, service count, tenure bin, payment risk, contract stability, high-value threshold. | `avg_monthly_charge` is better labelled a ratio than an aggregation. This is wording, not a functional gap. |
| A-07 | Aggregations, ratios, encodings, windows, or similar | **Complete** | The set includes a ratio, count aggregation, binning, Boolean/domain encoding, ordinal mapping, and train-derived threshold. | No temporal window is possible with this cross-sectional dataset; not required because the brief gives examples, not a mandatory mix. |
| A-08 | Document offline versus online features | **Complete** | Feature table marks all six as available offline and online and explains availability/skew control. | None. |
| A-09 | Awareness of training-serving skew | **Complete** | Shared module, persisted p75 threshold, train-only fitted preprocessors, and consistency tests are documented. Online transformation fails closed when the threshold is unavailable. | None. |
| A-10 | Same feature logic in training and serving | **Complete** | Both training and serving call `FeatureEngineer`; both serving modes load the saved threshold and preprocessor. A retained-artifact integration test proves offline batch and online API final-vector parity. | None. |
| A-11 | Prevent leakage | **Complete** | Raw data is split 60/20/20 before fitting the percentile, label encoders, and scaler. Promotion uses validation, with test retained for final estimates. | Strong implementation; preserve this ordering. |
| A-12 | Read new batch file(s) | **Complete** | CLI accepts input CSV and quality-checks it. | None. |
| A-13 | Append or merge to training table/file | **Complete** | Existing rows precede incoming rows; duplicate `customerID`s retain the last row by ingestion order. The no-event-timestamp boundary is explicit in code, tests, and documentation. | A future timestamp contract is optional production hardening, not an assignment gap. |
| A-14 | Log N rows and date | **Complete** | Summary includes timestamp, new, existing, total, deduplicated counts and the deduplication policy. `artifacts/logs/ingestion_20260808_082509.json` retains a representative successful replay. | None. |

**Section A readiness:** approximately **6.5-7.0/7**. The only realistic grading deductions would be evidence retention or imprecise source attribution.

## 5. Section B - Model Training & Offline Evaluation (7/28)

| ID | Requirement / sub-topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| B-01 | Repeatable training script/pipeline | **Complete** | Module CLI and configuration drive the complete workflow. | None. |
| B-02 | Load data | **Complete** | Configured raw CSV loaded in `TrainingPipeline.run`. | None. |
| B-03 | Train/validation or train/validation/test split | **Complete** | Deterministic, stratified 60/20/20 split. | None. |
| B-04 | Train model | **Complete** | Balanced Logistic Regression baseline and TensorFlow 2.13 neural candidate. | None. |
| B-05 | Evaluate model | **Complete** | Accuracy, precision, recall, F1, AUC, and confusion matrix on validation and test. | Add calibration or PR-AUC only as an optional improvement for imbalance; not required. |
| B-06 | Save artifacts | **Complete locally; repository evidence partial** | Models, preprocessor, feature threshold, evaluation JSON/Markdown, champion manifest, and MLflow DB exist locally. | Model binaries, preprocessor, and most evaluation reports are Git-ignored. Commit compact reports and either distribute small binaries or clearly require training before serving. |
| B-07 | Choose and justify metrics | **Complete** | AUC and recall are primary; precision/F1/accuracy and latency are secondary, with false-negative/false-positive reasoning. | Replace generic dollar ranges with sourced or explicitly hypothetical assumptions. |
| B-08 | Baseline model | **Complete** | Balanced Logistic Regression. | None. |
| B-09 | Candidate model | **Complete** | TensorFlow/Keras network `[64, 32, 16]`, dropout, early stopping, LR reduction, deterministic seeds, and balanced class weights. | None. |
| B-10 | Compare at least two versions | **Complete** | Validation comparison covers five metrics, followed by separate test estimates. | None. |
| B-11 | State promotion decision | **Complete** | Candidate is correctly rejected; baseline stays champion. | None. |
| B-12 | Optional promotion guardrail | **Complete** | Candidate requires AUC >= 0.80, recall >= 0.75, and validation AUC gain >= 0. | Consider a tolerance such as `-0.01` only if supported by the intended complexity trade-off; current stricter rule is defensible. |
| B-13 | Experiment tracking / registry (optional) | **Complete as bonus evidence** | Two finished MLflow runs and a tracked MLflow SQLite database; champion manifest acts as lightweight registry. | MLflow artifact directory is not committed; DB portability should be tested from a fresh clone. |

**Section B readiness:** approximately **6.75-7.0/7**. Functional requirements are complete; repository artifact policy is the main risk.

## 6. Section C - Serving & Inference (7/28)

| ID | Requirement / sub-topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| C-01 | Minimal API | **Complete** | FastAPI application is implemented. | None. |
| C-02 | `/predict` JSON endpoint | **Complete** | Pydantic request schema accepts the 19 model inputs and returns structured output. | Restrict categorical values with enums or validators to catch spelling errors earlier. |
| C-03 | Prediction in response | **Complete** | Probability, Yes/No class, and Low/Medium/High risk band returned. | Threshold is fixed at 0.5; document that it has not been economically optimized. |
| C-04 | Model version in response | **Complete** | Version comes from `models/current_best.json`. | For stronger lineage, include training run ID or model checksum. |
| C-05 | Choose inference pattern | **Complete** | Hybrid online request-response plus offline chunked batch scoring. | None. |
| C-06 | Explain whether a human waits | **Complete** | Customer-service agent use case is synchronous; marketing campaign path is batch. | None. |
| C-07 | Explain acceptable latency | **Complete** | p95 target below 200 ms. | None. |
| C-08 | Explain batch/streaming nature | **Complete** | Monthly marketing batch and live agent lookup are separated. No unjustified streaming claim. | None. |
| C-09 | Measure latency / throughput | **Complete** | Fresh retained benchmark: sequential average 9.56 ms, p95 10.21 ms; concurrent 126.29 req/s with 100/100 success. Batch output for 7,043 rows is retained locally. | Benchmark is localhost functional evidence, not a production capacity test; the PDF states this correctly. |
| C-10 | Batch total time and rows/sec | **Complete** | Batch code logs total time, rows/sec, and average per-row time; batch execution is described in the PDF. | Retain a dedicated batch benchmark JSON if stronger evidence is desired. |
| C-11 | Containerization (optional) | **Locally verified subset** | Clean TensorFlow/FastAPI image build and health-gated API/Prometheus/Grafana Compose path passed. | MLflow Compose startup was outside this verification; TensorFlow still makes the serving image larger than a baseline-only image. |

**Section C readiness:** approximately **7.0/7** for mandatory scope.

## 7. Section D - Monitoring, Data Quality & Retraining (5/28)

| ID | Requirement / sub-topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| D-01 | Infra: average latency | **Complete** | Benchmark reports average; Prometheus histogram records request latency. | A dashboard query for explicit average could be added. |
| D-02 | Infra: p95 latency | **Complete** | Benchmark and Prometheus histogram quantile. | None. |
| D-03 | Infra: error rate | **Complete in plan; partial metric semantics** | `prediction_errors_total` and alert rule exist. | Alert currently measures errors/second, not errors divided by all requests. Add a total request counter and ratio expression. |
| D-04 | Data counts | **Complete** | DQ and ingestion reports contain row counts; monitoring plan includes them. | Production freshness/count export to Prometheus is not implemented, but the assignment only requires a plan plus one lightweight check. |
| D-05 | Missing values | **Complete** | Per-column missing-rate check and threshold. | None. |
| D-06 | Basic drift signals | **Complete** | PSI and KS for continuous fields, chi-squared for categorical fields, saved report and warning behavior. | Current demo compares two partitions of one historical dataset; add a realistic “recent batch” fixture for stronger evidence. |
| D-07 | Model metric on labeled feedback | **Complete as plan** | Weekly AUC/recall and threshold alerts are documented. | No delayed-label join or automated calculation exists; correctly treated as future work. |
| D-08 | Business KPI | **Complete as plan** | Actual churn and retention-campaign ROI are documented. | No automated campaign outcome feed; correctly disclosed. |
| D-09 | Dashboards and alerts | **Complete; locally verified** | Prometheus loaded four mounted rules; Grafana provisioned the datasource/dashboard; Alertmanager routes to an internal audit sink and can fan out to credential-managed webhook, Slack, and email channels. | Real external destinations are intentionally uncommitted and still require target-environment delivery verification. |
| D-10 | State dashboard/alert audience | **Complete** | Engineers, data scientists, and business stakeholders have distinct views and signals. | None. |
| D-11 | Implement lightweight drift or quality check | **Complete** | Both data-quality checks and statistical drift checks are implemented and tested. | Exceeds the minimum. |
| D-12 | Log warning on drift/quality issue | **Complete** | Both modules log pass/warning/failure outcomes and save JSON. | None. |
| D-13 | Define 2-3 retraining signals | **Complete** | Four signals: labeled-data count, AUC drop, drift score, and model age. | None. |
| D-14 | Pseudocode or short function | **Complete** | Executable `RetrainingTrigger.should_retrain` plus four scenario logs. | None. |
| D-15 | Scheduler wiring not required | **Complete and correctly bounded** | Cron is configuration/proposal only; docs say external/human/CI execution remains necessary. | None. |
| D-16 | Incident scenario | **Complete** | MonthlyCharges schema/format change, detection, batch rejection, serving continuity, fix, revalidation, and post-mortem are described. | Avoid saying “rollback to previous day's data” unless that snapshot mechanism is actually implemented; frame as proposed response. |

**Section D readiness:** approximately **4.5-5.0/5**. Core required design and lightweight code are complete. Optional observability deployment remains incomplete.

## 8. Code quality and reproducibility (1/28)

| ID | Topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| QR-01 | Modular code structure | **Complete** | Separate data, features, models, training, serving, monitoring, and retraining packages. | None. |
| QR-02 | Configuration-driven behavior | **Complete** | Split ratios, thresholds, paths, model settings, and serving settings are in YAML. | Some file paths remain hard-coded in feature/API code; centralize them if polishing. |
| QR-03 | Dependency reproducibility | **Complete in current environment; portability risk** | Versions are pinned and the existing Python 3.9.6 environment executes TensorFlow 2.13 successfully. | A clean install was not performed. TensorFlow wheels vary by OS/CPU; document supported platform and Python version. |
| QR-04 | Tests | **Complete** | Fresh run: 100 unit and 17 integration tests passed, 0 failed. The integration suite includes five real startup/champion/serving checks and 12 dependency/documentation/notification contracts. | Retain the first successful hosted CI run as external execution evidence. |
| QR-05 | Coverage | **Adequate; improvement recommended** | Fresh source coverage is 69%; drift and retraining are high, while `train.py` is 31%. | Add orchestration error-path and end-to-end training smoke tests. |
| QR-06 | Fresh-checkout usability | **Complete after dependency setup** | Dataset, models, preprocessor, feature threshold, evaluation reports, configs, PDF, MLflow DB, benchmark evidence, and champion manifest are included. Champion loading was verified from an export of the Git index. | A clean dependency installation remains platform-sensitive because TensorFlow wheels vary by OS/CPU. |
| QR-07 | Documentation consistency | **Complete for the runbook** | `QUICKSTART.txt` now provides a saved-artifact fast path, full reproduction path, verified results, provenance note, evaluation exit-code behavior, explicit automation boundaries, and a prototype-status table. | Keep figures and statuses synchronized after future executions. |
| QR-08 | CI (optional) | **Implemented; hosted run pending** | A discoverable repository-root workflow passes artifacts between jobs, accepts either governed promotion outcome, runs 100 unit tests plus all 17 integration tests, smoke-tests the Docker image, and emits a non-deployment release summary. | Run it on GitHub-hosted infrastructure and retain the successful run URL before claiming hosted CI verification. |

## 9. Design document and communication (1/28)

| ID | Topic | Status | Evidence and assessment | Gap / recommended action |
|---|---|---|---|---|
| DOC-01 | Page/word constraint | **Complete** | Exactly 6 pages and 1,523 extracted words. | None. |
| DOC-02 | Visual quality | **Complete** | Six rendered pages show consistent hierarchy, legible tables, page numbering, and no clipping/overlap. | None. |
| DOC-03 | Architecture clarity | **Complete** | One-page workflow plus trigger and artifact-contract tables. | None. |
| DOC-04 | Evidence-backed communication | **Complete** | Four evidence panels cover DQ, model comparison, live API response, and latency benchmark. | Add one ingestion panel only if space permits; the current document already meets the brief. |
| DOC-05 | Honest scope boundaries | **Complete** | PDF distinguishes verified local paths from prototypes and external scheduling. | Preserve this language. |
| DOC-06 | Cross-document consistency | **Partial** | Final PDF, README, and quick-start reflect the current champion, results, provenance, and verification boundaries. | Long Markdown design text still contains older wording such as neural-network “better performance” in a trade-off subsection. |

## 10. Optional / bonus components

These items are useful but are not required by the attached assignment brief.

| Component | Status | Evidence | Required improvement before claiming completion |
|---|---|---|---|
| Docker API image | **Prototype** | `docker/Dockerfile.api` | Build and health-test from a clean checkout. |
| Docker Compose stack | **Partially verified** | API/Prometheus/Grafana path and Alertmanager/internal-sink path were each exercised locally | Verify MLflow separately and rerun all services together before claiming the complete stack as one deployment. |
| MLflow | **Locally verified** | Tracked `mlflow.db` with two finished runs | Confirm DB/artifact links work after cloning elsewhere. |
| Prometheus / Alertmanager | **Prometheus verified; notification routing implemented** | API target up, request metric scraped, four rules loaded, internal audit receiver configured, external webhook/Slack/email template uses file-backed secrets | Retain external delivery evidence after approved credentials are supplied. |
| Grafana | **Locally verified** | Prometheus datasource and churn dashboard provisioned; health/database APIs passed | Use production authentication and persistence controls for non-local deployment. |
| Streamlit | **Prototype** | `webapp/app.py` | Run and test; either add to Compose or keep documented as separate. |
| SHAP/LIME explanation | **Prototype / unit-tested** | explainer module and optional API endpoint | Execute `/explain` with the champion and retain response evidence. |
| CI/CD | **Implemented; locally validated** | Repository-root GitHub Actions workflow and five passing saved-artifact integration tests | Retain a successful hosted Actions run URL. |

## 11. Prioritized improvement backlog

### P0 - submission/reproducibility risk

None currently identified. The retained ingestion record is explicitly labelled as a reproducibility replay rather than a newly arrived production batch.

### P1 - strengthen correctness and grading confidence

1. **Add an end-to-end training-orchestration smoke test** covering load -> raw split -> train-only feature/preprocessor fit -> baseline training -> evaluation -> champion manifest -> API prediction.
2. **Evaluate categorical preprocessing migration.** The current fitted integer mappings and their unseen-value behavior are explicit and tested. Any future one-hot or unknown-sentinel migration must include full retraining, evaluation, artifact replacement, and renewed serving-parity evidence.
3. **Clarify threshold economics.** State that 0.5 is a demonstration threshold and add an optional validation-only cost/recall threshold analysis.
4. **Preserve dataset provenance.** Keep the source URLs, access date, rights statement, and checksum synchronized if the dataset file changes.

### P2 - optional production extensions

5. Supply approved external Alertmanager destinations and retain webhook/Slack/email delivery evidence.
6. Run the repaired GitHub Actions workflow on a hosted runner and retain the successful run URL.
7. Add delayed-label joins and scheduled AUC/recall computation plus retention-campaign outcome/ROI ingestion.
8. Activate recurring scheduling if the prototype is promoted beyond local demonstration.
9. Add a stable source event/update timestamp and snapshot version if the upstream data contract evolves beyond ingestion-order semantics.

## 12. Final conclusion

The project meets the assignment's mandatory breadth and demonstrates the intended M1-M11 production-ML concepts. Its strongest evidence is the leakage-safe training workflow, validation-only champion selection, shared online/offline feature path, executed API/batch benchmark, executable drift and retraining logic, comprehensive basic tests, and polished six-page submission.

The project handoff now includes runtime artifacts, an accurate runbook, and retained end-to-end ingestion evidence. Remaining work is optional production hardening and hosted-service verification rather than a core submission or reproducibility gap.
