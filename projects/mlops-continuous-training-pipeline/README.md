# Iris Continuous Training System

End-to-end MLOps pipeline on the Iris dataset — drift detection, automated
retraining, evaluation gates, and MLflow tracking throughout.

---

## Files at a Glance

| File | Purpose |
|------|---------|
| `config.yaml` | All thresholds, paths, SLOs, MLflow settings — edit here, not in code |
| `data_generator.py` | Creates `reference.csv`, `production.csv` (with drift), `golden.csv` |
| `train.py` | Trains RandomForest → logs to MLflow → saves candidate model |
| `evaluate.py` | Runs 5 gates → promotes candidate to MLflow Model Registry |
| `monitor.py` | Computes PSI / KS / JSD drift + EvidentlyAI report → MLflow run |
| `retrain_trigger.py` | Evaluates all 4 trigger types → logs decision to MLflow |
| `pipeline.py` | **One command to run everything** — monitor → trigger → train → evaluate → promote |
| `requirements.txt` | All Python dependencies |
| `tests/test_pipeline.py` | 14 unit tests |
| `.github/workflows/continuous_training.yml` | GitHub Actions CI/CD |

---

## Execution Order

### Step 1 — First-time setup

```bash
pip install -r requirements.txt
python data_generator.py        # creates data/reference.csv, production.csv, golden.csv
```

### Step 2 — Initial training and deployment

```bash
python train.py                 # trains model, logs to MLflow, saves models/candidate_model.pkl
python evaluate.py --promote    # runs 5 gates, promotes to MLflow Registry as Production
```

At this point your model is live in the MLflow Model Registry (stage = Production).

### Step 3 — Check drift and monitoring

```bash
python monitor.py               # runs PSI / KS / JSD on all features, saves drift_report.html
```

Open `reports/drift_report.html` in a browser for the interactive EvidentlyAI report.

### Step 4 — Check whether retraining is needed

```bash
python retrain_trigger.py       # evaluates time / performance / drift / active-learning triggers
```

Prints `DECISION: RETRAIN 🚨` or `DECISION: NO ACTION ✅`.

### Step 5 — Retrain (if triggered)

```bash
python train.py --combine       # retrains on reference + production data combined
python evaluate.py --promote    # runs gates again, promotes new version if it passes
```

---

## One-Command Automated Pipeline

Instead of running steps 3–5 manually, run the full pipeline:

```bash
python pipeline.py              # monitor → trigger → retrain (if needed) → evaluate → promote
python pipeline.py --retrain    # skip trigger check, force a retrain
python pipeline.py --dry-run    # run everything but do NOT promote
python pipeline.py --monitor-only  # check drift and stop, no retrain
```

This is what the GitHub Actions workflow runs on every push and weekly schedule.

---

## View Results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Every training run, monitoring cycle, trigger decision, and pipeline execution
is logged here with full metrics, artefacts, and tags.

---

## What Gets Logged to MLflow

| Script | Experiment | Logged |
|--------|-----------|--------|
| `train.py` | `iris_ct` | Hyperparams, CV scores, accuracy/F1/recall per class, feature importances, confusion matrix PNG, model → Registry |
| `evaluate.py` | `iris_ct` (same run) | Gate pass/fail, Registry transition None→Staging→Production |
| `monitor.py` | `iris_ct_monitoring` | PSI/KS/JSD per feature, class shift, EvidentlyAI HTML report |
| `retrain_trigger.py` | `iris_ct_monitoring` | Per-trigger metrics, final retrain=true/false tag |
| `pipeline.py` | `iris_ct` | Parent run with all child runs nested inside, pipeline duration, outcome |

Load the Production model at any time:
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/iris_rf_model/Production")
```

---

## The Four Retraining Triggers

| # | Trigger | Fires when | Config |
|---|---------|-----------|--------|
| 1 | **Time-based** | Last retrain > 7 days ago | `triggers.time_based.interval_days` |
| 2 | **Performance** | Accuracy < 0.85 or F1 < 0.82 on labelled production data | `slo.accuracy.critical` |
| 3 | **Drift-based** | PSI > 0.20 or KS p-value < 0.05 on any feature | `drift.psi.trigger` |
| 4 | **Active learning** | > 20% of predictions below 70% confidence | `retrain_trigger.py` |

## The Five Evaluation Gates

All must pass before a candidate is promoted to Production in the Registry:

| # | Gate | What it checks |
|---|------|---------------|
| 1 | Baseline comparison | New model ≥ current Production model accuracy (−0.5% margin) |
| 2 | Minimum thresholds | Accuracy ≥ 0.85, F1 ≥ 0.82 — hard floors |
| 3 | Golden dataset | 100% correct on 15 curated representative examples |
| 4 | Per-class recall | Every class ≥ 0.80 recall — no hidden class failures |
| 5 | Distribution check | No missing values, class balance ≥ 5%, at least 30 samples |

## Run Tests

```bash
pytest tests/ -v    # 14 tests
```
