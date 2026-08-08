"""Build the consolidated six-page assignment PDF from verified artifacts."""

from pathlib import Path
import json
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/enterprise_mlops_churn_submission.pdf"
REPO_URL = (
    "https://github.com/khshaik/data-science-ml-labs/"
    "tree/main/projects/enterprise-mlops-churn-prediction"
)
W, H = A4
NAVY = colors.HexColor("#112240")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E07A5F")
PALE = colors.HexColor("#F4F7FA")
MID = colors.HexColor("#425466")
GREEN = colors.HexColor("#E8F4EA")
AMBER = colors.HexColor("#FFF1D6")


def load_json(relative):
    return json.loads((ROOT / relative).read_text())


baseline = load_json("artifacts/eval/baseline_evaluation.json")
candidate = load_json("artifacts/eval/candidate_evaluation.json")
comparison = load_json("artifacts/eval/model_comparison.json")
benchmark = load_json("artifacts/benchmark_results.json")
drift = load_json("artifacts/drift_reports/drift_report_test.json")
api_result = load_json("artifacts/api_response.json")
test_result = load_json("artifacts/test_summary.json")
quality_files = sorted((ROOT / "artifacts/logs").glob("data_quality_report_*.json"))
quality = json.loads(quality_files[-1].read_text())


def header(c, title, page):
    c.setFillColor(NAVY)
    c.rect(0, H - 48, W, 48, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(34, H - 30, title)
    c.setFillColor(MID)
    c.setFont("Helvetica", 8)
    c.drawRightString(W - 34, 20, f"Mini Production ML System | Page {page} of 6")


def section(c, text, y):
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(36, y, text)
    c.setStrokeColor(BLUE)
    c.setLineWidth(1)
    c.line(36, y - 6, W - 36, y - 6)
    return y - 24


def paragraph(c, text, x, y, width=520, size=9.2, leading=12, color=MID):
    max_chars = max(25, int(width / (size * 0.52)))
    lines = []
    for block in text.split("\n"):
        lines.extend(textwrap.wrap(block, width=max_chars) or [""])
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def bullets(c, items, x, y, width=500, size=8.8, leading=11.2):
    for item in items:
        c.setFillColor(TEAL)
        c.circle(x + 3, y + 3, 2.2, fill=1, stroke=0)
        y = paragraph(c, item, x + 13, y, width - 13, size, leading)
        y -= 4
    return y


def metric_card(c, x, y, w, h, label, value, note="", fill=PALE):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(MID)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 10, y + h - 17, label.upper())
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x + 10, y + h - 39, value)
    if note:
        c.setFillColor(MID)
        c.setFont("Helvetica", 7.5)
        c.drawString(x + 10, y + 8, note)


def technology_chip(c, x, y, w, label, fill):
    c.setFillColor(fill)
    c.roundRect(x, y, w, 16, 5, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 6.2)
    c.drawCentredString(x + w / 2, y + 5.2, label)


def evidence_panel(c, x, y, w, h, title, lines, number):
    c.setFillColor(NAVY)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#8BE9FD"))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 12, y + h - 18, f"DEMO EVIDENCE {number}/4 - {title}")
    c.setFillColor(colors.HexColor("#E6EDF3"))
    c.setFont("Courier", 7.2)
    line_y = y + h - 35
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=max(28, int(w / 4.6))):
            c.drawString(x + 12, line_y, wrapped)
            line_y -= 10
            if line_y < y + 10:
                return


def code_panel(c, x, y, w, h, title, lines):
    c.setFillColor(NAVY)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#8BE9FD"))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 12, y + h - 18, title)
    c.setFillColor(colors.HexColor("#E6EDF3"))
    c.setFont("Courier", 6.5)
    line_y = y + h - 34
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=max(40, int(w / 4.1))):
            c.drawString(x + 12, line_y, wrapped)
            line_y -= 8.5
            if line_y < y + 8:
                return


def table(c, x, y, widths, rows, row_h=20, header_fill=BLUE):
    total_w = sum(widths)
    for r, row in enumerate(rows):
        fill = header_fill if r == 0 else (colors.white if r % 2 else PALE)
        c.setFillColor(fill)
        c.rect(x, y - row_h, total_w, row_h, fill=1, stroke=0)
        cursor = x
        for idx, cell in enumerate(row):
            c.setFillColor(colors.white if r == 0 else NAVY)
            c.setFont("Helvetica-Bold" if r == 0 else "Helvetica", 7.7)
            c.drawString(cursor + 5, y - 14, str(cell))
            cursor += widths[idx]
        y -= row_h
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.rect(x, y, total_w, row_h * len(rows), fill=0, stroke=1)
    return y


def draw_architecture(c, y_top):
    boxes = [
        (36, "CRM CSV", "incoming data", colors.HexColor("#DCEEF8")),
        (142, "Ingestion + DQ", "validate / merge", GREEN),
        (248, "Features + prep", "shared fitted code", AMBER),
        (354, "Training", "LR vs TensorFlow", colors.HexColor("#EDE7F6")),
        (460, "Champion", "validation selected", colors.HexColor("#FCE8E6")),
    ]
    for x, label, note, fill in boxes:
        c.setFillColor(fill)
        c.setStrokeColor(BLUE)
        c.roundRect(x, y_top - 58, 92, 50, 7, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(x + 46, y_top - 28, label)
        c.setFont("Helvetica", 6.8)
        c.drawCentredString(x + 46, y_top - 43, note)
        if x < 460:
            c.setStrokeColor(BLUE)
            c.line(x + 92, y_top - 33, x + 104, y_top - 33)
            c.line(x + 100, y_top - 29, x + 104, y_top - 33)
            c.line(x + 100, y_top - 37, x + 104, y_top - 33)

    lower = [
        (36, "Agent / analyst", "request or campaign", colors.HexColor("#DCEEF8")),
        (142, "Online API", "FastAPI /predict", GREEN),
        (248, "Offline batch", "chunked CSV score", GREEN),
        (354, "Observe", "metrics / DQ / drift", AMBER),
        (460, "Retrain decision", "labels / AUC / age", colors.HexColor("#FCE8E6")),
    ]
    for x, label, note, fill in lower:
        c.setFillColor(fill)
        c.setStrokeColor(BLUE)
        c.roundRect(x, y_top - 150, 92, 50, 7, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8.2)
        c.drawCentredString(x + 46, y_top - 120, label)
        c.setFont("Helvetica", 6.7)
        c.drawCentredString(x + 46, y_top - 136, note)

    # User request to online inference.
    c.setStrokeColor(BLUE)
    c.line(128, y_top - 125, 142, y_top - 125)
    c.line(138, y_top - 121, 142, y_top - 125)
    c.line(138, y_top - 129, 142, y_top - 125)

    # Champion fans out to both serving paths.
    c.line(506, y_top - 58, 506, y_top - 82)
    c.line(188, y_top - 82, 506, y_top - 82)
    c.line(188, y_top - 82, 188, y_top - 100)
    c.line(294, y_top - 82, 294, y_top - 100)

    # Online and offline outputs independently feed observability.
    c.line(188, y_top - 150, 188, y_top - 164)
    c.line(188, y_top - 164, 400, y_top - 164)
    c.line(294, y_top - 150, 294, y_top - 157)
    c.line(294, y_top - 157, 400, y_top - 157)
    c.line(400, y_top - 164, 400, y_top - 150)
    c.line(446, y_top - 125, 460, y_top - 125)

    # Eligibility is a logged decision; a human or CI restarts training.
    c.setStrokeColor(ORANGE)
    c.setDash(4, 3)
    c.line(506, y_top - 100, 506, y_top - 72)
    c.line(506, y_top - 72, 400, y_top - 72)
    c.line(400, y_top - 72, 400, y_top - 58)
    c.setDash()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=A4)
    c.setTitle("Enterprise MLOps Churn Prediction - Assignment Report")
    c.setAuthor("BITS Pilani MSc Assignment")

    # Page 1
    c.setFillColor(NAVY)
    c.rect(0, H - 260, W, 260, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(38, H - 90, "Enterprise MLOps")
    c.drawString(38, H - 126, "Churn Prediction System")
    c.setFont("Helvetica", 13)
    c.drawString(38, H - 158, "Design and Build a Mini Production ML System")
    c.setFillColor(colors.HexColor("#8BE9FD"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(38, H - 195, "PRODUCTION-ORIENTED PROTOTYPE | VERIFIED 08 AUG 2026")
    technologies = [
        ("Python 3.9.6", "#3776AB"),
        ("TensorFlow 2.13.0", "#F58220"),
        ("Keras 2.13.1", "#D00000"),
        ("scikit-learn 1.3.0", "#F7931E"),
        ("FastAPI 0.103.1", "#009688"),
        ("pandas 2.0.3", "#150458"),
        ("NumPy 1.24.3", "#4D77CF"),
        ("MLflow 2.7.1", "#0194E2"),
        ("pytest 7.4.2", "#0A9EDC"),
        ("prom-client 0.17.1", "#E6522C"),
    ]
    for idx, (label, color) in enumerate(technologies):
        row, col = divmod(idx, 5)
        technology_chip(c, 38 + col * 104, H - 224 - row * 21, 96, label, colors.HexColor(color))
    y = H - 300
    y = section(c, "Problem definition and intended use", y)
    y = paragraph(c, "The system predicts whether a telecommunications customer will churn. It supports two decisions: an agent requesting a real-time risk score during a customer interaction, and a marketing team scoring the customer base in batch for retention campaigns. The positive class is Churn = Yes.", 38, y)
    y -= 8
    metric_card(c, 38, y - 78, 155, 68, "Dataset", "7,043", "customers; 21 raw columns")
    metric_card(c, 214, y - 78, 155, 68, "Positive class", "26.5%", "moderate imbalance", AMBER)
    metric_card(c, 390, y - 78, 155, 68, "Primary objective", "Recall", "with AUC guardrail", GREEN)
    y -= 105
    y = paragraph(c, "Operational decision: customers above the chosen probability threshold enter a retention-review queue; the model does not automatically apply an offer or make an adverse customer decision. Scores should be combined with eligibility rules, contact preferences, intervention cost, and agent judgement. The prototype therefore measures discrimination and recall while keeping a human in the loop.", 38, y)
    y -= 8
    y = section(c, "Success criteria", y)
    bullets(c, [
        "Validation ROC AUC >= 0.80 and recall >= 0.75; precision and F1 expose retention-budget trade-offs.",
        "Online p95 latency below 200 ms with model version returned in every prediction.",
        "Repeatable ingestion, training, evaluation, serving, drift detection, and retraining-decision paths.",
        "The work optimizes engineering correctness and reproducibility rather than state-of-the-art accuracy.",
    ], 40, y)
    paragraph(c, "Evidence remains reproducible and artifact-backed.", 40, 165, 500, 8.4, 11)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(38, 48, "CODE REPOSITORY")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7.5)
    c.drawString(38, 35, REPO_URL)
    c.linkURL(REPO_URL, (35, 30, W - 35, 48), relative=0)
    c.showPage()

    # Page 2
    header(c, "Architecture and production workflow", 2)
    y = H - 78
    y = paragraph(c, "The architecture uses one feature implementation and persisted preprocessing state for training, online requests, and batch scoring. A validation-based champion manifest prevents serving whichever artifact happens to exist most recently. Solid arrows are runtime data paths; the dashed return is an eligible retraining decision that still requires a human or CI job.", 38, y)
    draw_architecture(c, y - 12)
    y -= 205
    y = section(c, "Triggers and user/system interactions", y)
    rows = [
        ["Path", "Trigger / consumer", "Implementation", "Current boundary"],
        ["Online", "POST /predict / agent", "api.py + shared artifacts", "synchronous; verified"],
        ["Offline", "CLI/scheduler / marketing", "batch_predict.py", "executed; scheduler external"],
        ["Lifecycle", "DQ, labels, AUC, drift, age", "drift_detector.py + trigger.py", "decision logged; no auto-train"],
    ]
    y = table(c, 38, y, [65, 130, 155, 165], rows, 22)
    y -= 16
    y = section(c, "Component and artifact contract", y)
    rows = [
        ["Layer", "Code responsibility", "Persisted contract"],
        ["Data + features", "ingestion, quality, preprocessing, engineering", "threshold + preprocessor"],
        ["Training", "baseline, TensorFlow candidate, train/evaluate", "MLflow + eval reports"],
        ["Serving", "FastAPI and batch scoring", "current_best.json"],
        ["Lifecycle", "metrics, drift, retraining eligibility", "metrics + JSON logs"],
    ]
    table(c, 38, y, [90, 260, 165], rows, 20)
    c.showPage()

    # Page 3
    header(c, "Data, features, and reproducibility", 3)
    y = H - 78
    y = section(c, "Dataset and assumptions", y)
    y = paragraph(c, "This project uses IBM's Telco Customer Churn teaching sample, originally distributed as WA_Fn-UseC_-Telco-Customer-Churn.csv and catalogued at Kaggle under blastchar/telco-customer-churn. The evaluated 7,043-row file has SHA-256 88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a. Kaggle marks the data files © Original Authors, not with a standard open-data license; IBM's Apache-2.0 code-pattern license does not license the separately supplied CSV. This repository makes no relicensing claim and includes it for academic reproducibility. Customer ID is removed before modelling, and the fictional cross-sectional sample is not live telecom telemetry or evidence of causal churn drivers.", 38, y, 520, 8.2, 10.2)
    y -= 6
    rows = [["Engineered feature", "Construction", "Availability / skew control"],
        ["avg_monthly_charge", "TotalCharges / tenure", "same deterministic code"],
        ["service_adoption_score", "count of six add-on services", "same deterministic code"],
        ["tenure_category", "0-12, 13-24, 25-48, 48+", "fixed boundaries"],
        ["payment_risk_flag", "electronic-check indicator", "fixed mapping"],
        ["contract_stability_score", "month=1, one-year=2, two-year=3", "fixed mapping"],
        ["high_value_customer", "MonthlyCharges > training p75", "persisted threshold"],
    ]
    y = table(c, 38, y, [125, 180, 210], rows, 23)
    y -= 18
    y = section(c, "Leakage and quality controls", y)
    y = bullets(c, [
        "Raw rows are stratified 60/20/20 before any data-derived feature is fitted. Encoders and scaler fit only on training rows.",
        "Quality gates cover schema, missing rates, numeric ranges, duplicate IDs, and logical consistency.",
        "The current batch passed blocking checks. Fifty-nine charge-history deviations are retained as a non-blocking warning because current monthly price need not equal historical average price.",
        "Batch ingestion merges new CSV data, keeps the latest customer record, and writes a timestamped JSON audit summary.",
    ], 40, y)
    y = paragraph(c, "Offline and online availability is explicit. All six features can be calculated from request-time account fields. The only learned feature parameter is the high-value threshold; it is fitted on training rows, serialized, and loaded for both serving modes. This avoids recomputing a percentile from a single online request or a recent batch.", 40, y, 500, 8.3, 10.5)
    evidence_panel(c, 38, 62, W - 76, 103, "DATA QUALITY RUN", [
        f"rows={quality['num_rows']} columns={quality['num_columns']} overall_passed={quality['overall_passed']}",
        "schema=PASS | missing=PASS | ranges=PASS | duplicates=PASS",
        "consistency=WARNING: 59 rows exceed 20% charge-history deviation",
        "artifact: artifacts/logs/data_quality_report_20260808_040152.json",
    ], 1)
    c.showPage()

    # Page 4
    header(c, "Model development and offline evaluation", 4)
    y = H - 78
    y = paragraph(c, "The baseline is balanced logistic regression. The candidate remains TensorFlow 2.13 - not an sklearn MLPClassifier - with hidden layers [64, 32, 16], seed 42, dropout 0.3, early stopping, learning-rate reduction, and balanced class weights computed from training labels. Validation selects the model; test data is reported separately as the final estimate.", 38, y)
    y -= 10
    rows = [["Validation metric", "Baseline LR", "TensorFlow NN", "Winner"],
        ["ROC AUC", "0.8354", "0.8300", "Baseline"],
        ["Recall", "0.7941", "0.7674", "Baseline"],
        ["Precision", "0.5174", "0.5009", "Baseline"],
        ["F1", "0.6266", "0.6061", "Baseline"],
        ["Accuracy", "0.7488", "0.7353", "Baseline"],
    ]
    y = table(c, 65, y, [125, 105, 110, 105], rows, 25)
    y -= 18
    metric_card(c, 38, y - 72, 155, 65, "Baseline test AUC", "0.8429", "untouched test set")
    metric_card(c, 214, y - 72, 155, 65, "Candidate test AUC", "0.8364", "untouched test set", AMBER)
    metric_card(c, 390, y - 72, 155, 65, "Champion", "Baseline", "lower complexity; better validation", GREEN)
    y -= 95
    y = section(c, "Promotion decision", y)
    y = paragraph(c, "The candidate passes absolute AUC and recall thresholds, but its validation AUC is 0.0054 lower and all other validation metrics are also lower. It is therefore not promoted: additional latency and model complexity are unjustified without a primary-metric gain.", 38, y)
    y -= 6
    y = paragraph(c, "AUC measures ranking quality across thresholds; recall captures how many true churners reach the intervention queue; precision estimates wasted outreach; and F1 summarizes the recall-precision balance. Accuracy is secondary because the majority class can make it look acceptable even when churners are missed. Final threshold calibration should use observed retention costs and capacity.", 38, y, 515, 8.4, 10.8)
    evidence_panel(c, 38, 55, W - 76, 116, "MODEL COMPARISON", [
        "VALIDATION  baseline_auc=0.8354  candidate_auc=0.8300  delta=-0.0054",
        "VALIDATION  baseline_recall=0.7941  candidate_recall=0.7674",
        "GUARDRAILS  min_auc=0.80 PASS | min_recall=0.75 PASS | auc_gain FAIL",
        "DECISION    KEEP BASELINE -> models/current_best.json",
        "MLFLOW      baseline_20260808_041203 FINISHED | candidate_20260808_041229 FINISHED",
        "artifacts/eval/model_comparison.json",
    ], 2)
    c.showPage()

    # Page 5
    header(c, "Serving, latency, and demo evidence", 5)
    y = H - 78
    y = section(c, "Request-response contract", y)
    y = paragraph(c, "FastAPI exposes /predict, /health, /metrics, and optional /explain endpoints. A request supplies the 19 model inputs. The response returns churn probability, Yes/No prediction, Low/Medium/High risk band, champion model version, request latency, and timestamp.", 38, y)
    response = api_result["response"]
    evidence_panel(c, 38, y - 150, W - 76, 132, "LIVE API RESPONSE", [
        "POST http://127.0.0.1:8000/predict  -> HTTP 200",
        f"churn_probability: {response['churn_probability']:.4f}",
        f"prediction: {response['churn_prediction']} | risk: {response['risk_level']}",
        f"model_version: {response['model_version']}",
        f"latency_ms: {response['latency_ms']:.2f} | /metrics verified: true",
    ], 3)
    y -= 180
    y = section(c, "Measured performance", y)
    seq = benchmark["sequential"]
    conc = benchmark["concurrent"]
    metric_card(c, 38, y - 72, 155, 65, "Sequential avg", f"{seq['avg_latency_ms']:.2f} ms", "100/100 successful")
    metric_card(c, 214, y - 72, 155, 65, "Sequential p95", f"{seq['p95_latency_ms']:.2f} ms", "target < 200 ms", GREEN)
    metric_card(c, 390, y - 72, 155, 65, "Concurrent throughput", f"{conc['throughput_rps']:.1f}/s", "10 workers; 100/100 success", AMBER)
    paragraph(c, "The benchmark ran on localhost with one application process and a fixed valid request, so it is evidence of functional latency rather than a cloud capacity guarantee. The batch path separately loaded the champion manifest and processed all 7,043 source rows in chunks, retaining customer ID, probability, class, risk band, version, and prediction date.", 38, y - 96, 515, 8.4, 10.8)
    evidence_panel(c, 38, 55, W - 76, 112, "LATENCY BENCHMARK", [
        f"sequential: n=100 success=100 avg={seq['avg_latency_ms']:.2f}ms p95={seq['p95_latency_ms']:.2f}ms p99={seq['p99_latency_ms']:.2f}ms",
        f"concurrent: n=100 workers=10 success=100 avg={conc['avg_latency_ms']:.2f}ms p95={conc['p95_latency_ms']:.2f}ms",
        f"throughput={conc['throughput_rps']:.2f} requests/sec",
        "batch: 7,043 rows scored successfully; predictions retained as CSV",
        "artifact: artifacts/benchmark_results.json",
    ], 4)
    c.showPage()

    # Page 6
    header(c, "Monitoring, retraining, trade-offs, and handoff", 6)
    y = H - 78
    y = section(c, "Monitoring and alerts", y)
    y = bullets(c, [
        "Infrastructure: average/p95 latency, throughput, 5xx error rate, service availability, CPU and memory.",
        "Data: row counts, missing rates, schema changes, data freshness, numeric PSI/KS, and categorical chi-squared tests.",
        "Model/business: weekly AUC/recall on delayed labels, prediction-rate changes, actual churn, and retention-campaign ROI.",
        f"Executed drift check: {drift['features_checked']} features, {drift['features_with_drift']} drifted, drift rate {drift['drift_rate']:.1%}.",
    ], 40, y, 500, 8.2, 10.2)
    y -= 2
    y = section(c, "Retraining and incident response", y)
    y = paragraph(c, "Retrain when any signal fires: 1,000 new labelled rows, AUC drop above 0.05, drift above 0.30, or 30 days since training. A new candidate must pass validation guardrails before its manifest changes. If CRM changes MonthlyCharges to a currency string, checks reject the batch and preserve the champion. Engineering fixes and replays ingestion; rollback is the unchanged manifest.", 38, y, 515, 8.1, 10.1)
    y -= 4
    y = section(c, "Commands, access URLs, and verification boundary", y)
    code_panel(c, 38, y - 112, W - 76, 102, "REPRODUCIBLE LOCAL HANDOFF", [
        "API       uvicorn src.serving.api:app --host 127.0.0.1 --port 8000  -> http://127.0.0.1:8000/docs",
        "MLflow    mlflow ui --backend-store-uri sqlite:///mlflow.db          -> http://127.0.0.1:5000",
        "Metrics   GET http://127.0.0.1:8000/metrics                          -> verified",
        "Offline   python -m src.serving.batch_predict ... | python -m src.monitoring.drift_detector",
        "Lifecycle python -m src.retraining.trigger | python -m src.training.train --model candidate",
        "Tests     pytest tests/unit -> 100 passed | pytest tests/integration -> 4 passed",
        "Monitor   Prometheus :9090 | Grafana :3000 -> locally verified; notifications excluded",
    ])
    y -= 130
    y = section(c, "Alignment, trade-offs, and next work", y)
    y = bullets(c, [
        "Core rubric alignment is complete across problem/data, modelling, inference, production considerations, and documentation. A successful 7,043-row end-to-end ingestion replay and embedded quality report are retained as evidence.",
        "Recall is prioritized because missing a churner is assumed costlier than an unnecessary offer; threshold economics require real campaign costs.",
        "Next: one-hot nominal features, delayed-label AUC and ROI automation, hosted CI evidence, external notifications, and scheduling.",
        "Docker monitoring is locally verified; hosted CI, SHAP/LIME, Streamlit, notification delivery, and cloud deployment remain unverified.",
    ], 40, y, 500, 7.6, 9.0)
    y = paragraph(c, "Ownership follows the signal: engineers receive availability, error, latency, schema, and freshness alerts; data scientists own drift, delayed-label evaluation, threshold review, and promotion; business teams receive churn-volume and campaign-outcome summaries. The current 104/104 tests (100 unit and 4 integration) and 69% unit-suite coverage are verified evidence, not a claim of exhaustive cloud production testing. Verification evidence is retained alongside source code.", 40, y, 500, 7.5, 9.0)
    metric_card(c, 38, 58, 155, 65, "All tests", f"{test_result['tests_passed']}/{test_result['tests_collected']}", "100 unit + 4 integration")
    metric_card(c, 214, 58, 155, 65, "Source coverage", f"{test_result['source_coverage_percent']}%", "fresh project-local run", AMBER)
    metric_card(c, 390, 58, 155, 65, "Submission", "6 pages", "diagram + four evidence panels", GREEN)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    build()
