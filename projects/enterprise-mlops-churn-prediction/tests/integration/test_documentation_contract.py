"""Keep operational documentation aligned with governed project artifacts."""

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN_DOCUMENT = ROOT / "docs/design_document.md"
CHAMPION_MANIFEST = ROOT / "models/current_best.json"
MODEL_COMPARISON = ROOT / "artifacts/eval/model_comparison.json"
TEST_SUMMARY = ROOT / "artifacts/test_summary.json"
TEST_COUNT_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "QUICKSTART.txt",
    ROOT / "docs/design_document.md",
    ROOT / "docs/assignment_alignment_and_workflow.md",
    ROOT / "docs/requirement_gap_analysis.md",
)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _collected_test_count(test_path: str):
    """Ask pytest for the effective count, including parametrized cases."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", test_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    match = re.search(r"collected (\d+) items?", output)
    assert match, output
    return int(match.group(1))


def test_design_model_selection_matches_governed_artifacts():
    """The design must name the champion selected by the evaluation outputs."""
    design = DESIGN_DOCUMENT.read_text()
    champion = _load_json(CHAMPION_MANIFEST)
    comparison = _load_json(MODEL_COMPARISON)
    decision = comparison["promotion_decision"]

    assert champion["model_type"] == "baseline"
    assert decision["should_promote"] is False
    assert decision["promoted_model"] == champion["model_type"]
    assert champion["model_version"] in design
    assert "KEEP BASELINE" in design
    assert "Logistic Regression" in design


def test_design_reports_validation_metrics_used_for_selection():
    """Selection claims must be traceable to validation, not final-test metrics."""
    design = DESIGN_DOCUMENT.read_text()
    comparison = _load_json(MODEL_COMPARISON)["comparison"]

    assert comparison["selection_dataset"] == "validation"
    assert f"AUC {comparison['baseline']['auc']:.4f}" in design
    assert f"AUC {comparison['candidate']['auc']:.4f}" in design
    assert f"AUC gain {comparison['differences']['auc']:.4f}" in design


def test_stale_candidate_selection_claims_cannot_return():
    """Reject the superseded recommendation that contradicted the manifest."""
    design = DESIGN_DOCUMENT.read_text()
    stale_claims = (
        "Neural Network: Better performance",
        "Logistic Regression: Explainable but lower accuracy",
        "Use Neural Network + SHAP/LIME for explanations",
    )

    for claim in stale_claims:
        assert claim not in design


def test_documented_test_counts_match_pytest_collection():
    """Keep retained evidence and active documentation tied to test discovery."""
    unit_count = _collected_test_count("tests/unit")
    integration_count = _collected_test_count("tests/integration")
    total_count = unit_count + integration_count
    summary = _load_json(TEST_SUMMARY)

    assert summary["unit_tests"] == unit_count
    assert summary["integration_tests"] == integration_count
    assert summary["tests_collected"] == total_count
    assert summary["tests_passed"] == total_count
    assert summary["tests_failed"] == 0

    for path in TEST_COUNT_DOCUMENTS:
        document = path.read_text().lower()
        assert f"{unit_count} unit" in document, path
        assert f"{integration_count} integration" in document, path
