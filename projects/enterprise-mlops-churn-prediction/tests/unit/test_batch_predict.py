"""Tests for file-based batch scoring."""

from unittest.mock import Mock
import yaml
import numpy as np
import pandas as pd

from src.serving.batch_predict import BatchPredictor


def test_predict_batch_writes_predictions(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"serving": {}}))
    predictor = BatchPredictor(str(config_path))

    raw = pd.DataFrame({"customerID": ["C1", "C2"], "value": [1.0, 2.0]})
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    raw.to_csv(input_path, index=False)

    predictor.feature_engineer = Mock()
    predictor.feature_engineer.create_features.return_value = raw
    predictor.preprocessor = Mock()
    predictor.preprocessor.preprocess_for_serving.return_value = pd.DataFrame(
        {"value": [1.0, 2.0]}
    )
    predictor.preprocessor.feature_names = ["value"]
    predictor.model = Mock()
    predictor.model.predict_proba.side_effect = [
        np.array([[0.8, 0.2]]),
        np.array([[0.3, 0.7]]),
    ]
    predictor.model_version = "baseline_v1.0.0"

    result = predictor.predict_batch(str(input_path), str(output_path), chunk_size=1)
    saved = pd.read_csv(output_path)

    assert result["total_rows"] == 2
    assert result["churn_count"] == 1
    assert saved["customerID"].tolist() == ["C1", "C2"]
    assert saved["churn_prediction"].tolist() == ["No", "Yes"]
    assert (saved["model_version"] == "baseline_v1.0.0").all()
