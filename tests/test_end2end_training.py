import os

import pytest

from src.models import run_train_pipeline
from src.params import (
    TrainingPipelineParams,
    SplittingParams,
    TrainingParams, TestParams, PreprocessingParams,
)


@pytest.mark.parametrize("dataset_path", ["./data/raw"])
def test_train_end2end(
        dataset_path: str,
        tmpdir
):
    expected_output_net_model_path = tmpdir.join("net.pt")
    expected_output_clf_model_path = tmpdir.join("clf.pkl")

    assert not os.path.exists(expected_output_net_model_path)
    assert not os.path.exists(expected_output_clf_model_path)

    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_net_model_path=expected_output_net_model_path,
        output_clf_model_path=expected_output_clf_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        test_params=TestParams(),
        train_params=TrainingParams(),
        preprocessing_params=PreprocessingParams()
    )
    metrics = run_train_pipeline(params)
    assert metrics["precision"] > 0
    assert os.path.exists(expected_output_net_model_path)
    assert os.path.exists(expected_output_clf_model_path)
    assert os.path.exists(params.metric_path)
