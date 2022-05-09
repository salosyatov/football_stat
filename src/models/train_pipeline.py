import json
import logging
import sys
from pathlib import Path
from typing import Dict

import click
import mlflow as mlflow
import yaml
from marshmallow import Schema

from ..models import predict_net_model, serialize_net_model, train_net_model
from ..data import split_train_val_data, cache_images
from ..models import train_clf_model, evaluate_clf_model, predict_clf_model, serialize_clf_model
from ..params import TrainingPipelineParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str) -> Dict:
    with open(config_path, "r") as input_stream:
        training_pipeline_params = Schema(TrainingPipelineParams).load(yaml.safe_load(input_stream))

    if training_pipeline_params.use_mlflow:
        mlflow.set_tracking_uri(training_pipeline_params.mlflow_uri)
        mlflow.set_experiment(training_pipeline_params.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params: TrainingPipelineParams):
    if training_pipeline_params.use_cache:
        if not Path(training_pipeline_params.preprocessing_params.cache_file_path).exists():
            cache_images(training_pipeline_params.preprocessing_params)
            logger.info("Cache is saved to file %s", training_pipeline_params.preprocessing_params.cache_file_path)
    else:
        raise ValueError("Training is supported with caching only.")

    train_data, val_data = split_train_val_data(training_pipeline_params.preprocessing_params.cache_file_path, training_pipeline_params.splitting_params)
    net_model, optimizer = train_net_model(train_data, val_data, training_pipeline_params.train_params)
    serialize_net_model(net_model, optimizer, training_pipeline_params.output_net_model_path)
    logger.info("Net model is saved to %s", training_pipeline_params.output_net_model_path)

    train_images, train_labels, _ = train_data
    val_images, val_labels, _ = val_data

    embeddings = predict_net_model(net_model, train_images, training_pipeline_params.test_params)
    clf_model = train_clf_model(embeddings, train_labels, training_pipeline_params.train_params)

    val_embeddings = predict_net_model(net_model, val_images, training_pipeline_params.test_params)
    predicts = predict_clf_model(clf_model, val_embeddings)
    metrics = evaluate_clf_model(val_labels, predicts)

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    serialize_clf_model(clf_model, training_pipeline_params.output_clf_model_path)
    logger.info("Clf model is saved to %s", training_pipeline_params.output_clf_model_path)

    return metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def main(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    main()
