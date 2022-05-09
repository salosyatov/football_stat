import pickle
from typing import Dict, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from ..params import TrainingParams

ClfModel = Union[RandomForestClassifier, XGBClassifier]


def train_clf_model(features: np.ndarray, target: np.ndarray, train_params: TrainingParams) -> ClfModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=train_params.random_state)
    elif train_params.model_type == "XGBClassifier":
        model = XGBClassifier(random_state=train_params.random_state)
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_clf_model(model: ClfModel, features: np.ndarray) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_clf_model(target: np.ndarray, predicts: np.ndarray) -> Dict[str, float]:
    precision, recall, fscore, support = precision_recall_fscore_support(target, predicts, average='micro')
    res = {"precision": precision, "recall": recall, "fscore": fscore, "support": support}
    return res


def serialize_clf_model(model: ClfModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
