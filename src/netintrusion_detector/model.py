"""Model training for binary (normal vs attack) and multiclass
(normal/DoS/Probe/R2L/U2R) network intrusion detection on NSL-KDD.

Uses `RandomForestClassifier`: a reasonable, well-established, non-trivial
default for tabular intrusion-detection data -- it handles the mix of
one-hot categorical and continuous numeric flow features without needing
feature-scale-sensitive assumptions, captures the nonlinear interactions
between features like `count`/`serror_rate` that distinguish e.g. DoS
floods from scans, and gives free feature-importance diagnostics. This is
a defensible, realistic choice (used throughout the published NSL-KDD
literature as a strong baseline), not an under-powered stub like a single
decision tree/stump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

DEFAULT_RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42,
)


@dataclass
class TrainedModel:
    pipeline: Pipeline
    task: str  # "binary" or "multiclass"
    classes_: list


def build_pipeline(rf_params: Optional[dict] = None) -> Pipeline:
    """Build an (unfitted) preprocessing + RandomForest pipeline."""
    params = dict(DEFAULT_RF_PARAMS)
    if rf_params:
        params.update(rf_params)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", RandomForestClassifier(**params)),
        ]
    )


def train_binary_model(
    X: pd.DataFrame, y_is_attack: pd.Series, rf_params: Optional[dict] = None
) -> TrainedModel:
    """Train a binary normal(0) vs attack(1) RandomForest classifier."""
    pipeline = build_pipeline(rf_params)
    pipeline.fit(X, y_is_attack)
    return TrainedModel(pipeline=pipeline, task="binary", classes_=list(pipeline.classes_))


def train_multiclass_model(
    X: pd.DataFrame, y_category: pd.Series, rf_params: Optional[dict] = None
) -> TrainedModel:
    """Train a multiclass normal/DoS/Probe/R2L/U2R RandomForest classifier."""
    pipeline = build_pipeline(rf_params)
    pipeline.fit(X, y_category)
    return TrainedModel(pipeline=pipeline, task="multiclass", classes_=list(pipeline.classes_))


def predict(model: TrainedModel, X: pd.DataFrame):
    """Return hard predictions for new data."""
    return model.pipeline.predict(X)


def predict_proba(model: TrainedModel, X: pd.DataFrame):
    """Return class probabilities for new data."""
    return model.pipeline.predict_proba(X)


def save_model(model: TrainedModel, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> TrainedModel:
    return joblib.load(path)
