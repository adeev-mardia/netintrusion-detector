"""Reusable sklearn preprocessing pipeline for NSL-KDD-schema features.

Builds a single `ColumnTransformer` that:

* One-hot encodes the three categorical flow features
  (`protocol_type`, `service`, `flag`), with `handle_unknown="ignore"` so
  a category value seen only at test/predict time (NSL-KDD's test set is
  known to include a couple of `service` values absent from training)
  never crashes inference -- it's just encoded as all-zeros.
* Standard-scales the remaining 38 numeric flow features.

Wrapping this in a `ColumnTransformer`/`Pipeline` (rather than doing ad hoc
`pd.get_dummies` + manual scaling) keeps the exact fitted encoding/scaling
statistics attached to the model artifact, so `predict` on new data is
guaranteed to use the identical transformation the model was trained with.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the NSL-KDD 41-feature schema."""
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable names for the preprocessor's output columns (post-fit)."""
    return list(preprocessor.get_feature_names_out())
