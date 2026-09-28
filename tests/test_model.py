from pathlib import Path

import numpy as np

from netintrusion_detector.data import load_nslkdd
from netintrusion_detector.model import (
    predict,
    predict_proba,
    train_binary_model,
    train_multiclass_model,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRAIN = ROOT / "data" / "sample" / "kddtrain_sample.csv"

# Small, fast RF config for tests.
FAST_RF = dict(n_estimators=20, random_state=42, n_jobs=1)


def _small_fixture(n=300):
    ds = load_nslkdd(str(SAMPLE_TRAIN))
    return ds.X.iloc[:n].reset_index(drop=True), ds


def test_binary_model_trains_and_predicts_expected_shape_and_labels():
    X, ds = _small_fixture()
    y = ds.is_attack.iloc[: len(X)].reset_index(drop=True)

    model = train_binary_model(X, y, rf_params=FAST_RF)
    preds = predict(model, X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)) <= {0, 1}

    proba = predict_proba(model, X)
    assert proba.shape == (len(X), 2)
    # Probabilities should sum to 1 per row.
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multiclass_model_trains_and_predicts_known_categories():
    X, ds = _small_fixture()
    y = ds.attack_category.iloc[: len(X)].reset_index(drop=True)

    model = train_multiclass_model(X, y, rf_params=FAST_RF)
    preds = predict(model, X)

    assert len(preds) == len(X)
    assert set(preds) <= set(y.unique())


def test_save_and_load_round_trip(tmp_path):
    from netintrusion_detector.model import load_model, save_model

    X, ds = _small_fixture()
    y = ds.is_attack.iloc[: len(X)].reset_index(drop=True)
    model = train_binary_model(X, y, rf_params=FAST_RF)

    out_path = tmp_path / "model.joblib"
    save_model(model, str(out_path))
    assert out_path.exists()

    loaded = load_model(str(out_path))
    preds_before = predict(model, X)
    preds_after = predict(loaded, X)
    assert np.array_equal(preds_before, preds_after)
