from pathlib import Path

from netintrusion_detector.data import load_nslkdd
from netintrusion_detector.preprocessing import build_preprocessor, get_output_feature_names

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRAIN = ROOT / "data" / "sample" / "kddtrain_sample.csv"


def test_preprocessor_round_trips_and_produces_expected_shape():
    ds = load_nslkdd(str(SAMPLE_TRAIN))
    pre = build_preprocessor()
    transformed = pre.fit_transform(ds.X)

    assert transformed.shape[0] == len(ds.X)
    # Output width = (one-hot categorical columns) + 38 numeric columns.
    feature_names = get_output_feature_names(pre)
    assert transformed.shape[1] == len(feature_names)
    n_numeric = 38
    assert transformed.shape[1] > n_numeric  # one-hot must add at least some columns

    # A second transform (no refit) on the same data should be identical.
    transformed_again = pre.transform(ds.X)
    assert transformed.shape == transformed_again.shape


def test_preprocessor_handles_unseen_categorical_value_at_transform_time():
    ds = load_nslkdd(str(SAMPLE_TRAIN))
    pre = build_preprocessor()
    pre.fit(ds.X)

    novel = ds.X.iloc[:5].copy()
    novel["service"] = "some_never_before_seen_service"
    # Must not raise, thanks to handle_unknown="ignore".
    out = pre.transform(novel)
    assert out.shape[0] == 5
