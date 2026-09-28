from pathlib import Path

import pandas as pd
import pytest

from netintrusion_detector.data import (
    ATTACK_CATEGORIES,
    CATEGORICAL_FEATURES,
    NSL_KDD_FEATURE_ORDER,
    NUMERIC_FEATURES,
    load_nslkdd,
    map_attack_to_category,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRAIN = ROOT / "data" / "sample" / "kddtrain_sample.csv"


def test_schema_covers_41_features():
    assert len(NSL_KDD_FEATURE_ORDER) == 41
    assert set(NSL_KDD_FEATURE_ORDER) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
    assert len(CATEGORICAL_FEATURES) == 3


@pytest.mark.parametrize(
    "attack_type,expected_category",
    [
        ("normal", "normal"),
        ("neptune", "DoS"),
        ("smurf", "DoS"),
        ("satan", "Probe"),
        ("portsweep", "Probe"),
        ("guess_passwd", "R2L"),
        ("ftp_write", "R2L"),
        ("buffer_overflow", "U2R"),
        ("rootkit", "U2R"),
    ],
)
def test_attack_category_mapping_known_types(attack_type, expected_category):
    assert map_attack_to_category(attack_type) == expected_category


def test_attack_category_mapping_is_case_insensitive_and_handles_unknown():
    assert map_attack_to_category("NEPTUNE") == "DoS"
    assert map_attack_to_category("totally_novel_attack_xyz") == "Unknown"


def test_load_nslkdd_bundled_sample_has_expected_columns_and_types():
    ds = load_nslkdd(str(SAMPLE_TRAIN))
    assert list(ds.X.columns) == NSL_KDD_FEATURE_ORDER
    assert len(ds.X) == len(ds.attack_type) == len(ds.attack_category) == len(ds.is_attack)
    assert set(ds.is_attack.unique()) <= {0, 1}
    assert set(ds.attack_category.unique()) <= set(ATTACK_CATEGORIES) | {"Unknown"}
    # is_attack should agree with attack_type == "normal"
    assert ((ds.attack_type == "normal") == (ds.is_attack == 0)).all()


def test_load_nslkdd_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_nslkdd("/nonexistent/path/does_not_exist.csv")


def test_load_nslkdd_rejects_file_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"duration": [1, 2], "label": ["normal", "neptune"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_nslkdd(str(bad_csv))


def test_generate_synthetic_nslkdd_like_is_deterministic_and_schema_correct():
    from netintrusion_detector.data import generate_synthetic_nslkdd_like

    df1 = generate_synthetic_nslkdd_like(n_samples=200, random_state=7)
    df2 = generate_synthetic_nslkdd_like(n_samples=200, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)

    assert set(NSL_KDD_FEATURE_ORDER) <= set(df1.columns)
    assert "label" in df1.columns
    assert len(df1) == 200
    # Should contain a mix of normal and attack labels, not be degenerate.
    assert df1["label"].nunique() >= 2
