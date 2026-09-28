"""Deterministically build small, stratified CSV samples of the real NSL-KDD
train/test sets for bundling in the repo (data/sample/), so the package is
instantly usable/testable without requiring a network download.

Run from the repo root:

    python3 scripts/make_sample_data.py

Requires data/raw/KDDTrain+.txt and data/raw/KDDTest+.txt to already be
present (see README for how to fetch the real dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from netintrusion_detector.data import ALL_COLUMNS_WITH_LABEL, load_nslkdd  # noqa: E402

RANDOM_STATE = 42


def stratified_sample(raw_path: Path, n_per_class_cap: int, out_path: Path) -> None:
    ds = load_nslkdd(str(raw_path))
    df = ds.X.copy()
    df["label"] = ds.attack_type
    df["difficulty"] = 0

    parts = []
    for category_label, group in df.groupby("label"):
        n = min(len(group), n_per_class_cap)
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    sample = sample[ALL_COLUMNS_WITH_LABEL]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"Wrote {len(sample)} rows ({sample['label'].nunique()} label types) to {out_path}")


if __name__ == "__main__":
    stratified_sample(
        ROOT / "data" / "raw" / "KDDTrain+.txt",
        n_per_class_cap=120,
        out_path=ROOT / "data" / "sample" / "kddtrain_sample.csv",
    )
    stratified_sample(
        ROOT / "data" / "raw" / "KDDTest+.txt",
        n_per_class_cap=40,
        out_path=ROOT / "data" / "sample" / "kddtest_sample.csv",
    )
