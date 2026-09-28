"""Command-line interface for netintrusion-detector.

    netintrusion-detector train    --train-path data/raw/KDDTrain+.txt
    netintrusion-detector evaluate --test-path data/raw/KDDTest+.txt
    netintrusion-detector predict  some_flows.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import evaluate as ev
from . import model as mdl
from .data import ATTACK_CATEGORIES, load_nslkdd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PACKAGE_ROOT / "data" / "raw" / "KDDTrain+.txt"
DEFAULT_TEST_PATH = PACKAGE_ROOT / "data" / "raw" / "KDDTest+.txt"
DEFAULT_SAMPLE_TRAIN = PACKAGE_ROOT / "data" / "sample" / "kddtrain_sample.csv"
DEFAULT_SAMPLE_TEST = PACKAGE_ROOT / "data" / "sample" / "kddtest_sample.csv"
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "models"


def _resolve_path(given: str | None, primary: Path, fallback: Path) -> str:
    if given:
        return given
    if primary.exists():
        return str(primary)
    if fallback.exists():
        print(
            f"[info] {primary} not found; falling back to bundled sample {fallback}",
            file=sys.stderr,
        )
        return str(fallback)
    raise FileNotFoundError(f"Neither {primary} nor fallback {fallback} exist.")


def cmd_train(args: argparse.Namespace) -> None:
    train_path = _resolve_path(args.train_path, DEFAULT_TRAIN_PATH, DEFAULT_SAMPLE_TRAIN)
    ds = load_nslkdd(train_path)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("binary", "both"):
        print(f"Training binary model on {len(ds.X)} rows from {train_path} ...")
        binary_model = mdl.train_binary_model(ds.X, ds.is_attack)
        mdl.save_model(binary_model, model_dir / "binary_model.joblib")
        print(f"Saved binary model to {model_dir / 'binary_model.joblib'}")

    if args.task in ("multiclass", "both"):
        print(f"Training multiclass model on {len(ds.X)} rows from {train_path} ...")
        multi_model = mdl.train_multiclass_model(ds.X, ds.attack_category)
        mdl.save_model(multi_model, model_dir / "multiclass_model.joblib")
        print(f"Saved multiclass model to {model_dir / 'multiclass_model.joblib'}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    test_path = _resolve_path(args.test_path, DEFAULT_TEST_PATH, DEFAULT_SAMPLE_TEST)
    ds = load_nslkdd(test_path)
    model_dir = Path(args.model_dir)

    results = {}

    if args.task in ("binary", "both"):
        binary_model = mdl.load_model(model_dir / "binary_model.joblib")
        y_pred = mdl.predict(binary_model, ds.X)
        y_score = mdl.predict_proba(binary_model, ds.X)[:, list(binary_model.classes_).index(1)]
        report = ev.evaluate_binary(ds.is_attack, y_pred, y_score)
        print(ev.format_binary_report(report))
        print()
        results["binary"] = report

    if args.task in ("multiclass", "both"):
        multi_model = mdl.load_model(model_dir / "multiclass_model.joblib")
        y_pred = mdl.predict(multi_model, ds.X)
        report = ev.evaluate_multiclass(
            ds.attack_category, y_pred, labels=sorted(set(ds.attack_category) | set(y_pred))
        )
        print(ev.format_multiclass_report(report))
        results["multiclass"] = report

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote JSON report to {args.output_json}")


def cmd_predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    task = args.task if args.task != "both" else "binary"
    model_path = model_dir / (
        "multiclass_model.joblib" if task == "multiclass" else "binary_model.joblib"
    )
    trained = mdl.load_model(model_path)

    df = pd.read_csv(args.csv_path)
    from .data import NSL_KDD_FEATURE_ORDER

    missing = set(NSL_KDD_FEATURE_ORDER) - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required feature columns: {sorted(missing)}")
    X = df[NSL_KDD_FEATURE_ORDER]

    preds = mdl.predict(trained, X)
    out = df.copy()
    out["prediction"] = preds
    if trained.task == "binary":
        out["prediction"] = out["prediction"].map({0: "normal", 1: "attack"})

    if args.output_csv:
        out.to_csv(args.output_csv, index=False)
        print(f"Wrote predictions to {args.output_csv}")
    else:
        print(out[["prediction"]].to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netintrusion-detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model(s) on NSL-KDD-schema training data.")
    p_train.add_argument("--train-path", default=None, help="Path to KDDTrain+.txt or a CSV.")
    p_train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p_train.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained model(s) on NSL-KDD-schema test data.")
    p_eval.add_argument("--test-path", default=None, help="Path to KDDTest+.txt or a CSV.")
    p_eval.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p_eval.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    p_eval.add_argument("--output-json", default=None, help="Optional path to write a JSON report.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_predict = sub.add_parser("predict", help="Predict on a CSV of NSL-KDD-schema flow features.")
    p_predict.add_argument("csv_path", help="CSV with the 41 NSL-KDD feature columns.")
    p_predict.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p_predict.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p_predict.add_argument("--output-csv", default=None)
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
