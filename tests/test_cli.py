import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRAIN = ROOT / "data" / "sample" / "kddtrain_sample.csv"
SAMPLE_TEST = ROOT / "data" / "sample" / "kddtest_sample.csv"


def _run_cli(tmp_path, *args):
    cmd = [sys.executable, "-m", "netintrusion_detector.cli", *args]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    return result


def test_cli_end_to_end_train_evaluate_predict(tmp_path):
    model_dir = tmp_path / "models"

    train_result = _run_cli(
        tmp_path,
        "train",
        "--train-path",
        str(SAMPLE_TRAIN),
        "--model-dir",
        str(model_dir),
        "--task",
        "both",
    )
    assert train_result.returncode == 0, train_result.stderr
    assert (model_dir / "binary_model.joblib").exists()
    assert (model_dir / "multiclass_model.joblib").exists()

    report_path = tmp_path / "report.json"
    eval_result = _run_cli(
        tmp_path,
        "evaluate",
        "--test-path",
        str(SAMPLE_TEST),
        "--model-dir",
        str(model_dir),
        "--task",
        "both",
        "--output-json",
        str(report_path),
    )
    assert eval_result.returncode == 0, eval_result.stderr
    assert "Binary (normal vs attack) evaluation" in eval_result.stdout
    assert "Multiclass" in eval_result.stdout
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "binary" in report and "multiclass" in report
    assert 0.0 <= report["binary"]["accuracy"] <= 1.0

    pred_out = tmp_path / "preds.csv"
    predict_result = _run_cli(
        tmp_path,
        "predict",
        str(SAMPLE_TEST),
        "--model-dir",
        str(model_dir),
        "--task",
        "binary",
        "--output-csv",
        str(pred_out),
    )
    assert predict_result.returncode == 0, predict_result.stderr
    assert pred_out.exists()

    import pandas as pd

    preds_df = pd.read_csv(pred_out)
    assert "prediction" in preds_df.columns
    assert set(preds_df["prediction"].unique()) <= {"normal", "attack"}
