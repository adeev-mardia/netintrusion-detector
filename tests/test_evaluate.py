import numpy as np
import pytest

from netintrusion_detector.evaluate import (
    evaluate_binary,
    evaluate_multiclass,
    false_positive_rate,
    fpr_at_thresholds,
)

# Hand-checkable tiny example.
# y_true: [0, 0, 1, 1, 1]
# y_pred: [0, 1, 1, 1, 0]
#
# idx0: true=0 pred=0 -> TN
# idx1: true=0 pred=1 -> FP
# idx2: true=1 pred=1 -> TP
# idx3: true=1 pred=1 -> TP
# idx4: true=1 pred=0 -> FN
#
# TN=1 FP=1 FN=1 TP=2
# accuracy  = (TN+TP)/5 = 3/5 = 0.6
# precision = TP/(TP+FP) = 2/2 = ... wait TP+FP = 2+1 = 3 -> 2/3
# recall    = TP/(TP+FN) = 2/3
# f1        = 2*p*r/(p+r) = 2*(2/3)*(2/3)/((2/3)+(2/3)) = (8/9)/(4/3) = 2/3
# FPR       = FP/(FP+TN) = 1/2 = 0.5

Y_TRUE = [0, 0, 1, 1, 1]
Y_PRED = [0, 1, 1, 1, 0]


def test_false_positive_rate_hand_checked():
    assert false_positive_rate(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_evaluate_binary_matches_hand_computed_metrics():
    report = evaluate_binary(Y_TRUE, Y_PRED)
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["f1"] == pytest.approx(2 / 3)
    assert report["false_positive_rate"] == pytest.approx(0.5)
    cm = report["confusion_matrix"]
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}


def test_fpr_at_thresholds_monotonic_in_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    results = fpr_at_thresholds(y_true, y_score, thresholds=(0.2, 0.5, 0.8))

    # At threshold 0.2: preds = [0,1,1,1,1,1] -> FP among first 3 normals (idx1,2)=2, TN=1
    assert results[0.2]["fp"] == 2
    assert results[0.2]["tn"] == 1

    # Raising the threshold should never increase FPR (fewer or equal false alarms).
    fprs = [results[t]["fpr"] for t in (0.2, 0.5, 0.8)]
    assert fprs == sorted(fprs, reverse=True)


def test_evaluate_multiclass_perfect_predictions():
    y_true = ["normal", "DoS", "Probe", "normal", "DoS"]
    y_pred = ["normal", "DoS", "Probe", "normal", "DoS"]
    report = evaluate_multiclass(y_true, y_pred)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    for cls, m in report["per_class"].items():
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)


def test_evaluate_multiclass_with_errors_hand_checked():
    # 3 classes, one misclassification: true DoS predicted as Probe.
    y_true = ["normal", "DoS", "DoS", "Probe"]
    y_pred = ["normal", "DoS", "Probe", "Probe"]
    report = evaluate_multiclass(y_true, y_pred, labels=["normal", "DoS", "Probe"])

    # DoS: true count=2, correctly predicted=1 -> recall=0.5; predicted DoS count=1, correct=1 -> precision=1.0
    assert report["per_class"]["DoS"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["DoS"]["precision"] == pytest.approx(1.0)
    # Probe: true count=1, predicted correctly -> recall=1.0; predicted Probe count=2, correct=1 -> precision=0.5
    assert report["per_class"]["Probe"]["recall"] == pytest.approx(1.0)
    assert report["per_class"]["Probe"]["precision"] == pytest.approx(0.5)
    assert report["accuracy"] == pytest.approx(3 / 4)
