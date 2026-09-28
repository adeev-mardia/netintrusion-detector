"""Evaluation metrics for intrusion-detection classifiers.

Beyond plain accuracy, an intrusion detector is judged operationally by:

* **Recall on attack traffic** -- missed attacks are the costly failure mode.
* **False positive rate (FPR)** -- how often normal traffic is flagged as an
  attack. A security analyst has to triage every alert, so a detector with
  great recall but a high FPR causes "alert fatigue" and gets ignored or
  disabled in practice; FPR at the deployed decision threshold is often the
  single most operationally important number for an anomaly detector.
* **Per-class recall in the multiclass setting** -- NSL-KDD is heavily
  class-imbalanced (R2L and especially U2R are rare), and a model can look
  good on macro accuracy while essentially never detecting the rare classes.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _confusion_counts_binary(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary labels {0,1}."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def false_positive_rate(y_true, y_pred) -> float:
    """FPR = FP / (FP + TN): fraction of normal traffic wrongly flagged."""
    tn, fp, fn, tp = _confusion_counts_binary(y_true, y_pred)
    denom = fp + tn
    return fp / denom if denom else 0.0


def fpr_at_thresholds(y_true, y_score, thresholds: Iterable[float] = (0.3, 0.5, 0.7)) -> dict:
    """Compute FPR (and recall/TPR, for context) at several decision
    thresholds on the predicted probability of the positive ("attack")
    class. Lowering the threshold catches more attacks (higher recall) at
    the cost of more false alarms (higher FPR) -- this trade-off is the
    core operating-point decision a real deployment has to make.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    results = {}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = _confusion_counts_binary(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        results[t] = {"fpr": fpr, "tpr_recall": tpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return results


def evaluate_binary(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> dict:
    """Full binary (normal=0 vs attack=1) evaluation report."""
    tn, fp, fn, tp = _confusion_counts_binary(y_true, y_pred)
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if y_score is not None:
        report["fpr_at_thresholds"] = fpr_at_thresholds(y_true, y_score)
    return report


def evaluate_multiclass(y_true, y_pred, labels: Optional[list] = None) -> dict:
    """Full multiclass evaluation report: overall + macro + per-class metrics."""
    if labels is None:
        labels = sorted(set(list(y_true)) | set(list(y_pred)))

    per_class = {}
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for cls, p, r, f in zip(labels, precisions, recalls, f1s):
        per_class[cls] = {"precision": float(p), "recall": float(r), "f1": float(f)}

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
    }
    return report


def format_binary_report(report: dict) -> str:
    lines = []
    lines.append("Binary (normal vs attack) evaluation")
    lines.append("-" * 40)
    lines.append(f"Accuracy:  {report['accuracy']:.4f}")
    lines.append(f"Precision: {report['precision']:.4f}")
    lines.append(f"Recall:    {report['recall']:.4f}")
    lines.append(f"F1:        {report['f1']:.4f}")
    lines.append(f"FPR:       {report['false_positive_rate']:.4f}")
    cm = report["confusion_matrix"]
    lines.append(
        f"Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"
    )
    if "fpr_at_thresholds" in report:
        lines.append("")
        lines.append("FPR / recall at operating thresholds:")
        lines.append(f"{'threshold':>10} {'FPR':>8} {'recall':>8}")
        for t, vals in report["fpr_at_thresholds"].items():
            lines.append(f"{t:>10} {vals['fpr']:>8.4f} {vals['tpr_recall']:>8.4f}")
    return "\n".join(lines)


def format_multiclass_report(report: dict) -> str:
    lines = []
    lines.append("Multiclass (normal/DoS/Probe/R2L/U2R) evaluation")
    lines.append("-" * 50)
    lines.append(f"Accuracy:      {report['accuracy']:.4f}")
    lines.append(f"Macro precision: {report['macro_precision']:.4f}")
    lines.append(f"Macro recall:    {report['macro_recall']:.4f}")
    lines.append(f"Macro F1:        {report['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for cls, m in report["per_class"].items():
        lines.append(f"{cls:>10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    lines.append("")
    lines.append(f"Confusion matrix (rows=true, cols=pred), labels={report['labels']}:")
    for row in report["confusion_matrix"]:
        lines.append(" ".join(f"{v:>6}" for v in row))
    return "\n".join(lines)
