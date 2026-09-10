"""Metrics, ordered by what actually costs money.

false_pass_rate is the PRIMARY safety metric: a truly unreliable pose that we
label Pass goes on to consume a wet-lab experiment. Accuracy is not the target.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

CLASSES = ["Pass", "Review", "Fail"]


def false_pass_rate(y_true, y_pred) -> float:
    """Fraction of true-Fail rows that were predicted Pass."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_fail = y_true == "Fail"
    if is_fail.sum() == 0:
        return 0.0
    return float((y_pred[is_fail] == "Pass").mean())


def false_fail_rate(y_true, y_pred) -> float:
    """Fraction of true-Pass rows that were predicted Fail (the cost of caution)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_pass = y_true == "Pass"
    if is_pass.sum() == 0:
        return 0.0
    return float((y_pred[is_pass] == "Fail").mean())


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0))


def brier_multiclass(y_true, proba, classes) -> float:
    """Calibration error. Lower is better. Only meaningful for probability output."""
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(proba, dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    for r, y in enumerate(y_true):
        onehot[r, idx[y]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def pass_yield(y_pred) -> float:
    """Fraction of all rows the gate actually lets through.

    Without this, false_pass_rate is gameable: a gate that never says Pass has
    a false-pass rate of zero and filters nothing.
    """
    y_pred = np.asarray(y_pred)
    return float((y_pred == "Pass").mean()) if len(y_pred) else 0.0


def pass_recall(y_true, y_pred) -> float:
    """Of the rows that really are Pass, how many did the gate recover."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_pass = y_true == "Pass"
    if is_pass.sum() == 0:
        return 0.0
    return float((y_pred[is_pass] == "Pass").mean())


def review_rate(y_pred) -> float:
    """Fraction routed to a human. This is the cost of being cautious."""
    y_pred = np.asarray(y_pred)
    return float((y_pred == "Review").mean()) if len(y_pred) else 0.0


def summary(y_true, y_pred, proba=None, classes=None) -> dict:
    out = {
        "false_pass_rate": false_pass_rate(y_true, y_pred),
        "false_fail_rate": false_fail_rate(y_true, y_pred),
        # false_pass_rate is only meaningful next to how much the gate passes.
        "pass_yield": pass_yield(y_pred),
        "pass_recall": pass_recall(y_true, y_pred),
        "review_rate": review_rate(y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "n": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "confusion_matrix_labels": CLASSES,
    }
    if proba is not None and classes is not None:
        out["brier"] = brier_multiclass(y_true, proba, classes)
    return out
