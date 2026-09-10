"""Pipeline step 2 -- train.

Purpose
-------
Compare a transparent rule baseline against three supervised models under a
GROUPED cross-validation, pick a winner on false-pass rate, and LOCK the
Pass/Review/Fail cut points using out-of-fold predictions only. The held-out
structures are never touched here.
"""
from __future__ import annotations

import io
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import contract, gates, io_utils, metrics, split

CLASSES = metrics.CLASSES


# --------------------------------------------------------------------------- #
# feature assembly
# --------------------------------------------------------------------------- #
GROUP_KEY = "structure_id"


def build_matrix(df: pd.DataFrame, medians: dict[str, float] | None = None):
    """Numeric matrix + explicit was-missing indicators + within-structure z-score.

    Conditional features may legitimately be absent. We median-impute them AND
    carry a boolean 'this was missing' column, so the model can learn from the
    absence instead of silently reading it as a real low value.

    interaction_score is reported on a scale that differs per structure, so its
    raw value is not comparable across structures and a boundary learned on one
    set of structures does not transfer to new ones. We therefore also supply
    its z-score within the pose ensemble it belongs to -- the same normalisation
    the pose-filter literature applies before combining scores.
    """
    x = df[contract.FEATURES].copy()
    if medians is None:
        medians = {}
        for c in contract.CONDITIONAL_FEATURES:
            med = x[c].median()
            # A conditional feature can be absent for an ENTIRE dataset (the
            # upstream tool never emitted it). Its median is then undefined; we
            # fall back to 0 and rely on the was-missing flag, which is 1 for
            # every row, to tell the model the column carries no information.
            medians[c] = 0.0 if pd.isna(med) else float(med)
    for c in contract.CONDITIONAL_FEATURES:
        x[f"{c}__was_missing"] = x[c].isna().astype(int)
        x[c] = x[c].fillna(medians[c])

    if GROUP_KEY in df.columns:
        g = df.groupby(GROUP_KEY)["interaction_score"]
        z = (df["interaction_score"] - g.transform("mean")) / (g.transform("std") + 1e-9)
        x["interaction_score__z_in_structure"] = z.fillna(0.0).to_numpy()
    else:
        x["interaction_score__z_in_structure"] = 0.0

    return x.astype(float), medians


def make_models(seed: int = 0) -> dict[str, SkPipeline]:
    return {
        "logistic_regression": SkPipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=2000))]
        ),
        "svm": SkPipeline(
            [("scale", StandardScaler()),
             ("clf", CalibratedClassifierCV(SVC(kernel="rbf", C=2.0, gamma="scale"), method="sigmoid", cv=3))]
        ),
        "gradient_boosting": SkPipeline(
            [("clf", GradientBoostingClassifier(random_state=seed))]
        ),
    }


# --------------------------------------------------------------------------- #
# decision rule
# --------------------------------------------------------------------------- #
def apply_cutpoints(proba: np.ndarray, classes: list[str], pass_cut: float, fail_cut: float) -> np.ndarray:
    """Probabilities -> Pass / Review / Fail, with a Review band in between.

    Fail is checked first: a suspected failure can never be relabelled Pass.
    """
    i_pass, i_fail = classes.index("Pass"), classes.index("Fail")
    out = np.full(len(proba), "Review", dtype=object)
    out[proba[:, i_pass] >= pass_cut] = "Pass"
    out[proba[:, i_fail] >= fail_cut] = "Fail"
    return out


def lock_cutpoints(y_true, proba, classes, max_false_pass_rate: float,
                   folds=None) -> tuple[float, float, dict]:
    """Choose cut points on OUT-OF-FOLD predictions, then never move them.

    Pooling all folds together and optimising on the pooled result is
    optimistic: a cut point can look safe on the pool while failing badly on
    one structure group. When `folds` is supplied we score each candidate cut
    point by its WORST fold, which is the number that actually transfers to an
    unseen structure.
    """
    y_true = np.asarray(y_true)
    val_idx = [va for _, va in folds] if folds else [np.arange(len(y_true))]

    best = None
    for fail_cut in np.arange(0.30, 0.75, 0.05):
        for pass_cut in np.arange(0.30, 0.96, 0.02):
            pred = apply_cutpoints(proba, classes, pass_cut, fail_cut)
            per_fold = [metrics.false_pass_rate(y_true[v], pred[v]) for v in val_idx]
            worst = float(max(per_fold))
            pooled = metrics.false_pass_rate(y_true, pred)
            f1 = metrics.macro_f1(y_true, pred)
            feasible = worst <= max_false_pass_rate
            key = (feasible, f1 if feasible else -worst)
            if best is None or key > best[0]:
                best = (key, float(pass_cut), float(fail_cut), {
                    "oof_false_pass_rate": pooled,
                    "oof_worst_fold_false_pass_rate": worst,
                    "oof_macro_f1": f1,
                })
    _, pass_cut, fail_cut, info = best
    return pass_cut, fail_cut, info


# --------------------------------------------------------------------------- #
# step entry point
# --------------------------------------------------------------------------- #
def train(
    train_uri: str,
    cutoffs_uri: str,
    output_prefix: str,
    threshold_config: str = "config/thresholds.v1.yaml",
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    df = io_utils.read_table(train_uri)
    thresholds = gates.load_thresholds(threshold_config)
    cutoffs = io_utils.read_json(cutoffs_uri)
    max_fpr = thresholds["registration_gate"]["max_false_pass_rate"]

    y = df["label"].to_numpy()
    x, medians = build_matrix(df)

    # ---- baseline: the versioned rule engine, evaluated on the same rows ----
    ruled = gates.rule_reliability(df, thresholds, cutoffs)
    baseline_cv = metrics.summary(y, ruled["rule_status"].to_numpy())

    # ---- supervised candidates under GroupKFold ----------------------------
    # GroupKFold cannot make more folds than there are groups
    n_groups = int(df[split.GROUP_KEY].nunique())
    n_splits = max(2, min(n_splits, n_groups))
    folds = list(split.grouped_folds(df, n_splits=n_splits))
    results, oof_store = {}, {}

    for name, model in make_models(seed).items():
        oof = np.zeros((len(df), len(CLASSES)))
        for tr, va in folds:
            m = make_models(seed)[name]
            m.fit(x.iloc[tr], y[tr])
            p = m.predict_proba(x.iloc[va])
            for j, c in enumerate(m.classes_):
                oof[va, CLASSES.index(c)] = p[:, j]
        pass_cut, fail_cut, info = lock_cutpoints(y, oof, CLASSES, max_fpr, folds=folds)
        pred = apply_cutpoints(oof, CLASSES, pass_cut, fail_cut)
        res = metrics.summary(y, pred, proba=oof, classes=CLASSES)
        res.update(pass_cut=pass_cut, fail_cut=fail_cut, **info)
        results[name] = res
        oof_store[name] = oof

    # ---- selection: worst-fold false-pass rate first, macro-F1 second ------
    # The worst fold, not the pooled average, is what an unseen structure looks
    # like. Selecting on the pooled number is how a model that fails on one
    # structure group gets promoted.
    winner = min(results, key=lambda k: (results[k]["oof_worst_fold_false_pass_rate"],
                                         -results[k]["macro_f1"]))

    final = make_models(seed)[winner]
    final.fit(x, y)

    bundle = {
        "model": final,
        "model_name": winner,
        "classes": CLASSES,
        "feature_columns": list(x.columns),
        "medians": medians,
        "pass_cut": results[winner]["pass_cut"],
        "fail_cut": results[winner]["fail_cut"],
        "threshold_version": thresholds["threshold_version"],
        "schema_version": contract.SCHEMA_VERSION,
        "cutoffs": cutoffs,
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    model_uri = io_utils.write_bytes(buf.getvalue(), os.path.join(output_prefix, "model.joblib"))

    report = {
        "selected_model": winner,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": seed,
        },
        "selection_rule": "lowest WORST-FOLD cross-validated false_pass_rate, macro_f1 as tie-break",
        "cv": "GroupKFold by structure_id",
        "n_splits": n_splits,
        "n_train_rows": int(len(df)),
        "n_train_structures": int(df[split.GROUP_KEY].nunique()),
        "rule_baseline_cv": baseline_cv,
        "model_cv": results,
        "locked_pass_cut": bundle["pass_cut"],
        "locked_fail_cut": bundle["fail_cut"],
        "model_uri": model_uri,
    }
    io_utils.write_json(report, os.path.join(output_prefix, "train_report.json"))
    return report


if __name__ == "__main__":  # pragma: no cover
    import json

    from .preprocess import preprocess

    p = preprocess(output_prefix="artifacts/local")
    r = train(p["train_uri"], p["cutoffs_uri"], "artifacts/local")
    print(json.dumps(r, indent=2, default=str))
