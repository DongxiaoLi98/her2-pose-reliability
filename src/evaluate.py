"""Pipeline step 3 -- evaluate.

Purpose
-------
One evaluation, on structures the model has never seen, with the cut points
already locked. This step compares the selected model against the rule baseline
on the SAME held-out rows and emits the numbers the condition step reads.
"""
from __future__ import annotations

import io
import os

import joblib

from . import gates, io_utils, metrics, split, train as train_mod


def evaluate(
    test_uri: str,
    model_uri: str,
    cutoffs_uri: str,
    output_prefix: str,
    threshold_config: str = "config/thresholds.v1.yaml",
) -> dict:
    df = io_utils.read_table(test_uri)
    bundle = joblib.load(io.BytesIO(io_utils.read_bytes(model_uri)))
    thresholds = gates.load_thresholds(threshold_config)
    cutoffs = io_utils.read_json(cutoffs_uri)

    y = df["label"].to_numpy()

    x, _ = train_mod.build_matrix(df, medians=bundle["medians"])
    x = x[bundle["feature_columns"]]
    proba = bundle["model"].predict_proba(x)
    # re-order columns to the canonical class order
    order = [list(bundle["model"].classes_).index(c) for c in bundle["classes"]]
    proba = proba[:, order]

    pred = train_mod.apply_cutpoints(proba, bundle["classes"], bundle["pass_cut"], bundle["fail_cut"])
    model_metrics = metrics.summary(y, pred, proba=proba, classes=bundle["classes"])

    ruled = gates.rule_reliability(df, thresholds, cutoffs)
    baseline_metrics = metrics.summary(y, ruled["rule_status"].to_numpy())

    result = {
        "evaluation_result": {
            "model_name": bundle["model_name"],
            "threshold_version": bundle["threshold_version"],
            "held_out_structures": int(df[split.GROUP_KEY].nunique()),
            "held_out_rows": int(len(df)),
            "model": model_metrics,
            "rule_baseline": baseline_metrics,
            "delta_false_pass_rate": baseline_metrics["false_pass_rate"] - model_metrics["false_pass_rate"],
            "delta_macro_f1": model_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        }
    }
    io_utils.write_json(result, os.path.join(output_prefix, "evaluation.json"))
    return result


if __name__ == "__main__":  # pragma: no cover
    import json

    from .preprocess import preprocess
    from .train import train

    p = preprocess(output_prefix="artifacts/local")
    t = train(p["train_uri"], p["cutoffs_uri"], "artifacts/local")
    e = evaluate(p["test_uri"], t["model_uri"], p["cutoffs_uri"], "artifacts/local")
    print(json.dumps(e, indent=2, default=str))
