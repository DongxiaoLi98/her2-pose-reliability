"""Pipeline step 4 -- register.

Purpose
-------
Publish a model version only when it clears the gate, and publish it together
with the context needed to reconstruct any future decision: which features,
which thresholds, which code commit, which held-out result.

Locally this writes a JSON manifest. On SageMaker the same function calls
mlflow.register_model(), which syncs into the SageMaker Model Registry.
"""
from __future__ import annotations

import os
import subprocess

from . import gates, io_utils


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.environ.get("CODE_COMMIT_SHA", "unknown")


def gate_passed(evaluation_result: dict, threshold_config: str = "config/thresholds.v1.yaml") -> bool:
    gate = gates.load_thresholds(threshold_config)["registration_gate"]
    m = evaluation_result["model"]
    return m["false_pass_rate"] <= gate["max_false_pass_rate"] and m["macro_f1"] >= gate["min_macro_f1"]


def register(
    evaluation_result: dict,
    model_uri: str,
    output_prefix: str,
    model_package_group_name: str = "her2-pose-reliability",
    approval_status: str = "PendingManualApproval",
    threshold_config: str = "config/thresholds.v1.yaml",
    mlflow_run_id: str | None = None,
) -> dict:
    passed = gate_passed(evaluation_result, threshold_config)
    manifest = {
        "model_package_group_name": model_package_group_name,
        "approval_status": approval_status if passed else "Rejected",
        "registered": passed,
        "model_uri": model_uri,
        "code_commit_sha": _code_commit(),
        "threshold_version": evaluation_result["threshold_version"],
        "evaluation_result": evaluation_result,
        "mlflow_run_id": mlflow_run_id,
    }

    if passed and os.environ.get("MLFLOW_TRACKING_URI"):
        try:  # pragma: no cover - only exercised inside SageMaker
            import mlflow

            mv = mlflow.register_model(
                model_uri=f"runs:/{mlflow_run_id}/model", name=model_package_group_name
            )
            manifest["mlflow_model_version"] = mv.version
        except Exception as exc:
            manifest["mlflow_error"] = str(exc)

    io_utils.write_json(manifest, os.path.join(output_prefix, "registration.json"))
    return manifest
