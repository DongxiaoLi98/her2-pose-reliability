"""Container entry point for the ProcessingStep pipeline.

One script, four steps, dispatched by --step. It is a thin adapter: it maps the
container's fixed directory layout onto the same functions that run locally.
No pipeline logic lives here.

    /opt/ml/processing/input/code     src/ and config/ (uploaded per run)
    /opt/ml/processing/input/data     the contract table
    /opt/ml/processing/input/prev*    outputs of earlier steps
    /opt/ml/processing/output         everything this step produces
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

BASE = os.environ.get("PROCESSING_BASE", "/opt/ml/processing")
CODE = f"{BASE}/input/code"
DATA = f"{BASE}/input/data"
PREV = f"{BASE}/input/prev"
PREV2 = f"{BASE}/input/prev2"
OUT = f"{BASE}/output"

sys.path.insert(0, CODE)

# The stock sklearn container does not ship PyYAML, which the threshold config
# needs. Install it here rather than baking a custom image for one dependency.
try:
    import yaml  # noqa: F401
except ImportError:  # pragma: no cover
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])


def _first_file(d: str, suffix: str = "") -> str:
    names = sorted(n for n in os.listdir(d) if n.endswith(suffix) and not n.startswith("."))
    if not names:
        raise FileNotFoundError(f"no {suffix or 'file'} under {d}: {os.listdir(d)}")
    return os.path.join(d, names[0])


def step_preprocess(args) -> dict:
    from src.preprocess import preprocess

    return preprocess(
        output_prefix=OUT,
        input_uri=_first_file(DATA, ".csv"),
        threshold_config=f"{CODE}/config/thresholds.v1.yaml",
        source="table",
        seed=args.seed,
    )


def step_train(args) -> dict:
    from src.train import train

    return train(
        train_uri=f"{PREV}/train.csv",
        cutoffs_uri=f"{PREV}/cutoffs.json",
        output_prefix=OUT,
        threshold_config=f"{CODE}/config/thresholds.v1.yaml",
        seed=args.seed,
    )


def step_evaluate(args) -> dict:
    from src.evaluate import evaluate

    result = evaluate(
        test_uri=f"{PREV}/test.csv",
        model_uri=f"{PREV2}/model.joblib",
        cutoffs_uri=f"{PREV}/cutoffs.json",
        output_prefix=OUT,
        threshold_config=f"{CODE}/config/thresholds.v1.yaml",
    )
    # carry the model forward so the register step has it in one place
    shutil.copy(f"{PREV2}/model.joblib", f"{OUT}/model.joblib")
    return result


def step_register(args) -> dict:
    from src.register import register

    with open(f"{PREV}/evaluation.json") as fh:
        evaluation = json.load(fh)["evaluation_result"]

    return register(
        evaluation_result=evaluation,
        model_uri=f"{PREV}/model.joblib",
        output_prefix=OUT,
        model_package_group_name=args.model_package_group_name,
        approval_status=args.approval_status,
        threshold_config=f"{CODE}/config/thresholds.v1.yaml",
    )


STEPS = {
    "preprocess": step_preprocess,
    "train": step_train,
    "evaluate": step_evaluate,
    "register": step_register,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=sorted(STEPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-package-group-name", default="her2-pose-reliability")
    ap.add_argument("--approval-status", default="PendingManualApproval")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"=== step {args.step} ===", flush=True)
    result = STEPS[args.step](args)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
