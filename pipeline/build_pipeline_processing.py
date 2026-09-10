"""SageMaker Pipeline built from ProcessingSteps.

WHY A SECOND PIPELINE FILE
--------------------------
`pipeline/build_pipeline.py` uses @step, which SageMaker runs as *training*
jobs. On a fresh account every training-instance quota is zero, so that path is
blocked until a quota increase is approved. Processing-job quotas are separate
and are not zero, so this file expresses the same DAG as ProcessingSteps.

The step logic is identical -- both files call the same functions in src/. The
difference is only which SageMaker job type executes them, which is exactly the
point of keeping the step functions free of any SageMaker imports.

    python -m pipeline.build_pipeline_processing \
        --bucket YOUR-BUCKET --role <execution-role-arn> \
        --data s3://YOUR-BUCKET/her2/pose_contract_table.csv
"""
from __future__ import annotations

import argparse
from time import gmtime, strftime

from sagemaker.core.image_uris import retrieve
from sagemaker.core.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep

BASE = "/opt/ml/processing"
ENTRY = "pipeline/entry/run_step.py"


def _processor(session, role, instance_type, region):
    return ScriptProcessor(
        image_uri=retrieve("sklearn", region, version="1.2-1", instance_type="ml.t3.medium"),
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        base_job_name="her2-pose",
        role=role,
        sagemaker_session=session,
    )


def _code_inputs():
    """src/ and config/ travel with every step, so any step can be re-run alone."""
    return [
        ProcessingInput(source="src", destination=f"{BASE}/input/code/src", input_name="src"),
        ProcessingInput(source="config", destination=f"{BASE}/input/code/config", input_name="config"),
    ]


def build(bucket: str, role: str, data_uri: str, region: str = "us-east-1",
          prefix: str = "her2-pose-reliability"):
    stamp = strftime("%d-%H-%M-%S", gmtime())
    out_root = f"s3://{bucket}/{prefix}/{stamp}"
    session = PipelineSession()

    p_data = ParameterString(name="InputDataUri", default_value=data_uri)
    p_group = ParameterString(name="ModelPackageGroupName", default_value=prefix)
    p_approval = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    p_max_fpr = ParameterFloat(name="MaxFalsePassRate", default_value=0.15)
    p_instance = ParameterString(name="InstanceType", default_value="ml.t3.medium")

    proc = _processor(session, role, p_instance, region)

    # ---- 1. preprocess ----------------------------------------------------
    pre_out = f"{out_root}/preprocess"
    s_pre = ProcessingStep(
        name="preprocess",
        step_args=proc.run(
            code=ENTRY,
            arguments=["--step", "preprocess"],
            inputs=_code_inputs() + [
                ProcessingInput(source=p_data, destination=f"{BASE}/input/data", input_name="data"),
            ],
            outputs=[ProcessingOutput(output_name="preprocess", source=f"{BASE}/output",
                                      destination=pre_out)],
        ),
    )

    # ---- 2. train ---------------------------------------------------------
    train_out = f"{out_root}/train"
    s_train = ProcessingStep(
        name="train",
        step_args=proc.run(
            code=ENTRY,
            arguments=["--step", "train"],
            inputs=_code_inputs() + [
                ProcessingInput(source=pre_out, destination=f"{BASE}/input/prev", input_name="prev"),
            ],
            outputs=[ProcessingOutput(output_name="train", source=f"{BASE}/output",
                                      destination=train_out)],
        ),
        depends_on=[s_pre],
    )

    # ---- 3. evaluate ------------------------------------------------------
    eval_out = f"{out_root}/evaluate"
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluate",
                               path="evaluation.json")
    s_eval = ProcessingStep(
        name="evaluate",
        step_args=proc.run(
            code=ENTRY,
            arguments=["--step", "evaluate"],
            inputs=_code_inputs() + [
                ProcessingInput(source=pre_out, destination=f"{BASE}/input/prev", input_name="prev"),
                ProcessingInput(source=train_out, destination=f"{BASE}/input/prev2", input_name="prev2"),
            ],
            outputs=[ProcessingOutput(output_name="evaluate", source=f"{BASE}/output",
                                      destination=eval_out)],
        ),
        property_files=[eval_report],
        depends_on=[s_train],
    )

    # ---- 4. register ------------------------------------------------------
    s_register = ProcessingStep(
        name="register",
        step_args=proc.run(
            code=ENTRY,
            arguments=[
                "--step", "register",
                "--model-package-group-name", p_group,
                "--approval-status", p_approval,
            ],
            inputs=_code_inputs() + [
                ProcessingInput(source=eval_out, destination=f"{BASE}/input/prev", input_name="prev"),
            ],
            outputs=[ProcessingOutput(output_name="register", source=f"{BASE}/output",
                                      destination=f"{out_root}/register")],
        ),
    )

    s_fail = FailStep(
        name="reliability-gate-failed",
        error_message="Held-out false-pass rate is above the configured ceiling",
    )

    # ---- the gate ---------------------------------------------------------
    # Registration is allowed only when the held-out false-pass rate clears the
    # ceiling. Not AUC: a false pass is what costs a downstream experiment.
    s_gate = ConditionStep(
        name="check-false-pass-rate",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(step_name=s_eval.name, property_file=eval_report,
                             json_path="evaluation_result.model.false_pass_rate"),
                right=p_max_fpr,
            )
        ],
        if_steps=[s_register],
        else_steps=[s_fail],
    )

    return Pipeline(
        name=f"{prefix}-proc-{stamp}",
        parameters=[p_data, p_group, p_approval, p_max_fpr, p_instance],
        steps=[s_pre, s_train, s_eval, s_gate],
        sagemaker_session=session,
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--data", required=True, help="s3:// uri of the contract table")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--prefix", default="her2-pose-reliability")
    ap.add_argument("--start", action="store_true")
    a = ap.parse_args()

    pipe = build(a.bucket, a.role, a.data, a.region, a.prefix)
    pipe.upsert(role_arn=a.role)
    print("upserted pipeline:", pipe.name)
    if a.start:
        ex = pipe.start()
        print("execution:", ex.describe()["PipelineExecutionArn"])


if __name__ == "__main__":  # pragma: no cover
    main()
