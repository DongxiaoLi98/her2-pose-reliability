"""SageMaker Pipeline definition.

The functions are NOT rewritten for the cloud. The same preprocess / train /
evaluate / register functions that run in run_local.py are wrapped with @step.
SageMaker infers the DAG from the data dependencies between them.

Run this from inside SageMaker Studio (JupyterLab), or from any machine that
has credentials for the execution role:

    python -m pipeline.build_pipeline --bucket <your-bucket> --role <role-arn>
"""
from __future__ import annotations

import argparse
import os
from time import gmtime, strftime

# SDK v3 (SageMaker Distribution 4.x) first, v2 as a fallback.
try:  # pragma: no cover
    from sagemaker.mlops.workflow.pipeline import Pipeline
    from sagemaker.mlops.workflow.function_step import step
    from sagemaker.mlops.workflow.condition_step import ConditionStep
    from sagemaker.mlops.workflow.fail_step import FailStep
    from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat
    from sagemaker.core.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.core.workflow.functions import Join
    from sagemaker.core.workflow.execution_variables import ExecutionVariables
except ImportError:  # pragma: no cover
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.function_step import step
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.parameters import ParameterString, ParameterFloat
    from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.workflow.functions import Join
    from sagemaker.workflow.execution_variables import ExecutionVariables

from src.evaluate import evaluate
from src.preprocess import preprocess
from src.register import register
from src.train import train


def build(bucket: str, prefix: str = "her2-pose-reliability", instance_type: str = "ml.m5.xlarge"):
    stamp = strftime("%d-%H-%M-%S", gmtime())
    pipeline_name = f"her2-pose-reliability-{stamp}"
    output_prefix = f"s3://{bucket}/{prefix}/{stamp}"

    p_input = ParameterString(name="InputDataUri", default_value="")  # s3:// contract table
    p_group = ParameterString(name="ModelPackageGroupName", default_value=prefix)
    p_approval = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    p_max_fpr = ParameterFloat(name="MaxFalsePassRate", default_value=0.15)
    p_instance = ParameterString(name="InstanceType", default_value=instance_type)

    s_pre = step(preprocess, name="preprocess", instance_type=p_instance)(
        output_prefix=output_prefix,
        input_uri=p_input,
        source="table",          # a contract-form table produced by
                                 # scripts/export_contract_table.py
    )
    s_train = step(train, name="train", instance_type=p_instance)(
        train_uri=s_pre["train_uri"],
        cutoffs_uri=s_pre["cutoffs_uri"],
        output_prefix=output_prefix,
    )
    s_eval = step(evaluate, name="evaluate", instance_type=p_instance)(
        test_uri=s_pre["test_uri"],
        model_uri=s_train["model_uri"],
        cutoffs_uri=s_pre["cutoffs_uri"],
        output_prefix=output_prefix,
    )
    s_register = step(register, name="register", instance_type=p_instance)(
        evaluation_result=s_eval["evaluation_result"],
        model_uri=s_train["model_uri"],
        output_prefix=output_prefix,
        model_package_group_name=p_group,
        approval_status=p_approval,
    )
    s_fail = FailStep(
        name="reliability-gate-failed",
        error_message=Join(on=" ", values=["Held-out false_pass_rate above", p_max_fpr]),
    )

    # THE GATE: registration is allowed only when the held-out false-pass rate
    # is at or below the configured ceiling. Not AUC -- false passes are what
    # cost a downstream experiment.
    s_gate = ConditionStep(
        name="check-false-pass-rate",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=s_eval["evaluation_result"]["model"]["false_pass_rate"],
                right=p_max_fpr,
            )
        ],
        if_steps=[s_register],
        else_steps=[s_fail],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[p_input, p_group, p_approval, p_max_fpr, p_instance],
        steps=[s_gate],
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--prefix", default="her2-pose-reliability")
    ap.add_argument("--start", action="store_true", help="start an execution after upsert")
    args = ap.parse_args()

    pipe = build(args.bucket, args.prefix)
    pipe.upsert(role_arn=args.role)
    print("upserted pipeline:", pipe.name)
    if args.start:
        ex = pipe.start()
        print("execution:", ex.describe()["PipelineExecutionArn"])


if __name__ == "__main__":  # pragma: no cover
    main()
