"""Pose-level input contract.

One row = one pose (one member of a predicted structure ensemble).
The contract is the single place that decides:
  - which fields are required,
  - what values are allowed,
  - what happens when a value is missing (never zero-imputation).
"""
from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "her2-pose-v1"

# --- identity / traceability -------------------------------------------------
ID_FIELDS = [
    "run_id",          # one reproducible pipeline run
    "candidate_id",    # one ADC candidate inside that run
    "structure_id",    # the structure the pose was generated from  -> GROUP KEY
    "pose_id",         # one ensemble member
    "tool_output_ref", # exact upstream artifact reference
]

# --- numeric model features --------------------------------------------------
REQUIRED_FEATURES = [
    "interaction_score",
    "structure_confidence",
    "structure_coverage",
    "mean_AEE",
    "std_AEE",
]
CONDITIONAL_FEATURES = [
    "pose_quality_score",
    "ensemble_consistency_score",
    "pose_pass_rate",
    "clash_count",
    "missing_interface_residues",
]
FEATURES = REQUIRED_FEATURES + CONDITIONAL_FEATURES

# --- metadata that must travel with the numbers ------------------------------
META_FIELDS = [
    "score_direction",     # higher_better | lower_better
    "confidence_source",
    "calibration_status",  # uncalibrated | provisionally_mapped | calibrated
    "runtime_status",      # complete | partial | failed | not_run | dependency_unavailable
    "schema_version",
]

LABEL_FIELDS = ["label", "label_source", "label_confidence"]

ALLOWED = {
    "score_direction": {"higher_better", "lower_better"},
    "calibration_status": {"uncalibrated", "provisionally_mapped", "calibrated"},
    "runtime_status": {"complete", "partial", "failed", "not_run", "dependency_unavailable"},
    "label": {"Pass", "Review", "Fail"},
    "label_source": {"experimental", "benchmark", "expert", "weak_rule"},
}

CLASSES = ["Pass", "Review", "Fail"]

RANGE_0_1 = ["structure_coverage", "pose_pass_rate", "label_confidence"]


class ContractError(ValueError):
    """Raised when a row violates a rule that cannot be expressed as a status."""


def validate_frame(df: pd.DataFrame, require_labels: bool = False) -> pd.DataFrame:
    """Structural validation. Raises on contract violations that are bugs.

    Missing *values* are NOT errors here -- they are handled downstream by the
    rankability gate, which turns them into on_hold / needs_rerun statuses.
    """
    missing_cols = [c for c in ID_FIELDS + REQUIRED_FEATURES + META_FIELDS if c not in df.columns]
    if missing_cols:
        raise ContractError(f"missing required columns: {missing_cols}")

    for col, allowed in ALLOWED.items():
        if col in df.columns:
            bad = set(df[col].dropna().unique()) - allowed
            if bad:
                raise ContractError(f"{col} has values outside the controlled vocabulary: {sorted(bad)}")

    for col in RANGE_0_1:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) and (s.lt(0).any() or s.gt(1).any()):
                raise ContractError(f"{col} must be within [0, 1]")

    if require_labels:
        require_label_columns(df)

    return df


def require_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Labels are only required on rows that will actually be scored.

    A row blocked by the rankability gate (failed tool run, missing input) has
    no label and must not be forced to have one -- inventing a label for an
    unavailable result is the same mistake as imputing zero for a missing value.
    """
    for col in LABEL_FIELDS:
        if col not in df.columns:
            raise ContractError(f"training data requires column {col}")
    if df["label"].isna().any():
        n = int(df["label"].isna().sum())
        raise ContractError(f"{n} scorable rows carry no label; check the label rule")
    return df


def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Make interaction_score comparable across tools.

    Some engines report 'more negative is better'. We flip those rows once, here,
    and record that we did it. Comparing raw values across tools without this is
    the classic score-direction bug.
    """
    out = df.copy()
    flip = out["score_direction"].eq("lower_better")
    out.loc[flip, "interaction_score"] = -out.loc[flip, "interaction_score"]
    out["interaction_score_direction_normalized"] = True
    return out
