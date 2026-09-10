"""Two gates that run BEFORE any scientific score exists.

Gate 1 -- rankability: is this row even scorable?
Gate 2 -- reliability (rule baseline): is the pose trustworthy enough to enter
          ranking? Worst applicable rule wins; a strong interaction score can
          never override a Fail.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import contract

RANKABLE = "rankable"
ON_HOLD = "on_hold"
NEEDS_RERUN = "needs_rerun"

FAILED_RUNTIME = {"failed", "not_run", "dependency_unavailable"}


def load_thresholds(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def rankability_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Attach rankability_status + missing_fields. Never imputes zero.

    - runtime failure          -> needs_rerun  (an availability problem)
    - missing required feature -> on_hold      (an availability problem)
    - otherwise                -> rankable
    """
    out = df.copy()
    missing_fields, status = [], []

    for _, row in out.iterrows():
        miss = [c for c in contract.REQUIRED_FEATURES if pd.isna(row.get(c))]
        if row.get("runtime_status") in FAILED_RUNTIME:
            status.append(NEEDS_RERUN)
        elif miss:
            status.append(ON_HOLD)
        else:
            status.append(RANKABLE)
        missing_fields.append(miss)

    out["rankability_status"] = status
    out["missing_fields"] = missing_fields
    return out


def fit_distribution_cutoffs(train_df: pd.DataFrame, thresholds: dict) -> dict[str, float]:
    """std_AEE has no confirmed scale, so its cut-offs are percentiles of the
    TRAIN split only. Fitting them on all data would leak the test set."""
    cfg = thresholds["rules"]["std_AEE"]
    s = train_df["std_AEE"].dropna()
    return {
        "std_AEE_pass_at_or_below": float(np.percentile(s, cfg["pass_at_or_below_percentile"])),
        "std_AEE_fail_above": float(np.percentile(s, cfg["fail_above_percentile"])),
    }


def rule_reliability(df: pd.DataFrame, thresholds: dict, cutoffs: dict) -> pd.DataFrame:
    """Rule baseline. Returns pose_reliability_status + reason_codes."""
    r = thresholds["rules"]
    statuses, reasons = [], []

    for _, row in df.iterrows():
        codes, worst = [], "Pass"

        def demote(level: str, code: str) -> None:
            nonlocal worst
            codes.append(code)
            order = {"Pass": 0, "Review": 1, "Fail": 2}
            if order[level] > order[worst]:
                worst = level

        cov = row["structure_coverage"]
        if cov < r["structure_coverage"]["fail_below"]:
            demote("Fail", "coverage_below_fail")
        elif cov < r["structure_coverage"]["pass_at_or_above"]:
            demote("Review", "coverage_in_review_band")

        conf = row["structure_confidence"]
        if conf < r["structure_confidence"]["fail_below"]:
            demote("Fail", "confidence_below_fail")
        elif conf < r["structure_confidence"]["pass_at_or_above"]:
            demote("Review", "confidence_in_review_band")

        sd = row["std_AEE"]
        if sd > cutoffs["std_AEE_fail_above"]:
            demote("Fail", "std_aee_above_p95")
        elif sd > cutoffs["std_AEE_pass_at_or_below"]:
            demote("Review", "std_aee_in_p75_p95")

        clash = row.get("clash_count")
        if pd.notna(clash):
            if clash > r["clash_count"]["fail_above"]:
                demote("Fail", "clash_count_above_fail")
            elif clash > r["clash_count"]["pass_at_or_below"]:
                demote("Review", "clash_count_nonzero")

        mres = row.get("missing_interface_residues")
        if pd.notna(mres) and mres > r["missing_interface_residues"]["fail_above"]:
            demote("Fail", "missing_critical_interface_residue")

        statuses.append(worst)
        reasons.append(codes or ["all_rules_pass"])

    out = df.copy()
    out["rule_status"] = statuses
    out["reason_codes"] = reasons
    out["threshold_version"] = thresholds["threshold_version"]
    return out
