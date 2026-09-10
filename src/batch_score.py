"""Pipeline step 5 -- batch scoring and candidate-level aggregation.

Purpose
-------
Score a whole pose ensemble in one batch (this workload is never a per-request
API call), attach reason codes and versions to every row, then roll poses up to
the candidate level. A candidate is never ranked from its single best pose.
"""
from __future__ import annotations

import io
import os
import time

import joblib
import numpy as np
import pandas as pd

from . import contract, gates, io_utils, split, train as train_mod


def score_batch(
    input_uri: str,
    model_uri: str,
    cutoffs_uri: str,
    output_prefix: str,
    threshold_config: str = "config/thresholds.v1.yaml",
) -> dict:
    raw = io_utils.read_table(input_uri)
    contract.validate_frame(raw)
    bundle = joblib.load(io.BytesIO(io_utils.read_bytes(model_uri)))
    thresholds = gates.load_thresholds(threshold_config)
    cutoffs = io_utils.read_json(cutoffs_uri)

    gated = gates.rankability_gate(raw)
    scorable = contract.normalize_direction(gated[gated["rankability_status"] == gates.RANKABLE].copy())
    blocked = gated[gated["rankability_status"] != gates.RANKABLE].copy()

    t0 = time.perf_counter()
    x, _ = train_mod.build_matrix(scorable, medians=bundle["medians"])
    x = x[bundle["feature_columns"]]
    proba = bundle["model"].predict_proba(x)
    order = [list(bundle["model"].classes_).index(c) for c in bundle["classes"]]
    proba = proba[:, order]
    pred = train_mod.apply_cutpoints(proba, bundle["classes"], bundle["pass_cut"], bundle["fail_cut"])
    elapsed = time.perf_counter() - t0

    ruled = gates.rule_reliability(scorable, thresholds, cutoffs)
    scorable["rule_status"] = ruled["rule_status"].to_numpy()
    scorable["reason_codes"] = ruled["reason_codes"].to_numpy()
    scorable["model_status"] = pred
    scorable["probability_pass"] = proba[:, bundle["classes"].index("Pass")]
    scorable["probability_fail"] = proba[:, bundle["classes"].index("Fail")]

    # A deterministic Fail from the rule engine can never be overridden by the
    # model. Disagreement anywhere else becomes Review, not an averaged guess.
    combined, disagreement = [], []
    for r_status, m_status in zip(scorable["rule_status"], scorable["model_status"]):
        if r_status == "Fail" or m_status == "Fail":
            combined.append("Fail")
        elif r_status == m_status:
            combined.append(r_status)
        else:
            combined.append("Review")
        disagreement.append(int(r_status != m_status))
    scorable["pose_reliability_status"] = combined
    scorable["model_disagreement"] = disagreement
    scorable["model_name"] = bundle["model_name"]
    scorable["threshold_version"] = bundle["threshold_version"]
    scorable["schema_version"] = bundle["schema_version"]

    scored_uri = io_utils.write_table(scorable, os.path.join(output_prefix, "scored_poses.csv"))
    blocked_uri = io_utils.write_table(blocked, os.path.join(output_prefix, "scored_blocked.csv"))

    latency = {
        "rows_scored": int(len(scorable)),
        "batch_seconds": round(elapsed, 4),
        "ms_per_row": round(1000 * elapsed / max(1, len(scorable)), 4),
        "cpu_count": os.cpu_count(),
    }
    io_utils.write_json(latency, os.path.join(output_prefix, "inference_latency.json"))
    return {"scored_uri": scored_uri, "blocked_uri": blocked_uri, **latency}


def aggregate_candidates(scored_uri: str, blocked_uri: str, output_prefix: str) -> str:
    scored = io_utils.read_table(scored_uri, list_columns=("reason_codes",))
    blocked = io_utils.read_table(blocked_uri, list_columns=("missing_fields",))

    rows = []
    for candidate_id, g in scored.groupby("candidate_id"):
        valid = len(g)
        n_pass = int((g["pose_reliability_status"] == "Pass").sum())
        n_fail = int((g["pose_reliability_status"] == "Fail").sum())
        accepted = g[g["pose_reliability_status"] == "Pass"]
        # crude "dominant cluster" proxy: the largest structure inside the candidate
        dominant = g["structure_id"].value_counts().iloc[0] / valid if valid else 0.0

        # Candidate policy, in the documented precedence order.
        if valid == 0:
            rec = "Hold"                       # not enough valid evidence
        elif n_fail / valid >= 0.5:
            rec = "Hold"                       # predominantly decoy-like
        elif n_pass / valid >= 0.5 and g["model_disagreement"].mean() <= 0.30:
            rec = "Advance"                    # reproducible and agreed
        else:
            rec = "Advance with review"        # mixed ensemble or disagreement

        rows.append(
            dict(
                candidate_id=candidate_id,
                valid_pose_count=valid,
                native_like_pose_rate=round(n_pass / valid, 4) if valid else 0.0,
                dominant_cluster_fraction=round(float(dominant), 4),
                median_accepted_interaction_score=(
                    round(float(accepted["interaction_score"].median()), 4) if len(accepted) else None
                ),
                accepted_score_iqr=(
                    round(float(np.subtract(*np.percentile(accepted["interaction_score"], [75, 25]))), 4)
                    if len(accepted) > 1
                    else None
                ),
                model_disagreement_rate=round(float(g["model_disagreement"].mean()), 4),
                blocked_pose_count=int((blocked["candidate_id"] == candidate_id).sum()),
                final_recommendation=rec,
            )
        )

    # candidates whose every pose was blocked still have to appear, as on_hold
    for candidate_id, g in blocked.groupby("candidate_id"):
        if candidate_id in scored["candidate_id"].values:
            continue
        rows.append(
            dict(
                candidate_id=candidate_id,
                valid_pose_count=0,
                native_like_pose_rate=0.0,
                dominant_cluster_fraction=0.0,
                median_accepted_interaction_score=None,
                accepted_score_iqr=None,
                model_disagreement_rate=None,
                blocked_pose_count=int(len(g)),
                final_recommendation="Rerun" if (g["rankability_status"] == gates.NEEDS_RERUN).any() else "Hold",
            )
        )

    out = pd.DataFrame(rows).sort_values("candidate_id")
    return io_utils.write_table(out, os.path.join(output_prefix, "candidate_summary.csv"))
