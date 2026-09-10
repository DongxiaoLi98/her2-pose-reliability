"""Synthetic pose rows that satisfy the contract.

WHY THIS EXISTS
---------------
Real pose-level AEE/PFE feature rows are not produced yet by the upstream
workflow. This generator emits rows with the same schema, the same grouping
structure (many poses per structure), the same missingness and runtime-failure
patterns, and labels drawn from a latent rule plus noise.

That makes the whole pipeline runnable and testable today, and swapping in real
data is a one-function change (replace load_raw() in preprocess.py).

It is a FIXTURE, not a benchmark. No performance number produced from it may be
presented as a biological result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract


def make_pose_rows(
    n_structures: int = 40,
    poses_per_structure: int = 25,
    seed: int = 0,
    missing_rate: float = 0.04,
    failed_run_rate: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for s in range(n_structures):
        structure_id = f"struct_{s:03d}"
        candidate_id = f"cand_{s // 2:03d}"          # 2 structures per candidate
        run_id = f"run_{s // 10:03d}"                # 10 structures per run
        # Structure-level latent quality: this is what makes a random row split
        # leak -- poses from the same structure share it.
        structure_quality = rng.beta(2.5, 2.0)
        direction = "higher_better" if s % 5 else "lower_better"

        for p in range(poses_per_structure):
            coverage = np.clip(rng.normal(0.55 + 0.45 * structure_quality, 0.08), 0, 1)
            confidence = np.clip(rng.normal(0.45 + 0.55 * structure_quality, 0.10), 0, 1)
            mean_aee = rng.normal(6.0 * structure_quality, 1.0)
            std_aee = abs(rng.normal(2.2 - 1.5 * structure_quality, 0.6)) + 0.05
            clash = rng.poisson(max(0.2, 4.0 * (1 - structure_quality)))
            missing_res = int(rng.random() < 0.10 * (1 - structure_quality))
            pose_quality = np.clip(rng.normal(structure_quality, 0.15), 0, 1)
            consistency = np.clip(rng.normal(structure_quality, 0.12), 0, 1)
            pass_rate = np.clip(rng.normal(structure_quality, 0.10), 0, 1)
            raw_interaction = rng.normal(2.0 * structure_quality + 0.5 * pose_quality, 0.8)

            # Latent labelling rule + noise. Deliberately NOT identical to the
            # rule baseline, so the supervised models have something to learn.
            latent = (
                1.4 * coverage
                + 1.2 * confidence
                + 0.8 * consistency
                - 0.35 * std_aee
                - 0.12 * clash
                - 0.9 * missing_res
                + rng.normal(0, 0.28)
            )
            if missing_res or clash > 8:
                label = "Fail"
            elif latent > 2.05:
                label = "Pass"
            elif latent > 1.45:
                label = "Review"
            else:
                label = "Fail"

            rows.append(
                dict(
                    run_id=run_id,
                    candidate_id=candidate_id,
                    structure_id=structure_id,
                    pose_id=f"{structure_id}_pose_{p:03d}",
                    tool_output_ref=f"s3://artifacts/{run_id}/{structure_id}/pose_{p:03d}.json",
                    interaction_score=(-raw_interaction if direction == "lower_better" else raw_interaction),
                    score_direction=direction,
                    structure_confidence=confidence,
                    confidence_source="structure_predictor_v1",
                    calibration_status="uncalibrated",
                    structure_coverage=coverage,
                    mean_AEE=mean_aee,
                    std_AEE=std_aee,
                    pose_quality_score=pose_quality,
                    ensemble_consistency_score=consistency,
                    pose_pass_rate=pass_rate,
                    clash_count=int(clash),
                    missing_interface_residues=missing_res,
                    runtime_status="complete",
                    schema_version=contract.SCHEMA_VERSION,
                    label=label,
                    label_source="weak_rule",
                    label_confidence=0.6,
                )
            )

    df = pd.DataFrame(rows)

    # Inject the two failure modes the contract must survive.
    n = len(df)
    fail_idx = rng.choice(n, size=int(failed_run_rate * n), replace=False)
    df.loc[fail_idx, "runtime_status"] = "failed"
    df.loc[fail_idx, contract.REQUIRED_FEATURES] = np.nan

    remaining = np.setdiff1d(np.arange(n), fail_idx)
    miss_idx = rng.choice(remaining, size=int(missing_rate * n), replace=False)
    for i in miss_idx:
        col = rng.choice(contract.REQUIRED_FEATURES)
        df.loc[i, col] = np.nan

    return df


if __name__ == "__main__":  # pragma: no cover
    d = make_pose_rows()
    print(d.shape)
    print(d["label"].value_counts())
    print(d["runtime_status"].value_counts())
