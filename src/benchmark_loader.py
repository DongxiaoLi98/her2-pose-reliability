"""Map a public benchmark table onto the pose-level contract.

Replaces the synthetic fixture with real data. Everything downstream --
gates, grouped split, training, cut-point locking, registration, batch
scoring -- is unchanged.

Honesty rules enforced here:
  * `rmsd` is used ONLY to derive the label. It never becomes a feature; a
    reference-dependent metric cannot be available for a novel candidate.
  * mean_AEE / std_AEE do not exist in public data. We substitute within-group
    score statistics and record that substitution in the returned provenance.
  * Rows whose score or rmsd could not be produced become runtime_status
    'failed' rather than being dropped, so the blocked path is exercised on
    real data too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import contract


class MappingError(ValueError):
    pass


def load_mapping(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _col(df: pd.DataFrame, name, required: bool, what: str):
    if name is None:
        if required:
            raise MappingError(f"'{what}' is required but mapped to null")
        return None
    if name not in df.columns:
        if required:
            raise MappingError(
                f"'{what}' maps to column '{name}', which is not in the table. "
                f"Run scripts/inspect_dataset.py and fix config/benchmark_mapping.yaml."
            )
        return None
    return df[name]


def load_benchmark(mapping_path: str = "config/benchmark_mapping.yaml",
                   table_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    m = load_mapping(mapping_path)
    c = m["columns"]
    src = pd.read_csv(table_path or m["input_table"])

    group = _col(src, c["group"], True, "group")
    pose = _col(src, c["pose"], True, "pose")
    rmsd = pd.to_numeric(_col(src, c["rmsd"], True, "rmsd"), errors="coerce")

    candidate = _col(src, c.get("candidate"), False, "candidate")
    method = _col(src, c.get("method"), False, "method")
    artifact = _col(src, c.get("artifact_ref"), False, "artifact_ref")

    # ---- score: mapped column, else a documented fallback -------------------
    provenance = {"source_name": m["source_name"], "substitutions": []}
    score = _col(src, c.get("score"), False, "score")
    if score is None:
        score = _col(src, c.get("fallback_score"), False, "fallback_score")
        if score is None:
            raise MappingError("neither 'score' nor 'fallback_score' resolves to a column")
        provenance["substitutions"].append(
            f"interaction_score <- {c['fallback_score']} (no docking score in table)"
        )
    score = pd.to_numeric(score, errors="coerce")

    # ---- QC pass fraction: always derivable ---------------------------------
    qc_cols = [q for q in m.get("qc_boolean_columns", []) if q in src.columns]
    if not qc_cols:
        raise MappingError("none of qc_boolean_columns are present in the table")
    qc = src[qc_cols].apply(lambda s: s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}))
    qc = qc.astype(float)
    pass_fraction = qc.mean(axis=1)
    provenance["qc_columns_used"] = qc_cols

    confidence = _col(src, c.get("confidence"), False, "confidence")
    if confidence is None:
        confidence = pass_fraction
        provenance["substitutions"].append(
            "structure_confidence <- fraction of QC checks passed (no tool confidence in table)"
        )
    confidence = pd.to_numeric(confidence, errors="coerce").clip(0, 1)

    out = pd.DataFrame({
        "run_id": m["source_name"],
        "candidate_id": (candidate if candidate is not None else group).astype(str),
        "structure_id": group.astype(str),
        "pose_id": pose.astype(str),
        "tool_output_ref": (artifact if artifact is not None else pose).astype(str),
        "interaction_score": score,
        "score_direction": c.get("score_direction", "lower_better"),
        "structure_confidence": confidence,
        "confidence_source": (method.astype(str) if method is not None else m["source_name"]),
        "calibration_status": "uncalibrated",
        "structure_coverage": pass_fraction,
        "runtime_status": "complete",
        "schema_version": contract.SCHEMA_VERSION,
    })

    # ---- conditional features ----------------------------------------------
    clash = _col(src, c.get("clash_count"), False, "clash_count")
    out["clash_count"] = pd.to_numeric(clash, errors="coerce") if clash is not None else np.nan

    mind = _col(src, c.get("min_distance_to_protein"), False, "min_distance_to_protein")
    if mind is not None:
        # a violated minimum distance is the small-molecule analogue of a
        # missing / clashing interface residue
        out["missing_interface_residues"] = (pd.to_numeric(mind, errors="coerce") < 0).astype(float)
    else:
        out["missing_interface_residues"] = np.nan

    noncov = _col(src, c.get("shortest_noncov"), False, "shortest_noncov")
    out["pose_quality_score"] = pd.to_numeric(noncov, errors="coerce").clip(0, 1) if noncov is not None else np.nan

    # ---- within-group statistics (documented proxies) -----------------------
    g = out.groupby("structure_id")["interaction_score"]
    out["mean_AEE"] = g.transform("mean")
    out["std_AEE"] = g.transform("std").fillna(0.0) + 1e-6
    out["ensemble_consistency_score"] = 1.0 / (1.0 + out["std_AEE"].abs())
    provenance["substitutions"].append(m["proxy_note"])

    # ---- label from rmsd, then discard rmsd ---------------------------------
    lr = m["label_rules"]
    label = pd.Series(pd.NA, index=out.index, dtype="object")
    label[rmsd > lr["review_at_or_below_rmsd"]] = "Fail"
    label[rmsd <= lr["review_at_or_below_rmsd"]] = "Review"
    label[rmsd <= lr["pass_at_or_below_rmsd"]] = "Pass"
    out["label"] = label
    out["label_source"] = lr["label_source"]
    out["label_confidence"] = lr["label_confidence"]

    # pose_pass_rate is a group statistic over the RULE-free notion of validity;
    # use the QC pass fraction so it never leaks the label.
    out["pose_pass_rate"] = out.groupby("structure_id")["structure_coverage"].transform(
        lambda s: (s >= 1.0).mean()
    )

    # ---- failed rows stay in, flagged --------------------------------------
    failed = score.isna() | rmsd.isna()
    out.loc[failed, "runtime_status"] = "failed"
    out.loc[failed, contract.REQUIRED_FEATURES] = np.nan
    out.loc[failed, "label"] = pd.NA

    contract.validate_frame(out)

    provenance.update(
        n_rows=int(len(out)),
        n_structures=int(out["structure_id"].nunique()),
        n_candidates=int(out["candidate_id"].nunique()),
        n_failed_rows=int(failed.sum()),
        label_counts={str(k): int(v) for k, v in
                      out["label"].value_counts(dropna=False).items()},
        rmsd_thresholds={"pass<=": lr["pass_at_or_below_rmsd"],
                         "review<=": lr["review_at_or_below_rmsd"]},
    )
    return out, provenance
