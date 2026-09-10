"""Pipeline step 1 -- preprocess.

Purpose
-------
Turn raw pose rows into (a) a scorable modelling table and (b) an explicit
status table for everything that is NOT scorable. The second output is the point
of the whole design: a failed tool run must leave the pipeline as
`needs_rerun`, not as a low biological score.
"""
from __future__ import annotations

import os

import pandas as pd

from . import contract, gates, io_utils, split


def load_raw(input_uri: str | None, source: str = "fixture",
             mapping_path: str = "config/benchmark_mapping.yaml") -> tuple[pd.DataFrame, dict]:
    """Three sources, one contract.

    'benchmark' -- a public benchmark table mapped through benchmark_mapping.yaml
    'table'     -- a CSV already in contract form (e.g. real AEE/PFE rows)
    'fixture'   -- synthetic rows, for tests and for running before data exists
    """
    if source == "benchmark":
        from .benchmark_loader import load_benchmark

        return load_benchmark(mapping_path, table_path=input_uri)
    if source == "table" or (source == "fixture" and input_uri):
        return io_utils.read_table(input_uri), {"source_name": "contract_table"}

    from .fixtures import make_pose_rows

    return make_pose_rows(), {"source_name": "synthetic_fixture",
                              "claim_boundary": "no metric from this source is a result"}


def preprocess(
    output_prefix: str,
    input_uri: str | None = None,
    threshold_config: str = "config/thresholds.v1.yaml",
    test_fraction: float = 0.25,
    seed: int = 0,
    source: str = "fixture",
    mapping_path: str = "config/benchmark_mapping.yaml",
) -> dict:
    raw, provenance = load_raw(input_uri, source=source, mapping_path=mapping_path)
    contract.validate_frame(raw)                      # structural checks only

    gated = gates.rankability_gate(raw)
    scorable = gated[gated["rankability_status"] == gates.RANKABLE].copy()
    blocked = gated[gated["rankability_status"] != gates.RANKABLE].copy()

    contract.require_label_columns(scorable)          # labels only where scoring happens

    # Direction normalisation happens exactly once, here.
    scorable = contract.normalize_direction(scorable)

    train_df, test_df = split.holdout_by_group(scorable, test_fraction=test_fraction, seed=seed)
    split.assert_no_group_leakage(train_df, test_df)

    thresholds = gates.load_thresholds(threshold_config)
    cutoffs = gates.fit_distribution_cutoffs(train_df, thresholds)  # TRAIN ONLY

    paths = {
        "train_uri": io_utils.write_table(train_df, os.path.join(output_prefix, "train.csv")),
        "test_uri": io_utils.write_table(test_df, os.path.join(output_prefix, "test.csv")),
        "blocked_uri": io_utils.write_table(blocked, os.path.join(output_prefix, "blocked_status.csv")),
        "cutoffs_uri": io_utils.write_json(
            {**cutoffs, "threshold_version": thresholds["threshold_version"]},
            os.path.join(output_prefix, "cutoffs.json"),
        ),
    }

    stats = {
        "n_raw_rows": int(len(raw)),
        "n_scorable_rows": int(len(scorable)),
        "n_blocked_rows": int(len(blocked)),
        "blocked_breakdown": blocked["rankability_status"].value_counts().to_dict(),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_structures": int(train_df[split.GROUP_KEY].nunique()),
        "n_test_structures": int(test_df[split.GROUP_KEY].nunique()),
        "train_label_counts": train_df["label"].value_counts().to_dict(),
        "test_label_counts": test_df["label"].value_counts().to_dict(),
        "schema_version": contract.SCHEMA_VERSION,
        "provenance": provenance,
    }
    io_utils.write_json(stats, os.path.join(output_prefix, "preprocess_stats.json"))
    return {**paths, **stats}


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(preprocess(output_prefix="artifacts/local"), indent=2, default=str))
