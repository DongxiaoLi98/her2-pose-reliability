"""Tests for the public-benchmark loader.

The loader is where real data meets the contract, so these tests check the
things that silently corrupt a benchmark: rmsd leaking in as a feature,
failed predictions being dropped, and a label rule that does not match the
documented thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import contract
from src.benchmark_loader import MappingError, load_benchmark, load_mapping
from src.preprocess import preprocess
from scripts.make_mock_benchmark import main as make_mock

MAPPING = "config/benchmark_mapping.yaml"
TABLE = "data/mock_benchmark_raw.csv"


@pytest.fixture(scope="module")
def mock_table():
    make_mock(n_structures=20, poses_per_structure=15, seed=11)
    return TABLE


def test_loader_produces_a_contract_compliant_frame(mock_table):
    df, prov = load_benchmark(MAPPING, table_path=mock_table)
    contract.validate_frame(df)                     # raises on any violation
    assert prov["n_structures"] == 20
    assert set(df["label"].dropna().unique()) <= {"Pass", "Review", "Fail"}


def test_rmsd_never_becomes_a_feature(mock_table):
    """A reference-dependent metric must not be available to the model."""
    df, _ = load_benchmark(MAPPING, table_path=mock_table)
    assert "rmsd" not in df.columns
    for f in contract.FEATURES:
        assert "rmsd" not in f.lower()


def test_failed_predictions_are_kept_and_flagged(mock_table):
    df, prov = load_benchmark(MAPPING, table_path=mock_table)
    failed = df[df["runtime_status"] == "failed"]
    assert prov["n_failed_rows"] == len(failed) > 0
    assert failed[contract.REQUIRED_FEATURES].isna().all().all()
    assert failed["label"].isna().all(), "a failed run must not be given a label"


def test_label_rule_matches_the_documented_thresholds(mock_table):
    raw = pd.read_csv(mock_table)
    df, _ = load_benchmark(MAPPING, table_path=mock_table)
    lr = load_mapping(MAPPING)["label_rules"]
    merged = df.assign(rmsd=raw["rmsd"].values).dropna(subset=["rmsd", "label"])
    assert (merged.loc[merged["label"] == "Pass", "rmsd"] <= lr["pass_at_or_below_rmsd"]).all()
    assert (merged.loc[merged["label"] == "Fail", "rmsd"] > lr["review_at_or_below_rmsd"]).all()


def test_mapping_error_is_actionable():
    with pytest.raises(MappingError) as e:
        load_benchmark(MAPPING, table_path="config/thresholds.v1.yaml")
    assert "inspect_dataset" in str(e.value) or "not in the table" in str(e.value)


def test_preprocess_routes_failed_rows_to_needs_rerun(mock_table, tmp_path):
    p = preprocess(output_prefix=str(tmp_path), input_uri=mock_table,
                   source="benchmark", mapping_path=MAPPING, seed=3)
    assert p["blocked_breakdown"].get("needs_rerun", 0) > 0
    assert p["n_train_structures"] + p["n_test_structures"] == 20
    # a label is required only where scoring happens
    train = pd.read_csv(p["train_uri"])
    assert train["label"].notna().all()


def test_a_fully_absent_conditional_feature_does_not_break_training(mock_table, tmp_path):
    """min_distance_to_protein is unmapped in the default config, so
    missing_interface_residues is NaN for every row. Training must still run."""
    from src.train import build_matrix

    df, _ = load_benchmark(MAPPING, table_path=mock_table)
    assert df["missing_interface_residues"].isna().all()
    # required features are only guaranteed present on rows the gate lets
    # through, so mirror the gate here
    scorable = df[df["runtime_status"] == "complete"]
    x, medians = build_matrix(scorable)
    assert np.isfinite(x.to_numpy()).all()
    assert medians["missing_interface_residues"] == 0.0
    assert x["missing_interface_residues__was_missing"].eq(1).all()
