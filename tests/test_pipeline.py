"""Integration tests.

These are the tests that would catch the four failure modes the design exists
to prevent: zero-imputation of missing data, group leakage, a high score
overriding a Fail, and a non-idempotent rerun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import contract, gates, metrics, split
from src.batch_score import aggregate_candidates, score_batch
from src.evaluate import evaluate
from src.fixtures import make_pose_rows
from src.preprocess import preprocess
from src.register import register
from src.train import train

CONFIG = "config/thresholds.v1.yaml"


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #
def test_contract_rejects_unknown_enum_value():
    df = make_pose_rows(n_structures=3, poses_per_structure=4)
    df.loc[0, "runtime_status"] = "kind_of_ok"
    with pytest.raises(contract.ContractError):
        contract.validate_frame(df)


def test_direction_normalisation_flips_lower_better_rows_only():
    df = make_pose_rows(n_structures=10, poses_per_structure=3).dropna(subset=["interaction_score"])
    out = contract.normalize_direction(df)
    lower = df["score_direction"] == "lower_better"
    assert np.allclose(out.loc[lower, "interaction_score"], -df.loc[lower, "interaction_score"])
    assert np.allclose(out.loc[~lower, "interaction_score"], df.loc[~lower, "interaction_score"])


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def test_failed_runtime_becomes_needs_rerun_not_a_low_score():
    df = make_pose_rows(n_structures=4, poses_per_structure=5, seed=1)
    df.loc[0, "runtime_status"] = "failed"
    df.loc[0, contract.REQUIRED_FEATURES] = np.nan
    out = gates.rankability_gate(df)
    assert out.loc[0, "rankability_status"] == gates.NEEDS_RERUN


def test_missing_required_feature_becomes_on_hold_and_is_never_zero():
    df = make_pose_rows(n_structures=4, poses_per_structure=5, seed=2)
    df.loc[0, "runtime_status"] = "complete"
    df.loc[0, "mean_AEE"] = np.nan
    out = gates.rankability_gate(df)
    assert out.loc[0, "rankability_status"] == gates.ON_HOLD
    assert "mean_AEE" in out.loc[0, "missing_fields"]
    assert pd.isna(out.loc[0, "mean_AEE"])  # not silently filled with 0


def test_high_interaction_score_cannot_override_a_reliability_fail():
    df = make_pose_rows(n_structures=4, poses_per_structure=5, seed=3).dropna(subset=contract.REQUIRED_FEATURES)
    df = df.reset_index(drop=True)
    df.loc[0, "interaction_score"] = 1e6          # spectacular score
    df.loc[0, "missing_interface_residues"] = 1   # critical structural defect
    df.loc[0, "score_direction"] = "higher_better"
    thresholds = gates.load_thresholds(CONFIG)
    cutoffs = gates.fit_distribution_cutoffs(df, thresholds)
    out = gates.rule_reliability(df, thresholds, cutoffs)
    assert out.loc[0, "rule_status"] == "Fail"
    assert "missing_critical_interface_residue" in out.loc[0, "reason_codes"]


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_grouped_split_puts_no_structure_on_both_sides():
    df = make_pose_rows(n_structures=12, poses_per_structure=6, seed=4)
    tr, te = split.holdout_by_group(df, test_fraction=0.25, seed=4)
    split.assert_no_group_leakage(tr, te)
    assert len(tr) + len(te) == len(df)


def test_leakage_assertion_actually_fires():
    df = make_pose_rows(n_structures=6, poses_per_structure=4, seed=5)
    with pytest.raises(AssertionError):
        split.assert_no_group_leakage(df, df)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_false_pass_rate_counts_only_true_fails_predicted_pass():
    y = np.array(["Fail", "Fail", "Pass", "Review"])
    p = np.array(["Pass", "Fail", "Pass", "Review"])
    assert metrics.false_pass_rate(y, p) == 0.5
    assert metrics.false_fail_rate(y, p) == 0.0


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("run"))
    p = preprocess(output_prefix=out, threshold_config=CONFIG, seed=7)
    t = train(p["train_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG, seed=7)
    e = evaluate(p["test_uri"], t["model_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG)
    return out, p, t, e


def test_pipeline_smoke_produces_a_gate_decision(run):
    out, p, t, e = run
    r = e["evaluation_result"]
    assert p["n_blocked_rows"] > 0, "fixture must exercise the blocked path"
    assert t["selected_model"] in {"logistic_regression", "svm", "gradient_boosting"}
    assert 0.0 <= r["model"]["false_pass_rate"] <= 1.0
    reg = register(r, t["model_uri"], out, threshold_config=CONFIG)
    assert isinstance(reg["registered"], bool)
    assert reg["threshold_version"] == "v1"


def test_batch_scoring_is_deterministic_and_idempotent(run):
    out, p, t, _ = run
    a = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG)
    first = pd.read_csv(a["scored_uri"])["pose_reliability_status"].tolist()
    b = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG)
    second = pd.read_csv(b["scored_uri"])["pose_reliability_status"].tolist()
    assert first == second, "re-scoring the same input must converge to the same state"
    assert a["ms_per_row"] > 0


def test_every_scored_row_carries_its_versions(run):
    out, p, t, _ = run
    a = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG)
    scored = pd.read_csv(a["scored_uri"])
    for col in ["threshold_version", "schema_version", "model_name", "reason_codes"]:
        assert col in scored.columns and scored[col].notna().all()


def test_candidate_aggregation_covers_every_candidate(run):
    out, p, t, _ = run
    a = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], out, threshold_config=CONFIG)
    uri = aggregate_candidates(a["scored_uri"], a["blocked_uri"], out)
    summary = pd.read_csv(uri)
    scored = pd.read_csv(a["scored_uri"])
    assert set(scored["candidate_id"]).issubset(set(summary["candidate_id"]))
    assert summary["final_recommendation"].isin(
        ["Advance", "Advance with review", "Hold", "Rerun", "Reference only"]
    ).all()
