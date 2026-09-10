"""Grouped splitting.

Poses generated from the same structure are near-duplicates. A random row split
puts siblings on both sides and inflates every metric. We split by structure_id
(leave-one-structure-out in spirit; GroupKFold / GroupShuffleSplit in practice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GROUP_KEY = "structure_id"


def holdout_by_group(
    df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 0, group_key: str = GROUP_KEY
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole groups, never rows."""
    groups = np.array(sorted(df[group_key].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    test_groups = set(groups[:n_test])
    is_test = df[group_key].isin(test_groups)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def grouped_folds(df: pd.DataFrame, n_splits: int = 5, group_key: str = GROUP_KEY):
    gkf = GroupKFold(n_splits=n_splits)
    return gkf.split(df, groups=df[group_key])


def assert_no_group_leakage(train: pd.DataFrame, test: pd.DataFrame, group_key: str = GROUP_KEY) -> None:
    overlap = set(train[group_key]) & set(test[group_key])
    if overlap:
        raise AssertionError(f"group leakage between train and test: {sorted(overlap)[:5]}")
