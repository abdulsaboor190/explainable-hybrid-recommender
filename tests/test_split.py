"""Tests for src/split.py – temporal per-user train/val/test split.

Checks:
  1. No rating row appears in two or more splits.
  2. For every user, max(train timestamp) <= min(test timestamp).
  3. Every user present in ratings is present in train.
  4. Split is deterministic (two calls produce identical DataFrames).
  5. Val rows never appear in test or train.
  6. Total row count is preserved across the three splits.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.split import split_per_user

# ---------------------------------------------------------------------------
# Shared fixtures – use the processed MovieLens ratings for speed
# ---------------------------------------------------------------------------

RATINGS_PATH = Path("data/processed/movielens/ratings.csv")


@pytest.fixture(scope="module")
def ratings_df():
    if not RATINGS_PATH.exists():
        pytest.skip(f"Ratings not found at {RATINGS_PATH}. Run preprocessing first.")
    return pd.read_csv(
        RATINGS_PATH,
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )


@pytest.fixture(scope="module")
def splits(ratings_df):
    """Run the split once; share across all tests in the module."""
    return split_per_user(ratings_df)


@pytest.fixture(scope="module")
def train_df(splits):
    return splits[0]


@pytest.fixture(scope="module")
def val_df(splits):
    return splits[1]


@pytest.fixture(scope="module")
def test_df(splits):
    return splits[2]


# ---------------------------------------------------------------------------
# 1. No row appears in two splits
# ---------------------------------------------------------------------------

def test_no_overlap_train_test(train_df, test_df, ratings_df):
    """Train and test index sets must be disjoint."""
    # Re-create a composite key to identify rows uniquely
    train_keys = set(zip(train_df["user_id"], train_df["item_id"]))
    test_keys  = set(zip(test_df["user_id"],  test_df["item_id"]))
    overlap = train_keys & test_keys
    assert not overlap, f"{len(overlap)} rows appear in both train and test"


def test_no_overlap_train_val(train_df, val_df):
    """Train and val (user, item) pairs must be disjoint."""
    train_keys = set(zip(train_df["user_id"], train_df["item_id"]))
    val_keys   = set(zip(val_df["user_id"],   val_df["item_id"]))
    overlap = train_keys & val_keys
    assert not overlap, f"{len(overlap)} rows appear in both train and val"


def test_no_overlap_val_test(val_df, test_df):
    """Val and test (user, item) pairs must be disjoint."""
    val_keys  = set(zip(val_df["user_id"],  val_df["item_id"]))
    test_keys = set(zip(test_df["user_id"], test_df["item_id"]))
    overlap = val_keys & test_keys
    assert not overlap, f"{len(overlap)} rows appear in both val and test"


# ---------------------------------------------------------------------------
# 2. Temporal ordering: max train timestamp ≤ min test timestamp (per user)
# ---------------------------------------------------------------------------

def test_temporal_order_train_test(train_df, test_df):
    """For every user, max train timestamp must be ≤ min test timestamp."""
    train_max = train_df.groupby("user_id")["timestamp"].max()
    test_min  = test_df.groupby("user_id")["timestamp"].min()

    # Only compare users present in both (some may have no test – shouldn't happen)
    common = train_max.index.intersection(test_min.index)
    violations = (train_max[common] > test_min[common]).sum()
    assert violations == 0, (
        f"{violations} users have train ratings newer than their test ratings"
    )


# ---------------------------------------------------------------------------
# 3. Every user is present in train
# ---------------------------------------------------------------------------

def test_every_user_in_train(ratings_df, train_df):
    """Every user from the full ratings set must appear in train."""
    all_users   = set(ratings_df["user_id"].unique())
    train_users = set(train_df["user_id"].unique())
    missing = all_users - train_users
    assert not missing, (
        f"{len(missing)} users missing from train: {list(missing)[:10]}"
    )


# ---------------------------------------------------------------------------
# 4. Determinism – two independent calls produce identical results
# ---------------------------------------------------------------------------

def test_split_deterministic(ratings_df):
    """Calling split_per_user twice on the same input must give identical splits."""
    tr1, va1, te1 = split_per_user(ratings_df)
    tr2, va2, te2 = split_per_user(ratings_df)

    for name, df1, df2 in [("train", tr1, tr2), ("val", va1, va2), ("test", te1, te2)]:
        df1_sorted = df1.sort_values(["user_id", "item_id"]).reset_index(drop=True)
        df2_sorted = df2.sort_values(["user_id", "item_id"]).reset_index(drop=True)
        assert df1_sorted.equals(df2_sorted), f"Non-deterministic result in {name} split"


# ---------------------------------------------------------------------------
# 5. Total row count is preserved
# ---------------------------------------------------------------------------

def test_total_row_count(ratings_df, train_df, val_df, test_df):
    """train + val + test must equal the total input rows."""
    total_split = len(train_df) + len(val_df) + len(test_df)
    assert total_split == len(ratings_df), (
        f"Row count mismatch: input {len(ratings_df)}, "
        f"split total {total_split}"
    )


# ---------------------------------------------------------------------------
# 6. Train split must have the largest share (~70-80 %)
# ---------------------------------------------------------------------------

def test_train_is_largest_split(train_df, val_df, test_df):
    """Train must contain more rows than val and test individually."""
    assert len(train_df) > len(val_df),  "Train is smaller than val"
    assert len(train_df) > len(test_df), "Train is smaller than test"
