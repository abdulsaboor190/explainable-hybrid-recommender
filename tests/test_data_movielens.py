"""Tests for src/data_movielens.py.

Checks structural correctness of the loaded MovieLens 100K dataset:
  - Exactly 100,000 ratings (ML-100K spec)
  - Exactly 943 unique users
  - Exactly 1,682 unique items
  - All ratings in [1, 5]
  - No duplicate (user_id, item_id) pairs
  - Every item has at least one genre token
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_movielens import load_movielens

RAW_DIR = Path("data/raw/ml-100k")


@pytest.fixture(scope="module")
def loaded_data():
    """Load data once per test module."""
    return load_movielens(raw_dir=RAW_DIR)


@pytest.fixture(scope="module")
def ratings_df(loaded_data):
    return loaded_data[0]


@pytest.fixture(scope="module")
def items_df(loaded_data):
    return loaded_data[1]


# ---------------------------------------------------------------------------
# Ratings tests
# ---------------------------------------------------------------------------

def test_total_ratings(ratings_df):
    """ML-100K must have exactly 100,000 ratings."""
    assert len(ratings_df) == 100_000, (
        f"Expected 100,000 ratings, got {len(ratings_df)}"
    )


def test_unique_users(ratings_df):
    """ML-100K must have exactly 943 unique users."""
    n_users = ratings_df["user_id"].nunique()
    assert n_users == 943, f"Expected 943 users, got {n_users}"


def test_unique_items_in_ratings(ratings_df):
    """ML-100K must have exactly 1,682 unique rated items."""
    n_items = ratings_df["item_id"].nunique()
    assert n_items == 1_682, f"Expected 1,682 items in ratings, got {n_items}"


def test_rating_range(ratings_df):
    """All ratings must be integers in [1, 5]."""
    assert ratings_df["rating"].between(1, 5).all(), (
        "Found ratings outside [1, 5]"
    )


def test_no_duplicate_interactions(ratings_df):
    """No (user_id, item_id) pair should appear more than once."""
    n_dupes = ratings_df.duplicated(subset=["user_id", "item_id"]).sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate (user_id, item_id) pairs"


def test_required_columns_ratings(ratings_df):
    """Ratings DataFrame must have the required columns."""
    required = {"user_id", "item_id", "rating", "timestamp"}
    assert required.issubset(ratings_df.columns), (
        f"Missing columns: {required - set(ratings_df.columns)}"
    )


# ---------------------------------------------------------------------------
# Items tests
# ---------------------------------------------------------------------------

def test_items_count(items_df):
    """Item catalogue must have exactly 1,682 items."""
    assert len(items_df) == 1_682, (
        f"Expected 1,682 items, got {len(items_df)}"
    )


def test_genre_tokens_not_empty(items_df):
    """Every item must have a non-empty genre_tokens string."""
    empty_mask = (
        items_df["genre_tokens"].isna()
        | (items_df["genre_tokens"].str.strip() == "")
    )
    n_empty = empty_mask.sum()
    assert n_empty == 0, f"{n_empty} items have empty genre_tokens"


def test_genre_tokens_at_least_one_token(items_df):
    """Every item's genre_tokens must contain at least one token."""
    token_counts = items_df["genre_tokens"].str.split().str.len()
    assert (token_counts >= 1).all(), (
        "Some items have zero genre tokens"
    )


def test_required_columns_items(items_df):
    """Items DataFrame must have the required columns."""
    required = {"item_id", "title", "genre_tokens"}
    assert required.issubset(items_df.columns), (
        f"Missing columns: {required - set(items_df.columns)}"
    )

