"""Tests for src/data_foodcom.py.

Checks structural correctness of the loaded Food.com dataset:
  - All ratings in [1, 5]
  - No duplicate (user_id, item_id) pairs
  - Every rated item_id exists in items_df
  - Every user has >= 20 ratings (post-filter, pre-sample threshold)
  - No empty 'text' values in items_df
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path("data/processed/foodcom")
MIN_RATINGS_PER_USER = 20


@pytest.fixture(scope="module")
def ratings_df():
    """Load pre-processed ratings from disk (run script first)."""
    path = PROCESSED_DIR / "ratings.csv"
    if not path.exists():
        pytest.skip(f"Processed ratings not found at {path}. Run script first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def items_df():
    """Load pre-processed items from disk (run script first)."""
    path = PROCESSED_DIR / "items.csv"
    if not path.exists():
        pytest.skip(f"Processed items not found at {path}. Run script first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def filter_info():
    path = PROCESSED_DIR / "filter_info.json"
    if not path.exists():
        pytest.skip(f"filter_info.json not found at {path}. Run script first.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ratings tests
# ---------------------------------------------------------------------------

def test_rating_range(ratings_df):
    """All ratings must be in [1, 5]."""
    assert ratings_df["rating"].between(1, 5).all(), (
        f"Found ratings outside [1, 5]: "
        f"{ratings_df['rating'].unique().tolist()}"
    )


def test_no_duplicate_interactions(ratings_df):
    """No (user_id, item_id) pair should appear more than once."""
    n_dupes = ratings_df.duplicated(subset=["user_id", "item_id"]).sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate (user_id, item_id) pairs"


def test_required_columns_ratings(ratings_df):
    """Ratings DataFrame must have canonical columns."""
    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(ratings_df.columns)
    assert not missing, f"Missing columns in ratings: {missing}"


def test_every_user_has_min_ratings(ratings_df):
    """Every user in the final ratings must have >= 20 ratings."""
    user_counts = ratings_df["user_id"].value_counts()
    below = (user_counts < MIN_RATINGS_PER_USER).sum()
    assert below == 0, (
        f"{below} users have fewer than {MIN_RATINGS_PER_USER} ratings"
    )


# ---------------------------------------------------------------------------
# Cross-reference test
# ---------------------------------------------------------------------------

def test_all_rated_items_in_items_df(ratings_df, items_df):
    """Every item_id in ratings must have a matching row in items_df."""
    rated_ids = set(ratings_df["item_id"].unique())
    item_ids = set(items_df["item_id"].unique())
    missing = rated_ids - item_ids
    assert not missing, (
        f"{len(missing)} rated item_ids not found in items_df: "
        f"{list(missing)[:10]}"
    )


# ---------------------------------------------------------------------------
# Items tests
# ---------------------------------------------------------------------------

def test_no_empty_text(items_df):
    """Every recipe must have a non-empty 'text' column for TF-IDF."""
    empty_mask = (
        items_df["text"].isna() | (items_df["text"].str.strip() == "")
    )
    n_empty = empty_mask.sum()
    assert n_empty == 0, f"{n_empty} items have empty 'text'"


def test_required_columns_items(items_df):
    """Items DataFrame must have required columns."""
    required = {"item_id", "name", "category", "text", "calories", "total_minutes"}
    missing = required - set(items_df.columns)
    assert not missing, f"Missing columns in items: {missing}"


def test_no_duplicate_items(items_df):
    """No duplicate item_id in items_df."""
    n_dupes = items_df.duplicated(subset=["item_id"]).sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate item_ids in items_df"


# ---------------------------------------------------------------------------
# Filter info sanity
# ---------------------------------------------------------------------------

def test_filter_info_counts_monotone(filter_info):
    """User/recipe/rating counts must decrease through the pipeline stages."""
    counts = filter_info["counts"]
    bf = counts["before_filter"]
    af = counts["after_filter"]
    afs = counts["after_sample"]

    assert af["users"]   <= bf["users"],   "Users increased after filter"
    assert af["recipes"] <= bf["recipes"], "Recipes increased after filter"
    assert af["ratings"] <= bf["ratings"], "Ratings increased after filter"
    assert afs["users"]  <= af["users"],   "Users increased after sampling"

