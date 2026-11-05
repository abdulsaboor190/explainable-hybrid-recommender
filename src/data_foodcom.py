"""Food.com recipes and reviews data loader.

Reads large CSV files in memory-efficient chunks, applies activity filtering,
samples users, and builds a TF-IDF-ready text column for content-based models.

Column mapping (actual → canonical):
  reviews  : RecipeId→item_id, AuthorId→user_id, Rating→rating,
             DateSubmitted→timestamp (unix seconds)
  recipes  : RecipeId→item_id, Name→name, RecipeCategory→category,
             Keywords→keywords_list, RecipeIngredientParts→ingredients_list,
             Calories→calories, TotalTime→total_minutes
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Columns we actually need (saves >70 % RAM on the 700 MB recipes file)
_REVIEW_COLS = ["RecipeId", "AuthorId", "Rating", "DateSubmitted"]
_RECIPE_COLS = [
    "RecipeId", "Name", "RecipeCategory",
    "Keywords", "RecipeIngredientParts",
    "Calories", "TotalTime",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_r_vector(val) -> List[str]:
    """Parse an R-style character vector string into a Python list of strings.

    Handles:
      c("a", "b", "c")  →  ["a", "b", "c"]
      "single value"     →  ["single value"]
      NA / NaN / None    →  []
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() in ("na", "nan", "none", "character(0)"):
        return []
    # Wrapped in c(...)
    if s.startswith("c("):
        inner = s[2:-1]  # strip c( and )
        # Extract all quoted strings
        tokens = re.findall(r'"([^"]*)"', inner)
        return [t.strip() for t in tokens if t.strip()]
    # Single quoted value
    m = re.match(r'^"([^"]*)"$', s)
    if m:
        return [m.group(1).strip()]
    # Bare string
    return [s]


def _parse_iso_duration(val) -> Optional[float]:
    """Parse ISO 8601 duration (e.g. PT1H30M, PT45M, PT24H) to total minutes.

    Returns None if the value is null or unparseable.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper()
    if not s or s in ("NA", "NAN", "NONE", ""):
        return None
    hours = re.search(r"(\d+(?:\.\d+)?)H", s)
    minutes = re.search(r"(\d+(?:\.\d+)?)M", s)
    total = 0.0
    if hours:
        total += float(hours.group(1)) * 60
    if minutes:
        total += float(minutes.group(1))
    return total if (hours or minutes) else None


def _tokens_to_text(items: List[str]) -> str:
    """Convert a list of ingredient/keyword strings to TF-IDF tokens.

    Multi-word terms have internal spaces replaced by underscores so they
    remain single tokens (e.g. "olive oil" → "olive_oil").
    """
    out = []
    for item in items:
        token = item.strip().lower()
        token = re.sub(r"\s+", "_", token)   # spaces → underscores
        if token:
            out.append(token)
    return " ".join(out)


def _load_reviews_chunked(raw_dir: Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Read reviews.csv in chunks, keeping only needed columns."""
    chunks = []
    for chunk in pd.read_csv(
        raw_dir / "reviews.csv",
        usecols=_REVIEW_COLS,
        chunksize=chunksize,
        dtype={"RecipeId": "int32", "AuthorId": "int32", "Rating": "int8"},
    ):
        # Keep only valid ratings inside the chunk to save RAM early
        chunk = chunk[chunk["Rating"].between(1, 5)].copy()
        chunk.dropna(subset=["RecipeId", "AuthorId", "Rating", "DateSubmitted"],
                     inplace=True)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def _iterative_filter(
    df: pd.DataFrame,
    min_user: int = 20,
    min_item: int = 20,
) -> pd.DataFrame:
    """Remove users/items below activity thresholds until convergence."""
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_foodcom(
    raw_dir: str | Path = "data/raw/foodcom",
    min_ratings_per_user: int = 20,
    min_ratings_per_recipe: int = 20,
    sample_users: int = 3_000,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load, filter, and sample the Food.com dataset.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing reviews.csv and recipes.csv.
    min_ratings_per_user : int
        Minimum number of ratings a user must have after iterative filtering.
    min_ratings_per_recipe : int
        Minimum number of ratings a recipe must have after iterative filtering.
    sample_users : int
        Number of users to sample (seed = random_state). If fewer remain,
        all users are kept.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    ratings_df : pd.DataFrame
        Columns: user_id, item_id, rating (int), timestamp (int unix seconds).
    items_df : pd.DataFrame
        Columns: item_id, name, category, ingredients_list, keywords_list,
                 calories, total_minutes, text.
    filter_info : dict
        Counts at each stage of the pipeline and config parameters.
    """
    raw_dir = Path(raw_dir)

    # ------------------------------------------------------------------
    # 1. Load reviews in chunks
    # ------------------------------------------------------------------
    print("Reading reviews.csv in chunks …", flush=True)
    reviews = _load_reviews_chunked(raw_dir)

    # Rename columns to canonical names
    reviews.rename(columns={
        "RecipeId": "item_id",
        "AuthorId": "user_id",
        "Rating": "rating",
        "DateSubmitted": "timestamp",
    }, inplace=True)

    # Parse timestamp to unix seconds (integer)
    reviews["timestamp"] = pd.to_datetime(
        reviews["timestamp"], utc=True, errors="coerce"
    ).astype("int64") // 10**9  # nanoseconds → seconds

    reviews.dropna(subset=["user_id", "item_id", "rating", "timestamp"],
                   inplace=True)

    # Drop duplicates keeping latest interaction
    reviews.sort_values("timestamp", ascending=False, inplace=True)
    reviews.drop_duplicates(subset=["user_id", "item_id"], keep="first",
                             inplace=True)
    reviews["user_id"] = reviews["user_id"].astype("int32")
    reviews["item_id"] = reviews["item_id"].astype("int32")
    reviews["rating"]  = reviews["rating"].astype("int8")
    reviews["timestamp"] = reviews["timestamp"].astype("int64")

    before_filter = {
        "users": int(reviews["user_id"].nunique()),
        "recipes": int(reviews["item_id"].nunique()),
        "ratings": int(len(reviews)),
    }
    print(f"  Before filter: {before_filter}", flush=True)

    # ------------------------------------------------------------------
    # 2. Iterative core filter
    # ------------------------------------------------------------------
    print("Applying iterative activity filter …", flush=True)
    reviews = _iterative_filter(reviews, min_ratings_per_user,
                                 min_ratings_per_recipe)

    after_filter = {
        "users": int(reviews["user_id"].nunique()),
        "recipes": int(reviews["item_id"].nunique()),
        "ratings": int(len(reviews)),
    }
    print(f"  After filter : {after_filter}", flush=True)

    # ------------------------------------------------------------------
    # 3. Sample users
    # ------------------------------------------------------------------
    all_users = reviews["user_id"].unique()
    if len(all_users) <= sample_users:
        sampled_users = all_users
        actual_sample = len(all_users)
        print(f"  Warning: only {len(all_users)} users remain — keeping all.",
              flush=True)
    else:
        rng = np.random.default_rng(random_state)
        sampled_users = rng.choice(all_users, size=sample_users, replace=False)
        actual_sample = sample_users

    ratings_df = reviews[reviews["user_id"].isin(sampled_users)].copy()
    ratings_df.sort_values(["user_id", "item_id"], inplace=True)
    ratings_df.reset_index(drop=True, inplace=True)

    after_sample = {
        "users": int(ratings_df["user_id"].nunique()),
        "recipes": int(ratings_df["item_id"].nunique()),
        "ratings": int(len(ratings_df)),
    }
    print(f"  After sample : {after_sample}", flush=True)

    sampled_item_ids = set(ratings_df["item_id"].unique())

    # ------------------------------------------------------------------
    # 4. Load recipes for sampled items only
    # ------------------------------------------------------------------
    print("Reading recipes.csv in chunks …", flush=True)
    recipe_chunks = []
    for chunk in pd.read_csv(
        raw_dir / "recipes.csv",
        usecols=_RECIPE_COLS,
        chunksize=100_000,
        dtype={"RecipeId": "int32"},
    ):
        sub = chunk[chunk["RecipeId"].isin(sampled_item_ids)]
        if not sub.empty:
            recipe_chunks.append(sub)

    items_raw = pd.concat(recipe_chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Parse R-style vectors, build text column
    # ------------------------------------------------------------------
    items_raw["ingredients_list"] = items_raw["RecipeIngredientParts"].apply(
        _parse_r_vector
    )
    items_raw["keywords_list"] = items_raw["Keywords"].apply(_parse_r_vector)

    # TF-IDF text: ingredients + keywords, underscore-tokenised
    items_raw["text"] = items_raw.apply(
        lambda row: _tokens_to_text(
            row["ingredients_list"] + row["keywords_list"]
        ),
        axis=1,
    )
    # Fallback: use recipe name if text is empty
    empty_text = items_raw["text"].str.strip() == ""
    items_raw.loc[empty_text, "text"] = (
        items_raw.loc[empty_text, "Name"].str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # ------------------------------------------------------------------
    # 6. Parse TotalTime → total_minutes; keep calories as numeric
    # ------------------------------------------------------------------
    items_raw["total_minutes"] = items_raw["TotalTime"].apply(_parse_iso_duration)
    items_raw["calories"] = pd.to_numeric(items_raw["Calories"], errors="coerce")

    items_df = items_raw[[
        "RecipeId", "Name", "RecipeCategory",
        "ingredients_list", "keywords_list",
        "calories", "total_minutes", "text",
    ]].rename(columns={
        "RecipeId": "item_id",
        "Name": "name",
        "RecipeCategory": "category",
    }).copy()

    items_df.drop_duplicates(subset=["item_id"], inplace=True)
    items_df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 7. Build filter_info
    # ------------------------------------------------------------------
    filter_info = {
        "config": {
            "min_ratings_per_user": min_ratings_per_user,
            "min_ratings_per_recipe": min_ratings_per_recipe,
            "sample_size": sample_users,
            "random_seed": random_state,
        },
        "counts": {
            "before_filter": before_filter,
            "after_filter": after_filter,
            "after_sample": after_sample,
        },
    }

    return ratings_df, items_df, filter_info


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Food.com dataset and write CSV files."
    )
    parser.add_argument("--raw-dir", default="data/raw/foodcom")
    parser.add_argument("--out-dir", default="data/processed/foodcom")
    parser.add_argument("--min-user", type=int, default=20)
    parser.add_argument("--min-recipe", type=int, default=20)
    parser.add_argument("--sample-users", type=int, default=3_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_df, items_df, filter_info = load_foodcom(
        raw_dir=args.raw_dir,
        min_ratings_per_user=args.min_user,
        min_ratings_per_recipe=args.min_recipe,
        sample_users=args.sample_users,
    )

    ratings_df.to_csv(out_dir / "ratings.csv", index=False)
    print(f"Saved {len(ratings_df):,} ratings → {out_dir / 'ratings.csv'}")

    # Serialise lists as JSON strings for CSV storage
    items_save = items_df.copy()
    items_save["ingredients_list"] = items_save["ingredients_list"].apply(json.dumps)
    items_save["keywords_list"]    = items_save["keywords_list"].apply(json.dumps)
    items_save.to_csv(out_dir / "items.csv", index=False)
    print(f"Saved {len(items_df):,} recipes  → {out_dir / 'items.csv'}")

    info_path = out_dir / "filter_info.json"
    with open(info_path, "w") as f:
        json.dump(filter_info, f, indent=2)
    print(f"Saved filter info → {info_path}")
    print(json.dumps(filter_info, indent=2))


if __name__ == "__main__":
    _main()
