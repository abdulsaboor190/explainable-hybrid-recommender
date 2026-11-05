"""MovieLens 100K data loader.

Reads raw u.data, u.item, and u.user files, cleans them, and returns
tidy DataFrames ready for the recommender pipeline.
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Genre name → single-token mapping (removes hyphens / spaces)
# ---------------------------------------------------------------------------
_GENRE_RENAME = {
    "Film-Noir": "FilmNoir",
    "Sci-Fi": "SciFi",
    "Children's": "Childrens",
}

# u.item genre columns in order (indices 5-23 after the fixed columns)
_GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation",
    "Childrens", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi",
    "Thriller", "War", "Western",
]


def load_movielens(
    raw_dir: str | Path = "data/raw/ml-100k",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the MovieLens 100K dataset.

    Parameters
    ----------
    raw_dir : str or Path
        Path to the directory containing u.data, u.item, and u.user.

    Returns
    -------
    ratings_df : pd.DataFrame
        Columns: user_id (int), item_id (int), rating (int), timestamp (int).
        Only ratings in [1, 5]. Duplicate (user_id, item_id) pairs are
        resolved by keeping the latest (highest timestamp).
    items_df : pd.DataFrame
        Columns: item_id (int), title (str), genre_tokens (str).
        genre_tokens is a space-separated string of genre names.
        Items with no genre flag set fall back to "unknown".
    """
    raw_dir = Path(raw_dir)

    # ------------------------------------------------------------------
    # 1. Load ratings (u.data)
    # ------------------------------------------------------------------
    ratings_df = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )

    # Keep only valid ratings in [1, 5]
    ratings_df = ratings_df[ratings_df["rating"].between(1, 5)].copy()

    # Drop rows with any missing value
    ratings_df.dropna(inplace=True)

    # Drop duplicates on (user_id, item_id), keeping the latest interaction
    ratings_df.sort_values("timestamp", ascending=False, inplace=True)
    ratings_df.drop_duplicates(subset=["user_id", "item_id"], keep="first", inplace=True)
    ratings_df.sort_values(["user_id", "item_id"], inplace=True)
    ratings_df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 2. Load item metadata (u.item)
    # ------------------------------------------------------------------
    # u.item has 24 columns: item_id | title | release date | video date |
    # IMDb URL | 19 genre flags
    item_col_names = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
    ] + _GENRE_NAMES

    items_raw = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        header=None,
        names=item_col_names,
        encoding="latin-1",
    )

    # Build genre_tokens: join active genre column names
    genre_flag_cols = _GENRE_NAMES  # 19 columns, in order

    def _build_genre_tokens(row: pd.Series) -> str:
        """Return space-joined active genre names; fallback to 'unknown'."""
        active = [g for g in genre_flag_cols if row[g] == 1]
        # If only "unknown" is flagged (or nothing), emit "unknown"
        if not active or active == ["unknown"]:
            return "unknown"
        # Drop the "unknown" token when real genres are present
        active = [g for g in active if g != "unknown"] or ["unknown"]
        return " ".join(active)

    items_raw["genre_tokens"] = items_raw.apply(_build_genre_tokens, axis=1)

    items_df = items_raw[["item_id", "title", "genre_tokens"]].copy()
    items_df.dropna(subset=["item_id", "title"], inplace=True)
    items_df["item_id"] = items_df["item_id"].astype(int)
    items_df.reset_index(drop=True, inplace=True)

    return ratings_df, items_df


# ---------------------------------------------------------------------------
# CLI / main block
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens 100K and write CSV files."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/ml-100k",
        help="Path to the ml-100k raw data directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/movielens",
        help="Output directory for ratings.csv and items.csv.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens 100K from: {args.raw_dir}")
    ratings_df, items_df = load_movielens(raw_dir=args.raw_dir)

    ratings_path = out_dir / "ratings.csv"
    items_path = out_dir / "items.csv"

    ratings_df.to_csv(ratings_path, index=False)
    items_df.to_csv(items_path, index=False)

    print(f"Ratings : {len(ratings_df):,} rows → {ratings_path}")
    print(f"Items   : {len(items_df):,} rows → {items_path}")
    print(f"Users   : {ratings_df['user_id'].nunique():,} unique")
    print(f"Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")


if __name__ == "__main__":
    _main()

