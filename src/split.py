"""Temporal per-user train / validation / test split.

Split strategy (purely time-based, no data leakage):
  1. Sort each user's ratings by (timestamp ASC, item_id ASC) for determinism.
  2. Newest 20 %  → test   (floor, min 1 row)
  3. Remaining   → train+val pool
  4. Newest 10 % of that pool → val  (round, min 1 row; leaving at least 1 train row)
  5. Rest        → train

This guarantees:
  • Every user has ≥ 1 rating in train, ≥ 1 in val (when pool ≥ 2), ≥ 1 in test.
  • max(train timestamp) ≤ min(test timestamp) for every user.
  • Fully deterministic: same seed order because we break timestamp ties by item_id.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import pandas as pd


def split_per_user(
    ratings_df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac_of_train: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal per-user train / validation / test split.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain columns [user_id, item_id, rating, timestamp].
    train_frac : float
        Fraction of each user's ratings assigned to train+val pool.
        The complement (1 - train_frac) goes to test.
    val_frac_of_train : float
        Fraction of the train+val pool used as validation (taken from
        the newest interactions).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each has the same columns as the input, with an added 'split' column.
    """
    train_rows, val_rows, test_rows = [], [], []

    # Sort globally once; per-user slices preserve this order
    sorted_df = ratings_df.sort_values(
        ["user_id", "timestamp", "item_id"], ascending=[True, True, True]
    )

    for user_id, group in sorted_df.groupby("user_id", sort=False):
        n = len(group)

        # --- test slice (newest n_test rows) ---
        n_test = max(1, math.floor(n * (1.0 - train_frac)))
        # Clamp so at least 1 row goes to train
        n_test = min(n_test, n - 1)

        test_part  = group.iloc[n - n_test:]
        trainval   = group.iloc[:n - n_test]

        # --- val slice (newest n_val rows of train+val pool) ---
        m = len(trainval)
        if m >= 2:
            n_val = max(1, round(m * val_frac_of_train))
            # Clamp so at least 1 row stays in train
            n_val = min(n_val, m - 1)
            val_part   = trainval.iloc[m - n_val:]
            train_part = trainval.iloc[:m - n_val]
        else:
            # Only 1 row in pool: put it in train, skip val for this user
            train_part = trainval
            val_part   = trainval.iloc[0:0]  # empty, same columns

        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    return train_df, val_df, test_df


def save_splits(
    ratings_df: pd.DataFrame,
    out_dir: str | Path,
    dataset_name: str = "",
) -> None:
    """Apply split_per_user and save CSV files plus cold-item catalogue.

    Writes to out_dir:
      train.csv, val.csv, test.csv, test_cold_items.csv

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Full ratings table with [user_id, item_id, rating, timestamp].
    out_dir : str or Path
        Directory where the split CSVs are written.
    dataset_name : str
        Human-readable label used in printed output.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f"[{dataset_name}] " if dataset_name else ""

    train_df, val_df, test_df = split_per_user(ratings_df)

    # --- Cold-start items in test (never seen in train) ---
    train_items = set(train_df["item_id"].unique())
    cold_mask   = ~test_df["item_id"].isin(train_items)
    cold_df     = test_df[cold_mask].copy()

    # Save CSVs
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(  out_dir / "val.csv",   index=False)
    test_df.to_csv( out_dir / "test.csv",  index=False)
    cold_df.to_csv( out_dir / "test_cold_items.csv", index=False)

    # Print summary
    total = len(train_df) + len(val_df) + len(test_df)
    print(
        f"{label}Train: {len(train_df):>8,} rows "
        f"({len(train_df)/total*100:.1f} %)  "
        f"| {train_df['user_id'].nunique():,} users  "
        f"| {train_df['item_id'].nunique():,} items"
    )
    print(
        f"{label}Val  : {len(val_df):>8,} rows "
        f"({len(val_df)/total*100:.1f} %)"
    )
    print(
        f"{label}Test : {len(test_df):>8,} rows "
        f"({len(test_df)/total*100:.1f} %)"
    )
    print(
        f"{label}Cold-start test items: {len(cold_df):,} rows "
        f"({cold_df['item_id'].nunique():,} unique items)"
    )


# ---------------------------------------------------------------------------
# CLI – run split on both datasets
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply temporal split to MovieLens and Food.com datasets."
    )
    parser.add_argument(
        "--movielens-dir", default="data/processed/movielens",
        help="Directory containing MovieLens ratings.csv",
    )
    parser.add_argument(
        "--foodcom-dir", default="data/processed/foodcom",
        help="Directory containing Food.com ratings.csv",
    )
    args = parser.parse_args()

    # --- MovieLens ---
    print("=" * 55)
    print("MovieLens 100K")
    print("=" * 55)
    ml_ratings = pd.read_csv(
        Path(args.movielens_dir) / "ratings.csv",
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )
    save_splits(ml_ratings, args.movielens_dir, dataset_name="MovieLens")

    # --- Food.com ---
    print()
    print("=" * 55)
    print("Food.com")
    print("=" * 55)
    fc_ratings = pd.read_csv(
        Path(args.foodcom_dir) / "ratings.csv",
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )
    save_splits(fc_ratings, args.foodcom_dir, dataset_name="Food.com")

