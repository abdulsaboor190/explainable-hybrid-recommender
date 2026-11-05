"""Dataset statistics computation and visualisation.

Computes descriptive statistics for any ratings DataFrame and optionally
produces matplotlib figures. Designed to work with both MovieLens 100K
and Food.com processed datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend – safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_stats(
    ratings_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute descriptive statistics for a ratings DataFrame.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Full ratings table with columns [user_id, item_id, rating].
    train_df, val_df, test_df : pd.DataFrame or None
        Pre-split DataFrames. If provided, their row counts are included.

    Returns
    -------
    dict
        Flat dictionary of all statistics, suitable for CSV / JSON export.
    """
    n_users   = int(ratings_df["user_id"].nunique())
    n_items   = int(ratings_df["item_id"].nunique())
    n_ratings = int(len(ratings_df))
    sparsity  = 1.0 - n_ratings / (n_users * n_items)

    # Rating distribution
    r_counts = ratings_df["rating"].value_counts().sort_index()
    rating_dist = {}
    for star in range(1, 6):
        cnt = int(r_counts.get(star, 0))
        pct = cnt / n_ratings * 100
        rating_dist[f"rating_{star}_count"] = cnt
        rating_dist[f"rating_{star}_pct"]   = round(pct, 2)

    # Ratings per user
    rpu = ratings_df.groupby("user_id")["rating"].count()
    # Ratings per item
    rpi = ratings_df.groupby("item_id")["rating"].count()

    stats = {
        # Global
        "n_users":    n_users,
        "n_items":    n_items,
        "n_ratings":  n_ratings,
        "sparsity":   round(sparsity, 6),
        # Rating value
        "mean_rating":   round(float(ratings_df["rating"].mean()), 4),
        "median_rating": float(ratings_df["rating"].median()),
        **rating_dist,
        # Per-user
        "rpu_min":    int(rpu.min()),
        "rpu_median": float(rpu.median()),
        "rpu_mean":   round(float(rpu.mean()), 2),
        "rpu_max":    int(rpu.max()),
        # Per-item
        "rpi_min":    int(rpi.min()),
        "rpi_median": float(rpi.median()),
        "rpi_mean":   round(float(rpi.mean()), 2),
        "rpi_max":    int(rpi.max()),
    }

    # Split counts (optional)
    if train_df is not None:
        stats["train_rows"] = int(len(train_df))
    if val_df is not None:
        stats["val_rows"] = int(len(val_df))
    if test_df is not None:
        stats["test_rows"] = int(len(test_df))

    return stats


def stats_to_csv(stats: dict, path: str | Path) -> None:
    """Write statistics dict as a two-column (metric, value) CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "value": v} for k, v in stats.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_rating_distribution(
    ratings_df: pd.DataFrame,
    dataset_name: str,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Bar chart of rating value distribution (1-5)."""
    r_counts = ratings_df["rating"].value_counts().sort_index()
    stars = [str(s) for s in range(1, 6)]
    counts = [int(r_counts.get(s, 0)) for s in range(1, 6)]
    pcts   = [c / len(ratings_df) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stars, counts, color="#4C72B0", edgecolor="white", linewidth=0.8)

    # Annotate bars with percentage
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    _style_axes(ax, f"{dataset_name} – Rating Distribution",
                "Rating (stars)", "Count")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_ratings_per_user(
    ratings_df: pd.DataFrame,
    dataset_name: str,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Histogram of number of ratings per user (log-scale y-axis)."""
    rpu = ratings_df.groupby("user_id")["rating"].count()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rpu, bins=50, color="#DD8452", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    _style_axes(ax, f"{dataset_name} – Ratings per User",
                "Number of ratings", "Users (log scale)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_ratings_per_item(
    ratings_df: pd.DataFrame,
    dataset_name: str,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Histogram of number of ratings per item (log-scale y-axis)."""
    rpi = ratings_df.groupby("item_id")["rating"].count()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rpi, bins=50, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    _style_axes(ax, f"{dataset_name} – Ratings per Item",
                "Number of ratings", "Items (log scale)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Markdown table
# ---------------------------------------------------------------------------

def build_summary_markdown(
    ml_stats: dict,
    fc_stats: dict,
) -> str:
    """Build a side-by-side Markdown comparison table."""
    rows = [
        ("Metric", "MovieLens 100K", "Food.com (sample)"),
        (":---", ":---:", ":---:"),
        ("Users",           f"{ml_stats['n_users']:,}",   f"{fc_stats['n_users']:,}"),
        ("Items",           f"{ml_stats['n_items']:,}",   f"{fc_stats['n_items']:,}"),
        ("Ratings",         f"{ml_stats['n_ratings']:,}", f"{fc_stats['n_ratings']:,}"),
        ("Sparsity",        f"{ml_stats['sparsity']:.4f}", f"{fc_stats['sparsity']:.4f}"),
        ("Mean rating",     f"{ml_stats['mean_rating']}", f"{fc_stats['mean_rating']}"),
        ("Median rating",   f"{ml_stats['median_rating']}", f"{fc_stats['median_rating']}"),
        ("★1 %",   f"{ml_stats['rating_1_pct']} %", f"{fc_stats['rating_1_pct']} %"),
        ("★2 %",   f"{ml_stats['rating_2_pct']} %", f"{fc_stats['rating_2_pct']} %"),
        ("★3 %",   f"{ml_stats['rating_3_pct']} %", f"{fc_stats['rating_3_pct']} %"),
        ("★4 %",   f"{ml_stats['rating_4_pct']} %", f"{fc_stats['rating_4_pct']} %"),
        ("★5 %",   f"{ml_stats['rating_5_pct']} %", f"{fc_stats['rating_5_pct']} %"),
        ("Ratings/user min",    str(ml_stats["rpu_min"]),    str(fc_stats["rpu_min"])),
        ("Ratings/user median", str(ml_stats["rpu_median"]), str(fc_stats["rpu_median"])),
        ("Ratings/user mean",   str(ml_stats["rpu_mean"]),   str(fc_stats["rpu_mean"])),
        ("Ratings/user max",    str(ml_stats["rpu_max"]),    str(fc_stats["rpu_max"])),
        ("Ratings/item min",    str(ml_stats["rpi_min"]),    str(fc_stats["rpi_min"])),
        ("Ratings/item median", str(ml_stats["rpi_median"]), str(fc_stats["rpi_median"])),
        ("Ratings/item mean",   str(ml_stats["rpi_mean"]),   str(fc_stats["rpi_mean"])),
        ("Ratings/item max",    str(ml_stats["rpi_max"]),    str(fc_stats["rpi_max"])),
        ("Train rows",   f"{ml_stats.get('train_rows','N/A'):,}" if isinstance(ml_stats.get('train_rows'), int) else "N/A",
                         f"{fc_stats.get('train_rows','N/A'):,}" if isinstance(fc_stats.get('train_rows'), int) else "N/A"),
        ("Val rows",     f"{ml_stats.get('val_rows','N/A'):,}" if isinstance(ml_stats.get('val_rows'), int) else "N/A",
                         f"{fc_stats.get('val_rows','N/A'):,}" if isinstance(fc_stats.get('val_rows'), int) else "N/A"),
        ("Test rows",    f"{ml_stats.get('test_rows','N/A'):,}" if isinstance(ml_stats.get('test_rows'), int) else "N/A",
                         f"{fc_stats.get('test_rows','N/A'):,}" if isinstance(fc_stats.get('test_rows'), int) else "N/A"),
    ]

    lines = [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}

    datasets = {
        "movielens": {
            "label": "MovieLens 100K",
            "ratings": "data/processed/movielens/ratings.csv",
            "train":   "data/processed/movielens/train.csv",
            "val":     "data/processed/movielens/val.csv",
            "test":    "data/processed/movielens/test.csv",
        },
        "foodcom": {
            "label": "Food.com",
            "ratings": "data/processed/foodcom/ratings.csv",
            "train":   "data/processed/foodcom/train.csv",
            "val":     "data/processed/foodcom/val.csv",
            "test":    "data/processed/foodcom/test.csv",
        },
    }

    for key, cfg in datasets.items():
        label = cfg["label"]
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        ratings_df = pd.read_csv(cfg["ratings"])
        train_df   = pd.read_csv(cfg["train"])
        val_df     = pd.read_csv(cfg["val"])
        test_df    = pd.read_csv(cfg["test"])

        stats = compute_stats(ratings_df, train_df, val_df, test_df)
        all_stats[key] = stats

        # Print key numbers
        for metric, val in stats.items():
            print(f"  {metric:<25} {val}")

        # Save CSV
        csv_path = results_dir / f"dataset_stats_{key}.csv"
        stats_to_csv(stats, csv_path)
        print(f"\n  → Saved {csv_path}")

        # Save plots
        slug = key
        plot_rating_distribution(
            ratings_df, label,
            figures_dir / f"{slug}_rating_dist.png",
        )
        plot_ratings_per_user(
            ratings_df, label,
            figures_dir / f"{slug}_rpu_hist.png",
        )
        plot_ratings_per_item(
            ratings_df, label,
            figures_dir / f"{slug}_rpi_hist.png",
        )
        print(f"  → Saved 3 figures to {figures_dir}/")

    # Build and save summary Markdown
    md_table = build_summary_markdown(all_stats["movielens"], all_stats["foodcom"])
    md_path = results_dir / "dataset_stats_summary.md"
    md_path.write_text(
        "# Dataset Statistics – Side-by-Side Comparison\n\n"
        + md_table
        + "\n\n> *Food.com counts reflect the 3,000-user activity-filtered sample.*\n",
        encoding="utf-8",
    )
    print(f"\n  → Saved {md_path}")
    print("\n" + "="*55)
    print("SUMMARY TABLE")
    print("="*55)
    print(md_table)

