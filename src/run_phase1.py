"""Phase 1 end-to-end pipeline runner.

Usage:
    python -m src.run_phase1

Runs all Phase 1 preprocessing steps in sequence:
  1. MovieLens 100K   → data/processed/movielens/{ratings,items}.csv
  2. Food.com         → data/processed/foodcom/{ratings,items,filter_info}.csv/.json
  3. Train/val/test split for both datasets
  4. Dataset statistics + figures + summary markdown

All paths are relative to the repository root (run from there).
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when invoked as `python -m src.run_phase1`
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.data_movielens import load_movielens
from src.data_foodcom import load_foodcom
from src.split import save_splits
from src.stats import compute_stats, stats_to_csv, build_summary_markdown
from src.stats import (
    plot_rating_distribution,
    plot_ratings_per_user,
    plot_ratings_per_item,
)

import json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    t_total = time.time()

    # --- Directories -------------------------------------------------------
    ml_processed  = _ROOT / "data/processed/movielens"
    fc_processed  = _ROOT / "data/processed/foodcom"
    results_dir   = _ROOT / "results"
    figures_dir   = results_dir / "figures"

    for d in [ml_processed, fc_processed, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1 – MovieLens 100K
    # =======================================================================
    _banner("Step 1 · MovieLens 100K preprocessing")
    t0 = time.time()
    ml_ratings, ml_items = load_movielens(raw_dir=_ROOT / "data/raw/ml-100k")
    ml_ratings.to_csv(ml_processed / "ratings.csv", index=False)
    ml_items.to_csv(ml_processed / "items.csv", index=False)
    print(f"  Ratings : {len(ml_ratings):,}  |  Items : {len(ml_items):,}  "
          f"[{_elapsed(t0)}]")

    # =======================================================================
    # Step 2 – Food.com
    # =======================================================================
    _banner("Step 2 · Food.com preprocessing")
    t0 = time.time()
    fc_ratings, fc_items, fc_filter_info = load_foodcom(
        raw_dir=_ROOT / "data/raw/foodcom"
    )
    fc_ratings.to_csv(fc_processed / "ratings.csv", index=False)

    fc_items_save = fc_items.copy()
    fc_items_save["ingredients_list"] = fc_items_save["ingredients_list"].apply(
        json.dumps
    )
    fc_items_save["keywords_list"] = fc_items_save["keywords_list"].apply(json.dumps)
    fc_items_save.to_csv(fc_processed / "items.csv", index=False)

    info_path = fc_processed / "filter_info.json"
    with open(info_path, "w") as f:
        json.dump(fc_filter_info, f, indent=2)
    print(f"  Ratings : {len(fc_ratings):,}  |  Items : {len(fc_items):,}  "
          f"[{_elapsed(t0)}]")

    # =======================================================================
    # Step 3 – Train / val / test splits
    # =======================================================================
    _banner("Step 3 · Train / val / test splits")
    t0 = time.time()

    print("  MovieLens:")
    save_splits(ml_ratings, ml_processed, dataset_name="MovieLens")

    print("  Food.com:")
    save_splits(fc_ratings, fc_processed, dataset_name="Food.com")
    print(f"  [{_elapsed(t0)}]")

    # =======================================================================
    # Step 4 – Dataset statistics
    # =======================================================================
    _banner("Step 4 · Dataset statistics & figures")
    t0 = time.time()

    all_stats: dict = {}
    datasets = {
        "movielens": ("MovieLens 100K", ml_ratings, ml_processed),
        "foodcom":   ("Food.com",        fc_ratings, fc_processed),
    }

    for key, (label, ratings_df, proc_dir) in datasets.items():
        train_df = pd.read_csv(proc_dir / "train.csv")
        val_df   = pd.read_csv(proc_dir / "val.csv")
        test_df  = pd.read_csv(proc_dir / "test.csv")

        stats = compute_stats(ratings_df, train_df, val_df, test_df)
        all_stats[key] = stats
        stats_to_csv(stats, results_dir / f"dataset_stats_{key}.csv")

        plot_rating_distribution(
            ratings_df, label, figures_dir / f"{key}_rating_dist.png"
        )
        plot_ratings_per_user(
            ratings_df, label, figures_dir / f"{key}_rpu_hist.png"
        )
        plot_ratings_per_item(
            ratings_df, label, figures_dir / f"{key}_rpi_hist.png"
        )
        print(f"  {label}: stats saved, 3 figures written")

    # Summary markdown
    md_table = build_summary_markdown(all_stats["movielens"], all_stats["foodcom"])
    md_path = results_dir / "dataset_stats_summary.md"
    md_path.write_text(
        "# Dataset Statistics – Side-by-Side Comparison\n\n"
        + md_table
        + "\n\n> *Food.com counts reflect the 3,000-user activity-filtered sample.*\n",
        encoding="utf-8",
    )
    print(f"  Summary markdown saved  [{_elapsed(t0)}]")

    # =======================================================================
    # Done
    # =======================================================================
    _banner("Phase 1 complete")
    print(f"  Total elapsed: {_elapsed(t_total)}")
    print()
    print("  MovieLens splits  :")
    for split in ("train", "val", "test"):
        n = len(pd.read_csv(ml_processed / f"{split}.csv"))
        print(f"    {split:5s}: {n:>8,}")
    print("  Food.com splits   :")
    for split in ("train", "val", "test"):
        n = len(pd.read_csv(fc_processed / f"{split}.csv"))
        print(f"    {split:5s}: {n:>8,}")


if __name__ == "__main__":
    run()
