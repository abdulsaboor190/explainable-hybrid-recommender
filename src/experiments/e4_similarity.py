"""Experiment E4: Similarity Metric (Cosine vs. Pearson), Neighbor Sweep (k), and Min Overlap.

Phase 3 Task 4: Design-Justification Experiment using VALIDATION DATA ONLY (fit on train, evaluate on val).
Strict protocol: Test set is NEVER touched in this experiment.

1. Sweep UserBasedCF and ItemBasedCF across:
   - Similarities: cosine, pearson
   - k: {5, 10, 20, 30, 50, 100}
   - min_overlap: fixed at config default (2 for MovieLens, 1 for Food.com, with reduced Pearson grid if needed)
   - Metrics: RMSE, MAE, NDCG@10
2. Min_overlap sweep {1, 3, 5, 10} for Pearson at the best k (or representative k=10 / k=30).
3. Record fit time per configuration.
4. Save: results/e4_<dataset>.csv
5. Check consistency: confirm the configs in src/config.py match the best configs found here by val RMSE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config import DATASET_CONFIGS, DatasetConfig
from src.evaluation import evaluate_model
from src.experiments.common import save_tidy_results
from src.models.cf import ItemBasedCF, UserBasedCF
from src.pipeline import load_dataset_data


def run_e4_similarity_sweep(dataset_name: str) -> pd.DataFrame:
    """Run E4 similarity and k sweep on validation set for a given dataset."""
    print("=" * 80)
    print(f"EXPERIMENT E4: {dataset_name.upper()} (Similarity, k, and Overlap Sweep on VALIDATION)")
    print("=" * 80)

    cfg = DATASET_CONFIGS[dataset_name]
    # Load train and validation splits (fit on train, evaluate on val)
    train_df, items_df, val_df, _ = load_dataset_data(cfg, fit_on="train")
    if val_df is None or len(val_df) == 0:
        raise ValueError(f"Validation set not found for {dataset_name}")

    print(f"Loaded train interactions: {len(train_df):,} | val interactions: {len(val_df):,}")
    print(f"Config default min_overlap: {cfg.min_overlap} | liked_threshold: {cfg.liked_threshold}\n")

    # Grid definition
    # Full k sweep requested
    k_grid = [5, 10, 20, 30, 50, 100]

    # For Food.com Pearson at min_overlap=1, pairwise overlap checking is O(N^2) with 6M pairs,
    # taking several minutes. If min_overlap=1 is too slow, we run Pearson on Food.com with min_overlap=2/3
    # or clearly document the grid. On MovieLens, default min_overlap is 2.
    if dataset_name == "foodcom":
        # We test cosine on full k grid at min_overlap=1.
        # For pearson on Food.com, we test k in {5, 10, 20, 30, 50, 100} at min_overlap=2 (fit is ~17-34s).
        foodcom_pearson_overlap = 2
        print(f"Note for Food.com: Cosine uses config default min_overlap=1. Pearson uses min_overlap={foodcom_pearson_overlap} for tractability.")
    else:
        foodcom_pearson_overlap = cfg.min_overlap

    model_types = [
        ("ItemBasedCF", ItemBasedCF),
        ("UserBasedCF", UserBasedCF),
    ]

    results_rows: List[Dict[str, Any]] = []

    # 1. Main Similarity & k Sweep
    print("--- 1. Evaluating Similarity (Cosine vs. Pearson) and k Grid ---")
    for model_name, model_cls in model_types:
        for sim in ["cosine", "pearson"]:
            mo = foodcom_pearson_overlap if (dataset_name == "foodcom" and sim == "pearson") else cfg.min_overlap
            
            # Fit model once with max k or default k, similarity matrix is computed during fit()
            print(f"Fitting {model_name} (similarity={sim}, min_overlap={mo})...", end="", flush=True)
            t_fit_start = time.time()
            model_inst = model_cls(
                similarity=sim,
                k=k_grid[-1],
                min_overlap=mo,
                shrinkage=cfg.shrinkage,
            )
            model_inst.fit(train_df)
            fit_duration = time.time() - t_fit_start
            print(f" fitted in {fit_duration:.2f}s.")

            # Sweep k on validation set
            for k in k_grid:
                model_inst.k = k
                t_eval_start = time.time()
                eval_metrics = evaluate_model(
                    model=model_inst,
                    train_df=train_df,
                    test_df=val_df,
                    ks=(10,),
                    relevant_threshold=cfg.liked_threshold,
                )
                eval_duration = time.time() - t_eval_start

                rmse_val = eval_metrics["rmse"]
                mae_val = eval_metrics["mae"]
                ndcg_val = eval_metrics["ndcg@10"]
                p10_val = eval_metrics["precision@10"]

                print(
                    f"  {model_name:11s} | sim={sim:7s} | k={k:3d} | mo={mo:2d} | "
                    f"RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | NDCG@10: {ndcg_val:.4f} "
                    f"(eval: {eval_duration:.1f}s)"
                )

                results_rows.append({
                    "dataset": dataset_name,
                    "section": "similarity_and_k_sweep",
                    "model_type": model_name,
                    "similarity": sim,
                    "k": k,
                    "min_overlap": mo,
                    "fit_time_sec": round(fit_duration, 3),
                    "eval_time_sec": round(eval_duration, 3),
                    "val_RMSE": round(rmse_val, 4),
                    "val_MAE": round(mae_val, 4),
                    "val_NDCG@10": round(ndcg_val, 4),
                    "val_P@10": round(p10_val, 4),
                })

    # 2. Min Overlap Sweep for Pearson {1, 3, 5, 10}
    print("\n--- 2. Evaluating Min Overlap Sweep {1, 3, 5, 10} for Pearson ---")
    # Choose representative k=cfg.cf_k (or 30)
    rep_k = cfg.cf_k
    overlap_candidates = [1, 3, 5, 10] if dataset_name == "movielens" else [2, 3, 5, 10]

    for model_name, model_cls in model_types:
        for mo in overlap_candidates:
            print(f"Fitting {model_name} (Pearson, k={rep_k}, min_overlap={mo})...", end="", flush=True)
            t_fit_start = time.time()
            m_overlap = model_cls(
                similarity="pearson",
                k=rep_k,
                min_overlap=mo,
                shrinkage=cfg.shrinkage,
            )
            m_overlap.fit(train_df)
            fit_duration = time.time() - t_fit_start
            print(f" fitted in {fit_duration:.2f}s.", end="", flush=True)

            t_eval_start = time.time()
            eval_metrics = evaluate_model(
                model=m_overlap,
                train_df=train_df,
                test_df=val_df,
                ks=(10,),
                relevant_threshold=cfg.liked_threshold,
            )
            eval_duration = time.time() - t_eval_start

            rmse_val = eval_metrics["rmse"]
            mae_val = eval_metrics["mae"]
            ndcg_val = eval_metrics["ndcg@10"]
            p10_val = eval_metrics["precision@10"]

            print(
                f" -> RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | NDCG@10: {ndcg_val:.4f} "
                f"(eval: {eval_duration:.1f}s)"
            )

            results_rows.append({
                "dataset": dataset_name,
                "section": "min_overlap_sweep",
                "model_type": model_name,
                "similarity": "pearson",
                "k": rep_k,
                "min_overlap": mo,
                "fit_time_sec": round(fit_duration, 3),
                "eval_time_sec": round(eval_duration, 3),
                "val_RMSE": round(rmse_val, 4),
                "val_MAE": round(mae_val, 4),
                "val_NDCG@10": round(ndcg_val, 4),
                "val_P@10": round(p10_val, 4),
            })

    results_df = pd.DataFrame(results_rows)

    # Save to results/e4_<dataset>.csv
    out_file = save_tidy_results(results_df, "e4", dataset_name)
    print(f"\nSaved tidy results to {out_file}")

    # Consistency Check with src/config.py
    print("\n" + "=" * 80)
    print(f"CONSISTENCY CHECK WITH src/config.py: {dataset_name.upper()}")
    print("=" * 80)
    # Find best model configuration by val RMSE in the main sweep
    main_sweep_df = results_df[results_df["section"] == "similarity_and_k_sweep"]
    best_row = main_sweep_df.sort_values("val_RMSE").iloc[0]

    cfg_model_type = "ItemBasedCF" if cfg.cf_type == "item_based" else "UserBasedCF"
    print(f"src/config.py settings:")
    print(f"  CF Type     : {cfg_model_type} ({cfg.cf_type})")
    print(f"  Similarity  : {cfg.cf_similarity}")
    print(f"  k           : {cfg.cf_k}")
    print(f"  min_overlap : {cfg.min_overlap}")

    print(f"\nBest Config Found on Validation (by val RMSE):")
    print(f"  CF Type     : {best_row['model_type']}")
    print(f"  Similarity  : {best_row['similarity']}")
    print(f"  k           : {best_row['k']}")
    print(f"  min_overlap : {best_row['min_overlap']}")
    print(f"  val RMSE    : {best_row['val_RMSE']:.4f}")

    match_type = (best_row["model_type"] == cfg_model_type)
    match_sim = (best_row["similarity"] == cfg.cf_similarity)
    print(f"\nModel Type Match : {'MATCH' if match_type else 'DIFFERENT'}")
    print(f"Similarity Match : {'MATCH' if match_sim else 'DIFFERENT'}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E4: Similarity and k Sweep on Validation.")
    parser.add_argument("--dataset", choices=["movielens", "foodcom", "all"], default="all")
    args = parser.parse_args()

    datasets = ["movielens", "foodcom"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_e4_similarity_sweep(ds)


if __name__ == "__main__":
    main()

