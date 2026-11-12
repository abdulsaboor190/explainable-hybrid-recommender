"""Experiment E1: Comprehensive Baseline and Model Comparison on Test Split.

Evaluates on TEST (fitted on train_val):
  - Global Mean
  - User Mean
  - Item Mean
  - Popularity
  - Best User-CF (UserBasedCF)
  - Best Item-CF (ItemBasedCF)
  - Content (ContentBased)
  - Fixed Hybrid (HybridRecommender, alpha=0.5)
  - Adaptive Hybrid (HybridRecommender, tuned k0)

Metrics:
  - RMSE, MAE
  - P@5, P@10, R@5, R@10, NDCG@5, NDCG@10
  - Coverage@10, Diversity@10
  - 95% Bootstrap Confidence Intervals for NDCG@10 and RMSE
  - Automatic baseline check: beats_global_mean (RMSE) and beats_popularity (NDCG@10)

Saves:
  - results/e1_<dataset>.csv (tidy summary table with CIs and checks)
  - results/e1_per_user_<dataset>.csv (per-user metric values)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config import DATASET_CONFIGS
from src.experiments.common import (
    bootstrap_ci,
    run_final_evaluation,
    save_tidy_results,
)


E1_MODELS = [
    ("global_mean", "Global Mean"),
    ("user_mean", "User Mean"),
    ("item_mean", "Item Mean"),
    ("popularity", "Popularity"),
    ("user_cf", "User-CF"),
    ("item_cf", "Item-CF"),
    ("content", "Content"),
    ("hybrid_fixed", "Fixed Hybrid"),
    ("hybrid_adaptive", "Adaptive Hybrid"),
]


def run_e1_dataset(dataset_name: str, force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute E1 benchmark on TEST split for a single dataset."""
    print("=" * 80)
    print(f"EXPERIMENT E1: {dataset_name.upper()} (Test Set Evaluation, refit on train_val)")
    print("=" * 80)

    cfg = DATASET_CONFIGS[dataset_name]
    t0 = time.perf_counter()

    # 1. Run final evaluation across all models with per-user breakdown
    agg_results, per_user_details, cfg_hash = run_final_evaluation(
        dataset=cfg,
        model_names=[k for k, _ in E1_MODELS],
        ks=(5, 10),
        force_rebuild=force_rebuild,
    )
    t_eval = time.perf_counter() - t0
    print(f"Evaluated {len(agg_results)} models in {t_eval:.2f}s.\n")

    # 2. Extract baseline reference numbers for beats_* checks
    gm_rmse = agg_results["global_mean"]["rmse"]
    pop_ndcg10 = agg_results["popularity"]["ndcg@10"]

    summary_rows: List[Dict[str, Any]] = []
    per_user_rows: List[Dict[str, Any]] = []

    for key, display_name in E1_MODELS:
        if key not in agg_results:
            continue

        res = agg_results[key]
        u_det = per_user_details[key]

        # Bootstrap 95% CIs
        # A. NDCG@10 (resampling evaluated users)
        ndcg_values = [u_det["per_user_ndcg"][10][u] for u in u_det["evaluated_users"]]
        _, ndcg_ci_low, ndcg_ci_high = bootstrap_ci(ndcg_values, n_boot=1000, seed=42)

        # B. RMSE (resampling per-user mean squared error, statistic=sqrt(mean))
        sq_err_values = [u_det["per_user_squared_error"][u] for u in u_det["user_ids"]]
        _, rmse_ci_low, rmse_ci_high = bootstrap_ci(
            sq_err_values,
            n_boot=1000,
            seed=42,
            statistic_fn=lambda arr: np.sqrt(np.mean(arr)),
        )

        # Checks against baselines
        # beats_global_mean: RMSE strictly lower than Global Mean
        # beats_popularity: NDCG@10 strictly higher than Popularity
        is_gm = key == "global_mean"
        is_pop = key == "popularity"

        beats_gm = "BASELINE" if is_gm else ("YES" if res["rmse"] < gm_rmse else "NO")
        beats_pop = "BASELINE" if is_pop else ("YES" if res["ndcg@10"] > pop_ndcg10 else "NO")

        row = {
            "dataset": dataset_name,
            "experiment": "E1",
            "model": display_name,
            "model_key": key,
            "RMSE": round(res["rmse"], 4),
            "RMSE_95CI": f"[{rmse_ci_low:.4f}, {rmse_ci_high:.4f}]",
            "MAE": round(res["mae"], 4),
            "P@5": round(res["precision@5"], 4),
            "P@10": round(res["precision@10"], 4),
            "R@5": round(res["recall@5"], 4),
            "R@10": round(res["recall@10"], 4),
            "NDCG@5": round(res["ndcg@5"], 4),
            "NDCG@10": round(res["ndcg@10"], 4),
            "NDCG@10_95CI": f"[{ndcg_ci_low:.4f}, {ndcg_ci_high:.4f}]",
            "coverage@10": round(res.get("coverage@10", 0.0), 4),
            "diversity@10": round(res.get("diversity@10", 0.0), 4),
            "users_evaluated": int(res["users_evaluated"]),
            "users_skipped": int(res["users_skipped_no_relevant"]),
            "beats_global_mean": beats_gm,
            "beats_popularity": beats_pop,
        }
        summary_rows.append(row)

        # Per-user records
        for u in u_det["user_ids"]:
            has_rel = u in u_det["evaluated_users"]
            per_user_rows.append({
                "dataset": dataset_name,
                "model": display_name,
                "user_id": u,
                "has_relevant": has_rel,
                "mean_sq_error": u_det["per_user_squared_error"].get(u, np.nan),
                "mean_abs_error": u_det["per_user_absolute_error"].get(u, np.nan),
                "P@5": u_det["per_user_precision"][5].get(u, np.nan) if has_rel else np.nan,
                "P@10": u_det["per_user_precision"][10].get(u, np.nan) if has_rel else np.nan,
                "R@5": u_det["per_user_recall"][5].get(u, np.nan) if has_rel else np.nan,
                "R@10": u_det["per_user_recall"][10].get(u, np.nan) if has_rel else np.nan,
                "NDCG@5": u_det["per_user_ndcg"][5].get(u, np.nan) if has_rel else np.nan,
                "NDCG@10": u_det["per_user_ndcg"][10].get(u, np.nan) if has_rel else np.nan,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_per_user = pd.DataFrame(per_user_rows)

    # 3. Save files
    out_summary_path = save_tidy_results(
        df=df_summary,
        experiment_name="e1",
        dataset_name=dataset_name,
        config_hash=cfg_hash,
    )
    out_per_user_path = Path("results") / f"e1_per_user_{dataset_name}.csv"
    df_per_user.to_csv(out_per_user_path, index=False)

    print(f"Saved E1 summary to {out_summary_path}")
    print(f"Saved per-user metrics to {out_per_user_path}")

    # Display clean table
    display_cols = [
        "model", "RMSE", "RMSE_95CI", "MAE", "P@10", "R@10", "NDCG@10",
        "NDCG@10_95CI", "coverage@10", "diversity@10", "beats_global_mean", "beats_popularity"
    ]
    print("\n" + "-" * 110)
    print(f"E1 RESULTS SUMMARY: {dataset_name.upper()}")
    print("-" * 110)
    print(df_summary[display_cols].to_string(index=False))
    print("-" * 110 + "\n")

    return df_summary, df_per_user


def run_all(force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    """Execute E1 across MovieLens first, then Food.com."""
    datasets = ["movielens", "foodcom"]
    summaries = {}
    for ds in datasets:
        df_sum, _ = run_e1_dataset(ds, force_rebuild=force_rebuild)
        summaries[ds] = df_sum
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E1 (Model & Baseline Comparison).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "movielens", "foodcom"],
        help="Dataset to evaluate ('movielens', 'foodcom', or 'all').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of model cache.",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        run_all(force_rebuild=args.force)
    else:
        run_e1_dataset(args.dataset, force_rebuild=args.force)


if __name__ == "__main__":
    main()
