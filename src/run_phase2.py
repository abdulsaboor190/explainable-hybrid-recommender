"""Phase 2 Verification and Sanity Evaluation Runner.

Fits all models on train_val (train + validation splits combined) and evaluates
on the held-out TEST set across both MovieLens 100K and Food.com:
  - Global Mean
  - User Mean
  - Item Mean
  - Popularity
  - User-CF (UserBasedCF)
  - Item-CF (ItemBasedCF)
  - Content (ContentBased)
  - Fixed Hybrid (HybridRecommender, alpha=0.5)
  - Adaptive Hybrid (HybridRecommender, tuned k0)

Metrics: RMSE, MAE, P@5, P@10, R@10, NDCG@10.
Results saved to results/phase2_sanity_<dataset>.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation import evaluate_model
from src.pipeline import build_models, load_dataset_data


MODELS_TO_EVALUATE = [
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


def run_dataset_sanity(dataset_name: str, force_rebuild: bool = False) -> pd.DataFrame:
    """Run sanity evaluation on TEST split for a single dataset fitted on train_val."""
    print("=" * 80)
    print(f"PHASE 2 SANITY EVALUATION: {dataset_name.upper()} (fitted on train_val -> evaluated on TEST)")
    print("=" * 80)

    # 1. Load data
    train_val_df, items_df, _, test_df = load_dataset_data(dataset_name, fit_on="train_val")
    if test_df is None or len(test_df) == 0:
        raise ValueError(f"No test set found for dataset '{dataset_name}'")

    print(f"Loaded {len(train_val_df):,} train_val interactions and {len(test_df):,} test interactions.")

    # 2. Build or load models fitted on train_val
    t_start = time.perf_counter()
    models = build_models(dataset_name, fit_on="train_val", force_rebuild=force_rebuild)
    t_build = time.perf_counter() - t_start
    print(f"Models ready in {t_build:.2f}s.\n")

    # 3. Evaluate each model on test split
    rows: List[Dict[str, Any]] = []

    for key, display_name in MODELS_TO_EVALUATE:
        if key not in models:
            print(f"Warning: {key} not found in models dict. Skipping.")
            continue

        model = models[key]
        t0 = time.perf_counter()
        res = evaluate_model(
            model=model,
            train_df=train_val_df,
            test_df=test_df,
            ks=(5, 10),
            relevant_threshold=4.0,
        )
        elapsed = time.perf_counter() - t0

        row = {
            "Model": display_name,
            "RMSE": round(res["rmse"], 4),
            "MAE": round(res["mae"], 4),
            "P@5": round(res["precision@5"], 4),
            "P@10": round(res["precision@10"], 4),
            "R@10": round(res["recall@10"], 4),
            "NDCG@10": round(res["ndcg@10"], 4),
        }
        rows.append(row)
        print(f"  {display_name:<16} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | NDCG@10: {row['NDCG@10']:.4f} ({elapsed:.1f}s)")

    df_results = pd.DataFrame(rows)

    # 4. Save results to results/phase2_sanity_<dataset>.csv
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"phase2_sanity_{dataset_name}.csv"
    df_results.to_csv(out_path, index=False)

    print("\n" + "-" * 80)
    print(f"SUMMARY TABLE: {dataset_name.upper()}")
    print("-" * 80)
    print(df_results.to_string(index=False))
    print(f"\nSaved sanity table to {out_path}\n")

    return df_results


def run_all(force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    """Run Phase 2 sanity evaluation on all datasets."""
    datasets = ["movielens", "foodcom"]
    results = {}
    for ds in datasets:
        results[ds] = run_dataset_sanity(ds, force_rebuild=force_rebuild)
    # Also integrate exit criteria check
    print("\n" + "=" * 80)
    print("RUNNING EXIT CRITERIA CHECKS (all must PASS)")
    print("=" * 80)
    from scripts.verify_exit_criteria import verify_all_criteria
    criteria_results = verify_all_criteria()
    all_pass = all(v["status"] == "PASS" for v in criteria_results.values())
    print("\n--- Exit Criteria Summary ---")
    for name, outcome in criteria_results.items():
        print(f"  [{outcome['status']}] {name}")
    if all_pass:
        print("\n✓ ALL 8 EXIT CRITERIA PASSED. Phase 2 is complete.")
    else:
        print("\n✗ SOME EXIT CRITERIA FAILED — see details above.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 verification and test sanity evaluation.")
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
        run_dataset_sanity(args.dataset, force_rebuild=args.force)


if __name__ == "__main__":
    main()

