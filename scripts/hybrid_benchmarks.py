"""Hybrid recommender tuning and evaluation runner on MovieLens and Food.com."""

import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

from src.config import DATASET_CONFIGS
from src.models.cf import ItemBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender
from src.evaluation import evaluate_model


def run_hybrid_eval():
    all_records = []

    for key, cfg in DATASET_CONFIGS.items():
        print("=" * 80)
        print(f"HYBRID EVALUATION ON {cfg.name.upper()} (Validation Split)")
        print("=" * 80)

        train_df = pd.read_csv(cfg.data_dir / "train.csv")
        val_df = pd.read_csv(cfg.data_dir / "val.csv")
        items_df = pd.read_csv(cfg.data_dir / "items.csv")

        # 1. Fit base models once
        print("Fitting ItemBasedCF base model...")
        cf_model = ItemBasedCF(similarity="cosine", k=30, min_overlap=5)
        cf_model.fit(train_df)

        print("Fitting ContentBased base model...")
        content_model = ContentBased(
            items_df=items_df,
            text_column=cfg.text_column,
            liked_threshold=cfg.liked_threshold,
            k_content=30,
        )
        content_model.fit(train_df)

        # 2. Hybrid instance
        hybrid = HybridRecommender(cf_model, content_model, mode="adaptive", k0=10.0)
        hybrid.fit(train_df)

        # 3. Tune k0 on validation split
        print("\nTuning k0 over grid (1, 2, 5, 10, 20, 50, 100)...")
        grid = [1, 2, 5, 10, 20, 50, 100]
        best_k0 = hybrid.tune_k0(val_df, grid=grid, dataset_name=cfg.name, save_results=True)
        print(f"-> Selected best k0 for {cfg.name}: {best_k0}")

        # Load and print the full tuning curve
        df_grid = pd.read_csv(f"results/k0_tuning_{cfg.name}.csv")
        print(f"\n--- {cfg.name} k0 Tuning Grid Results ---")
        print(df_grid.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Alpha distribution summary
        hybrid.k0 = best_k0
        hybrid.mode = "adaptive"
        alpha_dist = hybrid.alpha_distribution()
        print(f"\n--- {cfg.name} Alpha Distribution (k0={best_k0}) ---")
        print(f"  Min:    {alpha_dist['min']:.4f}")
        print(f"  Median: {alpha_dist['median']:.4f}")
        print(f"  Mean:   {alpha_dist['mean']:.4f}")
        print(f"  Max:    {alpha_dist['max']:.4f}")
        print(f"  Share with alpha < 0.5: {alpha_dist['share_below_0_5']*100:.2f}%")

        # 4. Evaluate ablations on validation split
        modes_to_test = [
            ("cf_only", "cf_only", best_k0),
            ("content_only", "content_only", best_k0),
            ("fixed_0.5", "fixed", best_k0),
            (f"adaptive_k0={best_k0}", "adaptive", best_k0),
        ]

        for label, mode, k0_val in modes_to_test:
            hybrid.mode = mode
            hybrid.k0 = k0_val
            t0 = time.time()
            metrics = evaluate_model(
                model=hybrid,
                train_df=train_df,
                test_df=val_df,
                ks=(5, 10),
                relevant_threshold=4.0,
                seen_df=None,
                total_catalogue_items=len(items_df),
                item_vectors=None,
            )
            eval_time = time.time() - t0

            all_records.append({
                "Dataset": cfg.name,
                "Mode": label,
                "Val_RMSE": metrics["rmse"],
                "Val_MAE": metrics["mae"],
                "Val_P@10": metrics["precision@10"],
                "Val_R@10": metrics["recall@10"],
                "Val_NDCG@10": metrics["ndcg@10"],
                "Val_Cov@10": metrics["coverage@10"],
                "Eval_s": eval_time,
            })
            print(f"[{cfg.name}] {label:<20} | RMSE: {metrics['rmse']:.4f} | NDCG@10: {metrics['ndcg@10']:.4f} | Cov@10: {metrics['coverage@10']:.4f}")

    df_summary = pd.DataFrame(all_records)
    print("\n" + "=" * 90)
    print("HYBRID ABLATION RESULTS ON VALIDATION SPLIT")
    print("=" * 90)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_csv = Path("results/hybrid_val_results.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    run_hybrid_eval()

