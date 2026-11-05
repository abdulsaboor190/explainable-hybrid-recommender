"""Content-Based evaluation and comparison runner on MovieLens 100K and Food.com."""

import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

from src.config import DATASET_CONFIGS
from src.models.baselines import GlobalMean, UserMean, ItemMean, Popularity
from src.models.cf import ItemBasedCF
from src.models.content import ContentBased
from src.evaluation import evaluate_model


def run_benchmark():
    records = []

    for key, cfg in DATASET_CONFIGS.items():
        print("=" * 80)
        print(f"Running Validation Evaluation on {cfg.name.upper()}...")
        print("=" * 80)

        train_df = pd.read_csv(cfg.data_dir / "train.csv")
        val_df = pd.read_csv(cfg.data_dir / "val.csv")
        items_df = pd.read_csv(cfg.data_dir / "items.csv")

        # Models to evaluate side-by-side
        models = [
            ("GlobalMean", GlobalMean()),
            ("UserMean", UserMean()),
            ("ItemMean", ItemMean()),
            ("Popularity", Popularity()),
            ("ItemBasedCF", ItemBasedCF(similarity="cosine", k=30, min_overlap=5)),
            ("ContentBased", ContentBased(items_df=items_df, text_column=cfg.text_column, liked_threshold=cfg.liked_threshold, k_content=30)),
        ]

        for name, model in models:
            t0 = time.time()
            model.fit(train_df)
            fit_time = time.time() - t0

            # Evaluate on validation split
            metrics = evaluate_model(
                model=model,
                train_df=train_df,
                test_df=val_df,
                ks=(5, 10),
                relevant_threshold=4.0,
                seen_df=None,  # Seen is train_df
                total_catalogue_items=len(items_df),
                item_vectors=None,
            )

            records.append({
                "Dataset": cfg.name,
                "Model": name,
                "Val_RMSE": metrics["rmse"],
                "Val_MAE": metrics["mae"],
                "Val_P@10": metrics["precision@10"],
                "Val_R@10": metrics["recall@10"],
                "Val_NDCG@10": metrics["ndcg@10"],
                "Val_Cov@10": metrics["coverage@10"],
                "Fit_s": fit_time,
            })
            print(f"[{cfg.name}] {name:<14} | RMSE: {metrics['rmse']:.4f} | NDCG@10: {metrics['ndcg@10']:.4f} | Cov@10: {metrics['coverage@10']:.4f}")

        # Show 2 example users for ContentBased
        cb_model = models[-1][1]
        sample_users = val_df["user_id"].unique()[:2]
        print(f"\n--- Content-Based Sanity Check Examples ({cfg.name}) ---")
        for u in sample_users:
            recs = cb_model.recommend(u, n=5, exclude_seen=True)
            print(f"\nUser {u}:")
            for it, score in recs:
                top_terms = cb_model.get_top_terms(u, it, n=4)
                term_str = ", ".join([f"{t}:{val:.3f}" for t, val in top_terms]) if top_terms else "no common terms"
                # Lookup item title/name
                item_row = items_df[items_df["item_id"] == it]
                name_val = item_row[cfg.title_column].values[0] if len(item_row) > 0 else f"item_{it}"
                print(f"  * [{it}] {name_val[:45]:<45} (Score: {score:.3f}) | Top Terms: {term_str}")

    df_res = pd.DataFrame(records)
    print("\n" + "=" * 90)
    print("COMPARATIVE EVALUATION ON VALIDATION SPLIT")
    print("=" * 90)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save to results/
    df_res.to_csv("results/content_vs_baselines_val.csv", index=False)


if __name__ == "__main__":
    run_benchmark()

