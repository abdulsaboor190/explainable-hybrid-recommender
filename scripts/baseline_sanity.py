"""Baseline sanity evaluation runner on MovieLens 100K and Food.com."""

import sys
import time
from pathlib import Path
from typing import Dict, Any

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.baselines import GlobalMean, UserMean, ItemMean, Popularity
from src.evaluation import evaluate_model


def build_movielens_item_vectors(items_path: Path) -> Dict[int, np.ndarray]:
    """Build item feature vectors for MovieLens using genre TF-IDF."""
    items_df = pd.read_csv(items_path)
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # Genre tokens column contains space-separated genre names
    matrix = tfidf.fit_transform(items_df["genre_tokens"].fillna("unknown")).toarray()
    return {item_id: matrix[idx] for idx, item_id in enumerate(items_df["item_id"])}


def build_foodcom_item_vectors(items_path: Path) -> Dict[int, np.ndarray]:
    """Build item feature vectors for Food.com using ingredient/keyword TF-IDF."""
    items_df = pd.read_csv(items_path)
    tfidf = TfidfVectorizer(max_features=1000)
    matrix = tfidf.fit_transform(items_df["text"].fillna("")).toarray()
    return {item_id: matrix[idx] for idx, item_id in enumerate(items_df["item_id"])}


def run_dataset_eval(dataset_name: str, data_dir: Path, vector_builder) -> pd.DataFrame:
    print(f"\n{'='*70}\nRunning Baselines on {dataset_name}...\n{'='*70}")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    items_df = pd.read_csv(data_dir / "items.csv")
    total_items = len(items_df)

    item_vectors = vector_builder(data_dir / "items.csv")

    models = [
        ("GlobalMean", GlobalMean()),
        ("UserMean", UserMean()),
        ("ItemMean", ItemMean()),
        ("Popularity", Popularity()),
    ]

    records = []
    for name, model in models:
        t0 = time.time()
        model.fit(train_df)
        fit_time = time.time() - t0

        t1 = time.time()
        metrics = evaluate_model(
            model=model,
            train_df=train_df,
            test_df=test_df,
            ks=(5, 10),
            relevant_threshold=4.0,
            seen_df=val_df,
            total_catalogue_items=total_items,
            item_vectors=item_vectors,
        )
        eval_time = time.time() - t1

        row = {
            "Dataset": dataset_name,
            "Model": name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "Cold_RMSE": metrics["cold_rmse"],
            "P@5": metrics["precision@5"],
            "R@5": metrics["recall@5"],
            "NDCG@5": metrics["ndcg@5"],
            "P@10": metrics["precision@10"],
            "R@10": metrics["recall@10"],
            "NDCG@10": metrics["ndcg@10"],
            "Cov@10": metrics["coverage@10"],
            "Div@10": metrics["diversity@10"],
            "Fit_s": fit_time,
            "Eval_s": eval_time,
        }
        records.append(row)
        print(f"[{dataset_name}] {name:<12} | RMSE: {row['RMSE']:.4f} | NDCG@10: {row['NDCG@10']:.4f} | Cov@10: {row['Cov@10']:.4f} | Fit: {fit_time:.2f}s | Eval: {eval_time:.2f}s")

    return pd.DataFrame(records)


def main():
    root = Path(__file__).parent.parent
    ml_dir = root / "data" / "processed" / "movielens"
    fc_dir = root / "data" / "processed" / "foodcom"

    ml_res = run_dataset_eval("MovieLens 100K", ml_dir, build_movielens_item_vectors)
    fc_res = run_dataset_eval("Food.com", fc_dir, build_foodcom_item_vectors)

    all_res = pd.concat([ml_res, fc_res], ignore_index=True)
    
    print("\n" + "="*95)
    print("ALL BASELINE RESULTS SUMMARY TABLE")
    print("="*95)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 1000)
    print(all_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Also save to results/
    out_path = root / "results" / "baseline_sanity_results.csv"
    all_res.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
