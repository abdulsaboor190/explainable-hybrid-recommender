"""CF tuning and benchmarking script across MovieLens and Food.com."""

import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
from src.models.cf import UserBasedCF, ItemBasedCF
from src.evaluation import rmse


def run_movielens_grid():
    print("=" * 80)
    print("Evaluating Collaborative Filtering on MovieLens 100K (Train -> Val)")
    print("=" * 80)

    train_df = pd.read_csv("data/processed/movielens/train.csv")
    val_df = pd.read_csv("data/processed/movielens/val.csv")

    results = []

    # Configs to evaluate
    configs = [
        ("UserBasedCF", "cosine", [10, 30, 50]),
        ("UserBasedCF", "pearson", [10, 30, 50]),
        ("ItemBasedCF", "cosine", [10, 30, 50]),
        ("ItemBasedCF", "pearson", [10, 30, 50]),
    ]

    for model_type, sim, ks in configs:
        # Pre-fit on similarity to avoid recomputing similarity matrix 3 times
        base_cls = UserBasedCF if model_type == "UserBasedCF" else ItemBasedCF
        print(f"Fitting {model_type} ({sim})...")
        t0 = time.time()
        base_model = base_cls(similarity=sim, k=50, min_overlap=5)
        base_model.fit(train_df)
        train_time = time.time() - t0

        for k in ks:
            model = base_cls(similarity=sim, k=k, min_overlap=5)
            # Share fitted matrices
            model.global_mean_ = base_model.global_mean_
            model.user2idx_ = base_model.user2idx_
            model.idx2user_ = base_model.idx2user_
            model.item2idx_ = base_model.item2idx_
            model.idx2item_ = base_model.idx2item_
            model.user_means_ = base_model.user_means_
            model.sim_matrix_ = base_model.sim_matrix_
            model.R_sparse_ = base_model.R_sparse_
            model.user_seen_set_ = base_model.user_seen_set_
            if hasattr(base_model, "R_csc_"):
                model.R_csc_ = base_model.R_csc_
            if hasattr(base_model, "R_centered_"):
                model.R_centered_ = base_model.R_centered_
            if hasattr(base_model, "R_centered_csc_"):
                model.R_centered_csc_ = base_model.R_centered_csc_
            if hasattr(base_model, "item_means_"):
                model.item_means_ = base_model.item_means_

            # Val RMSE
            val_rmse = rmse(model, val_df)

            # Recommend time per user (sample 100 users for stable timing)
            test_users = val_df["user_id"].unique()[:100]
            t_rec_start = time.time()
            for u in test_users:
                _ = model.recommend(u, n=10, exclude_seen=True)
            rec_time_per_user_ms = ((time.time() - t_rec_start) / len(test_users)) * 1000.0

            results.append({
                "Model": model_type,
                "Similarity": sim,
                "k": k,
                "Val_RMSE": val_rmse,
                "Train_Time_s": train_time,
                "Rec_Time_ms_user": rec_time_per_user_ms
            })
            print(f"  k={k:2d} | Val RMSE: {val_rmse:.4f} | Rec Time: {rec_time_per_user_ms:.2f} ms/user")

    df_res = pd.DataFrame(results)
    return df_res


def run_foodcom_best(best_config):
    print("\n" + "=" * 80)
    print("Evaluating Best CF Config on Food.com (Train -> Val)")
    print("=" * 80)

    train_df = pd.read_csv("data/processed/foodcom/train.csv")
    val_df = pd.read_csv("data/processed/foodcom/val.csv")

    model_type = best_config["Model"]
    sim = best_config["Similarity"]
    k = best_config["k"]

    base_cls = UserBasedCF if model_type == "UserBasedCF" else ItemBasedCF
    print(f"Fitting {model_type} ({sim}, k={k}) on Food.com...")
    t0 = time.time()
    model = base_cls(similarity=sim, k=k, min_overlap=5)
    model.fit(train_df)
    train_time = time.time() - t0

    print("Evaluating Val RMSE...")
    val_rmse = rmse(model, val_df)

    print("Benchmarking recommend() latency on 100 users...")
    test_users = val_df["user_id"].unique()[:100]
    t_rec_start = time.time()
    for u in test_users:
        _ = model.recommend(u, n=10, exclude_seen=True)
    rec_time_per_user_ms = ((time.time() - t_rec_start) / len(test_users)) * 1000.0

    fc_res = pd.DataFrame([{
        "Dataset": "Food.com",
        "Model": model_type,
        "Similarity": sim,
        "k": k,
        "Val_RMSE": val_rmse,
        "Train_Time_s": train_time,
        "Rec_Time_ms_user": rec_time_per_user_ms
    }])
    return fc_res


def main():
    ml_results = run_movielens_grid()
    
    print("\n" + "=" * 80)
    print("MOVIELENS 100K CF GRID SEARCH RESULTS")
    print("=" * 80)
    print(ml_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Determine best config by lowest Val RMSE
    best_idx = ml_results["Val_RMSE"].idxmin()
    best_config = ml_results.iloc[best_idx].to_dict()
    print(f"\nBest Config on MovieLens: {best_config['Model']} ({best_config['Similarity']}, k={best_config['k']}) with Val RMSE: {best_config['Val_RMSE']:.4f}")

    fc_result = run_foodcom_best(best_config)
    print("\n" + "=" * 80)
    print("FOOD.COM BEST CF RESULT")
    print("=" * 80)
    print(fc_result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
