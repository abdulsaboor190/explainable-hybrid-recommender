"""Systematic timing benchmarks for cold build, cached load, and recommend + explain latency."""

import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.pipeline import build_models, load_dataset_data


def benchmark():
    random.seed(42)
    datasets = ["movielens", "foodcom"]
    results = []

    print("=" * 80)
    print("RUNNING TASK 7 PERFORMANCE AND LATENCY BENCHMARKS")
    print("=" * 80)

    for ds in datasets:
        print(f"\n--- Benchmarking {ds.upper()} ---")

        # 1. Cold build (fit_on="train")
        t0 = time.perf_counter()
        _ = build_models(dataset=ds, fit_on="train", force_rebuild=True)
        t_cold_train = time.perf_counter() - t0
        print(f"  Cold Build (train)     : {t_cold_train:.3f} s")

        # 2. Cached load (fit_on="train")
        t0 = time.perf_counter()
        models = build_models(dataset=ds, fit_on="train", force_rebuild=False)
        t_cached_train = time.perf_counter() - t0
        print(f"  Cached Load (train)    : {t_cached_train:.3f} s")

        # 3. Cold build (fit_on="train_val")
        t0 = time.perf_counter()
        _ = build_models(dataset=ds, fit_on="train_val", force_rebuild=True)
        t_cold_train_val = time.perf_counter() - t0
        print(f"  Cold Build (train_val) : {t_cold_train_val:.3f} s")

        # 4. Cached load (fit_on="train_val")
        t0 = time.perf_counter()
        _ = build_models(dataset=ds, fit_on="train_val", force_rebuild=False)
        t_cached_train_val = time.perf_counter() - t0
        print(f"  Cached Load (train_val): {t_cached_train_val:.3f} s")

        # 5. Latency for 100 random users on adaptive hybrid
        hyb = models["hybrid_adaptive"]
        train_df, _, _, _ = load_dataset_data(ds, fit_on="train")
        unique_users = train_df["user_id"].unique().tolist()
        sample_users = random.sample(unique_users, min(100, len(unique_users)))

        # Time recommend(n=10) + explain(top-1)
        t0 = time.perf_counter()
        for u in sample_users:
            recs = hyb.recommend(u, n=10)
            if recs:
                _ = hyb.explain(u, recs[0][0])
        t_rec_top1 = time.perf_counter() - t0
        per_user_top1_ms = (t_rec_top1 / len(sample_users)) * 1000.0
        print(f"  100 Users (rec + explain top-1) : {t_rec_top1:.3f} s total ({per_user_top1_ms:.2f} ms/user)")

        # Time recommend(n=10) + explain(all 10)
        t0 = time.perf_counter()
        for u in sample_users:
            recs = hyb.recommend(u, n=10)
            for it, _ in recs:
                _ = hyb.explain(u, it)
        t_rec_all10 = time.perf_counter() - t0
        per_user_all10_ms = (t_rec_all10 / len(sample_users)) * 1000.0
        print(f"  100 Users (rec + explain all 10): {t_rec_all10:.3f} s total ({per_user_all10_ms:.2f} ms/user)")

        results.append({
            "Dataset": "MovieLens 100K" if ds == "movielens" else "Food.com",
            "Cold Build (train)": f"{t_cold_train:.3f}s",
            "Cached Load (train)": f"{t_cached_train:.3f}s",
            "Cold Build (train_val)": f"{t_cold_train_val:.3f}s",
            "Cached Load (train_val)": f"{t_cached_train_val:.3f}s",
            "Rec+Explain (top-1)": f"{per_user_top1_ms:.2f} ms/user",
            "Rec+Explain (all 10)": f"{per_user_all10_ms:.2f} ms/user",
            "Target (<1000 ms)": "PASSED" if per_user_all10_ms < 1000.0 else "FAILED",
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("TIMING BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(df_res.to_string(index=False))

    # Also save to results/
    out_csv = Path("results/pipeline_benchmarks.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")


if __name__ == "__main__":
    benchmark()
