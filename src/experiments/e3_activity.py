"""Experiment E3: Performance by User Activity Level and Cold-Start Simulation.

Phase 3 Task 3:
1. Define activity groups from each user's number of TRAINING ratings (train+val) using
   terciles computed per dataset (few / medium / many). Print group boundaries and user counts.
2. For each group, evaluate on TEST: User Mean, Popularity, Best CF, Content, Fixed Hybrid, Adaptive Hybrid.
   Metrics: RMSE, MAE, NDCG@10, P@10. Report 95% bootstrap CIs per group.
3. Compute and report mean alpha_u of the adaptive hybrid per group.
4. Specifically test the claim "the adaptive blend helps low-activity users most":
   Report adaptive minus fixed difference per group with a paired bootstrap CI (seed 42).
5. Simulated cold-start experiment: For users with deliberately truncated history, keep only
   the first 5 training ratings per user in a copy of train, refit adaptive/fixed/CF/content on
   that truncated data, and evaluate on the same TEST ratings.
6. Save:
   - results/e3_<dataset>.csv (activity terciles performance, CIs, mean alpha, and paired differences)
   - results/e3_coldstart_<dataset>.csv (simulated cold-start 5-rating history evaluation)
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
from src.evaluation import evaluate_model_per_user
from src.experiments.common import bootstrap_ci, save_tidy_results
from src.models.baselines import Popularity, UserMean
from src.models.cf import ItemBasedCF, UserBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender
from src.pipeline import build_models, compute_config_hash, load_dataset_data


def get_activity_terciles(train_val_df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
    """Compute activity terciles (few, medium, many) based on user interaction counts in train_val."""
    user_counts = train_val_df.groupby("user_id").size()
    q_low = float(user_counts.quantile(1.0 / 3.0))
    q_high = float(user_counts.quantile(2.0 / 3.0))

    def assign_group(count: int) -> str:
        if count <= q_low:
            return "few"
        elif count <= q_high:
            return "medium"
        else:
            return "many"

    group_series = user_counts.map(assign_group)
    boundaries = {
        "q_low": q_low,
        "q_high": q_high,
        "min_count": int(user_counts.min()),
        "max_count": int(user_counts.max()),
        "median_count": float(user_counts.median()),
        "few_range": f"[{int(user_counts.min())}, {int(q_low)}]",
        "medium_range": f"({int(q_low)}, {int(q_high)}]",
        "many_range": f"({int(q_high)}, {int(user_counts.max())}]",
        "counts_per_group": group_series.value_counts().to_dict(),
    }
    return group_series, boundaries


def run_e3_activity_analysis(
    dataset_name: str,
    e1_per_user_df: pd.DataFrame,
    group_series: pd.Series,
    boundaries: Dict[str, Any],
    cfg: DatasetConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate performance across activity terciles using per-user test evaluation data."""
    best_cf_display = "Item-CF" if cfg.cf_type == "item_based" else "User-CF"
    target_models = [
        ("User Mean", "User Mean"),
        ("Popularity", "Popularity"),
        (best_cf_display, "Best CF"),
        ("Content", "Content"),
        ("Fixed Hybrid", "Fixed Hybrid"),
        ("Adaptive Hybrid", "Adaptive Hybrid"),
    ]

    # Map groups to per-user DataFrame
    df = e1_per_user_df.copy()
    df["activity_group"] = df["user_id"].map(group_series)

    # Calculate theoretical mean alpha_u per group for adaptive hybrid
    k0 = cfg.k0
    user_counts = df.groupby("user_id").size()  # count rows or lookup
    # More accurately: look up true training counts
    train_counts = group_series.index.map(lambda u: df[df["user_id"] == u]["user_id"].count())
    # Let's map alpha_u from training rating count: n_u / (n_u + k0)
    user_train_counts = {}
    for uid, grp in group_series.items():
        pass

    results_rows: List[Dict[str, Any]] = []

    # Evaluate each group
    for grp in ["few", "medium", "many", "all"]:
        if grp == "all":
            sub_df = df
        else:
            sub_df = df[df["activity_group"] == grp]

        n_users_in_grp = sub_df["user_id"].nunique()

        for model_key, display_name in target_models:
            m_df = sub_df[sub_df["model"] == model_key]
            if len(m_df) == 0:
                continue

            # RMSE from per-user mean_sq_error
            se_vals = m_df["mean_sq_error"].values
            rmse_pt, rmse_ci_low, rmse_ci_high = bootstrap_ci(
                se_vals,
                n_boot=1000,
                seed=42,
                statistic_fn=lambda arr: np.sqrt(np.mean(arr)),
            )

            # MAE from per-user mean_abs_error
            ae_vals = m_df["mean_abs_error"].values
            mae_pt, mae_ci_low, mae_ci_high = bootstrap_ci(
                ae_vals,
                n_boot=1000,
                seed=42,
                statistic_fn=np.mean,
            )

            # NDCG@10 (users with relevant items)
            rel_df = m_df[m_df["has_relevant"] == True]
            if len(rel_df) > 0:
                ndcg_vals = rel_df["NDCG@10"].values
                ndcg_pt, ndcg_ci_low, ndcg_ci_high = bootstrap_ci(
                    ndcg_vals,
                    n_boot=1000,
                    seed=42,
                    statistic_fn=np.mean,
                )
                p10_vals = rel_df["P@10"].values
                p10_pt, p10_ci_low, p10_ci_high = bootstrap_ci(
                    p10_vals,
                    n_boot=1000,
                    seed=42,
                    statistic_fn=np.mean,
                )
            else:
                ndcg_pt, ndcg_ci_low, ndcg_ci_high = 0.0, 0.0, 0.0
                p10_pt, p10_ci_low, p10_ci_high = 0.0, 0.0, 0.0

            results_rows.append({
                "dataset": dataset_name,
                "activity_group": grp,
                "model": display_name,
                "n_users": n_users_in_grp,
                "RMSE": rmse_pt,
                "RMSE_ci_95": f"[{rmse_ci_low:.4f}, {rmse_ci_high:.4f}]",
                "MAE": mae_pt,
                "MAE_ci_95": f"[{mae_ci_low:.4f}, {mae_ci_high:.4f}]",
                "NDCG@10": ndcg_pt,
                "NDCG@10_ci_95": f"[{ndcg_ci_low:.4f}, {ndcg_ci_high:.4f}]",
                "P@10": p10_pt,
                "P@10_ci_95": f"[{p10_ci_low:.4f}, {p10_ci_high:.4f}]",
            })

    # Paired differences: Adaptive minus Fixed per group
    paired_rows: List[Dict[str, Any]] = []
    fixed_df = df[df["model"] == "Fixed Hybrid"].set_index("user_id")
    adapt_df = df[df["model"] == "Adaptive Hybrid"].set_index("user_id")

    common_users = fixed_df.index.intersection(adapt_df.index)
    paired_data = pd.DataFrame({
        "user_id": common_users,
        "activity_group": [group_series.get(u, "unknown") for u in common_users],
        "has_relevant": fixed_df.loc[common_users, "has_relevant"].values,
        "diff_se": adapt_df.loc[common_users, "mean_sq_error"].values - fixed_df.loc[common_users, "mean_sq_error"].values,
        "diff_ae": adapt_df.loc[common_users, "mean_abs_error"].values - fixed_df.loc[common_users, "mean_abs_error"].values,
        "diff_ndcg10": adapt_df.loc[common_users, "NDCG@10"].values - fixed_df.loc[common_users, "NDCG@10"].values,
        "diff_p10": adapt_df.loc[common_users, "P@10"].values - fixed_df.loc[common_users, "P@10"].values,
    })

    for grp in ["few", "medium", "many", "all"]:
        sub_p = paired_data if grp == "all" else paired_data[paired_data["activity_group"] == grp]
        n_u = len(sub_p)

        # SE diff
        d_se = sub_p["diff_se"].values
        mean_d_se, d_se_low, d_se_high = bootstrap_ci(d_se, n_boot=1000, seed=42)

        # AE diff
        d_ae = sub_p["diff_ae"].values
        mean_d_ae, d_ae_low, d_ae_high = bootstrap_ci(d_ae, n_boot=1000, seed=42)

        # NDCG@10 diff (relevant users)
        sub_rel = sub_p[sub_p["has_relevant"] == True]
        d_ndcg = sub_rel["diff_ndcg10"].values
        mean_d_ndcg, d_ndcg_low, d_ndcg_high = bootstrap_ci(d_ndcg, n_boot=1000, seed=42)

        # P@10 diff
        d_p10 = sub_rel["diff_p10"].values
        mean_d_p10, d_p10_low, d_p10_high = bootstrap_ci(d_p10, n_boot=1000, seed=42)

        paired_rows.append({
            "dataset": dataset_name,
            "activity_group": grp,
            "n_users": n_u,
            "diff_SE_mean": mean_d_se,
            "diff_SE_95ci": f"[{d_se_low:.5f}, {d_se_high:.5f}]",
            "diff_AE_mean": mean_d_ae,
            "diff_AE_95ci": f"[{d_ae_low:.5f}, {d_ae_high:.5f}]",
            "diff_NDCG10_mean": mean_d_ndcg,
            "diff_NDCG10_95ci": f"[{d_ndcg_low:.5f}, {d_ndcg_high:.5f}]",
            "diff_P10_mean": mean_d_p10,
            "diff_P10_95ci": f"[{d_p10_low:.5f}, {d_p10_high:.5f}]",
        })

    return pd.DataFrame(results_rows), pd.DataFrame(paired_rows)


def run_simulated_coldstart(
    cfg: DatasetConfig,
    train_val_df: pd.DataFrame,
    items_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Simulated cold-start experiment: keep first 5 ratings per user in train, refit and evaluate."""
    print("\n--- Running Simulated Cold-Start Experiment (5 interactions per user) ---")
    t0 = time.time()

    # Truncate training history to first 5 interactions per user
    trunc_train = (
        train_val_df.groupby("user_id", group_keys=False)
        .apply(lambda g: g.head(5))
        .reset_index(drop=True)
    )
    print(f"Truncated train interactions: {len(trunc_train)} across {trunc_train['user_id'].nunique()} users")

    # Fit models on truncated training data
    # 1. Best CF
    if cfg.cf_type == "item_based":
        cf = ItemBasedCF(similarity=cfg.cf_similarity, k=cfg.cf_k, min_overlap=cfg.min_overlap, shrinkage=cfg.shrinkage)
    else:
        cf = UserBasedCF(similarity=cfg.cf_similarity, k=cfg.cf_k, min_overlap=cfg.min_overlap, shrinkage=cfg.shrinkage)
    cf.fit(trunc_train)

    # 2. Content
    content = ContentBased(
        items_df=items_df,
        text_column=cfg.text_column,
        liked_threshold=cfg.liked_threshold,
        k_content=cfg.k_content,
    )
    content.fit(trunc_train)

    # 3. Fixed Hybrid (alpha=0.5)
    hyb_fixed = HybridRecommender(cf, content, mode="fixed")
    hyb_fixed.fit(trunc_train)

    # 4. Adaptive Hybrid (tuned k0)
    hyb_adaptive = HybridRecommender(cf, content, mode="adaptive", k0=cfg.k0)
    hyb_adaptive.fit(trunc_train)

    # 5. User Mean Baseline
    um = UserMean()
    um.fit(trunc_train)

    # 6. Popularity Baseline
    pop = Popularity()
    pop.fit(trunc_train)

    cold_models = [
        ("User Mean", um),
        ("Popularity", pop),
        (f"CF Only ({cfg.cf_type})", cf),
        ("Content Only", content),
        ("Fixed Hybrid (0.5)", hyb_fixed),
        (f"Adaptive Hybrid (k0={cfg.k0})", hyb_adaptive),
    ]

    cold_rows: List[Dict[str, Any]] = []

    for name, model in cold_models:
        t_m = time.time()
        metrics, u_det = evaluate_model_per_user(
            model=model,
            train_df=trunc_train,
            test_df=test_df,
            ks=(10,),
            relevant_threshold=cfg.liked_threshold,
        )

        # Bootstrap CIs
        se_vals = list(u_det["per_user_squared_error"].values())
        _, r_low, r_high = bootstrap_ci(se_vals, n_boot=1000, seed=42, statistic_fn=lambda arr: np.sqrt(np.mean(arr)))

        eval_users = u_det["evaluated_users"]
        if len(eval_users) > 0:
            ndcg_vals = [u_det["per_user_ndcg"][10][u] for u in eval_users]
            _, n_low, n_high = bootstrap_ci(ndcg_vals, n_boot=1000, seed=42, statistic_fn=np.mean)
            p10_vals = [u_det["per_user_precision"][10][u] for u in eval_users]
            _, p_low, p_high = bootstrap_ci(p10_vals, n_boot=1000, seed=42, statistic_fn=np.mean)
        else:
            n_low, n_high, p_low, p_high = 0.0, 0.0, 0.0, 0.0

        print(f"  {name:25s} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | NDCG@10: {metrics['ndcg@10']:.4f} | P@10: {metrics['precision@10']:.4f} ({time.time()-t_m:.1f}s)")

        cold_rows.append({
            "dataset": cfg.name,
            "model": name,
            "training_history": "simulated_cold_start_5_ratings",
            "RMSE": metrics["rmse"],
            "RMSE_ci_95": f"[{r_low:.4f}, {r_high:.4f}]",
            "MAE": metrics["mae"],
            "NDCG@10": metrics["ndcg@10"],
            "NDCG@10_ci_95": f"[{n_low:.4f}, {n_high:.4f}]",
            "P@10": metrics["precision@10"],
            "P@10_ci_95": f"[{p_low:.4f}, {p_high:.4f}]",
            "evaluated_users": len(eval_users),
        })

    print(f"Cold-start experiment complete in {time.time()-t0:.2f}s.\n")
    return pd.DataFrame(cold_rows)


def run_e3_dataset(dataset_name: str) -> None:
    """Run complete E3 experiment pipeline for the specified dataset."""
    print("=" * 80)
    print(f"EXPERIMENT E3: {dataset_name.upper()} (Activity Terciles & Cold-Start Simulation)")
    print("=" * 80)

    cfg = DATASET_CONFIGS[dataset_name]
    train_val_df, items_df, _, test_df = load_dataset_data(cfg, fit_on="train_val")

    # 1. Activity Terciles
    group_series, boundaries = get_activity_terciles(train_val_df)
    print(f"Activity Tercile Boundaries for {dataset_name}:")
    print(f"  Total Users : {len(group_series)}")
    print(f"  Ratings Range: min={boundaries['min_count']}, median={boundaries['median_count']}, max={boundaries['max_count']}")
    print(f"  FEW Group   : {boundaries['few_range']} ratings -> {boundaries['counts_per_group']['few']} users")
    print(f"  MEDIUM Group: {boundaries['medium_range']} ratings -> {boundaries['counts_per_group']['medium']} users")
    print(f"  MANY Group  : {boundaries['many_range']} ratings -> {boundaries['counts_per_group']['many']} users")

    # 2. Mean alpha_u of Adaptive Hybrid per group
    user_train_counts = train_val_df.groupby("user_id").size()
    alpha_per_user = user_train_counts / (user_train_counts + cfg.k0)
    df_alpha = pd.DataFrame({"group": group_series, "alpha_u": alpha_per_user})
    alpha_means = df_alpha.groupby("group")["alpha_u"].mean().to_dict()
    print("\nAdaptive Hybrid Mean Alpha (CF weight alpha_u) per group:")
    for g in ["few", "medium", "many"]:
        print(f"  {g.upper():6s} group: mean alpha_u = {alpha_means[g]:.4f} (Content weight = {1.0 - alpha_means[g]:.4f})")
    print(f"  OVERALL    : mean alpha_u = {alpha_per_user.mean():.4f}\n")

    # 3. Load per-user evaluation data from E1
    e1_path = Path("results") / f"e1_per_user_{dataset_name}.csv"
    if not e1_path.exists():
        raise FileNotFoundError(f"Missing {e1_path}. Please run E1 first.")
    e1_per_user_df = pd.read_csv(e1_path)

    # 4. Activity breakdown evaluation
    results_df, paired_df = run_e3_activity_analysis(
        dataset_name=dataset_name,
        e1_per_user_df=e1_per_user_df,
        group_series=group_series,
        boundaries=boundaries,
        cfg=cfg,
    )

    # Merge mean alpha and boundaries info into results_df
    results_df["mean_adaptive_alpha"] = results_df["activity_group"].map(
        lambda g: alpha_means.get(g, float(alpha_per_user.mean()))
    )
    results_df["group_rating_range"] = results_df["activity_group"].map(
        lambda g: boundaries.get(f"{g}_range", "all")
    )

    # 5. Simulated cold-start evaluation
    coldstart_df = run_simulated_coldstart(
        cfg=cfg,
        train_val_df=train_val_df,
        items_df=items_df,
        test_df=test_df,
    )

    # 6. Save results
    save_tidy_results(results_df, "e3", dataset_name)
    save_tidy_results(paired_df, "e3_significance", dataset_name)
    save_tidy_results(coldstart_df, "e3_coldstart", dataset_name)

    # Print summary tables
    print("=" * 95)
    print(f"E3 ACTIVITY TERCILE TABLE: {dataset_name.upper()}")
    print("=" * 95)
    cols_show = ["activity_group", "group_rating_range", "model", "n_users", "mean_adaptive_alpha", "RMSE", "MAE", "NDCG@10", "P@10"]
    print(results_df[cols_show].to_string(index=False))

    print("\n" + "=" * 95)
    print(f"E3 PAIRED DIFFERENCE (Adaptive minus Fixed) PER GROUP: {dataset_name.upper()}")
    print("=" * 95)
    p_cols = ["activity_group", "n_users", "diff_SE_mean", "diff_SE_95ci", "diff_AE_mean", "diff_AE_95ci", "diff_NDCG10_mean", "diff_NDCG10_95ci"]
    print(paired_df[p_cols].to_string(index=False))

    print("\n" + "=" * 95)
    print(f"E3 SIMULATED COLD-START (5 Ratings/User) TABLE: {dataset_name.upper()}")
    print("=" * 95)
    cs_cols = ["model", "RMSE", "RMSE_ci_95", "MAE", "NDCG@10", "NDCG@10_ci_95", "P@10"]
    print(coldstart_df[cs_cols].to_string(index=False))
    print("=" * 95 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E3: Activity terciles and simulated cold start.")
    parser.add_argument("--dataset", choices=["movielens", "foodcom", "all"], default="all")
    args = parser.parse_args()

    datasets = ["movielens", "foodcom"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_e3_dataset(ds)


if __name__ == "__main__":
    main()

