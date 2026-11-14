"""Experiment E2: Adaptive Weight vs. Fixed 50/50 Hybrid Ablation & Sensitivity Analysis.

Tests the project's central novelty claim:
1. Ablation comparison on TEST (refit on train+val):
   - cf_only (alpha = 1.0)
   - content_only (alpha = 0.0)
   - fixed (alpha = 0.5)
   - adaptive (k0 tuned on val: k0=20.0 for MovieLens, k0=100.0 for Food.com)
   Metrics: RMSE, MAE, NDCG@10, P@10, R@10

2. Paired statistical significance:
   Adaptive vs. Fixed on per-user paired differences:
   - Mean per-user difference with 95% paired bootstrap CI over users (seed 42)
   - Paired Wilcoxon signed-rank test and paired sign test p-values
   - Clear plain-language statement on statistical significance

3. Sensitivity analyses (clearly labelled, NOT used to choose hyperparameters):
   - k0 sensitivity: grid (1, 2, 5, 10, 20, 50, 100) on TEST
   - Fixed alpha sensitivity: grid (0.0, 0.25, 0.5, 0.75, 1.0) on TEST

Saves:
  - results/e2_<dataset>.csv (main ablation + sensitivity tables)
  - results/e2_significance_<dataset>.csv (paired significance metrics and tests)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.config import DATASET_CONFIGS
from src.evaluation import evaluate_model_per_user
from src.experiments.common import bootstrap_ci, save_tidy_results
from src.models.hybrid import HybridRecommender
from src.pipeline import build_models, compute_config_hash, load_dataset_data


def paired_sign_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int, int]:
    """Compute paired sign test (two-sided binomial test on non-zero differences).

    Returns:
        (stat, p_value, n_pos, n_neg)
    """
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0, 1.0, 0, 0
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    res = stats.binomtest(k=n_pos, n=n, p=0.5, alternative="two-sided")
    return float(n_pos), float(res.pvalue), n_pos, n_neg


def run_e2_dataset(dataset_name: str, force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute E2 ablation, significance, and sensitivity benchmarks for a single dataset."""
    print("=" * 80)
    print(f"EXPERIMENT E2: {dataset_name.upper()} (Adaptive vs. Fixed Hybrid Ablation)")
    print("=" * 80)

    cfg = DATASET_CONFIGS[dataset_name]
    train_val_df, items_df, _, test_df = load_dataset_data(cfg, fit_on="train_val")
    cfg_hash = compute_config_hash(cfg, fit_on="train_val", fit_df=train_val_df)
    models = build_models(cfg, fit_on="train_val", force_rebuild=force_rebuild)

    cf_model = models["cf"]
    content_model = models["content"]
    val_k0 = cfg.k0

    print(f"Dataset: {dataset_name} | Validation-tuned k0: {val_k0}")
    print(f"Train+Val interactions: {len(train_val_df):,} | Test interactions: {len(test_df):,}")

    # =========================================================================
    # Part 1: Main Ablation Models (cf_only, content_only, fixed 0.5, adaptive val_k0)
    # =========================================================================
    print("\n--- Part 1: Evaluating Main Ablation Variants on TEST ---")
    ablation_variants = [
        ("cf_only", "CF Only (alpha=1.0)", "cf_only", 0.5, val_k0),
        ("content_only", "Content Only (alpha=0.0)", "content_only", 0.0, val_k0),
        ("fixed_0.5", "Fixed Hybrid (alpha=0.5)", "fixed", 0.5, val_k0),
        ("adaptive_tuned", f"Adaptive Hybrid (k0={val_k0:g})", "adaptive", 0.5, val_k0),
    ]

    ablation_metrics: Dict[str, Dict[str, float]] = {}
    ablation_user_details: Dict[str, Dict[str, Any]] = {}

    for var_id, label, mode, fixed_alpha, k0_val in ablation_variants:
        print(f"  Evaluating {label}...")
        hyb = HybridRecommender(cf_model, content_model, mode=mode, k0=k0_val)
        hyb.fit(train_val_df)
        if mode == "fixed":
            hyb.get_alpha = lambda u, a=fixed_alpha: a

        m_res, u_det = evaluate_model_per_user(
            model=hyb,
            train_df=train_val_df,
            test_df=test_df,
            ks=(5, 10),
            relevant_threshold=cfg.liked_threshold,
        )
        ablation_metrics[var_id] = m_res
        ablation_user_details[var_id] = u_det

    # =========================================================================
    # Part 2: Paired Statistical Significance (Adaptive vs Fixed 0.5)
    # =========================================================================
    print("\n--- Part 2: Paired Significance Testing (Adaptive vs. Fixed) ---")
    u_det_ad = ablation_user_details["adaptive_tuned"]
    u_det_fix = ablation_user_details["fixed_0.5"]

    all_users = sorted(u_det_ad["user_ids"])
    eval_users = sorted(set(u_det_ad["evaluated_users"]).intersection(u_det_fix["evaluated_users"]))

    significance_records: List[Dict[str, Any]] = []

    # A. Metric: Squared Error (Error metric, all users; lower is better)
    ad_se = np.array([u_det_ad["per_user_squared_error"][u] for u in all_users])
    fix_se = np.array([u_det_fix["per_user_squared_error"][u] for u in all_users])
    diff_se = ad_se - fix_se  # negative means adaptive has lower error

    mean_diff_se, ci_low_se, ci_high_se = bootstrap_ci(diff_se, n_boot=1000, seed=42)
    # Wilcoxon signed-rank test
    diff_se_nonzero = diff_se[diff_se != 0]
    if len(diff_se_nonzero) > 0:
        w_stat_se, w_pval_se = stats.wilcoxon(diff_se)
    else:
        w_stat_se, w_pval_se = 0.0, 1.0
    _, sign_pval_se, n_pos_se, n_neg_se = paired_sign_test(ad_se, fix_se)

    # Statistical distinction check
    se_sig = "YES" if (w_pval_se < 0.05 and (ci_low_se > 0 or ci_high_se < 0)) else "NO"
    if se_sig == "YES":
        se_plain = "Statistically distinguishable from zero (p < 0.05, 95% CI excludes 0)"
    else:
        se_plain = "Not statistically distinguishable from zero (p >= 0.05 or 95% CI contains 0)"

    significance_records.append({
        "dataset": dataset_name,
        "comparison": "Adaptive vs Fixed (alpha=0.5)",
        "metric": "Mean Squared Error (SE)",
        "n_users": len(all_users),
        "adaptive_mean": float(np.mean(ad_se)),
        "fixed_mean": float(np.mean(fix_se)),
        "mean_difference": float(mean_diff_se),
        "ci_95_lower": float(ci_low_se),
        "ci_95_upper": float(ci_high_se),
        "wilcoxon_stat": float(w_stat_se),
        "wilcoxon_pvalue": float(w_pval_se),
        "sign_test_pvalue": float(sign_pval_se),
        "statistically_significant": se_sig,
        "interpretation": se_plain,
    })

    # B. Metric: Absolute Error (MAE, all users; lower is better)
    ad_ae = np.array([u_det_ad["per_user_absolute_error"][u] for u in all_users])
    fix_ae = np.array([u_det_fix["per_user_absolute_error"][u] for u in all_users])
    diff_ae = ad_ae - fix_ae

    mean_diff_ae, ci_low_ae, ci_high_ae = bootstrap_ci(diff_ae, n_boot=1000, seed=42)
    if len(diff_ae[diff_ae != 0]) > 0:
        w_stat_ae, w_pval_ae = stats.wilcoxon(diff_ae)
    else:
        w_stat_ae, w_pval_ae = 0.0, 1.0
    _, sign_pval_ae, _, _ = paired_sign_test(ad_ae, fix_ae)
    ae_sig = "YES" if (w_pval_ae < 0.05 and (ci_low_ae > 0 or ci_high_ae < 0)) else "NO"
    ae_plain = "Statistically distinguishable from zero" if ae_sig == "YES" else "Not statistically distinguishable from zero"

    significance_records.append({
        "dataset": dataset_name,
        "comparison": "Adaptive vs Fixed (alpha=0.5)",
        "metric": "Mean Absolute Error (AE)",
        "n_users": len(all_users),
        "adaptive_mean": float(np.mean(ad_ae)),
        "fixed_mean": float(np.mean(fix_ae)),
        "mean_difference": float(mean_diff_ae),
        "ci_95_lower": float(ci_low_ae),
        "ci_95_upper": float(ci_high_ae),
        "wilcoxon_stat": float(w_stat_ae),
        "wilcoxon_pvalue": float(w_pval_ae),
        "sign_test_pvalue": float(sign_pval_ae),
        "statistically_significant": ae_sig,
        "interpretation": ae_plain,
    })

    # C. Metric: NDCG@10 (Ranking metric, evaluated users with relevant items; higher is better)
    ad_ndcg = np.array([u_det_ad["per_user_ndcg"][10][u] for u in eval_users])
    fix_ndcg = np.array([u_det_fix["per_user_ndcg"][10][u] for u in eval_users])
    diff_ndcg = ad_ndcg - fix_ndcg  # positive means adaptive ranks better

    mean_diff_ndcg, ci_low_ndcg, ci_high_ndcg = bootstrap_ci(diff_ndcg, n_boot=1000, seed=42)
    diff_ndcg_nonzero = diff_ndcg[diff_ndcg != 0]
    if len(diff_ndcg_nonzero) > 0:
        w_stat_ndcg, w_pval_ndcg = stats.wilcoxon(diff_ndcg)
    else:
        w_stat_ndcg, w_pval_ndcg = 0.0, 1.0
    _, sign_pval_ndcg, _, _ = paired_sign_test(ad_ndcg, fix_ndcg)

    ndcg_sig = "YES" if (w_pval_ndcg < 0.05 and (ci_low_ndcg > 0 or ci_high_ndcg < 0)) else "NO"
    if ndcg_sig == "YES":
        ndcg_plain = "Statistically distinguishable from zero (p < 0.05, 95% CI excludes 0)"
    else:
        ndcg_plain = "Not statistically distinguishable from zero (p >= 0.05 or 95% CI contains 0)"

    significance_records.append({
        "dataset": dataset_name,
        "comparison": "Adaptive vs Fixed (alpha=0.5)",
        "metric": "NDCG@10",
        "n_users": len(eval_users),
        "adaptive_mean": float(np.mean(ad_ndcg)),
        "fixed_mean": float(np.mean(fix_ndcg)),
        "mean_difference": float(mean_diff_ndcg),
        "ci_95_lower": float(ci_low_ndcg),
        "ci_95_upper": float(ci_high_ndcg),
        "wilcoxon_stat": float(w_stat_ndcg),
        "wilcoxon_pvalue": float(w_pval_ndcg),
        "sign_test_pvalue": float(sign_pval_ndcg),
        "statistically_significant": ndcg_sig,
        "interpretation": ndcg_plain,
    })

    # D. Metric: P@10 (Precision@10)
    ad_p10 = np.array([u_det_ad["per_user_precision"][10][u] for u in eval_users])
    fix_p10 = np.array([u_det_fix["per_user_precision"][10][u] for u in eval_users])
    diff_p10 = ad_p10 - fix_p10

    mean_diff_p10, ci_low_p10, ci_high_p10 = bootstrap_ci(diff_p10, n_boot=1000, seed=42)
    if len(diff_p10[diff_p10 != 0]) > 0:
        w_stat_p10, w_pval_p10 = stats.wilcoxon(diff_p10)
    else:
        w_stat_p10, w_pval_p10 = 0.0, 1.0
    _, sign_pval_p10, _, _ = paired_sign_test(ad_p10, fix_p10)
    p10_sig = "YES" if (w_pval_p10 < 0.05 and (ci_low_p10 > 0 or ci_high_p10 < 0)) else "NO"

    significance_records.append({
        "dataset": dataset_name,
        "comparison": "Adaptive vs Fixed (alpha=0.5)",
        "metric": "Precision@10",
        "n_users": len(eval_users),
        "adaptive_mean": float(np.mean(ad_p10)),
        "fixed_mean": float(np.mean(fix_p10)),
        "mean_difference": float(mean_diff_p10),
        "ci_95_lower": float(ci_low_p10),
        "ci_95_upper": float(ci_high_p10),
        "wilcoxon_stat": float(w_stat_p10),
        "wilcoxon_pvalue": float(w_pval_p10),
        "sign_test_pvalue": float(sign_pval_p10),
        "statistically_significant": p10_sig,
        "interpretation": "Statistically distinguishable" if p10_sig == "YES" else "Not statistically distinguishable",
    })

    # E. Metric: R@10 (Recall@10)
    ad_r10 = np.array([u_det_ad["per_user_recall"][10][u] for u in eval_users])
    fix_r10 = np.array([u_det_fix["per_user_recall"][10][u] for u in eval_users])
    diff_r10 = ad_r10 - fix_r10

    mean_diff_r10, ci_low_r10, ci_high_r10 = bootstrap_ci(diff_r10, n_boot=1000, seed=42)
    if len(diff_r10[diff_r10 != 0]) > 0:
        w_stat_r10, w_pval_r10 = stats.wilcoxon(diff_r10)
    else:
        w_stat_r10, w_pval_r10 = 0.0, 1.0
    _, sign_pval_r10, _, _ = paired_sign_test(ad_r10, fix_r10)
    r10_sig = "YES" if (w_pval_r10 < 0.05 and (ci_low_r10 > 0 or ci_high_r10 < 0)) else "NO"

    significance_records.append({
        "dataset": dataset_name,
        "comparison": "Adaptive vs Fixed (alpha=0.5)",
        "metric": "Recall@10",
        "n_users": len(eval_users),
        "adaptive_mean": float(np.mean(ad_r10)),
        "fixed_mean": float(np.mean(fix_r10)),
        "mean_difference": float(mean_diff_r10),
        "ci_95_lower": float(ci_low_r10),
        "ci_95_upper": float(ci_high_r10),
        "wilcoxon_stat": float(w_stat_r10),
        "wilcoxon_pvalue": float(w_pval_r10),
        "sign_test_pvalue": float(sign_pval_r10),
        "statistically_significant": r10_sig,
        "interpretation": "Statistically distinguishable" if r10_sig == "YES" else "Not statistically distinguishable",
    })

    df_significance = pd.DataFrame(significance_records)

    # =========================================================================
    # Part 3: k0 Sensitivity Analysis (TEST grid: 1, 2, 5, 10, 20, 50, 100)
    # =========================================================================
    print("\n--- Part 3: Evaluating k0 Sensitivity Grid on TEST ---")
    k0_grid = [1, 2, 5, 10, 20, 50, 100]
    k0_metrics: Dict[float, Dict[str, float]] = {}

    for k0_cand in k0_grid:
        print(f"  Evaluating adaptive k0={k0_cand}...")
        hyb_k0 = HybridRecommender(cf_model, content_model, mode="adaptive", k0=k0_cand)
        hyb_k0.fit(train_val_df)
        m_res, _ = evaluate_model_per_user(
            model=hyb_k0,
            train_df=train_val_df,
            test_df=test_df,
            ks=(5, 10),
            relevant_threshold=cfg.liked_threshold,
        )
        k0_metrics[k0_cand] = m_res

    # =========================================================================
    # Part 4: Fixed Alpha Sensitivity Analysis (TEST grid: 0.0, 0.25, 0.5, 0.75, 1.0)
    # =========================================================================
    print("\n--- Part 4: Evaluating Fixed Alpha Sensitivity Grid on TEST ---")
    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_metrics: Dict[float, Dict[str, float]] = {}

    for a_val in alpha_grid:
        print(f"  Evaluating fixed alpha={a_val:.2f}...")
        if a_val == 1.0:
            mode = "cf_only"
        elif a_val == 0.0:
            mode = "content_only"
        else:
            mode = "fixed"

        hyb_alpha = HybridRecommender(cf_model, content_model, mode=mode, k0=val_k0)
        hyb_alpha.fit(train_val_df)
        if mode == "fixed":
            hyb_alpha.get_alpha = lambda u, a=a_val: a

        m_res, _ = evaluate_model_per_user(
            model=hyb_alpha,
            train_df=train_val_df,
            test_df=test_df,
            ks=(5, 10),
            relevant_threshold=cfg.liked_threshold,
        )
        alpha_metrics[a_val] = m_res

    # =========================================================================
    # Compile Master Summary Table (Tidy Format)
    # =========================================================================
    summary_rows: List[Dict[str, Any]] = []

    # 1. Main Ablation rows
    for var_id, label, _, _, k0_val in ablation_variants:
        m = ablation_metrics[var_id]
        summary_rows.append({
            "dataset": dataset_name,
            "section": "main_ablation",
            "variant": var_id,
            "description": label,
            "is_reported_result": True,
            "RMSE": round(m["rmse"], 4),
            "MAE": round(m["mae"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "P@10": round(m["precision@10"], 4),
            "R@10": round(m["recall@10"], 4),
        })

    # 2. k0 Sensitivity rows
    for k0_cand in k0_grid:
        m = k0_metrics[k0_cand]
        is_chosen = (k0_cand == val_k0)
        summary_rows.append({
            "dataset": dataset_name,
            "section": "k0_sensitivity (NOT used to choose k0)",
            "variant": f"adaptive_k0_{k0_cand}",
            "description": f"Adaptive Hybrid (k0={k0_cand})" + (" [VALIDATION WINNER]" if is_chosen else ""),
            "is_reported_result": is_chosen,
            "RMSE": round(m["rmse"], 4),
            "MAE": round(m["mae"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "P@10": round(m["precision@10"], 4),
            "R@10": round(m["recall@10"], 4),
        })

    # 3. Fixed Alpha Sensitivity rows
    for a_val in alpha_grid:
        m = alpha_metrics[a_val]
        summary_rows.append({
            "dataset": dataset_name,
            "section": "alpha_sensitivity (NOT used to choose alpha)",
            "variant": f"fixed_alpha_{a_val:.2f}",
            "description": f"Fixed Hybrid (alpha={a_val:.2f})",
            "is_reported_result": (a_val == 0.5),
            "RMSE": round(m["rmse"], 4),
            "MAE": round(m["mae"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "P@10": round(m["precision@10"], 4),
            "R@10": round(m["recall@10"], 4),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Save to results/
    out_e2 = save_tidy_results(
        df=df_summary,
        experiment_name="e2",
        dataset_name=dataset_name,
        config_hash=cfg_hash,
    )
    out_sig = save_tidy_results(
        df=df_significance,
        experiment_name="e2_significance",
        dataset_name=dataset_name,
        config_hash=cfg_hash,
    )

    print(f"\nSaved E2 main results to {out_e2}")
    print(f"Saved E2 significance results to {out_sig}")

    # Display console output
    print("\n" + "=" * 90)
    print(f"E2 MAIN ABLATION TABLE: {dataset_name.upper()}")
    print("=" * 90)
    df_main_disp = df_summary[df_summary["section"] == "main_ablation"][["description", "RMSE", "MAE", "NDCG@10", "P@10", "R@10"]]
    print(df_main_disp.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"E2 PAIRED SIGNIFICANCE TEST (Adaptive vs. Fixed 0.5): {dataset_name.upper()}")
    print("=" * 90)
    df_sig_disp = df_significance[["metric", "mean_difference", "ci_95_lower", "ci_95_upper", "wilcoxon_pvalue", "statistically_significant", "interpretation"]]
    print(df_sig_disp.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"E2 SENSITIVITY ANALYSES: {dataset_name.upper()}")
    print("=" * 90)
    df_sens_disp = df_summary[df_summary["section"] != "main_ablation"][["section", "description", "RMSE", "MAE", "NDCG@10", "P@10", "R@10"]]
    print(df_sens_disp.to_string(index=False))
    print("=" * 90 + "\n")

    return df_summary, df_significance


def run_all(force_rebuild: bool = False) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Execute E2 on MovieLens first, then Food.com."""
    datasets = ["movielens", "foodcom"]
    results = {}
    for ds in datasets:
        results[ds] = run_e2_dataset(ds, force_rebuild=force_rebuild)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E2 (Adaptive vs Fixed Hybrid Ablation & Sensitivity).")
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
        run_e2_dataset(args.dataset, force_rebuild=args.force)


if __name__ == "__main__":
    main()

