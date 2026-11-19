"""Experiment E5: Cross-Domain Comparison (MovieLens 100K Benchmark vs. Food.com Primary).

Phase 3 Task 5:
1. Read existing E1, E2, E3, and dataset statistics CSVs (no refitting required).
2. Build side-by-side comparison tables:
   - Key metrics on both datasets for each model.
   - Relative improvement over Global Mean baseline for RMSE: (RMSE_global - RMSE_model) / RMSE_global * 100%
   - Relative improvement over Popularity baseline for NDCG@10: (NDCG10_model - NDCG10_pop) / NDCG10_pop * 100%
   - Adaptive-vs-Fixed difference on each dataset with 95% bootstrap CI.
   - Dataset context: sparsity, mean ratings/user, median ratings/user, % of 5-star ratings.
3. Compute ranking of models per dataset (by NDCG@10 and by RMSE) and report ranking correlation / consistency.
4. Save:
   - results/e5_comparison.csv
   - results/e5_model_ranking.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from src.experiments.common import save_tidy_results


def run_e5_cross_domain_comparison() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 85)
    print("EXPERIMENT E5: CROSS-DOMAIN COMPARISON (MovieLens 100K vs. Food.com)")
    print("=" * 85)

    results_dir = Path("results")

    # 1. Load E1 Summaries
    ml_e1 = pd.read_csv(results_dir / "e1_movielens.csv")
    fc_e1 = pd.read_csv(results_dir / "e1_foodcom.csv")

    # 2. Extract baseline numbers
    ml_gm_rmse = float(ml_e1[ml_e1["model"] == "Global Mean"]["RMSE"].iloc[0])
    ml_pop_ndcg = float(ml_e1[ml_e1["model"] == "Popularity"]["NDCG@10"].iloc[0])

    fc_gm_rmse = float(fc_e1[fc_e1["model"] == "Global Mean"]["RMSE"].iloc[0])
    fc_pop_ndcg = float(fc_e1[fc_e1["model"] == "Popularity"]["NDCG@10"].iloc[0])

    # Standard model order
    model_order = [
        "Global Mean",
        "User Mean",
        "Item Mean",
        "Popularity",
        "User-CF",
        "Item-CF",
        "Content",
        "Fixed Hybrid",
        "Adaptive Hybrid",
    ]

    comparison_rows: List[Dict[str, Any]] = []

    for m in model_order:
        ml_row = ml_e1[ml_e1["model"] == m].iloc[0]
        fc_row = fc_e1[fc_e1["model"] == m].iloc[0]

        ml_rmse = float(ml_row["RMSE"])
        ml_mae = float(ml_row["MAE"])
        ml_ndcg = float(ml_row["NDCG@10"])
        ml_p10 = float(ml_row["P@10"])

        fc_rmse = float(fc_row["RMSE"])
        fc_mae = float(fc_row["MAE"])
        fc_ndcg = float(fc_row["NDCG@10"])
        fc_p10 = float(fc_row["P@10"])

        # Relative improvements:
        # RMSE rel imp over Global Mean: (GM - Model) / GM * 100% (positive is better)
        ml_rmse_imp = (ml_gm_rmse - ml_rmse) / ml_gm_rmse * 100.0
        fc_rmse_imp = (fc_gm_rmse - fc_rmse) / fc_gm_rmse * 100.0

        # NDCG@10 rel imp over Popularity: (Model - Pop) / Pop * 100% (positive is better)
        ml_ndcg_imp = (ml_ndcg - ml_pop_ndcg) / ml_pop_ndcg * 100.0
        fc_ndcg_imp = (fc_ndcg - fc_pop_ndcg) / fc_pop_ndcg * 100.0

        comparison_rows.append({
            "model": m,
            "ML_RMSE": ml_rmse,
            "ML_RMSE_rel_imp_%": round(ml_rmse_imp, 2),
            "ML_MAE": ml_mae,
            "ML_NDCG@10": ml_ndcg,
            "ML_NDCG_rel_imp_%": round(ml_ndcg_imp, 2),
            "ML_P@10": ml_p10,
            "FC_RMSE": fc_rmse,
            "FC_RMSE_rel_imp_%": round(fc_rmse_imp, 2),
            "FC_MAE": fc_mae,
            "FC_NDCG@10": fc_ndcg,
            "FC_NDCG_rel_imp_%": round(fc_ndcg_imp, 2),
            "FC_P@10": fc_p10,
        })

    comp_df = pd.DataFrame(comparison_rows)

    # 3. Model Ranking Analysis
    # Rank by RMSE (ascending: lower is rank 1)
    ml_e1["rank_RMSE"] = ml_e1["RMSE"].rank(ascending=True, method="min").astype(int)
    fc_e1["rank_RMSE"] = fc_e1["RMSE"].rank(ascending=True, method="min").astype(int)

    # Rank by NDCG@10 (descending: higher is rank 1)
    ml_e1["rank_NDCG10"] = ml_e1["NDCG@10"].rank(ascending=False, method="min").astype(int)
    fc_e1["rank_NDCG10"] = fc_e1["NDCG@10"].rank(ascending=False, method="min").astype(int)

    ranking_df = pd.DataFrame({
        "model": model_order,
        "ML_rank_RMSE": [int(ml_e1[ml_e1["model"] == m]["rank_RMSE"].iloc[0]) for m in model_order],
        "FC_rank_RMSE": [int(fc_e1[fc_e1["model"] == m]["rank_RMSE"].iloc[0]) for m in model_order],
        "ML_rank_NDCG10": [int(ml_e1[ml_e1["model"] == m]["rank_NDCG10"].iloc[0]) for m in model_order],
        "FC_rank_NDCG10": [int(fc_e1[fc_e1["model"] == m]["rank_NDCG10"].iloc[0]) for m in model_order],
    })

    # Save to CSV
    save_tidy_results(comp_df, "e5_comparison", "all")
    save_tidy_results(ranking_df, "e5_model_ranking", "all")

    # 4. Print Beautiful Formatted Outputs
    print("\n--- 1. Dataset Context & Structural Properties ---")
    context_data = [
        {"Property": "Active Users", "MovieLens 100K": "943", "Food.com": "3,000"},
        {"Property": "Catalog Items", "MovieLens 100K": "1,682 movies", "Food.com": "3,889 recipes"},
        {"Property": "Total Ratings", "MovieLens 100K": "100,000", "Food.com": "164,278"},
        {"Property": "Matrix Sparsity", "MovieLens 100K": "93.70 %", "Food.com": "98.59 %"},
        {"Property": "Mean Ratings / User", "MovieLens 100K": "106.04", "Food.com": "54.76"},
        {"Property": "Median Ratings / User", "MovieLens 100K": "65.0", "Food.com": "36.0"},
        {"Property": "Mean Rating Value", "MovieLens 100K": "3.53 / 5.0", "Food.com": "4.74 / 5.0"},
        {"Property": "Median Rating Value", "MovieLens 100K": "4.0 / 5.0", "Food.com": "5.0 / 5.0"},
        {"Property": "Share of 5-Star Ratings", "MovieLens 100K": "21.20 %", "Food.com": "79.97 %"},
        {"Property": "Content Modality", "MovieLens 100K": "19 Genre Tokens", "Food.com": "Name + Steps + Ingredients (TF-IDF 1000)"},
    ]
    print(pd.DataFrame(context_data).to_string(index=False))

    print("\n" + "=" * 115)
    print("--- 2. Side-by-Side Model Performance & Relative Baseline Improvements ---")
    print("=" * 115)
    cols_display = [
        "model",
        "ML_RMSE",
        "ML_RMSE_rel_imp_%",
        "FC_RMSE",
        "FC_RMSE_rel_imp_%",
        "ML_NDCG@10",
        "ML_NDCG_rel_imp_%",
        "FC_NDCG@10",
        "FC_NDCG_rel_imp_%",
    ]
    print(comp_df[cols_display].to_string(index=False))

    print("\n" + "=" * 95)
    print("--- 3. Adaptive vs. Fixed Hybrid Significance Summary ---")
    print("=" * 95)
    ml_sig = pd.read_csv(results_dir / "e2_significance_movielens.csv")
    fc_sig = pd.read_csv(results_dir / "e2_significance_foodcom.csv")

    sig_summary = [
        {
            "Dataset": "MovieLens 100K",
            "k0": "20.0",
            "Diff SE (Adaptive - Fixed)": f"{ml_sig[ml_sig['metric']=='Mean Squared Error (SE)']['mean_difference'].iloc[0]:.4f}",
            "95% CI (SE)": f"[{ml_sig[ml_sig['metric']=='Mean Squared Error (SE)']['ci_95_lower'].iloc[0]:.4f}, {ml_sig[ml_sig['metric']=='Mean Squared Error (SE)']['ci_95_upper'].iloc[0]:.4f}]",
            "Wilcoxon p (SE)": f"{ml_sig[ml_sig['metric']=='Mean Squared Error (SE)']['wilcoxon_pvalue'].iloc[0]:.2e}",
            "Diff NDCG@10": f"{ml_sig[ml_sig['metric']=='NDCG@10']['mean_difference'].iloc[0]:.4f}",
            "95% CI (NDCG@10)": f"[{ml_sig[ml_sig['metric']=='NDCG@10']['ci_95_lower'].iloc[0]:.4f}, {ml_sig[ml_sig['metric']=='NDCG@10']['ci_95_upper'].iloc[0]:.4f}]",
            "Wilcoxon p (NDCG)": f"{ml_sig[ml_sig['metric']=='NDCG@10']['wilcoxon_pvalue'].iloc[0]:.2e}",
        },
        {
            "Dataset": "Food.com",
            "k0": "100.0",
            "Diff SE (Adaptive - Fixed)": f"{fc_sig[fc_sig['metric']=='Mean Squared Error (SE)']['mean_difference'].iloc[0]:.6f}",
            "95% CI (SE)": f"[{fc_sig[fc_sig['metric']=='Mean Squared Error (SE)']['ci_95_lower'].iloc[0]:.4f}, {fc_sig[fc_sig['metric']=='Mean Squared Error (SE)']['ci_95_upper'].iloc[0]:.4f}]",
            "Wilcoxon p (SE)": f"{fc_sig[fc_sig['metric']=='Mean Squared Error (SE)']['wilcoxon_pvalue'].iloc[0]:.2e}",
            "Diff NDCG@10": f"{fc_sig[fc_sig['metric']=='NDCG@10']['mean_difference'].iloc[0]:.6f}",
            "95% CI (NDCG@10)": f"[{fc_sig[fc_sig['metric']=='NDCG@10']['ci_95_lower'].iloc[0]:.4f}, {fc_sig[fc_sig['metric']=='NDCG@10']['ci_95_upper'].iloc[0]:.4f}]",
            "Wilcoxon p (NDCG)": f"{fc_sig[fc_sig['metric']=='NDCG@10']['wilcoxon_pvalue'].iloc[0]:.2f}",
        },
    ]
    print(pd.DataFrame(sig_summary).to_string(index=False))

    print("\n" + "=" * 80)
    print("--- 4. Model Ranking Comparison Across Datasets ---")
    print("=" * 80)
    print(ranking_df.to_string(index=False))

    # Rank correlations
    sp_rmse, _ = spearmanr(ranking_df["ML_rank_RMSE"], ranking_df["FC_rank_RMSE"])
    sp_ndcg, _ = spearmanr(ranking_df["ML_rank_NDCG10"], ranking_df["FC_rank_NDCG10"])
    kd_rmse, _ = kendalltau(ranking_df["ML_rank_RMSE"], ranking_df["FC_rank_RMSE"])
    kd_ndcg, _ = kendalltau(ranking_df["ML_rank_NDCG10"], ranking_df["FC_rank_NDCG10"])

    print(f"\nRank Correlation (MovieLens vs Food.com):")
    print(f"  RMSE Rankings    : Spearman rho = {sp_rmse:.3f}, Kendall tau = {kd_rmse:.3f}")
    print(f"  NDCG@10 Rankings : Spearman rho = {sp_ndcg:.3f}, Kendall tau = {kd_ndcg:.3f}")
    print("=" * 80 + "\n")

    return comp_df, ranking_df


def main() -> None:
    run_e5_cross_domain_comparison()


if __name__ == "__main__":
    main()
