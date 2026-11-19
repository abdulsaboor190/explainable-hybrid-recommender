"""Plotting routines for Phase 3 experiments (E1 to E5).

Strict constraint: Reads ONLY from results/*.csv. No models are fitted or evaluated.
Saves high-resolution (dpi=200) figures to results/figures/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure writable matplotlib config directory in sandbox
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
})

# Consistent model color palette across all figures
MODEL_COLORS: Dict[str, str] = {
    "Global Mean": "#8c8c8c",        # Grey
    "User Mean": "#7f7f7f",          # Darker grey
    "Item Mean": "#a6a6a6",          # Lighter grey
    "Popularity": "#e377c2",         # Pink
    "User-CF": "#1f77b4",           # Blue
    "Item-CF": "#17becf",           # Cyan
    "Best CF": "#17becf",           # Cyan
    "CF Only": "#1f77b4",           # Blue
    "Content": "#2ca02c",           # Green
    "Content Only": "#2ca02c",      # Green
    "Fixed Hybrid": "#ff7f0e",      # Orange
    "Fixed Hybrid (0.5)": "#ff7f0e",# Orange
    "Adaptive Hybrid": "#d62728",   # Red / Crimson
}

FIGURES_DIR = Path("results/figures")


def _ensure_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _parse_ci(ci_str: str) -> Tuple[float, float]:
    """Parse string representation of 95% CI '[low, high]' to floats."""
    if not isinstance(ci_str, str):
        return 0.0, 0.0
    clean = ci_str.strip("[]() ")
    parts = clean.split(",")
    if len(parts) == 2:
        return float(parts[0].strip()), float(parts[1].strip())
    return 0.0, 0.0


def plot_e1_main(dataset: str) -> Path:
    """E1: Grouped bar chart of NDCG@10 and RMSE per model with CI error bars."""
    _ensure_dir()
    csv_path = Path("results") / f"e1_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    models = df["model"].tolist()

    rmse_vals = df["RMSE"].values
    rmse_err_low = []
    rmse_err_high = []
    for val, ci_str in zip(rmse_vals, df["RMSE_95CI"]):
        low, high = _parse_ci(ci_str)
        rmse_err_low.append(max(0.0, val - low))
        rmse_err_high.append(max(0.0, high - val))

    ndcg_vals = df["NDCG@10"].values
    ndcg_err_low = []
    ndcg_err_high = []
    for val, ci_str in zip(ndcg_vals, df["NDCG@10_95CI"]):
        low, high = _parse_ci(ci_str)
        ndcg_err_low.append(max(0.0, val - low))
        ndcg_err_high.append(max(0.0, high - val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#4c72b0") for m in models]

    # Panel 1: RMSE (Lower is better)
    bars1 = ax1.bar(x, rmse_vals, yerr=[rmse_err_low, rmse_err_high], capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title(f"Rating Prediction Error (RMSE)\n[Lower is better]", fontweight="bold")
    ax1.set_ylabel("RMSE on Test Set")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=40, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval * 0.5, f"{yval:.3f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    # Panel 2: NDCG@10 (Higher is better)
    bars2 = ax2.bar(x, ndcg_vals, yerr=[ndcg_err_low, ndcg_err_high], capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title(f"Top-10 Ranking Performance (NDCG@10)\n[Higher is better]", fontweight="bold")
    ax2.set_ylabel("NDCG@10 on Test Set")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=40, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=8)

    ds_title = "MovieLens 100K Benchmark" if dataset == "movielens" else "Food.com Recipes Primary Domain"
    fig.suptitle(f"Experiment E1: Test Split Model Comparison — {ds_title}", fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = FIGURES_DIR / f"e1_main_{dataset}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_e2_ablation_and_sensitivity(dataset: str) -> Path:
    """E2: Bar chart of ablation variants + Line plots of alpha sensitivity and k0 sensitivity."""
    _ensure_dir()
    csv_path = Path("results") / f"e2_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Main ablation variants
    main_df = df[df["section"] == "main_ablation"].copy()

    # 2. Fixed alpha sensitivity
    alpha_df = df[df["section"].str.startswith("alpha_sensitivity")].copy()
    alpha_vals = [float(d.split("alpha=")[1].replace(")", "")) for d in alpha_df["description"]]
    alpha_df["alpha"] = alpha_vals
    alpha_df = alpha_df.sort_values("alpha")

    # 3. k0 sensitivity
    k0_df = df[df["section"].str.startswith("k0_sensitivity")].copy()
    k0_vals = [float(d.split("k0=")[1].split(" ")[0].replace(")", "")) for d in k0_df["description"]]
    k0_df["k0"] = k0_vals
    k0_df = k0_df.sort_values("k0")

    # Tuned adaptive reference
    adaptive_row = main_df[main_df["variant"] == "adaptive_tuned"].iloc[0]
    ad_rmse = adaptive_row["RMSE"]
    ad_ndcg = adaptive_row["NDCG@10"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: Ablation Bar Chart (RMSE & NDCG@10)
    variants = ["CF Only", "Content Only", "Fixed (0.5)", "Adaptive"]
    var_colors = [MODEL_COLORS["CF Only"], MODEL_COLORS["Content Only"], MODEL_COLORS["Fixed Hybrid"], MODEL_COLORS["Adaptive Hybrid"]]
    v_rmse = main_df["RMSE"].values
    ax1 = axes[0]
    bars = ax1.bar(variants, v_rmse, color=var_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title("Ablation: Rating Error (RMSE)\n[Lower is better]", fontweight="bold")
    ax1.set_ylabel("RMSE on Test Set")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval * 0.5, f"{yval:.4f}", ha="center", va="center", color="white", fontweight="bold")

    # Panel 2: Fixed Alpha Sensitivity (0.0 to 1.0) with Adaptive Reference
    ax2 = axes[1]
    ax2.plot(alpha_df["alpha"], alpha_df["RMSE"], marker="o", color="#ff7f0e", label="Fixed Hybrid (varying alpha)")
    ax2.axhline(ad_rmse, color="#d62728", linestyle="--", label=f"Adaptive Hybrid (k0 tuned: {ad_rmse:.4f})")
    ax2.set_title("Fixed alpha Sensitivity Grid\n[Effect of Global CF Weight]", fontweight="bold")
    ax2.set_xlabel("CF Weight alpha (0.0 = Content Only, 1.0 = CF Only)")
    ax2.set_ylabel("RMSE")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: k0 Sensitivity Grid (Adaptive)
    ax3 = axes[2]
    ax3.plot(k0_df["k0"], k0_df["RMSE"], marker="s", color="#d62728", label="Adaptive Hybrid (varying k0)")
    fixed_row = main_df[main_df["variant"] == "fixed_0.5"].iloc[0]
    ax3.axhline(fixed_row["RMSE"], color="#ff7f0e", linestyle=":", label=f"Fixed 50/50 Baseline ({fixed_row['RMSE']:.4f})")
    ax3.set_xscale("log")
    ax3.set_title("Adaptive k0 Sensitivity\n[Effect of Profile-Length Scaling]", fontweight="bold")
    ax3.set_xlabel("k0 Parameter (Log Scale)")
    ax3.set_ylabel("RMSE")
    ax3.legend(loc="best")
    ax3.grid(True, which="both", linestyle="--", alpha=0.6)

    ds_title = "MovieLens 100K Benchmark" if dataset == "movielens" else "Food.com Recipes Primary Domain"
    fig.suptitle(f"Experiment E2: Hybrid Weight Ablation & Sensitivity Analysis — {ds_title}", fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = FIGURES_DIR / f"e2_ablation_{dataset}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_e3_activity_and_alpha(dataset: str) -> Path:
    """E3: Grouped bar chart by activity terciles + Mean alpha per group."""
    _ensure_dir()
    csv_path = Path("results") / f"e3_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter to the three tercile groups: few, medium, many
    terc_df = df[df["activity_group"].isin(["few", "medium", "many"])].copy()

    # Models to plot
    key_models = ["Popularity", "Best CF", "Content", "Fixed Hybrid", "Adaptive Hybrid"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))
    groups = ["few", "medium", "many"]
    x = np.arange(len(groups))
    width = 0.15

    # Panel 1: RMSE across activity groups
    for i, m in enumerate(key_models):
        m_vals = [terc_df[(terc_df["activity_group"] == g) & (terc_df["model"] == m)]["RMSE"].iloc[0] for g in groups]
        ax1.bar(x + (i - 2) * width, m_vals, width, label=m, color=MODEL_COLORS.get(m, "#333"), alpha=0.85, edgecolor="black", linewidth=0.4)
    ax1.set_title("RMSE by Activity Tercile\n[Lower is better]", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Few\nRatings", "Medium\nRatings", "Many\nRatings"])
    ax1.set_ylabel("RMSE on Test Set")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 2: NDCG@10 across activity groups
    for i, m in enumerate(key_models):
        m_vals = [terc_df[(terc_df["activity_group"] == g) & (terc_df["model"] == m)]["NDCG@10"].iloc[0] for g in groups]
        ax2.bar(x + (i - 2) * width, m_vals, width, label=m, color=MODEL_COLORS.get(m, "#333"), alpha=0.85, edgecolor="black", linewidth=0.4)
    ax2.set_title("NDCG@10 by Activity Tercile\n[Higher is better]", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Few\nRatings", "Medium\nRatings", "Many\nRatings"])
    ax2.set_ylabel("NDCG@10 on Test Set")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 3: Mean alpha_u of Adaptive Hybrid per group
    alpha_vals = [terc_df[(terc_df["activity_group"] == g) & (terc_df["model"] == "Adaptive Hybrid")]["mean_adaptive_alpha"].iloc[0] for g in groups]
    bars3 = ax3.bar(groups, alpha_vals, color="#d62728", alpha=0.85, edgecolor="black", width=0.45)
    ax3.axhline(0.5, color="#ff7f0e", linestyle="--", label="Fixed Hybrid (α = 0.50)")
    ax3.set_title("Mean Adaptive CF Weight (α_u)\n[Monotonic scaling with profile size]", fontweight="bold")
    ax3.set_ylabel("Mean CF Weight α_u (1 - α_u = Content Weight)")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_xticks(range(len(groups)))
    ax3.set_xticklabels(["Few\nRatings", "Medium\nRatings", "Many\nRatings"])
    ax3.legend(loc="upper left")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")

    ds_title = "MovieLens 100K Benchmark" if dataset == "movielens" else "Food.com Recipes Primary Domain"
    fig.suptitle(f"Experiment E3: Performance and Adaptive Weight by User Activity Level — {ds_title}", fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = FIGURES_DIR / f"e3_activity_{dataset}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_e4_similarity_curves(dataset: str) -> Path:
    """E4: Line plots of RMSE vs k for cosine and Pearson, separate panels for User-CF and Item-CF."""
    _ensure_dir()
    csv_path = Path("results") / f"e4_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    sweep_df = df[df["section"] == "similarity_and_k_sweep"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))

    # Panel 1: ItemBasedCF (RMSE vs k)
    item_df = sweep_df[sweep_df["model_type"] == "ItemBasedCF"]
    for sim, color, marker in [("cosine", "#1f77b4", "o"), ("pearson", "#ff7f0e", "s")]:
        sub = item_df[item_df["similarity"] == sim].sort_values("k")
        ax1.plot(sub["k"], sub["val_RMSE"], marker=marker, color=color, label=f"Item-CF ({sim.capitalize()})")
    ax1.set_title("Item-Based CF: Validation RMSE vs. k", fontweight="bold")
    ax1.set_xlabel("Number of Neighbors (k)")
    ax1.set_ylabel("Validation RMSE")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: UserBasedCF (RMSE vs k)
    user_df = sweep_df[sweep_df["model_type"] == "UserBasedCF"]
    for sim, color, marker in [("cosine", "#2ca02c", "o"), ("pearson", "#d62728", "s")]:
        sub = user_df[user_df["similarity"] == sim].sort_values("k")
        ax2.plot(sub["k"], sub["val_RMSE"], marker=marker, color=color, label=f"User-CF ({sim.capitalize()})")
    ax2.set_title("User-Based CF: Validation RMSE vs. k", fontweight="bold")
    ax2.set_xlabel("Number of Neighbors (k)")
    ax2.set_ylabel("Validation RMSE")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ds_title = "MovieLens 100K Benchmark" if dataset == "movielens" else "Food.com Recipes Primary Domain"
    fig.suptitle(f"Experiment E4: Similarity Metric & Neighbor Sensitivity (Validation Set) — {ds_title}", fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = FIGURES_DIR / f"e4_similarity_{dataset}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_e5_cross_domain_comparison() -> Path:
    """E5: Side-by-side comparison chart of relative improvements over baselines."""
    _ensure_dir()
    csv_path = Path("results") / "e5_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    models = df["model"].tolist()

    x = np.arange(len(models))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Relative Improvement in RMSE over Global Mean (%)
    ml_rmse_imp = df["ML_RMSE_rel_imp_%"].values
    fc_rmse_imp = df["FC_RMSE_rel_imp_%"].values

    ax1.bar(x - width/2, ml_rmse_imp, width, label="MovieLens 100K", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.bar(x + width/2, fc_rmse_imp, width, label="Food.com Recipes", color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Relative Rating Error Reduction over Global Mean (%)\n[Higher is better]", fontweight="bold")
    ax1.set_ylabel("Error Reduction (%) = (GM - Model) / GM")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=40, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 2: Relative NDCG@10 vs Popularity Baseline (%)
    ml_ndcg_imp = df["ML_NDCG_rel_imp_%"].values
    fc_ndcg_imp = df["FC_NDCG_rel_imp_%"].values

    ax2.bar(x - width/2, ml_ndcg_imp, width, label="MovieLens 100K", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.bar(x + width/2, fc_ndcg_imp, width, label="Food.com Recipes", color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Popularity Baseline (0%)")
    ax2.set_title("Relative Top-10 Ranking vs. Popularity Baseline (%)\n[Relative NDCG@10 Deficit]", fontweight="bold")
    ax2.set_ylabel("Relative Change (%) = (Model - Pop) / Pop")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=40, ha="right")
    ax2.legend(loc="lower left")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Experiment E5: Cross-Domain Performance Comparison Normalized to Baselines", fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = FIGURES_DIR / "e5_cross_domain_comparison.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def generate_all_plots() -> List[Path]:
    """Generate all figures for both datasets and cross-domain analysis."""
    generated: List[Path] = []
    print("Generating Experiment Plots (results/figures/)...")

    # E1
    generated.append(plot_e1_main("movielens"))
    generated.append(plot_e1_main("foodcom"))

    # E2
    generated.append(plot_e2_ablation_and_sensitivity("movielens"))
    generated.append(plot_e2_ablation_and_sensitivity("foodcom"))

    # E3
    generated.append(plot_e3_activity_and_alpha("movielens"))
    generated.append(plot_e3_activity_and_alpha("foodcom"))

    # E4
    generated.append(plot_e4_similarity_curves("movielens"))
    generated.append(plot_e4_similarity_curves("foodcom"))

    # E5
    generated.append(plot_e5_cross_domain_comparison())

    for p in generated:
        print(f"  Saved figure: {p}")
    return generated


if __name__ == "__main__":
    generate_all_plots()

