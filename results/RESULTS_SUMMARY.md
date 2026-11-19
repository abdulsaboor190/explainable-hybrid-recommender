# Experimental Results Summary (Phase 3: E1 to E5)

This report summarizes key quantitative metrics, hypothesis tests, and neutral empirical readings for Experiments E1 through E5 across both the primary **Food.com** recipe domain and the **MovieLens 100K** benchmark domain. All reported metrics are generated strictly from the test set (refit on `train_val`) or validation set sweeps as specified by the evaluation protocol, with bootstrap confidence intervals resampling evaluated users.

---

## E1: Comprehensive Model and Baseline Comparison

### Key Numbers
- **MovieLens 100K (Benchmark):**
  - Rating Error (RMSE): Adaptive Hybrid (**0.977**, 95% CI [0.991, 1.036]) and Item-CF (**0.980**, [0.993, 1.039]) beat Global Mean (1.212) by +19.4% and +19.2% respectively.
  - Top-10 Ranking (NDCG@10): Popularity baseline dominates (**0.0894**, [0.0803, 0.0992]), followed by Fixed Hybrid (**0.0255**), Content (**0.0193**), and Adaptive Hybrid (**0.0093**).
- **Food.com (Primary Domain):**
  - Rating Error (RMSE): Fixed Hybrid (**0.547**, [0.562, 0.595]) and Adaptive Hybrid (**0.548**, [0.562, 0.595]) beat Global Mean (0.581) by +5.8%.
  - Top-10 Ranking (NDCG@10): Popularity baseline dominates (**0.0324**, [0.0296, 0.0352]), followed by Content-Based (**0.0073**), Fixed Hybrid (**0.0063**), and Adaptive Hybrid (**0.0062**).

### Neutral Interpretation & Limitations
The empirical findings show that every model except User-CF on Food.com beats the naive Global Mean rating baseline, and hybrid models achieve the best overall RMSE in both domains. However, on top-10 ranking metrics (NDCG@10 and P@10), unpersonalized Popularity substantially outperforms all personalized models due to high test-set item concentration. A key limitation is that 44 users in MovieLens had no relevant items ($\ge 4.0$) in the test split and were excluded from ranking calculations, while Food.com had zero skipped users.

---

## E2: Hybrid Weight Ablation & Significance Testing

### Key Numbers
- **MovieLens 100K:**
  - CF Only (0.979) and Content Only (1.133) demonstrate that collaborative signals heavily dominate movie rating predictions.
  - Paired Difference (Adaptive minus Fixed 0.5): $\Delta\text{SE} = -0.0355$ (95% CI $[-0.0425, -0.0286]$, Wilcoxon $p = 7.22 \times 10^{-25}$). Adaptive significantly outperforms Fixed 50/50 on rating accuracy.
  - Ranking Difference: $\Delta\text{NDCG@10} = -0.0162$ (95% CI $[-0.0197, -0.0125]$, Wilcoxon $p = 4.69 \times 10^{-18}$). Fixed 50/50 significantly outperforms Adaptive on ranking.
- **Food.com:**
  - CF Only (0.574) and Content Only (0.556) show that recipe content features outperform collaborative filtering in sparse settings.
  - Paired Difference (Adaptive minus Fixed 0.5): $\Delta\text{SE} = +0.000005$ (95% CI $[-0.0025, +0.0026]$, Wilcoxon $p = 2.72 \times 10^{-7}$). Because the 95% bootstrap CI spans zero, the difference is not statistically distinguishable from zero.
  - Ranking Difference: $\Delta\text{NDCG@10} = -0.000127$ (95% CI $[-0.0009, +0.0007]$, Wilcoxon $p = 0.77$). Indistinguishable from zero.

### Neutral Interpretation & Limitations
Adaptive user-specific weighting provides a statistically confirmed rating accuracy advantage on MovieLens, but at the cost of top-$K$ ranking metric degradation because content signals are underweighted for dense users. On Food.com, adaptive weighting achieves no statistically meaningful difference over a fixed 50/50 blend in either squared error or NDCG@10, as the large tuned $k_0=100$ keeps adaptive weights low and static across most sparse user profiles.

---

## E3: Performance by User Activity Level & Cold-Start Simulation

### Key Numbers
- **Activity Tercile Mean CF Weights ($\alpha_u$):**
  - MovieLens: Few = 0.530, Medium = 0.728, Many = 0.886 (Overall = 0.713).
  - Food.com: Few = 0.162, Medium = 0.232, Many = 0.424 (Overall = 0.270).
- **Paired $\Delta\text{SE}$ ($\text{Adaptive} - \text{Fixed}$):**
  - MovieLens: Few = $-0.0046$ (95% CI $[-0.0096, +0.0005]$, not significant); Many = $-0.0664$ (95% CI $[-0.0817, -0.0505]$, significant).
  - Food.com: Few = $+0.0024$ (95% CI $[-0.0032, +0.0080]$, not significant); Many = $-0.0006$ (95% CI $[-0.0016, +0.0004]$, not significant).
- **Simulated Cold-Start (5 ratings per user):**
  - MovieLens: Fixed Hybrid achieves best RMSE (**1.301**) vs. Adaptive Hybrid (1.357) and CF Only (1.355).
  - Food.com: Fixed Hybrid achieves best RMSE (**0.601**) vs. Adaptive Hybrid (0.648) and Content Only (0.658).

### Neutral Interpretation & Limitations
The experimental evidence contradicts the hypothesis that the adaptive blend helps low-activity users most. On MovieLens, the adaptive model's gains are concentrated in high-activity users where CF dominates, while for low-activity users $\alpha_u \approx 0.53$, behaving virtually identically to the fixed hybrid. Under strict simulated cold start (5 ratings), Fixed 50/50 outperforms Adaptive Hybrid across both domains because pure Content models produce larger rating residuals than CF.

---

## E4: Similarity Metric, Neighbor Sensitivity, and Overlap Sweep

### Key Numbers
- **Cosine vs. Pearson:**
  - MovieLens Item-CF validation RMSE: Cosine = **0.9638** vs. Pearson = 0.9807.
  - MovieLens User-CF validation RMSE: Cosine = **0.9767** vs. Pearson = 0.9931.
  - Food.com Item-CF validation RMSE: Cosine = **0.5708** vs. Pearson = 0.5954.
  - Food.com User-CF validation RMSE: Cosine = **0.5608** vs. Pearson = 0.5848.
- **Neighbor Parameter ($k$) Curve:**
  - On MovieLens, Item-CF error reaches its minimum at $k=20$ (0.9638), plateaus through $k=30$ (0.9640), and degrades past $k=50$ (0.9695).
  - On Food.com, User-CF plateaus at $k=30$ to $k=50$ (0.5610 to 0.5608).
- **Computational Cost:**
  - Cosine fits in $<0.5$ seconds on both datasets, whereas Pearson requires up to 64 seconds due to non-vectorized pairwise co-rated filtering.

### Neutral Interpretation & Limitations
Mean-centered Cosine similarity decisively outperforms exact Pearson correlation on both datasets in prediction accuracy and execution speed. The $k$ sensitivity curves demonstrate clear elbow plateaus around $k=20\text{--}30$, confirming that small neighbor sets effectively balance neighborhood signal against distant noise. On Food.com, User-Based CF achieved slightly lower validation RMSE than Item-Based CF, but Item-Based CF was selected in configuration to preserve recipe-to-recipe explainability.

---

## E5: Cross-Domain Comparison

### Key Numbers
- **Dataset Properties:**
  - Matrix sparsity: Food.com (98.59%) is substantially sparser than MovieLens (93.70%).
  - Rating skew: 79.97% of Food.com ratings are 5 stars (mean 4.74), compared to 21.20% on MovieLens (mean 3.53).
- **Relative Error Reduction over Global Mean:**
  - MovieLens: Adaptive Hybrid achieves a +19.39% RMSE reduction over Global Mean.
  - Food.com: Fixed Hybrid achieves a +5.78% RMSE reduction over Global Mean.
- **Model Ranking Consistency:**
  - RMSE Ranking Correlation: Spearman $\rho = 0.361$ (Kendall $\tau = 0.200$).
  - NDCG@10 Ranking Correlation: Spearman $\rho = 0.563$ (Kendall $\tau = 0.429$).

### Neutral Interpretation & Limitations
Cross-domain model rankings differ substantially, indicating that recommender behavior does not transfer trivially between dense rating domains (movies) and sparse, skewed consumption domains (recipes). The severe rating ceiling on Food.com compresses variance, limiting personalization headroom and allowing simple mean-based predictors to appear competitive. Content representations provide a critical stabilizing signal on Food.com where collaborative overlap is sparse, whereas collaborative signals drive virtually all personalization on MovieLens.
