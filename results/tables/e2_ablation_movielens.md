# Experiment E2: Hybrid Weight Ablation & Significance (MovieLens 100K)

### Main Ablation Variants
| Variant | RMSE | MAE | NDCG@10 | P@10 |
| :--- | :---: | :---: | :---: | :---: |
| CF Only (alpha=1.0) | 0.980 | **0.768** | 0.0022 | 0.0017 |
| Content Only (alpha=0.0) | 1.133 | 0.897 | 0.0193 | 0.0155 |
| Fixed Hybrid (alpha=0.5) | 1.006 | 0.797 | **0.0255** | **0.0202** |
| Adaptive Hybrid (k0=20) | **0.977** | 0.769 | 0.0093 | 0.0070 |

### Paired Significance Testing (Adaptive vs. Fixed 0.5)
| Metric | Mean Difference | 95% Bootstrap CI | Wilcoxon p-value | Significant (p<0.05 & CI excludes 0) |
| :--- | :---: | :---: | :---: | :---: |
| Mean Squared Error (SE) | -0.0355 | [-0.0425, -0.0286] | 7.22e-25 | **YES** |
| Mean Absolute Error (AE) | -0.0167 | [-0.0199, -0.0137] | 9.59e-23 | **YES** |
| NDCG@10 | -0.0162 | [-0.0197, -0.0125] | 4.69e-18 | **YES** |
| Precision@10 | -0.0132 | [-0.0165, -0.0102] | 8.36e-17 | **YES** |
| Recall@10 | -0.0091 | [-0.0121, -0.0064] | 3.91e-14 | **YES** |