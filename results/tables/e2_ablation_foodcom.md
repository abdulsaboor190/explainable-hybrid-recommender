# Experiment E2: Hybrid Weight Ablation & Significance (Food.com)

### Main Ablation Variants
| Variant | RMSE | MAE | NDCG@10 | P@10 |
| :--- | :---: | :---: | :---: | :---: |
| CF Only (alpha=1.0) | 0.574 | 0.359 | 0.0029 | 0.0024 |
| Content Only (alpha=0.0) | 0.556 | **0.346** | **0.0073** | **0.0063** |
| Fixed Hybrid (alpha=0.5) | **0.547** | 0.351 | 0.0063 | 0.0055 |
| Adaptive Hybrid (k0=100) | 0.547 | 0.349 | 0.0062 | 0.0050 |

### Paired Significance Testing (Adaptive vs. Fixed 0.5)
| Metric | Mean Difference | 95% Bootstrap CI | Wilcoxon p-value | Significant (p<0.05 & CI excludes 0) |
| :--- | :---: | :---: | :---: | :---: |
| Mean Squared Error (SE) | 0.0000 | [-0.0025, 0.0026] | 2.72e-07 | **NO** |
| Mean Absolute Error (AE) | -0.0043 | [-0.0061, -0.0026] | 6.44e-17 | **YES** |
| NDCG@10 | -0.0001 | [-0.0009, 0.0007] | 7.68e-01 | **NO** |
| Precision@10 | -0.0005 | [-0.0011, 0.0001] | 1.37e-01 | **NO** |
| Recall@10 | 0.0001 | [-0.0007, 0.0009] | 9.15e-01 | **NO** |