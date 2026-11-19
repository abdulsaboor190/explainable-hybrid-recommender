# Experiment E5: Cross-Domain Comparison (MovieLens 100K vs. Food.com)

### Side-by-Side Model Performance and Relative Baseline Gains
| Model | ML RMSE | ML RMSE Imp (%) | FC RMSE | FC RMSE Imp (%) | ML NDCG@10 | ML NDCG Imp (%) | FC NDCG@10 | FC NDCG Imp (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Global Mean | 1.212 | +0.00% | 0.581 | +0.00% | 0.0309 | -65.44% | 0.0034 | -89.51% |
| User Mean | 1.141 | +5.87% | 0.550 | +5.25% | 0.0309 | -65.44% | 0.0034 | -89.51% |
| Item Mean | 1.072 | +11.54% | 0.575 | +1.03% | 0.0008 | -99.11% | 0.0020 | -93.83% |
| Popularity | 1.072 | +11.54% | 0.575 | +1.03% | **0.0894** | +0.00% | **0.0324** | +0.00% |
| User-CF | 1.064 | +12.24% | 0.589 | -1.34% | 0.0159 | -82.21% | 0.0053 | -83.64% |
| Item-CF | 0.980 | +19.18% | 0.574 | +1.21% | 0.0022 | -97.54% | 0.0029 | -91.05% |
| Content | 1.133 | +6.55% | 0.556 | +4.29% | 0.0193 | -78.41% | 0.0073 | -77.47% |
| Fixed Hybrid | 1.006 | +17.03% | **0.547** | +5.78% | 0.0255 | -71.48% | 0.0063 | -80.56% |
| Adaptive Hybrid | **0.977** | +19.39% | 0.547 | +5.75% | 0.0093 | -89.60% | 0.0062 | -80.86% |

### Model Rankings Across Domains
| Model | ML Rank (RMSE) | FC Rank (RMSE) | ML Rank (NDCG@10) | FC Rank (NDCG@10) |
| :--- | :---: | :---: | :---: | :---: |
| Global Mean | 9 | 8 | 2 | 6 |
| User Mean | 8 | 3 | 2 | 6 |
| Item Mean | 5 | 6 | 9 | 9 |
| Popularity | 5 | 6 | 1 | 1 |
| User-CF | 4 | 9 | 6 | 5 |
| Item-CF | 2 | 5 | 8 | 8 |
| Content | 7 | 4 | 5 | 2 |
| Fixed Hybrid | 3 | 1 | 4 | 3 |
| Adaptive Hybrid | 1 | 2 | 7 | 4 |