# Explainable Hybrid Recommender System

An explainable hybrid recommender system designed for recipes (Food.com dataset as primary) and movies (MovieLens 100K benchmark), operating through a shared evaluation and recommendation pipeline.

The project implements classical, fully transparent recommendation algorithms:
- Global, user, and item mean baselines, popularity baseline
- User-based and item-based k-Nearest Neighbors Collaborative Filtering (Cosine and Pearson similarities)
- TF-IDF content-based filtering
- Adaptive hybrid weighting: $\alpha_u = \frac{n_u}{n_u + k_0}$
- Built-in explainability generation for every predicted recommendation

---

## Setup Instructions

1. **Clone the repository and enter the project directory:**
   ```bash
   git clone <repo-url>
   cd RS_Project
   ```

2. **Create and activate a virtual environment (Python 3.10+):**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install pinned dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full Phase 1 pipeline** (preprocessing -> splits -> statistics):
   ```bash
   python -m src.run_phase1
   ```
   This requires both datasets to be downloaded first (see **Data download** below).
   Outputs are written to `data/processed/` and `results/`.

---

## Model Building & Caching Pipeline

All models (baselines, CF, ContentBased, and all Hybrid variants) can be built and cached with a single command:

```bash
# Build models on training split (cached under data/cache/<dataset>/)
python -m src.build_models --dataset movielens
python -m src.build_models --dataset foodcom

# Refit on combined train+val split for final test evaluation
python -m src.build_models --dataset movielens --fit-on train_val
python -m src.build_models --dataset foodcom --fit-on train_val

# Force a clean rebuild ignoring cache
python -m src.build_models --dataset movielens --force
```

### Evaluation Protocol Convention: Train -> Val -> Train+Val Refit

The project strictly adheres to rigorous evaluation partitioning:
1. **Model Tuning & Validation**: All hyperparameter searches (CF neighborhood $k$, similarity metric, min overlap, TF-IDF representations, and hybrid half-saturation threshold $k_0$) are performed exclusively by fitting models on `train.csv` and evaluating on `val.csv`. The test set (`test.csv`) is strictly held out and remains untouched during all tuning and exploratory work.
2. **Final Test Evaluation ("train_val")**: For final reporting against the unseen test split, models are refitted using `fit_on="train_val"` on the combined training and validation data (`train_df + val_df`), using the best hyperparameter configuration determined during validation.

---

## Benchmarking & Performance

Run the systematic pipeline benchmark:
```bash
python scripts/benchmark_pipeline.py
```

### Performance & Latency Summary

| Dataset | Cold Build (`train`) | Cached Load (`train`) | Cold Build (`train_val`) | Cached Load (`train_val`) | Rec + Explain (Top 1) | Rec + Explain (All 10) | Target (<1000 ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MovieLens 100K** | 1.74s | 0.47s | 1.82s | 0.52s | 3.79 ms/user | 8.08 ms/user | **PASSED** |
| **Food.com** | 4.39s | 1.24s | 5.30s | 1.55s | 5.54 ms/user | 10.94 ms/user | **PASSED** |

Recommendation latency and transparent explanation generation execute in under 11 ms per user across both datasets (~90x-260x faster than the 1.0s target).

---

## Data Download

### MovieLens 100K (Benchmark)
- Download URL: https://files.grouplens.org/datasets/movielens/ml-100k.zip
- Extract the contents into `data/raw/ml-100k/` such that files like `u.data`, `u.item`, and `u.user` reside directly inside `data/raw/ml-100k/`.

### Food.com Recipes & Reviews (Primary)
- Download `recipes.csv` and `reviews.csv` from Kaggle:
  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- Place both files directly into `data/raw/foodcom/` (`data/raw/foodcom/recipes.csv` and `data/raw/foodcom/reviews.csv`).
