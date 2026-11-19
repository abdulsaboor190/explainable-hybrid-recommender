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

### Phase 2 Full Verification (Sanity Tables + Exit Criteria)

Run the complete Phase 2 verification for both datasets:

```bash
python -m src.run_phase2
```

This command:
1. Builds all models on `train_val` (from cache if available).
2. Evaluates all models on the held-out TEST split (RMSE, MAE, P@5, P@10, R@10, NDCG@10).
3. Saves tables to `results/phase2_sanity_movielens.csv` and `results/phase2_sanity_foodcom.csv`.
4. Runs all 8 Phase 2 exit criteria checks and reports PASS/FAIL.

### Evaluation Protocol Convention: Train -> Val -> Train+Val Refit

The project strictly adheres to rigorous evaluation partitioning:
1. **Model Tuning & Validation**: All hyperparameter searches (CF neighborhood $k$, similarity metric, min overlap, TF-IDF representations, and hybrid half-saturation threshold $k_0$) are performed exclusively by fitting models on `train.csv` and evaluating on `val.csv`. The test set (`test.csv`) is strictly held out and remains untouched during all tuning and exploratory work.
2. **Final Test Evaluation ("train_val")**: For final reporting against the unseen test split, models are refitted using `fit_on="train_val"` on the combined training and validation data (`train_df + val_df`), using the best hyperparameter configuration determined during validation.

---

## Streamlit Interactive Demo Application

An interactive web application allows instructors and evaluators to explore user rating histories, inspect personalized recommendations with transparent structured explanations, compare multiple recommendation models side-by-side, filter by dietary requirements and nutritional constraints, and experiment with pantry kitchen mode.

### Running the Demo App

You can launch the app directly with Streamlit:
```bash
streamlit run app/app.py
```

Or using the automated wrapper which checks/builds pipeline caches first:
```bash
python -m src.run_app
```

### Key Demo Features:
- **Sidebar**: Domain picker (Food.com Recipes vs. MovieLens 100K), user selector with dynamic empirical archetypes (*Low-activity user*, *Mid-activity user*, *Heavy user*) plus arbitrary custom User ID input, model selector (`Adaptive Hybrid`, `Fixed Hybrid`, `CF Only`, `Content Only`, `Popularity`), and top-$N$ recommendation slider.
- **Recipe Options (Food.com)**: Dietary restrictions (Vegetarian, Vegan, Gluten-Free, Nut-Free, Dairy-Free), max prep time slider, max calorie slider, excluded ingredients list, and **Kitchen Mode** pantry ingredient matching.
- **Main Tabs**:
  - **User Profile**: Complete rating history sorted by rating, with rating count and plain-English adaptive weight explanation ($\alpha_u$).
  - **Recommendations & Explanations**: Top-$N$ list displaying predicted rating (1 to 5 stars), prep time, calories, categories, kitchen overlap %, and structured natural-language explanations.
  - **Compare Models**: Side-by-side Top-5 recommendations across all 5 models under the same active filter constraints.

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
