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

4. **Run the full Phase 1 pipeline** (preprocessing → splits → statistics):
   ```bash
   python -m src.run_phase1
   ```
   This requires both datasets to be downloaded first (see **Data download** below).
   Outputs are written to `data/processed/` and `results/`.

---

## Data download

### MovieLens 100K (Benchmark)
- Download URL: https://files.grouplens.org/datasets/movielens/ml-100k.zip
- Extract the contents into `data/raw/ml-100k/` such that files like `u.data`, `u.item`, and `u.user` reside directly inside `data/raw/ml-100k/`.

### Food.com Recipes & Reviews (Primary)
- Download `recipes.csv` and `reviews.csv` from Kaggle:
  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- Place both files directly into `data/raw/foodcom/` (`data/raw/foodcom/recipes.csv` and `data/raw/foodcom/reviews.csv`).

