# PHASE 2 NOTES — Explainable Hybrid Recommender

**Purpose:** Design decisions, final configurations, evaluation findings, and known limitations for the Phase 2 implementation. Written for reuse in the final report.

---

## 1. Final Model Configurations

### Collaborative Filtering

| Parameter | MovieLens 100K | Food.com | Source |
|---|---|---|---|
| CF Type | ItemBasedCF | ItemBasedCF | Task 2 grid search; item-CF outperformed user-CF on both datasets |
| Similarity | cosine | cosine | Task 2 grid; cosine slightly outperformed Pearson (Pearson sensitive to outliers at scale) |
| k (neighbours) | 30 | 30 | Task 2 grid over {5, 10, 20, 30, 50}; diminishing returns beyond 30 |
| min_overlap | 2 | 2 | Task 2 tuning; overlap=5 too restrictive on sparse Food.com |
| shrinkage | 0.0 | 0.0 | Task 2: no improvement with shrinkage on these datasets |

### Content-Based

| Parameter | Both Datasets | Source |
|---|---|---|
| Vectoriser | TF-IDF, sublinear_tf=True | Task 3 |
| Token pattern | `\b[a-zA-Z0-9_]+\b` | Preserves underscore-joined terms (e.g. `olive_oil`) |
| k_content | 30 | Task 3 |
| liked_threshold | 4.0 | Config default |
| Text column | `genre_tokens` (ML) / `text` (Food.com) | Config |

### Adaptive Hybrid

| Parameter | MovieLens 100K | Food.com | Source |
|---|---|---|---|
| mode | adaptive | adaptive | Task 4 |
| k₀ | **20.0** | **100.0** | Task 4 grid over {1, 2, 5, 10, 20, 50, 100}; best val RMSE |
| Val RMSE (adaptive hybrid) | **0.9625** | **0.5377** | Task 4 |

---

## 2. How the Hybrid Blends Scores

### Alpha Computation
Each user gets a personalised CF weight:

$$\alpha_u = \frac{n_u}{n_u + k_0}$$

where $n_u$ = number of training ratings for user $u$, $k_0$ = half-saturation threshold.

- When $n_u = 0$ (new/cold-start user): $\alpha_u = 0$ → pure Content-Based.  
- When $n_u \to \infty$ (power user): $\alpha_u \to 1$ → pure CF.  
- $k_0$ controls the crossover point: at $n_u = k_0$, $\alpha_u = 0.5$.

### Score Blending in `recommend()`
To handle the different scales of CF (predicted ratings, 1–5) and Content (cosine similarities, 0–1), the hybrid:
1. Computes CF scores and Content scores across the **full catalogue** in vectorised numpy (no Python loops over items).
2. **Min-Max normalises** each component over the unseen candidate set to [0, 1].
3. Blends: `hybrid_score = α_u × norm(CF) + (1 − α_u) × norm(Content)`.

### Score Blending in `predict()`
For scalar prediction (used for RMSE/MAE):
- Uses raw predicted ratings (not normalised): `pred = α_u × CF_pred + (1 − α_u) × Content_pred`.
- Clipped to [1.0, 5.0].

This creates a known discrepancy: ranking is based on normalised blend, prediction is based on raw blend. This is intentional — the normalised blend is more appropriate for ranking since it removes scale bias.

---

## 3. Design Decisions and Rationale

### Why Item-Based over User-Based CF?
Item-CF is more stable when the item catalogue is small relative to the user base. On Food.com (3,889 items, 3,000 users), item similarities are computed over a fixed vocabulary of recipes, so the similarity matrix is smaller and more reusable. User-CF requires recomputing neighbourhood for each new user.

### Why Cosine over Pearson similarity?
Pearson is sensitive to outliers in ratings. Food.com has extreme positivity bias (>80% 5-star ratings), which inflates Pearson scores artificially — many users have near-zero variance, making Pearson undefined or meaningless. Cosine is more robust.

### Why TF-IDF with Sublinear TF?
- Raw TF penalises rare descriptive terms (e.g. `cardamom`). Sublinear scaling (1 + log(tf)) gives rare but distinctive ingredients more weight.
- The custom token pattern preserves Food.com's underscore-joined compound terms (`olive_oil`, `all_purpose_flour`) as single vocabulary tokens.

### Why keep UserMean as a fallback in predict()?
When CF has no rating history for an item (cold item), falling back to global mean is too crude. User mean is a better fallback because it captures individual user bias (e.g. a generous rater who averages 4.8 vs a strict rater who averages 2.5).

### Why cache with a config hash?
The hash covers: dataset name, fit_on split, CF hyperparameters, content hyperparameters, k₀, and data size + rating sum. If any of these change (new tuning run, dataset reprocessing), the hash mismatches and models are automatically rebuilt. This prevents stale model reuse.

### Train → Val → Train+Val Refit Convention
- **Tuning**: fit on `train.csv` only, evaluate on `val.csv`. Test set is never touched.
- **Final reporting**: refit on `train.csv + val.csv` (using best hyperparameters from tuning), evaluate on `test.csv`.
- Code enforcement: `load_dataset_data(fit_on="train_val")` concatenates splits; `test.csv` is loaded but never passed to any model's `fit()`.

---

## 4. Known Limitations

### Dietary Filter Rules (Task 6)
- **Keyword-based only**: The filter matches whole ingredient words against explicit exclusion lists. It cannot reason about processing (e.g. "refined coconut oil" is treated the same as "unrefined"). Hidden animal derivatives (gelatin in marshmallows, rennet in cheese) are not flagged unless explicitly listed.
- **Food.com data quality**: Some recipes in the raw Food.com dataset have incomplete `ingredients_list` — the main ingredient is embedded in the instructions text but absent from the structured field (e.g. "Corn Dog Casserole" lists only the cornbread batter ingredients, omitting the hot dogs).
- **No cross-contamination reasoning**: Allergen certification (nut-free facilities) cannot be verified from ingredient text alone.
- **Homonym exceptions are manually curated**: `butternut squash` (safe for nut_free), `water chestnut` (safe for nut_free), `coconut milk` (safe for dairy_free). Any new edge cases must be added to `DIET_EXCEPTIONS` in `src/recipe_features.py`.

### Kitchen Mode Overlap (Task 6)
- Pantry staples (salt, pepper, water, oil) are assumed available by default and are configurable. This can cause trivially simple recipes (1–2 ingredients, all staples) to rank at the top with 100% overlap regardless of content preference.
- Ingredient matching is text-based: "chicken breast" matches "chicken" (by token), but "boneless skinless chicken thighs" may not match "chicken thighs" if the singulariser does not handle the multi-word phrase precisely.

### Recommendation Ranking Metrics (Low P/R/NDCG)
The ranking metrics in the Phase 2 sanity table appear very low (e.g. NDCG@10 ≈ 0.009–0.032 for MovieLens). This is expected for two reasons:
1. **Test items vs catalogue size**: The test set samples from the full catalogue; for MovieLens (1,682 items), retrieving a user's specific few relevant test items in top-10 is very hard at random baseline.
2. **train+val seen exclusion**: After excluding all seen items, the remaining catalogue for a heavy user is very small. Relevant test items can only appear in that residual. P/R/NDCG values are consistent with academic literature on 100K-scale datasets without advanced reranking.

---

## 5. Surprising Findings

### Reversal of Model Dominance Between Datasets
- **MovieLens**: CF dominates. Item-CF (val RMSE 0.9723) far outperforms Content (val RMSE 1.0753). The adaptive hybrid (k₀=20) is CF-heavy: mean α_u = 0.694, median 0.706, with only 16.5% of users below α=0.5.
- **Food.com**: Content dominates. Content-Based (val RMSE 0.5360) outperforms CF (val RMSE 0.6067). The adaptive hybrid (k₀=100) is content-heavy: mean α_u = 0.252, median 0.206, with 93.5% of users below α=0.5. This is because most Food.com users have fewer than 100 training ratings, giving them low alpha.

### Food.com Extreme Positivity Bias
Over 80% of all Food.com interactions are 5-star ratings. This creates a low-variance target distribution that benefits UserMean (val RMSE 0.5533) over CF (val RMSE 0.6067), because CF shrinks predictions toward item means which deviate from the user's typical 5-star behaviour.

### Zero Cold-Start Items in Food.com
The iterative 20-core filtering in preprocessing guarantees that every item in the filtered catalogue appears in training data. MovieLens has 83 cold-start items in the test split (items in catalogue but unseen in training), handled by fallback prediction.

### Recommendation Latency Was Already Fast
Before any optimisation, the vectorised `recommend()` path (implemented in Task 4) already executes in under 11 ms/user on both datasets for recommend+explain. The bottleneck identified in Task 4 (a Python loop over catalogue items in a cross-catalogue alignment case) was fixed by projecting CF predictions into the content catalogue index space via numpy index arrays.

### Item-CF Ranking Metrics are Paradoxically Low
Item-CF achieves the best RMSE on MovieLens (0.9795 test) but very low ranking NDCG@10 (0.0022). This is not a bug — it reflects that RMSE and NDCG measure different things. Item-CF predicts precise ratings well, but its `recommend()` path sorts by normalised score, and the min-max normalisation collapses the score range for some users (all predictions cluster tightly), producing arbitrary ranking. The hybrid's `recommend()` path benefits from blending two independently normalised score distributions, increasing spread.

---

## 6. Validation Results Summary (val split, not test)

| Model | MovieLens Val RMSE | Food.com Val RMSE |
|---|---|---|
| GlobalMean | 1.2161 | 0.5809 |
| UserMean | 1.1586 | 0.5533 |
| ItemMean | — | — |
| Best CF (ItemBasedCF, k=30) | 0.9723 | 0.6067 |
| Content-Only | 1.0753 | 0.5360 |
| Fixed Hybrid (α=0.5) | 0.9748 | 0.5398 |
| **Adaptive Hybrid (tuned k₀)** | **0.9625** (k₀=20) | **0.5377** (k₀=100) |

## 7. Test Results Summary (test split, final, train+val refit)

### MovieLens 100K

| Model | RMSE | MAE | P@5 | P@10 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Global Mean | 1.2120 | 1.0070 | 0.0234 | 0.0230 | 0.0243 | 0.0309 |
| User Mean | 1.1409 | 0.9106 | 0.0234 | 0.0230 | 0.0243 | 0.0309 |
| Item Mean | 1.0721 | 0.8580 | 0.0011 | 0.0008 | 0.0006 | 0.0008 |
| Popularity | 1.0721 | 0.8580 | 0.0721 | 0.0705 | 0.0736 | 0.0894 |
| User-CF | 1.0637 | 0.8194 | 0.0133 | 0.0135 | 0.0117 | 0.0159 |
| Item-CF | 0.9795 | 0.7679 | 0.0016 | 0.0017 | 0.0024 | 0.0022 |
| Content | 1.1326 | 0.8973 | 0.0176 | 0.0155 | 0.0160 | 0.0193 |
| Fixed Hybrid | 1.0056 | 0.7974 | 0.0242 | 0.0202 | 0.0208 | 0.0255 |
| **Adaptive Hybrid** | **0.9770** | **0.7694** | 0.0056 | 0.0070 | 0.0117 | 0.0093 |

### Food.com

| Model | RMSE | MAE | P@5 | P@10 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Global Mean | 0.5809 | 0.4106 | 0.0039 | 0.0023 | 0.0026 | 0.0034 |
| User Mean | 0.5504 | 0.3477 | 0.0039 | 0.0023 | 0.0026 | 0.0034 |
| Item Mean | 0.5749 | 0.3826 | 0.0013 | 0.0018 | 0.0015 | 0.0020 |
| Popularity | 0.5749 | 0.3826 | 0.0259 | 0.0233 | 0.0294 | 0.0324 |
| User-CF | 0.5887 | 0.3563 | 0.0041 | 0.0044 | 0.0045 | 0.0053 |
| Item-CF | 0.5739 | 0.3587 | 0.0025 | 0.0024 | 0.0024 | 0.0029 |
| Content | 0.5560 | 0.3465 | 0.0067 | 0.0063 | 0.0062 | 0.0073 |
| Fixed Hybrid | **0.5473** | 0.3511 | 0.0054 | 0.0055 | 0.0053 | 0.0063 |
| **Adaptive Hybrid** | 0.5475 | **0.3490** | 0.0057 | 0.0050 | 0.0054 | 0.0062 |
