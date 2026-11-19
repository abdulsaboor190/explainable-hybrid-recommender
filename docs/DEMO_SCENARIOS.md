# Streamlit Demo Scenarios & Scripted Walkthrough

This document details four scripted demonstration scenarios for the Explainable Hybrid Recommender web application (`app/app.py`). Each scenario uses real user IDs and inputs from the underlying dataset distributions, verifies actual system behavior, explains what the instructor should observe, and identifies exact UI locations for verification screenshots.

---

## Latency & Performance Summary (Warm Cache)

Page interaction latency measured across both domains after models and data are cached in memory:

| Interaction Type | Food.com Recipes | MovieLens 100K | Target Latency | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Switching User** (History + Recommendations) | 24.6 ms | 21.6 ms | < 2,000 ms | **PASSED** (80x faster) |
| **Changing Model** (Adaptive $\to$ Fixed $\to$ CF) | 16.6 ms | 18.0 ms | < 2,000 ms | **PASSED** (110x faster) |
| **Applying Filters** (Dietary + Prep Time + Calories) | 643.6 ms | N/A | < 2,000 ms | **PASSED** (3x faster) |
| **Kitchen Mode** (Pantry Matching & Overlap Scoring) | 1,748.7 ms | N/A | < 2,000 ms | **PASSED** (under 2s) |

---

## Scenario 1: Low-Activity User (Content-Driven Personalization)

- **Dataset Domain:** MovieLens 100K (Benchmark)
- **User ID:** `19` (Selected via preset: *"Low-activity user (ID 19, 17 ratings)"*)
- **Active Model:** `Adaptive Hybrid` ($k_0 = 20.0$)
- **Filter Settings:** None
- **Demonstration Goal:** Demonstrates that for a user with limited interaction history ($n_u = 17$), the adaptive formulation ($\alpha_u = \frac{17}{17 + 20} = 0.46$) properly limits collaborative weight and delegates majority influence ($54\%$) to item content representations.

### What the Instructor Sees:
1. **Profile Tab:**
   - Total Rated Items: **17** | Mean Rating: **3.47 / 5.0** | Adaptive CF Weight: **0.46**.
   - Plain-English banner: *"This user has 17 ratings in training history (k₀ = 20.0). The adaptive blend assigns 0.46 (46%) to collaborative filtering and 0.54 (54%) to item content features."*
2. **Recommendations Tab:**
   - Top-1 Item: **The Magic Hour (1998)** — Predicted Rating: ⭐ **4.26 / 5.0**.
   - Structured Explanation:
     - Component: `hybrid` ($\alpha = 0.46$).
     - Summary: *"You have rated 17 items, so we leaned mostly on item content (weight 0.54 vs 0.46). Shares drama with movies you liked (e.g. Eve's Bayou (1997))."*
   - Top-2 Item: **Beat the Devil (1954)** — Predicted Rating: ⭐ **3.81 / 5.0**.
     - Summary: *"You have rated 17 items, so we leaned mostly on item content (weight 0.54 vs 0.46). Shares comedy and drama with movies you liked (e.g. Liar Liar (1997)). Additionally, rating patterns connect this to Evil Dead II (1987)."*

```
[SCREENSHOT PLACEHOLDER: docs/screenshots/scenario1_low_activity_user.png]
Description: Full view of Recommendations tab for User 19 showing 0.46 CF / 0.54 Content adaptive blend banner and content-driven explanation expanders.
```

---

## Scenario 2: Heavy User (Collaborative-Dominant Personalization)

- **Dataset Domain:** MovieLens 100K (Benchmark)
- **User ID:** `405` (Selected via preset: *"Heavy user (ID 405, 590 ratings)"*)
- **Active Model:** `Adaptive Hybrid` ($k_0 = 20.0$)
- **Filter Settings:** None
- **Demonstration Goal:** Demonstrates that for a dense user with extensive ratings ($n_u = 590$), the adaptive formula automatically transitions to near-pure collaborative filtering ($\alpha_u = \frac{590}{590 + 20} = 0.97$), leveraging rich co-rating neighborhoods.

### What the Instructor Sees:
1. **Profile Tab:**
   - Total Rated Items: **590** | Mean Rating: **2.01 / 5.0** | Adaptive CF Weight: **0.97**.
   - Plain-English banner: *"This user has 590 ratings in training history (k₀ = 20.0). The adaptive blend assigns 0.97 (97%) to collaborative filtering and 0.03 (3%) to item content features."*
2. **Recommendations Tab:**
   - Top-1 Item: **Santa with Muscles (1996)** — Predicted Rating: ⭐ **4.90 / 5.0**.
   - Structured Explanation:
     - Component: `hybrid` ($\alpha = 0.97$).
     - Summary: *"You have rated 590 items, so we leaned mostly on collaborative filtering (weight 0.97 vs 0.03). Because you rated Raiders of the Lost Ark (1981) (5 stars) and Terminator, The (1984) (5 stars), which are often rated similarly to this item. It also matches your preference for comedy."*
   - Top-2 Item: **Incognito (1997)** — Predicted Rating: ⭐ **4.90 / 5.0**.
     - Summary: *"You have rated 590 items, so we leaned mostly on collaborative filtering (weight 0.97 vs 0.03). Because you rated Scream (1996) (5 stars) and Star Wars (1977) (5 stars), which are often rated similarly to this item. It also matches your preference for thriller and crime."*

```
[SCREENSHOT PLACEHOLDER: docs/screenshots/scenario2_heavy_user.png]
Description: Recommendations tab for User 405 showing 0.97 CF weight and collaborative-neighbor explanations citing previously rated 5-star movies.
```

---

## Scenario 3: Recipes with Dietary Restriction & Preparation Time Limit

- **Dataset Domain:** Food.com Recipes (Primary Domain)
- **User ID:** `434952` (Selected via preset: *"Mid-activity user (ID 434952, 29 ratings)"*)
- **Active Model:** `Adaptive Hybrid` ($k_0 = 100.0$)
- **Filter Settings:**
  - Dietary Restriction: `Vegan` (Strict zero meat, poultry, seafood, dairy, egg, or honey ingredients)
  - Time Limit: Checkbox enabled $\to$ Max Time = **30 minutes**
- **Demonstration Goal:** Demonstrates pre-ranking constraint filtering. The system guarantees that 100% of returned recipes strictly conform to the dietary exclusion list and preparation time cap, without crashing or falling back to unfiltered recipes.

### What the Instructor Sees:
1. **Recipe Options Expander:**
   - Diet selected: `Vegan`.
   - Time limit active: `30 mins`.
2. **Recommendations Tab:**
   - Total items returned: **10** (Full top-N list preserved).
   - Top-1 Item: **Beth's Pizza Crust** — Predicted: ⭐ **4.57 / 5.0** | Prep Time: **30 mins** | Category: **Pizza**.
     - Verified Ingredients: `water, flour, olive oil, sugar, salt` (100% plant-based, zero dairy/eggs).
     - Explanation: *"You have rated 29 items, so we leaned mostly on item content (weight 0.78 vs 0.22). Shares olive oil, oven and 30 mins with recipes you rated highly (e.g. Roasted Asparagus)."*
   - Top-2 Item: **Chocolate Sauce (Throw Away Your Hershey's!)** — Predicted: ⭐ **4.68 / 5.0** | Prep Time: **10 mins** | Category: **Dessert**.
     - Verified Ingredients: `granulated sugar, salt, water, vanilla` (Dairy-free vegan chocolate syrup).
   - Top-3 Item: **Brownie Mix - Amazing Every Single Dang Time** — Predicted: ⭐ **4.67 / 5.0** | Prep Time: **30 mins** | Category: **Dessert**.

> **Note on Data Limitation (Keyword vs. Ground Truth):** In Food.com, recipe names and metadata occasionally contain culinary descriptor keywords (e.g. "Chicken-Fried Steak" or "Buffalo Chicken Pizza") whose parsed ingredients are simply flour, eggs, or mozzarella cheese. As documented in Phase 2, our rule-based filter strictly excludes recipes containing banned ingredients. When tested with `Vegetarian`, baked goods like **Holiday Snickerdoodles** and **Cinnamon Cookies** pass cleanly, while testing with `Vegan` eliminates all eggs and dairy products.

```
[SCREENSHOT PLACEHOLDER: docs/screenshots/scenario3_diet_time_filters.png]
Description: View of Food.com recommendations filtered by Vegan and <= 30 mins, showing prep time badges and vegan ingredients.
```

---

## Scenario 4: Kitchen Mode Pantry Matching with Missing Ingredients

- **Dataset Domain:** Food.com Recipes (Primary Domain)
- **User ID:** `434952`
- **Active Model:** `Adaptive Hybrid`
- **Pantry Input:** `"chicken, garlic, yogurt, rice, onion"`
- **Demonstration Goal:** Demonstrates pantry-matching kitchen mode. Recipes are scored by combining ingredient coverage % with user hybrid preference scores, surfacing the exact missing ingredients required to cook each meal.

### What the Instructor Sees:
1. **Kitchen Mode Status:**
   - Active pantry input: `chicken, garlic, yogurt, rice, onion`.
2. **Recommendations Tab:**
   - Top-1 Item: **Easiest Chicken Pot Pie** — Predicted: ⭐ **4.70 / 5.0**.
     - **Kitchen Match:** **100% of ingredients on hand**.
     - Missing: **None (Fully stocked!)**.
   - Top-2 Item: **Easy!!! Easiest Chicken Dinner Ever** — Predicted: ⭐ **4.69 / 5.0**.
     - **Kitchen Match:** **75% of ingredients on hand**.
     - Missing: **rotel** (canned diced tomatoes with green chilies).
   - Top-3 Item: **Mermaid's Tender Roast Chicken** — Predicted: ⭐ **4.65 / 5.0**.
     - **Kitchen Match:** **75% of ingredients on hand**.
     - Missing: **butter**.
   - Top-4 Item: **Smothered Chicken** — Predicted: ⭐ **4.67 / 5.0**.
     - **Kitchen Match:** **66% of ingredients on hand**.
     - Missing: **sour cream**.
   - Top-5 Item: **Grilled Teriyaki Chicken** — Predicted: ⭐ **4.63 / 5.0**.
     - **Kitchen Match:** **66% of ingredients on hand**.
     - Missing: **fresh lemon juice**.

```
[SCREENSHOT PLACEHOLDER: docs/screenshots/scenario4_kitchen_pantry_mode.png]
Description: Recommendations tab in Kitchen Mode displaying ingredient overlap percentages and explicit missing ingredients for pantry inventory.
```

---

## Scenario 5: Model Comparison Tab (Side-by-Side Architectural Contrast)

- **Dataset Domain:** Food.com Recipes
- **User ID:** `434952`
- **Demonstration Goal:** Demonstrates the **"Compare models"** tab, where the top-5 recommendations from `Adaptive Hybrid`, `Fixed Hybrid (50/50)`, `Collaborative Filtering Only`, `Content-Based Only`, and `Popularity Baseline` are rendered in 5 side-by-side columns under identical active filter constraints.

### What the Instructor Sees:
- **Columns 1 & 2 (Hybrids):** Balanced blend of highly relevant personalized recipes combining user taste and text representations.
- **Column 3 (CF Only):** Surfaces items based strictly on co-rating neighbors, exhibiting higher rating error residuals on sparse recipes.
- **Column 4 (Content Only):** Surfaces recipes rich in ingredients and tags matching the user's past 5-star history.
- **Column 5 (Popularity Baseline):** Shows overall top-rated community staples across the entire user base.

```
[SCREENSHOT PLACEHOLDER: docs/screenshots/scenario5_compare_models_tab.png]
Description: Five-column side-by-side comparison tab illustrating rank order differences across models for the same user.
```
