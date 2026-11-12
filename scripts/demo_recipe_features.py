"""Demonstration of recipe-specific features on Food.com data.

Shows:
  1. Top-5 recommendations for a real user with diet=vegetarian and max_minutes=45.
  2. Top-5 kitchen-mode recommendations for "chicken, garlic, yogurt, rice, onion".
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.items import get_item_name
from src.models.cf import ItemBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender
from src.recipe_features import (
    filter_recipes,
    kitchen_recommend,
    parse_ingredients_field,
    recommend_with_filters,
)


def run_demo():
    print("Loading Food.com datasets...")
    train_df = pd.read_csv("Data/processed/foodcom/train.csv")
    items_df = pd.read_csv("Data/processed/foodcom/items.csv")

    print("Fitting Content-Based model...")
    cb = ContentBased(items_df, text_column="text")
    cb.fit(train_df)

    print("Fitting ItemBasedCF model (cosine, k=30, min_overlap=2)...")
    cf = ItemBasedCF(similarity="cosine", k=30, min_overlap=2)
    cf.fit(train_df)

    print("Initializing Adaptive HybridRecommender (k0=100.0)...")
    hybrid = HybridRecommender(cf, cb, mode="adaptive", k0=100.0)
    hybrid.fit(train_df)

    # Pick an active real user from train_df
    user_counts = train_df["user_id"].value_counts()
    sample_user = user_counts.index[0]  # user with high engagement
    n_ratings = user_counts.iloc[0]
    print(f"\nSelected Real User ID: {sample_user} ({n_ratings} ratings in training data)")

    # -----------------------------------------------------------------------
    # Part 1: Top-5 Vegetarian Recipes with Prep Time <= 45 minutes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: Top-5 Recommendations for User with diet='vegetarian' & max_minutes=45")
    print("=" * 80)

    filters = {"diet": "vegetarian", "max_minutes": 45}
    veg_recs = recommend_with_filters(
        model=hybrid,
        user=sample_user,
        n=5,
        filters=filters,
        items_df=items_df,
    )

    items_lookup = items_df.set_index("item_id")
    for rank, (item_id, score) in enumerate(veg_recs, start=1):
        row = items_lookup.loc[item_id]
        name = row["name"]
        cal = row.get("calories", "N/A")
        mins = row.get("total_minutes", "N/A")
        ings = parse_ingredients_field(row.get("ingredients_list", []))
        print(f"\n{rank}. [{item_id}] {name}")
        print(f"   Score: {score:.4f} | Calories: {cal} | Total Minutes: {mins}")
        print(f"   Ingredients ({len(ings)}): {ings}")

    # -----------------------------------------------------------------------
    # Part 2: Top-5 Kitchen Mode Recommendations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: Top-5 Kitchen Mode Recommendations for:")
    print("        'chicken, garlic, yogurt, rice, onion'")
    print("=" * 80)

    pantry_query = "chicken, garlic, yogurt, rice, onion"
    kitchen_recs = kitchen_recommend(
        model=hybrid,
        user=sample_user,
        available_ingredients=pantry_query,
        n=5,
        min_overlap=0.5,
        alpha_kitchen=0.5,
        items_df=items_df,
    )

    for rank, rec in enumerate(kitchen_recs, start=1):
        item_id = rec["item_id"]
        row = items_lookup.loc[item_id] if item_id in items_lookup.index else {}
        cal = row.get("calories", "N/A") if isinstance(row, pd.Series) else "N/A"
        mins = row.get("total_minutes", "N/A") if isinstance(row, pd.Series) else "N/A"
        all_ings = parse_ingredients_field(row.get("ingredients_list", [])) if isinstance(row, pd.Series) else []

        print(f"\n{rank}. [{item_id}] {rec['name']}")
        print(f"   Final Score: {rec['final_score']:.4f} (Overlap: {rec['overlap']:.2f}, Hybrid Score: {rec['hybrid_score']})")
        print(f"   Total Ingredients ({len(all_ings)}): {all_ings}")
        print(f"   Missing ({len(rec['missing_ingredients'])}): {rec['missing_ingredients']}")
        print(f"   Explanation: {rec['explanation']}")


if __name__ == "__main__":
    run_demo()
