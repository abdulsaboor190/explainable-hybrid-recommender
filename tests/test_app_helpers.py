"""Unit and regression tests for app helper functions and UI scenarios.

Covers:
- Recommendation helper returns requested number of items with non-empty explanations for 20 random users per dataset.
- Dietary filters never return banned ingredients.
- Kitchen mode with empty ingredient list handles gracefully without crashing.
- Unknown users fall back gracefully.
- Identical inputs produce strictly identical outputs (determinism).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from app.helpers import (
    get_adaptive_alpha_explanation,
    get_cached_data,
    get_cached_models,
    get_recommendations_for_model,
    get_user_profile,
)
from src.recipe_features import DIET_EXCLUSIONS, parse_ingredients_field


def test_random_users_recommendation_and_explanation():
    """Verify recommendation helper returns requested count and non-empty explanations for 20 random users per dataset."""
    rng = np.random.RandomState(42)

    for ds in ["movielens", "foodcom"]:
        data = get_cached_data(ds)
        models = get_cached_models(ds)
        model = models["hybrid_adaptive"]
        all_users = data["all_users"]

        # Pick 20 random users
        sample_users = rng.choice(all_users, size=20, replace=False)

        for u in sample_users:
            recs, warn = get_recommendations_for_model(
                model=model,
                dataset_name=ds,
                user_id=int(u),
                n=10,
                items_df=data["items_df"],
                name_map=data["name_map"],
            )
            assert len(recs) == 10, f"Expected 10 recs for User {u} in {ds}, got {len(recs)}"
            for r in recs:
                assert r["name"] and len(r["name"].strip()) > 0
                assert 1.0 <= r["predicted_rating"] <= 5.0
                struct = r["structured_explanation"]
                assert struct and len(struct.get("summary", "").strip()) > 0


def test_diet_filter_all_categories():
    """Verify that every dietary filter excludes all banned ingredients."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    model = models["hybrid_adaptive"]
    user_id = data["archetypes"]["Low-activity user (ID 18344, 17 ratings)"]
    items_lookup = data["items_df"].set_index("item_id")

    diets_to_test = ["vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free"]

    for d in diets_to_test:
        recs, warn = get_recommendations_for_model(
            model=model,
            dataset_name="foodcom",
            user_id=user_id,
            n=10,
            items_df=data["items_df"],
            name_map=data["name_map"],
            filters={"diet": d},
        )
        assert len(recs) > 0, f"Filter {d} returned 0 recipes"
        banned = set(DIET_EXCLUSIONS[d])

        for r in recs:
            raw_ings = items_lookup.loc[r["item_id"], "ingredients_list"]
            parsed = parse_ingredients_field(raw_ings)
            for ing in parsed:
                ing_tokens = set(ing.lower().split())
                for b in banned:
                    # check exact token or exact match
                    if b in ing_tokens or ing.lower() == b:
                        pytest.fail(f"Diet {d} recipe '{r['name']}' contains banned item '{b}' in ingredient '{ing}'")


def test_kitchen_mode_empty_input_graceful():
    """Verify kitchen mode with empty or whitespace-only input does not crash and defaults gracefully."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    model = models["hybrid_adaptive"]
    user_id = data["archetypes"]["Mid-activity user (ID 434952, 29 ratings)"]

    for empty_val in ["", "   ", None]:
        recs, warn = get_recommendations_for_model(
            model=model,
            dataset_name="foodcom",
            user_id=user_id,
            n=10,
            items_df=data["items_df"],
            name_map=data["name_map"],
            kitchen_ingredients=empty_val,
        )
        assert len(recs) == 10
        # should fall back to standard recommendation without crashing
        assert "kitchen_overlap" not in recs[0]


def test_unknown_user_fallback():
    """Verify unknown user falls back gracefully on both datasets."""
    for ds in ["movielens", "foodcom"]:
        data = get_cached_data(ds)
        models = get_cached_models(ds)
        unknown_id = 888888888

        # Profile is empty dataframe
        profile = get_user_profile(unknown_id, data["train_val_df"], data["items_df"], data["name_map"])
        assert len(profile) == 0

        # Alpha is 0.0
        alpha, text = get_adaptive_alpha_explanation(unknown_id, data["user_counts"], 20.0)
        assert alpha == 0.0
        assert "new/unseen user" in text

        # Recommendations return top-N
        recs, warn = get_recommendations_for_model(
            model=models["hybrid_adaptive"],
            dataset_name=ds,
            user_id=unknown_id,
            n=5,
            items_df=data["items_df"],
            name_map=data["name_map"],
        )
        assert len(recs) == 5
        for r in recs:
            assert 1.0 <= r["predicted_rating"] <= 5.0


def test_strict_determinism():
    """Verify identical inputs produce identical outputs across multiple calls."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    model = models["hybrid_adaptive"]
    user_id = list(data["archetypes"].values())[2]  # Heavy user

    recs_a, _ = get_recommendations_for_model(
        model=model,
        dataset_name="foodcom",
        user_id=user_id,
        n=10,
        items_df=data["items_df"],
        name_map=data["name_map"],
        filters={"diet": "vegetarian", "max_minutes": 45},
    )

    recs_b, _ = get_recommendations_for_model(
        model=model,
        dataset_name="foodcom",
        user_id=user_id,
        n=10,
        items_df=data["items_df"],
        name_map=data["name_map"],
        filters={"diet": "vegetarian", "max_minutes": 45},
    )

    assert len(recs_a) == len(recs_b)
    for ra, rb in zip(recs_a, recs_b):
        assert ra["item_id"] == rb["item_id"]
        assert ra["predicted_rating"] == rb["predicted_rating"]
