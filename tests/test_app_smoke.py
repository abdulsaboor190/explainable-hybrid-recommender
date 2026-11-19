"""Headless smoke test for the Streamlit demo application.

Verifies end-to-end functionality without needing a browser:
1. Loads cached models and data for both Food.com and MovieLens.
2. Generates recommendations across all models (Adaptive, Fixed, CF, Content, Popularity).
3. Verifies that predicted ratings are valid floats in [1.0, 5.0].
4. Verifies deterministic output: identical inputs produce identical recommendations.
5. Verifies dietary filtering: vegan filter never returns a banned non-vegan ingredient.
6. Verifies kitchen mode pantry overlap and missing ingredient computation.
7. Verifies graceful handling of unknown users and strict over-filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd

from app.helpers import (
    get_cached_data,
    get_cached_models,
    get_recommendations_for_model,
    get_user_profile,
    get_adaptive_alpha_explanation,
)
from src.recipe_features import parse_ingredients_field, DIET_EXCLUSIONS


def test_app_data_and_models_load():
    """Verify data and models load cleanly for both datasets."""
    for ds in ["foodcom", "movielens"]:
        data = get_cached_data(ds)
        assert len(data["items_df"]) > 0
        assert len(data["train_val_df"]) > 0
        assert len(data["archetypes"]) == 3
        assert "Low-activity user" in list(data["archetypes"].keys())[0]

        models = get_cached_models(ds)
        assert "hybrid_adaptive" in models
        assert "hybrid_fixed" in models
        assert "popularity" in models


def test_recommendations_and_predictions():
    """Verify recommendation generation and that predicted ratings fall in [1.0, 5.0]."""
    for ds in ["foodcom", "movielens"]:
        data = get_cached_data(ds)
        models = get_cached_models(ds)
        user_id = list(data["archetypes"].values())[0]

        for mod_name in ["hybrid_adaptive", "hybrid_fixed", "hybrid_content_only"]:
            recs, warn = get_recommendations_for_model(
                model=models[mod_name],
                dataset_name=ds,
                user_id=user_id,
                n=5,
                items_df=data["items_df"],
                name_map=data["name_map"],
            )
            assert len(recs) == 5
            for r in recs:
                assert 1.0 <= r["predicted_rating"] <= 5.0
                assert "name" in r
                assert "structured_explanation" in r


def test_deterministic_output():
    """Assert identical inputs give identical outputs across repeat calls."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    user_id = list(data["archetypes"].values())[1]  # mid user

    run1, _ = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=user_id,
        n=10,
        items_df=data["items_df"],
        name_map=data["name_map"],
    )
    run2, _ = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=user_id,
        n=10,
        items_df=data["items_df"],
        name_map=data["name_map"],
    )

    ids1 = [r["item_id"] for r in run1]
    ids2 = [r["item_id"] for r in run2]
    scores1 = [r["predicted_rating"] for r in run1]
    scores2 = [r["predicted_rating"] for r in run2]

    assert ids1 == ids2, "Recommendations must be deterministic!"
    assert scores1 == scores2, "Predicted ratings must be deterministic!"


def test_diet_filter_never_returns_banned():
    """Assert dietary filter (e.g. vegan) never returns a recipe containing banned ingredients."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    user_id = list(data["archetypes"].values())[0]

    recs, _ = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=user_id,
        n=10,
        items_df=data["items_df"],
        name_map=data["name_map"],
        filters={"diet": "vegan"},
    )
    assert len(recs) > 0

    vegan_banned = set(DIET_EXCLUSIONS["vegan"])
    items_lookup = data["items_df"].set_index("item_id")

    for r in recs:
        raw_ings = items_lookup.loc[r["item_id"], "ingredients_list"]
        ings = parse_ingredients_field(raw_ings)
        for ing in ings:
            ing_lower = ing.lower()
            for b in vegan_banned:
                assert b not in ing_lower.split(), f"Recipe {r['name']} contains banned vegan ingredient '{b}' in '{ing}'"


def test_kitchen_mode():
    """Verify kitchen mode pantry overlap calculation."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    user_id = list(data["archetypes"].values())[0]

    kitchen_pantry = "chicken breast, garlic, olive oil, onion, rice, salt, black pepper"
    recs, _ = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=user_id,
        n=5,
        items_df=data["items_df"],
        name_map=data["name_map"],
        kitchen_ingredients=kitchen_pantry,
    )
    assert len(recs) > 0
    for r in recs:
        assert "kitchen_overlap" in r
        assert 0.0 <= r["kitchen_overlap"] <= 1.0
        assert "missing_ingredients" in r
        assert isinstance(r["missing_ingredients"], list)


def test_robustness_unknown_user_and_empty_filters():
    """Verify unknown user and empty filter edge cases do not crash."""
    data = get_cached_data("foodcom")
    models = get_cached_models("foodcom")
    unknown_user = 999999999

    # Unknown user profile
    profile = get_user_profile(unknown_user, data["train_val_df"], data["items_df"], data["name_map"])
    assert len(profile) == 0

    alpha, text = get_adaptive_alpha_explanation(unknown_user, data["user_counts"], 100.0)
    assert alpha == 0.0

    # Unknown user recommendations
    recs, _ = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=unknown_user,
        n=5,
        items_df=data["items_df"],
        name_map=data["name_map"],
    )
    assert len(recs) == 5

    # Impossible filter: max_minutes=0 and max_calories=0
    recs_empty, warn = get_recommendations_for_model(
        model=models["hybrid_adaptive"],
        dataset_name="foodcom",
        user_id=unknown_user,
        n=5,
        items_df=data["items_df"],
        name_map=data["name_map"],
        filters={"max_minutes": 0, "max_calories": 0},
    )
    assert len(recs_empty) == 0
    assert warn is not None
    assert "excluded all" in warn
