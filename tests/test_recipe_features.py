"""Tests for recipe-specific features: filters, kitchen mode, and constrained recommendations."""

import numpy as np
import pandas as pd
import pytest

from src.base import BaseRecommender
from src.recipe_features import (
    DEFAULT_PANTRY_STAPLES,
    filter_recipes,
    is_ingredient_covered,
    kitchen_recommend,
    normalize_ingredient,
    recommend_with_filters,
)


# ---------------------------------------------------------------------------
# Mock Recommender for Unit Testing
# ---------------------------------------------------------------------------

class MockRecommender(BaseRecommender):
    """Simple mock recommender with controllable user history and predictions."""

    def __init__(self, predictions=None, known_users=None, user_seen=None):
        self.predictions = predictions or {}
        self.user_counts_ = {u: 10 for u in (known_users or [])}
        self.user_seen_set_ = user_seen or {}

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def predict(self, user, item) -> float:
        return float(self.predictions.get((user, item), 3.5))

    def recommend(self, user, n=10, exclude_seen=True):
        seen = self.user_seen_set_.get(user, set()) if exclude_seen else set()
        items = [
            (it, score)
            for (u, it), score in self.predictions.items()
            if u == user and it not in seen
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def explain(self, user, item) -> str:
        return f"Mock explanation for {user} and {item}"


# ---------------------------------------------------------------------------
# Test 1: Vegetarian removes chicken recipe and keeps vegetable one
# ---------------------------------------------------------------------------

def test_vegetarian_removes_chicken_keeps_vegetable():
    df = pd.DataFrame([
        {
            "item_id": 1,
            "name": "Roast Chicken",
            "ingredients_list": ["chicken breast", "olive oil", "garlic", "rosemary"],
            "calories": 450,
            "total_minutes": 60,
        },
        {
            "item_id": 2,
            "name": "Steamed Broccoli & Tofu",
            "ingredients_list": ["broccoli", "tofu", "soy sauce", "sesame oil"],
            "calories": 250,
            "total_minutes": 20,
        },
    ])

    allowed = filter_recipes(df, diet="vegetarian")
    assert 2 in allowed, "Vegetable/tofu recipe should be allowed for vegetarian"
    assert 1 not in allowed, "Chicken recipe must be excluded for vegetarian"


# ---------------------------------------------------------------------------
# Test 2: 'peanut butter' blocked for nut_free but 'butternut squash' allowed
# ---------------------------------------------------------------------------

def test_nut_free_blocks_peanut_butter_allows_butternut_squash():
    df = pd.DataFrame([
        {
            "item_id": 10,
            "name": "PB&J Sandwich",
            "ingredients_list": ["peanut butter", "grape jelly", "white bread"],
            "calories": 350,
            "total_minutes": 5,
        },
        {
            "item_id": 20,
            "name": "Roasted Butternut Squash",
            "ingredients_list": ["butternut squash", "olive oil", "sea salt"],
            "calories": 180,
            "total_minutes": 40,
        },
    ])

    allowed = filter_recipes(df, diet="nut_free")
    assert 20 in allowed, "Butternut squash must be allowed under nut_free (not a tree nut)"
    assert 10 not in allowed, "Peanut butter must be blocked under nut_free"


# ---------------------------------------------------------------------------
# Test 3: Time and Calorie Caps (including missing value handling)
# ---------------------------------------------------------------------------

def test_time_and_calorie_caps():
    df = pd.DataFrame([
        {
            "item_id": 101,
            "name": "Quick Low-Cal Salad",
            "ingredients_list": ["lettuce", "cucumber", "vinegar"],
            "calories": 150.0,
            "total_minutes": 10.0,
        },
        {
            "item_id": 102,
            "name": "Slow High-Cal Feast",
            "ingredients_list": ["potatoes", "cheese", "cream"],
            "calories": 850.0,
            "total_minutes": 90.0,
        },
        {
            "item_id": 103,
            "name": "Missing Calories Recipe",
            "ingredients_list": ["apple", "cinnamon"],
            "calories": np.nan,
            "total_minutes": 15.0,
        },
        {
            "item_id": 104,
            "name": "Missing Time Recipe",
            "ingredients_list": ["banana", "honey"],
            "calories": 200.0,
            "total_minutes": np.nan,
        },
    ])

    # Calorie cap alone
    cal_filtered = filter_recipes(df, max_calories=500.0)
    assert 101 in cal_filtered
    assert 104 in cal_filtered
    assert 102 not in cal_filtered, "850 calories exceeds 500 cap"
    assert 103 not in cal_filtered, "Missing calorie value must be excluded when max_calories is active"

    # Time cap alone
    time_filtered = filter_recipes(df, max_minutes=30.0)
    assert 101 in time_filtered
    assert 103 in time_filtered
    assert 102 not in time_filtered, "90 minutes exceeds 30 cap"
    assert 104 not in time_filtered, "Missing time value must be excluded when max_minutes is active"

    # Inactive filters do NOT exclude recipes with missing values
    no_filter = filter_recipes(df)
    assert len(no_filter) == 4, "When filters are inactive, missing values should not be excluded"


# ---------------------------------------------------------------------------
# Test 4: Kitchen overlap computed by hand on a toy example
# ---------------------------------------------------------------------------

def test_kitchen_overlap_hand_computed():
    df = pd.DataFrame([
        {
            "item_id": 201,
            "name": "Tomato Basil Pasta",
            # 4 recipe ingredients: pasta, tomato, basil, garlic
            "ingredients_list": ["pasta", "tomato", "basil", "garlic"],
            "calories": 400,
            "total_minutes": 25,
        },
    ])

    # User has 3 of 4 ingredients: tomato, basil, garlic
    # We pass pantry_staples=[] so only user ingredients count
    user_ings = ["tomato", "basil", "garlic"]

    model = MockRecommender()
    recs = kitchen_recommend(
        model=model,
        user=999,  # Unknown user -> overlap only
        available_ingredients=user_ings,
        min_overlap=0.5,
        alpha_kitchen=0.5,
        items_df=df,
        pantry_staples=[],
    )

    assert len(recs) == 1
    rec = recs[0]
    # Hand calculation: 3 covered / 4 total = 0.75 exactly
    assert pytest.approx(rec["overlap"], 0.001) == 0.75
    assert pytest.approx(rec["final_score"], 0.001) == 0.75
    assert rec["item_id"] == 201


# ---------------------------------------------------------------------------
# Test 5: Missing ingredients listed correctly
# ---------------------------------------------------------------------------

def test_missing_ingredients_listed_correctly():
    df = pd.DataFrame([
        {
            "item_id": 301,
            "name": "Omelette",
            "ingredients_list": ["eggs", "cheddar cheese", "mushrooms", "bell pepper", "onion"],
            "calories": 300,
            "total_minutes": 15,
        },
    ])

    # User has eggs, cheese, onion
    user_ings = ["eggs", "cheddar cheese", "onion"]

    model = MockRecommender()
    recs = kitchen_recommend(
        model=model,
        user=999,
        available_ingredients=user_ings,
        min_overlap=0.5,
        items_df=df,
        pantry_staples=[],
    )

    assert len(recs) == 1
    missing = recs[0]["missing_ingredients"]
    assert "mushrooms" in missing
    assert "bell pepper" in missing
    assert "eggs" not in missing
    assert "cheddar cheese" not in missing
    assert "onion" not in missing
    assert "Matches 3 of 5 ingredients" in recs[0]["explanation"]


# ---------------------------------------------------------------------------
# Test 6: Empty ingredient input does not crash
# ---------------------------------------------------------------------------

def test_empty_ingredient_input_does_not_crash():
    df = pd.DataFrame([
        {
            "item_id": 401,
            "name": "Simple Tea",
            "ingredients_list": ["water", "tea bag"],
            "calories": 0,
            "total_minutes": 5,
        },
    ])

    model = MockRecommender()

    # Empty list
    recs_empty_list = kitchen_recommend(
        model=model,
        user=999,
        available_ingredients=[],
        items_df=df,
        pantry_staples=[],
    )
    assert isinstance(recs_empty_list, list)

    # Empty string
    recs_empty_str = kitchen_recommend(
        model=model,
        user=999,
        available_ingredients="",
        items_df=df,
        pantry_staples=[],
    )
    assert isinstance(recs_empty_str, list)

    # None
    recs_none = kitchen_recommend(
        model=model,
        user=999,
        available_ingredients=None,
        items_df=df,
        pantry_staples=[],
    )
    assert isinstance(recs_none, list)


# ---------------------------------------------------------------------------
# Test 7: recommend_with_filters keeps top-n full when enough recipes pass
# ---------------------------------------------------------------------------

def test_recommend_with_filters_full_top_n():
    # 20 recipes: even are vegetarian, odd are meat
    rows = []
    preds = {}
    u = 42
    for i in range(20):
        is_meat = (i % 2 == 1)
        rows.append({
            "item_id": i,
            "name": f"Recipe {i}",
            "ingredients_list": ["chicken", "rice"] if is_meat else ["tofu", "rice"],
            "calories": 200,
            "total_minutes": 20,
        })
        preds[(u, i)] = 5.0 - (i * 0.1)  # score: 5.0, 4.9, 4.8...

    df = pd.DataFrame(rows)
    model = MockRecommender(predictions=preds, known_users=[u])

    # Request top 5 vegetarian recipes
    recs = recommend_with_filters(
        model=model,
        user=u,
        n=5,
        filters={"diet": "vegetarian"},
        items_df=df,
    )

    # Exactly 5 recommendations returned
    assert len(recs) == 5
    # All 5 returned recipes must be vegetarian (even IDs: 0, 2, 4, 6, 8)
    for item_id, score in recs:
        assert item_id % 2 == 0, f"Item {item_id} should be vegetarian"

