"""Unit tests for explanation engine across all models."""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import GlobalMean, ItemMean, Popularity, UserMean
from src.models.cf import ItemBasedCF, UserBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender
from src.explanations import explain, explain_structured
from src.items import get_item_name_map


@pytest.fixture
def toy_data():
    """Toy catalogue and ratings for deterministic explanation testing."""
    items_df = pd.DataFrame([
        {"item_id": 1, "name": "Garlic Chicken", "text": "chicken garlic olive_oil"},
        {"item_id": 2, "name": "Lemon Pasta", "text": "lemon pasta garlic olive_oil"},
        {"item_id": 3, "name": "Tomato Soup", "text": "tomato basil soup"},
        {"item_id": 4, "name": "Roast Chicken", "text": "chicken rosemary garlic"},
        {"item_id": 5, "name": "Apple Tart", "text": "apple cinnamon sugar butter"},
    ])
    train_df = pd.DataFrame([
        # User 1 likes chicken and pasta
        {"user_id": 101, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 101, "item_id": 2, "rating": 4.0, "timestamp": 2},
        # User 2 also likes chicken and pasta (similar to User 1)
        {"user_id": 102, "item_id": 1, "rating": 5.0, "timestamp": 3},
        {"user_id": 102, "item_id": 2, "rating": 4.0, "timestamp": 4},
        {"user_id": 102, "item_id": 4, "rating": 5.0, "timestamp": 5},
        # User 3 likes desserts
        {"user_id": 103, "item_id": 5, "rating": 5.0, "timestamp": 6},
    ])
    item_name_map = {
        1: "Garlic Chicken",
        2: "Lemon Pasta",
        3: "Tomato Soup",
        4: "Roast Chicken",
        5: "Apple Tart",
    }
    return items_df, train_df, item_name_map


def test_item_cf_explanation(toy_data):
    items_df, train_df, name_map = toy_data
    model = ItemBasedCF(similarity="cosine", k=5, min_overlap=1)
    model.fit(train_df)

    # User 1 hasn't rated item 4 (Roast Chicken)
    text = model.explain(101, 4, item_name_map=name_map)
    assert isinstance(text, str)
    assert len(text) > 10
    # Must mention item name and stars, no math jargon like 'cosine'
    assert "cosine" not in text.lower()
    assert ("Garlic Chicken" in text or "Lemon Pasta" in text)
    assert ("stars" in text)
    assert text.startswith("Because you rated")

    # Structured output
    struct = model.explain_structured(101, 4, item_name_map=name_map)
    assert struct["component"] == "cf_item"
    assert struct["alpha"] == 1.0
    assert len(struct["reasons"]) > 0
    assert struct["reasons"][0]["type"] == "cf_item"
    assert len(struct["reasons"][0]["items"]) > 0


def test_user_cf_explanation(toy_data):
    items_df, train_df, name_map = toy_data
    model = UserBasedCF(similarity="cosine", k=5, min_overlap=1)
    model.fit(train_df)

    # User 1 evaluating item 4 (rated 5.0 by similar User 102)
    text = model.explain(101, 4, item_name_map=name_map)
    assert isinstance(text, str)
    assert "cosine" not in text.lower()
    assert "Users similar to you also rated this highly" in text
    assert "similar users" in text

    struct = model.explain_structured(101, 4, item_name_map=name_map)
    assert struct["component"] == "cf_user"
    assert struct["alpha"] == 1.0
    assert struct["reasons"][0]["similar_users_count"] >= 1


def test_content_explanation(toy_data):
    items_df, train_df, name_map = toy_data
    model = ContentBased(items_df, text_column="text", liked_threshold=4.0)
    model.fit(train_df)

    # User 1 on item 4 (shares chicken and garlic)
    text = model.explain(101, 4, item_name_map=name_map, item_type="recipe")
    assert isinstance(text, str)
    assert "cosine" not in text.lower()
    # Underscores should be converted to spaces
    assert "_" not in text
    assert ("chicken" in text or "garlic" in text)
    assert ("recipes you rated highly" in text)
    assert ("Garlic Chicken" in text or "Lemon Pasta" in text)

    struct = model.explain_structured(101, 4, item_name_map=name_map, item_type="recipe")
    assert struct["component"] == "content"
    assert struct["alpha"] == 0.0
    assert len(struct["reasons"][0]["terms"]) > 0


def test_hybrid_explanation(toy_data):
    items_df, train_df, name_map = toy_data
    cf = ItemBasedCF(similarity="cosine", k=5, min_overlap=1)
    cf.fit(train_df)
    cnt = ContentBased(items_df, text_column="text", liked_threshold=4.0)
    cnt.fit(train_df)

    hybrid = HybridRecommender(cf, cnt, mode="adaptive", k0=5.0)
    hybrid.fit(train_df)

    # User 101 has 2 ratings -> alpha = 2 / (2 + 5) = 2/7 ≈ 0.29 -> Content dominated
    text = hybrid.explain(101, 4, item_name_map=name_map, item_type="recipe")
    assert "You have rated 2 items" in text
    assert "we leaned mostly on item content" in text
    assert "weight" in text
    assert "cosine" not in text.lower()

    struct = hybrid.explain_structured(101, 4, item_name_map=name_map, item_type="recipe")
    assert struct["component"] == "hybrid"
    assert abs(struct["alpha"] - 2.0 / 7.0) < 1e-4
    assert struct["n_ratings"] == 2
    assert len(struct["reasons"]) >= 1


def test_cold_start_fallbacks_never_crash(toy_data):
    items_df, train_df, name_map = toy_data
    cf = ItemBasedCF(similarity="cosine", k=5, min_overlap=1)
    cf.fit(train_df)
    cnt = ContentBased(items_df, text_column="text", liked_threshold=4.0)
    cnt.fit(train_df)
    hybrid = HybridRecommender(cf, cnt, mode="adaptive", k0=5.0)
    hybrid.fit(train_df)

    # Completely unobserved user 999 on unobserved item 888
    models = [
        GlobalMean(),
        UserMean(),
        ItemMean(),
        Popularity(),
        cf,
        cnt,
        hybrid,
    ]
    for m in models:
        m.fit(train_df)
        text = m.explain(999, 888, item_name_map=name_map)
        assert isinstance(text, str)
        assert len(text) > 0
        assert not text.isspace()
        # Must acknowledge lack of personal history honestly
        assert ("not enough" in text.lower() or "catalog" in text.lower() or "average" in text.lower() or "popular" in text.lower() or "no rating history" in text.lower())

        struct = m.explain_structured(999, 888, item_name_map=name_map)
        assert isinstance(struct, dict)
        assert "component" in struct
        assert "reasons" in struct


def test_baseline_explanations(toy_data):
    _, train_df, name_map = toy_data
    pop = Popularity()
    pop.fit(train_df)
    text = pop.explain(101, 1, item_name_map=name_map)
    assert "popular" in text.lower()
    assert "ratings" in text.lower()

    im = ItemMean()
    im.fit(train_df)
    text = im.explain(101, 1, item_name_map=name_map)
    assert "average rating" in text.lower()

    um = UserMean()
    um.fit(train_df)
    text = um.explain(101, 1, item_name_map=name_map)
    assert "average rating tendency" in text.lower()


def test_real_dataset_50_random_pairs():
    """Verify explain() returns non-empty string for 50 random pairs on MovieLens and Food.com."""
    # 1. MovieLens
    ml_train = pd.read_csv("data/processed/movielens/train.csv")
    ml_items = pd.read_csv("data/processed/movielens/items.csv")
    ml_name_map = dict(zip(ml_items["item_id"], ml_items["title"]))

    cf_ml = ItemBasedCF(similarity="cosine", k=10, min_overlap=2)
    cf_ml.fit(ml_train)
    cnt_ml = ContentBased(ml_items, text_column="genre_tokens")
    cnt_ml.fit(ml_train)
    hybrid_ml = HybridRecommender(cf_ml, cnt_ml, mode="adaptive", k0=20.0)
    hybrid_ml.fit(ml_train)

    rng = np.random.default_rng(42)
    sample_users = rng.choice(ml_train["user_id"].unique(), size=50, replace=True)
    sample_items = rng.choice(ml_items["item_id"].unique(), size=50, replace=True)

    for u, i in zip(sample_users, sample_items):
        exp_str = hybrid_ml.explain(u, i, item_name_map=ml_name_map, item_type="movie")
        assert isinstance(exp_str, str)
        assert len(exp_str.strip()) > 0
        assert " nan" not in exp_str.lower()
        assert "none" not in exp_str.lower()

    # 2. Food.com sample
    fc_train = pd.read_csv("data/processed/foodcom/train.csv")
    fc_items = pd.read_csv("data/processed/foodcom/items.csv")
    fc_name_map = dict(zip(fc_items["item_id"], fc_items["name"]))

    cf_fc = ItemBasedCF(similarity="cosine", k=10, min_overlap=2)
    cf_fc.fit(fc_train)
    cnt_fc = ContentBased(fc_items, text_column="text")
    cnt_fc.fit(fc_train)
    hybrid_fc = HybridRecommender(cf_fc, cnt_fc, mode="adaptive", k0=100.0)
    hybrid_fc.fit(fc_train)

    sample_fc_users = rng.choice(fc_train["user_id"].unique(), size=50, replace=True)
    sample_fc_items = rng.choice(fc_items["item_id"].unique(), size=50, replace=True)

    for u, i in zip(sample_fc_users, sample_fc_items):
        exp_str = hybrid_fc.explain(u, i, item_name_map=fc_name_map, item_type="recipe")
        assert isinstance(exp_str, str)
        assert len(exp_str.strip()) > 0
        assert " nan" not in exp_str.lower()
        assert "none" not in exp_str.lower()
