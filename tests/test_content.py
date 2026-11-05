"""Tests for ContentBased recommender model."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.content import ContentBased


@pytest.fixture
def toy_content_data():
    """Create a 5-item toy catalogue with distinct text attributes.
    
    Item 1: "italian pasta garlic olive_oil"
    Item 2: "italian pizza cheese tomato"
    Item 3: "asian noodles soy_sauce garlic"
    Item 4: "mexican tacos beef salsa"
    Item 5: "french dessert pastry chocolate"
    """
    items_df = pd.DataFrame([
        {"item_id": 1, "text": "italian pasta garlic olive_oil"},
        {"item_id": 2, "text": "italian pizza cheese tomato"},
        {"item_id": 3, "text": "asian noodles soy_sauce garlic"},
        {"item_id": 4, "text": "mexican tacos beef salsa"},
        {"item_id": 5, "text": "french dessert pastry chocolate"},
    ])

    # User 1 rated Item 1 (5.0, liked) and Item 2 (4.0, liked)
    # User 2 rated only Item 4 with rating 2.0 (no liked items!)
    train_df = pd.DataFrame([
        {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 1, "item_id": 2, "rating": 4.0, "timestamp": 2},
        {"user_id": 2, "item_id": 4, "rating": 2.0, "timestamp": 3},
    ])

    return items_df, train_df


def test_token_pattern_keeps_underscore(toy_content_data):
    """Verify that terms with underscores (like olive_oil) are treated as single tokens."""
    items_df, _ = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text")
    assert "olive_oil" in model.vocab_
    assert "soy_sauce" in model.vocab_


def test_content_recommendations_and_top_terms(toy_content_data):
    """Verify recommendation ranking and top shared terms for user 1.
    
    User 1 liked Item 1 (garlic, italian, olive_oil, pasta) and Item 2 (cheese, italian, pizza, tomato).
    Candidate unseen items are 3, 4, 5.
    Item 3 shares 'garlic' with user 1's profile.
    Items 4 and 5 share nothing.
    Therefore, Item 3 must be the top unseen recommendation!
    """
    items_df, train_df = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text", liked_threshold=4.0)
    model.fit(train_df)

    recs = model.recommend(user=1, n=3, exclude_seen=True)
    rec_items = [it for it, _ in recs]
    assert rec_items[0] == 3

    # Check top terms between User 1 and Item 3
    top_terms = model.get_top_terms(user=1, item=3, n=5)
    term_names = [t for t, _ in top_terms]
    assert "garlic" in term_names


def test_user_with_no_liked_items(toy_content_data):
    """A user with no ratings >= liked_threshold must not crash, and falls back gracefully."""
    items_df, train_df = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text", liked_threshold=4.0)
    model.fit(train_df)

    # User 2 rated only item 4 with rating 2.0 (< 4.0)
    recs = model.recommend(user=2, n=3, exclude_seen=True)
    assert len(recs) > 0
    # Predict rating must be in [1, 5]
    pred = model.predict(user=2, item=1)
    assert 1.0 <= pred <= 5.0


def test_recommend_excludes_seen(toy_content_data):
    """recommend() must never return seen items when exclude_seen=True."""
    items_df, train_df = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text")
    model.fit(train_df)

    recs = model.recommend(user=1, n=5, exclude_seen=True)
    rec_items = [it for it, _ in recs]
    assert 1 not in rec_items
    assert 2 not in rec_items


def test_predict_bounds_and_fallbacks(toy_content_data):
    """All predictions must reside in [1, 5] and unknown users/items must fall back properly."""
    items_df, train_df = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text")
    model.fit(train_df)

    # Unknown user -> global mean
    pred_unk_u = model.predict(user=999, item=1)
    assert np.isclose(pred_unk_u, train_df["rating"].mean())

    # Unknown item -> user mean
    u1_mean = train_df[train_df["user_id"] == 1]["rating"].mean()
    pred_unk_i = model.predict(user=1, item=999)
    assert np.isclose(pred_unk_i, u1_mean)

    for u in [1, 2, 999]:
        for i in [1, 2, 3, 4, 5, 999]:
            r = model.predict(u, i)
            assert 1.0 <= r <= 5.0
def test_save_and_load_content(toy_content_data, tmp_path):
    """Verify joblib save and load restores ContentBased model identically."""
    items_df, train_df = toy_content_data
    model = ContentBased(items_df=items_df, text_column="text")
    model.fit(train_df)

    save_path = tmp_path / "cb_model.joblib"
    model.save(save_path)

    loaded = ContentBased.load(save_path)
    assert np.allclose(model.user_profiles_[1], loaded.user_profiles_[1])
    assert np.isclose(model.predict(1, 3), loaded.predict(1, 3))
