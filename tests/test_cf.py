"""Tests for UserBasedCF and ItemBasedCF collaborative filtering models."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.cf import UserBasedCF, ItemBasedCF


@pytest.fixture
def toy_matrix():
    """Create a toy 4-user x 5-item rating dataset.
    
    Ratings table:
       i1  i2  i3  i4  i5
    u1  5   3   .   1   .
    u2  4   .   .   1   .
    u3  .   4   2   .   5
    u4  1   .   .   5   .
    
    Let's check user 1 (u1): ratings = {i1:5, i2:3, i4:1}, mean(u1) = 3.0
    u2: {i1:4, i4:1}, mean(u2) = 2.5
    u3: {i2:4, i3:2, i5:5}, mean(u3) = 3.6667
    u4: {i1:1, i4:5}, mean(u4) = 3.0
    
    Overlap between u1 and u2: {i1, i4} (2 items)
    Overlap between u1 and u4: {i1, i4} (2 items)
    """
    rows = [
        {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 1, "item_id": 2, "rating": 3.0, "timestamp": 2},
        {"user_id": 1, "item_id": 4, "rating": 1.0, "timestamp": 3},
        {"user_id": 2, "item_id": 1, "rating": 4.0, "timestamp": 4},
        {"user_id": 2, "item_id": 4, "rating": 1.0, "timestamp": 5},
        {"user_id": 3, "item_id": 2, "rating": 4.0, "timestamp": 6},
        {"user_id": 3, "item_id": 3, "rating": 2.0, "timestamp": 7},
        {"user_id": 3, "item_id": 5, "rating": 5.0, "timestamp": 8},
        {"user_id": 4, "item_id": 1, "rating": 1.0, "timestamp": 9},
        {"user_id": 4, "item_id": 4, "rating": 5.0, "timestamp": 10},
    ]
    return pd.DataFrame(rows)


def test_hand_calculated_user_cf(toy_matrix):
    """Hand-verify UserBasedCF prediction on toy matrix.
    
    Users: u1, u2, u4 all rated i1 and i4.
    Mean(u1) = (5 + 3 + 1)/3 = 3.0. Centered: i1: +2, i2: 0, i4: -2.
    Mean(u2) = (4 + 1)/2 = 2.5. Centered: i1: +1.5, i4: -1.5.
    Mean(u4) = (1 + 5)/2 = 3.0. Centered: i1: -2.0, i4: +2.0.
    
    Cosine similarity on centered vectors (with min_overlap=2):
      u1 centered: [2, 0, 0, -2, 0], norm = sqrt(4 + 4) = sqrt(8)
      u2 centered: [1.5, 0, 0, -1.5, 0], norm = sqrt(2.25 + 2.25) = sqrt(4.5)
      dot(u1, u2) = 2*1.5 + (-2)*(-1.5) = 3 + 3 = 6
      sim(u1, u2) = 6 / (sqrt(8) * sqrt(4.5)) = 6 / sqrt(36) = 6 / 6 = 1.0!
      
      u4 centered: [-2, 0, 0, 2, 0], norm = sqrt(8)
      dot(u1, u4) = 2*(-2) + (-2)*2 = -4 - 4 = -8
      sim(u1, u4) = -8 / 8 = -1.0.
      
    Now, predict for u1 on item 4 (which u2 and u4 also rated):
      Top-k=1 with highest similarity: neighbor is u2 (sim = 1.0).
      Rating of u2 on i4: r(u2, i4) = 1.0. Centered: 1.0 - 2.5 = -1.5.
      r_hat(u1, i4) = mean(u1) + sim(u1, u2) * (-1.5) / |sim(u1, u2)|
                    = 3.0 + 1.0 * (-1.5) / 1.0 = 1.5.
    """
    model = UserBasedCF(similarity="cosine", k=1, min_overlap=2)
    model.fit(toy_matrix)
    
    pred = model.predict(user=1, item=4)
    assert np.isclose(pred, 1.5, atol=1e-4)


def test_cosine_vs_pearson_toy_data():
    """Cosine and Pearson must produce different similarities / predictions on asymmetric data."""
    # Build dataset where user 1 rated 3 items: [5, 4, 1] (mean = 3.333)
    # User 2 rated 3 items: [4, 4, 1] (mean = 3.0)
    # User 3 rated [1, 2, 4]
    rows = [
        {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 1, "item_id": 2, "rating": 4.0, "timestamp": 2},
        {"user_id": 1, "item_id": 3, "rating": 1.0, "timestamp": 3},
        {"user_id": 2, "item_id": 1, "rating": 4.0, "timestamp": 4},
        {"user_id": 2, "item_id": 2, "rating": 2.0, "timestamp": 5},
        {"user_id": 2, "item_id": 4, "rating": 5.0, "timestamp": 6},  # item 4 rated by u2 but not u1
    ]
    df = pd.DataFrame(rows)
    cf_cos = UserBasedCF(similarity="cosine", k=2, min_overlap=2)
    cf_pea = UserBasedCF(similarity="pearson", k=2, min_overlap=2)
    
    cf_cos.fit(df)
    cf_pea.fit(df)
    
    # Cosine centers by entire user mean (including unshared item 4 for u2)
    # Pearson centers ONLY by co-rated items (items 1 and 2)
    sim_cos = cf_cos.sim_matrix_[0, 1]
    sim_pea = cf_pea.sim_matrix_[0, 1]
    assert not np.isclose(sim_cos, sim_pea)


def test_rating_bounds_and_fallbacks(toy_matrix):
    """Verify that predictions are strictly in [1.0, 5.0] and fallbacks function correctly."""
    model = UserBasedCF(similarity="cosine", k=5, min_overlap=1)
    model.fit(toy_matrix)

    # 1. Unknown user fallback -> global mean
    global_mean = toy_matrix["rating"].mean()
    pred_unknown_user = model.predict(user=999, item=1)
    assert np.isclose(pred_unknown_user, global_mean)

    # 2. Unknown item fallback -> user mean
    u1_mean = toy_matrix[toy_matrix["user_id"] == 1]["rating"].mean()
    pred_unknown_item = model.predict(user=1, item=999)
    assert np.isclose(pred_unknown_item, u1_mean)

    # 3. Both unknown
    pred_both_unknown = model.predict(user=999, item=999)
    assert np.isclose(pred_both_unknown, global_mean)

    # 4. Strictly in [1, 5]
    for u in [1, 2, 3, 4, 999]:
        for i in [1, 2, 3, 4, 5, 999]:
            r = model.predict(u, i)
            assert 1.0 <= r <= 5.0


def test_recommend_excludes_seen(toy_matrix):
    """Vectorized recommend() must never return items already in the user's training set."""
    model = UserBasedCF(similarity="cosine", k=2, min_overlap=1)
    model.fit(toy_matrix)

    user = 1
    seen_items = set(toy_matrix[toy_matrix["user_id"] == user]["item_id"])
    recs = model.recommend(user=user, n=5, exclude_seen=True)

    rec_items = [it for it, _ in recs]
    assert len(seen_items.intersection(rec_items)) == 0

    # Test item-based version
    item_model = ItemBasedCF(similarity="cosine", k=2, min_overlap=1)
    item_model.fit(toy_matrix)
    recs_item = item_model.recommend(user=user, n=5, exclude_seen=True)
    rec_items_item = [it for it, _ in recs_item]
    assert len(seen_items.intersection(rec_items_item)) == 0


def test_save_and_load_cf(toy_matrix, tmp_path):
    """Verify joblib save and load restores model state identically."""
    model = ItemBasedCF(similarity="cosine", k=2, min_overlap=1)
    model.fit(toy_matrix)

    save_path = tmp_path / "item_cf.joblib"
    model.save(save_path)

    loaded = ItemBasedCF.load(save_path)
    assert np.allclose(model.sim_matrix_, loaded.sim_matrix_)
    assert np.isclose(model.predict(1, 3), loaded.predict(1, 3))
