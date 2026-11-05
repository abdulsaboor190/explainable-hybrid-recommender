"""Tests for HybridRecommender model."""

import pytest
import numpy as np
import pandas as pd

from src.models.hybrid import HybridRecommender
from src.models.cf import ItemBasedCF
from src.models.content import ContentBased


class MockCF(ItemBasedCF):
    """Mock CF returning constant ratings."""
    def __init__(self, pred_val=4.0):
        self.pred_val = pred_val
        self.item2idx_ = {1: 0, 2: 1, 3: 2, 4: 3}
        self.idx2item_ = {0: 1, 1: 2, 2: 3, 3: 4}

    def predict(self, user, item):
        return self.pred_val

    def recommend(self, user, n=10, exclude_seen=True):
        return [(i, self.pred_val) for i in [1, 2, 3, 4][:n]]


class MockContent(ContentBased):
    """Mock Content model returning constant ratings."""
    def __init__(self, pred_val=2.0):
        self.pred_val = pred_val
        self.item_ids_ = [1, 2, 3, 4]
        self.item2idx_ = {1: 0, 2: 1, 3: 2, 4: 3}
        self.idx2item_ = {0: 1, 1: 2, 2: 3, 3: 4}
        self.user_profiles_ = {1: np.array([1.0, 0.0])}
        self.item_tfidf_ = np.eye(4)

    def predict(self, user, item):
        return self.pred_val

    def recommend(self, user, n=10, exclude_seen=True):
        return [(i, self.pred_val) for i in [1, 2, 3, 4][:n]]


def test_alpha_formula():
    """Verify adaptive alpha formula: alpha_u = n_u / (n_u + k0).
    
    Given k0 = 10:
      n_u = 0   -> 0 / (0 + 10) = 0.0
      n_u = 5   -> 5 / (5 + 10) = 5/15 = 0.33333333
      n_u = 50  -> 50 / (50 + 10) = 50/60 = 0.83333333
    """
    cf = MockCF()
    content = MockContent()
    hybrid = HybridRecommender(cf, content, mode="adaptive", k0=10.0)

    # Mock user counts directly
    hybrid.user_counts_ = {100: 0, 101: 5, 102: 50}

    assert np.isclose(hybrid.get_alpha(100), 0.0)
    assert np.isclose(hybrid.get_alpha(101), 5.0 / 15.0)
    assert np.isclose(hybrid.get_alpha(102), 50.0 / 60.0)

    # Unknown user (not in dictionary) -> 0.0
    assert np.isclose(hybrid.get_alpha(999), 0.0)


def test_fixed_mode_is_always_half():
    """Fixed mode must always return alpha = 0.5 regardless of rating count."""
    cf = MockCF()
    content = MockContent()
    hybrid = HybridRecommender(cf, content, mode="fixed", k0=10.0)
    hybrid.user_counts_ = {1: 0, 2: 5, 3: 100}

    assert hybrid.get_alpha(1) == 0.5
    assert hybrid.get_alpha(2) == 0.5
    assert hybrid.get_alpha(3) == 0.5


def test_cf_and_content_only_modes():
    """cf_only mode must match CF exactly, content_only must match Content exactly."""
    cf = MockCF(pred_val=4.5)
    content = MockContent(pred_val=2.5)

    h_cf = HybridRecommender(cf, content, mode="cf_only")
    h_cnt = HybridRecommender(cf, content, mode="content_only")

    assert np.isclose(h_cf.predict(1, 1), 4.5)
    assert np.isclose(h_cnt.predict(1, 1), 2.5)


def test_blended_prediction_bounds():
    """Blended predictions must remain in [1, 5]."""
    cf = MockCF(pred_val=5.5)  # will be clipped
    content = MockContent(pred_val=0.5)  # will be clipped
    hybrid = HybridRecommender(cf, content, mode="fixed")

    pred = hybrid.predict(1, 1)
    assert 1.0 <= pred <= 5.0


def test_recommend_excludes_seen():
    """recommend() must exclude seen items for hybrid models."""
    items_df = pd.DataFrame([
        {"item_id": 1, "text": "action thriller"},
        {"item_id": 2, "text": "comedy romance"},
        {"item_id": 3, "text": "drama crime"},
        {"item_id": 4, "text": "scifi action"},
    ])
    train_df = pd.DataFrame([
        {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 1, "item_id": 2, "rating": 3.0, "timestamp": 2},
    ])
    cf = ItemBasedCF(similarity="cosine", k=5, min_overlap=1)
    cf.fit(train_df)
    content = ContentBased(items_df=items_df, text_column="text")
    content.fit(train_df)

    hybrid = HybridRecommender(cf, content, mode="fixed")
    hybrid.fit(train_df)

    recs = hybrid.recommend(user=1, n=5, exclude_seen=True)
    rec_items = [it for it, _ in recs]
    assert 1 not in rec_items
    assert 2 not in rec_items

