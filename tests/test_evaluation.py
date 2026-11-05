"""Tests for evaluation module metrics with hand-calculated reference values."""

import pytest
import numpy as np
import pandas as pd

from src.base import BaseRecommender
from src.evaluation import (
    rmse,
    mae,
    compute_ndcg_at_k,
    catalogue_coverage,
    intra_list_diversity,
    ranking_metrics,
)


class DummyModel(BaseRecommender):
    """Dummy model for deterministic evaluation tests."""

    def __init__(self, pred_val: float = 3.0, rec_items: list = None):
        self.pred_val = pred_val
        self.rec_items = rec_items or []

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def predict(self, user: int, item: int) -> float:
        return self.pred_val

    def recommend(self, user: int, n: int = 10, exclude_seen: bool = True):
        return [(it, self.pred_val) for it in self.rec_items[:n]]

    def explain(self, user: int, item: int) -> str:
        return "Dummy"


def test_rmse_and_mae_known_values():
    """Verify RMSE and MAE against exact hand-calculated figures.
    
    Actuals: [2.0, 4.0, 5.0, 1.0]
    Predictions: [3.0, 3.0, 3.0, 3.0]
    Errors: [-1.0, 1.0, 2.0, -2.0]
    Squared errors: [1.0, 1.0, 4.0, 4.0] -> sum = 10.0, mean = 2.5 -> RMSE = sqrt(2.5) ~= 1.58113883
    Absolute errors: [1.0, 1.0, 2.0, 2.0] -> sum = 6.0, mean = 1.5 -> MAE = 1.5
    """
    model = DummyModel(pred_val=3.0)
    test_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 30],
        "rating": [2.0, 4.0, 5.0, 1.0]
    })
    
    calc_rmse = rmse(model, test_df)
    calc_mae = mae(model, test_df)
    
    assert np.isclose(calc_rmse, np.sqrt(2.5), atol=1e-5)
    assert np.isclose(calc_mae, 1.5, atol=1e-5)


def test_ndcg_perfect_and_zero():
    """Verify NDCG@K boundary cases.
    
    Case 1: Perfect ranking
    Relevant items = {1, 2, 3}
    Recommended = [1, 2, 3, 4, 5]
    At K=3:
      DCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63092975 + 0.5 = 2.13092975
      IDCG@3 = 2.13092975
      NDCG@3 = 1.0
      
    Case 2: All miss
    Recommended = [4, 5, 6]
    Hits = [0, 0, 0] -> NDCG = 0.0
    """
    rel = {1, 2, 3}
    perfect_recs = [1, 2, 3, 4, 5]
    zero_recs = [4, 5, 6, 7, 8]
    
    assert np.isclose(compute_ndcg_at_k(perfect_recs, rel, k=3), 1.0)
    assert np.isclose(compute_ndcg_at_k(zero_recs, rel, k=3), 0.0)


def test_precision_recall_toy_list():
    """Verify Precision@K and Recall@K on a 5-item toy list.
    
    User has relevant test items: {10, 20, 30} (3 items)
    Recommended top-5: [10, 99, 20, 88, 77]
    Hits at K=5: {10, 20} -> 2 hits
    Hits at K=3: {10, 20} -> 2 hits
    
    Precision@5 = 2 / 5 = 0.4
    Recall@5    = 2 / 3 = 0.6666667
    
    Precision@3 = 2 / 3 = 0.6666667
    Recall@3    = 2 / 3 = 0.6666667
    """
    model = DummyModel(rec_items=[10, 99, 20, 88, 77])
    train_df = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]})
    test_df = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_id": [10, 20, 30, 40],
        "rating": [5.0, 4.0, 4.5, 2.0]  # item 40 has rating < 4, so not relevant
    })
    
    metrics, _ = ranking_metrics(model, train_df, test_df, ks=(3, 5), relevant_threshold=4.0)
    
    assert np.isclose(metrics["precision@5"], 2.0 / 5.0)
    assert np.isclose(metrics["recall@5"], 2.0 / 3.0)
    assert np.isclose(metrics["precision@3"], 2.0 / 3.0)
    assert np.isclose(metrics["recall@3"], 2.0 / 3.0)


def test_catalogue_coverage_toy():
    """Verify catalogue coverage calculation.
    
    Catalogue size: 10 items (ids 1..10)
    User 1 recs: [1, 2, 3]
    User 2 recs: [2, 3, 4]
    Total unique recommended items = {1, 2, 3, 4} (4 items)
    Coverage = 4 / 10 = 0.40
    """
    recs = [
        [1, 2, 3],
        [2, 3, 4]
    ]
    cov = catalogue_coverage(recs, n_items=10)
    assert np.isclose(cov, 0.4)


def test_intra_list_diversity_toy():
    """Verify intra-list diversity (ILD) with orthogonal vectors.
    
    List of 2 items: item A = [1, 0], item B = [0, 1]
    Cosine similarity = 0.0 -> Cosine distance = 1.0 - 0.0 = 1.0
    ILD should equal 1.0.
    """
    recs = [[101, 102]]
    vectors = {
        101: np.array([1.0, 0.0]),
        102: np.array([0.0, 1.0])
    }
    ild = intra_list_diversity(recs, vectors)
    assert np.isclose(ild, 1.0)

