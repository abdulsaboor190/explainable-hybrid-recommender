"""Baseline recommendation models inheriting from BaseRecommender."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.base import BaseRecommender


def _clip_rating(val: float) -> float:
    """Ensure rating prediction is strictly bounded in [1.0, 5.0]."""
    return float(np.clip(val, 1.0, 5.0))


class GlobalMean(BaseRecommender):
    """Predicts the overall global average rating for all user-item pairs."""

    def __init__(self) -> None:
        self.global_mean_: float = 3.0
        self.all_items_: List[Any] = []
        self.user_seen_: Dict[Any, Set[Any]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit global mean model."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))
        self.all_items_ = sorted(train_df["item_id"].unique().tolist())
        self.user_seen_ = (
            train_df.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

    def predict(self, user: Any, item: Any) -> float:
        """Predict the global mean rating."""
        return self.global_mean_

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Recommend items with highest predicted rating (constant global mean)."""
        seen = self.user_seen_.get(user, set()) if exclude_seen else set()
        cands = [item for item in self.all_items_ if item not in seen]
        return [(item, self.global_mean_) for item in cands[:n]]

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Explain recommendation."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation for recommendation."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)



class UserMean(BaseRecommender):
    """Predicts a user's average rating across all items they have rated.
    
    Fallback:
      - Unknown user -> Global training mean.
    """

    def __init__(self) -> None:
        self.global_mean_: float = 3.0
        self.user_means_: Dict[Any, float] = {}
        self.all_items_: List[Any] = []
        self.user_seen_: Dict[Any, Set[Any]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit user mean model."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))
        user_grp = train_df.groupby("user_id")["rating"].mean()
        self.user_means_ = {
            u: _clip_rating(float(m)) for u, m in user_grp.items()
        }
        self.all_items_ = sorted(train_df["item_id"].unique().tolist())
        self.user_seen_ = (
            train_df.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

    def predict(self, user: Any, item: Any) -> float:
        """Predict user mean rating with global mean fallback."""
        if user in self.user_means_:
            return self.user_means_[user]
        return self.global_mean_

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Recommend items ordered by predicted user mean."""
        pred = self.predict(user, None)
        seen = self.user_seen_.get(user, set()) if exclude_seen else set()
        cands = [item for item in self.all_items_ if item not in seen]
        return [(item, pred) for item in cands[:n]]

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Explain recommendation."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation for recommendation."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)



class ItemMean(BaseRecommender):
    """Predicts an item's average rating across all users who rated it.
    
    Fallback:
      - Unknown item -> User mean (if user known) -> Global training mean.
    """

    def __init__(self) -> None:
        self.global_mean_: float = 3.0
        self.item_means_: Dict[Any, float] = {}
        self.user_means_: Dict[Any, float] = {}
        self.all_items_: List[Any] = []
        self.user_seen_: Dict[Any, Set[Any]] = {}
        self.ranked_items_: List[Tuple[Any, float]] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit item mean model."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))
        item_grp = train_df.groupby("item_id")["rating"].mean()
        self.item_means_ = {
            i: _clip_rating(float(m)) for i, m in item_grp.items()
        }
        user_grp = train_df.groupby("user_id")["rating"].mean()
        self.user_means_ = {
            u: _clip_rating(float(m)) for u, m in user_grp.items()
        }
        self.all_items_ = sorted(train_df["item_id"].unique().tolist())
        self.user_seen_ = (
            train_df.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        # Pre-rank items by mean rating descending, tie-break by item_id
        self.ranked_items_ = sorted(
            self.item_means_.items(),
            key=lambda x: (-x[1], x[0])
        )

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating with ItemMean -> UserMean -> GlobalMean fallback."""
        if item in self.item_means_:
            return self.item_means_[item]
        if user in self.user_means_:
            return self.user_means_[user]
        return self.global_mean_

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Recommend items with highest item average rating."""
        seen = self.user_seen_.get(user, set()) if exclude_seen else set()
        recs: List[Tuple[Any, float]] = []
        for item, score in self.ranked_items_:
            if item not in seen:
                recs.append((item, score))
                if len(recs) == n:
                    break
        return recs

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Explain recommendation."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation for recommendation."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)



class Popularity(BaseRecommender):
    """Ranks items by popularity (number of training ratings).
    
    Ties broken by mean rating descending, then item_id ascending.
    predict() returns the item mean rating (with user mean -> global mean fallback).
    """

    def __init__(self) -> None:
        self.global_mean_: float = 3.0
        self.item_means_: Dict[Any, float] = {}
        self.user_means_: Dict[Any, float] = {}
        self.item_counts_: Dict[Any, int] = {}
        self.user_seen_: Dict[Any, Set[Any]] = {}
        self.ranked_items_: List[Any] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit popularity model."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))
        
        stats = train_df.groupby("item_id")["rating"].agg(["count", "mean"])
        self.item_counts_ = stats["count"].to_dict()
        self.item_means_ = {
            i: _clip_rating(float(m)) for i, m in stats["mean"].items()
        }
        
        user_grp = train_df.groupby("user_id")["rating"].mean()
        self.user_means_ = {
            u: _clip_rating(float(m)) for u, m in user_grp.items()
        }
        
        self.user_seen_ = (
            train_df.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        
        # Rank by count descending, mean descending, item_id ascending
        sorted_tuples = sorted(
            stats.itertuples(),
            key=lambda row: (-row.count, -row.mean, row.Index)
        )
        self.ranked_items_ = [row.Index for row in sorted_tuples]

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating falling back to item mean -> user mean -> global mean."""
        if item in self.item_means_:
            return self.item_means_[item]
        if user in self.user_means_:
            return self.user_means_[user]
        return self.global_mean_

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Recommend most frequently rated items."""
        seen = self.user_seen_.get(user, set()) if exclude_seen else set()
        recs: List[Tuple[Any, float]] = []
        for item in self.ranked_items_:
            if item not in seen:
                # Top recommendation returns (item, predicted_rating)
                pred_rating = self.predict(user, item)
                recs.append((item, pred_rating))
                if len(recs) == n:
                    break
        return recs

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Explain recommendation."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation for recommendation."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)


