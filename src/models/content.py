"""Content-Based Recommender System using TF-IDF item representations and user profiling."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.base import BaseRecommender


def _clip_rating(val: float) -> float:
    return float(np.clip(val, 1.0, 5.0))


class ContentBased(BaseRecommender):
    """Content-Based Recommender model based on TF-IDF item vectors and user profiles.

    Parameters
    ----------
    items_df : pd.DataFrame
        Catalogue dataframe containing 'item_id' and the specified text column.
    text_column : str
        Column name containing item text tokens (e.g. 'genre_tokens' or 'text').
    liked_threshold : float, default=4.0
        Threshold at or above which an interaction is considered 'liked'.
    k_content : int, default=30
        Number of most content-similar items used for rating prediction.
    min_df : int or float, default=1
        Minimum document frequency for TF-IDF vectorizer.
    ngram_range : tuple of (int, int), default=(1, 1)
        N-gram range for TF-IDF.
    sublinear_tf : bool, default=True
        Apply sublinear term frequency scaling (1 + log(tf)).
    """

    def __init__(
        self,
        items_df: pd.DataFrame,
        text_column: str = "text",
        liked_threshold: float = 4.0,
        k_content: int = 30,
        min_df: int | float = 1,
        ngram_range: Tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
    ) -> None:
        self.text_column = text_column
        self.liked_threshold = liked_threshold
        self.k_content = k_content
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

        # Fit TF-IDF on item text immediately
        # Token pattern keeps underscore-joined terms like olive_oil as one token
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b",
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
        )

        texts = items_df[text_column].fillna("").astype(str).tolist()
        self.item_tfidf_: sp.csr_matrix = self.vectorizer.fit_transform(texts)
        # Ensure L2 normalization per item vector
        self.item_tfidf_ = normalize(self.item_tfidf_, norm="l2", axis=1)

        # Mapping dictionaries
        self.item_ids_ = items_df["item_id"].tolist()
        self.item2idx_: Dict[Any, int] = {
            item_id: idx for idx, item_id in enumerate(self.item_ids_)
        }
        self.idx2item_: Dict[int, Any] = {
            idx: item_id for idx, item_id in enumerate(self.item_ids_)
        }
        self.vocab_: Dict[str, int] = self.vectorizer.vocabulary_
        self.idx2term_: Dict[int, str] = {
            idx: term for term, idx in self.vocab_.items()
        }

        # Training state
        self.global_mean_: float = 3.0
        self.user_means_: Dict[Any, float] = {}
        self.user_profiles_: Dict[Any, np.ndarray] = {}  # user_id -> 1D dense vector
        self.user_ratings_: Dict[Any, Dict[Any, float]] = {}  # user_id -> {item_id: rating}
        self.user_seen_set_: Dict[Any, Set[Any]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit ContentBased model on user interaction history."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))

        # Compute user means
        user_grp = train_df.groupby("user_id")["rating"].mean()
        self.user_means_ = {
            u: _clip_rating(float(m)) for u, m in user_grp.items()
        }

        # Store interaction ratings and seen set
        self.user_seen_set_ = (
            train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_ratings_ = {
            u: dict(zip(grp["item_id"], grp["rating"]))
            for u, grp in train_df.groupby("user_id")
        }

        n_terms = self.item_tfidf_.shape[1]

        # Build user profile for each user
        self.user_profiles_ = {}
        for user, ratings_dict in self.user_ratings_.items():
            # Identify liked items
            liked_items = [
                it for it, r in ratings_dict.items()
                if r >= self.liked_threshold and it in self.item2idx_
            ]

            if len(liked_items) > 0:
                item_indices = [self.item2idx_[it] for it in liked_items]
                weights = np.array([ratings_dict[it] for it in liked_items], dtype=np.float64)
            else:
                # If a user has no liked items, use all rated items weighted by rating
                valid_items = [it for it in ratings_dict.keys() if it in self.item2idx_]
                if len(valid_items) > 0:
                    item_indices = [self.item2idx_[it] for it in valid_items]
                    weights = np.array([ratings_dict[it] for it in valid_items], dtype=np.float64)
                else:
                    item_indices = []
                    weights = np.array([], dtype=np.float64)

            if len(item_indices) == 0:
                # Fallback: empty profile
                self.user_profiles_[user] = np.zeros(n_terms, dtype=np.float64)
            else:
                # Weighted average of TF-IDF vectors
                sub_matrix = self.item_tfidf_[item_indices, :].tocsc()
                profile = np.array(sub_matrix.T @ weights).ravel()
                norm = np.linalg.norm(profile)
                if norm > 0.0:
                    profile = profile / norm
                self.user_profiles_[user] = profile

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating using item-item content kNN centered by user mean."""
        if user not in self.user_means_:
            return self.global_mean_

        u_mean = self.user_means_[user]

        if item not in self.item2idx_:
            return u_mean

        target_idx = self.item2idx_[item]
        target_vec = self.item_tfidf_[target_idx]  # 1 x n_terms

        user_rated = self.user_ratings_.get(user, {})
        rated_in_catalogue = [it for it in user_rated.keys() if it in self.item2idx_]

        if len(rated_in_catalogue) == 0:
            return u_mean

        rated_indices = [self.item2idx_[it] for it in rated_in_catalogue]
        rated_ratings = np.array([user_rated[it] for it in rated_in_catalogue], dtype=np.float64)

        # Content similarity between target item and user's rated items
        rated_tfidf = self.item_tfidf_[rated_indices, :]  # n_rated x n_terms
        sims = (rated_tfidf @ target_vec.T).toarray().ravel()  # 1D numpy array

        # Filter positive/non-zero similarities
        valid_mask = sims > 1e-9
        if not np.any(valid_mask):
            return u_mean

        sims = sims[valid_mask]
        rated_ratings = rated_ratings[valid_mask]

        # Top-k most content-similar rated items
        top_k = min(self.k_content, len(sims))
        top_k_idx = np.argsort(-sims)[:top_k]

        top_sims = sims[top_k_idx]
        top_ratings = rated_ratings[top_k_idx]

        denom = np.sum(np.abs(top_sims))
        if denom == 0.0:
            return u_mean

        pred = u_mean + (np.sum(top_sims * (top_ratings - u_mean)) / denom)
        return _clip_rating(pred)

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Recommend items based on cosine similarity between user profile and candidate items."""
        if user not in self.user_profiles_:
            # Unknown user: predict global mean on catalogue
            seen = set()
            cands = self.item_ids_
            return [(it, self.global_mean_) for it in cands[:n]]

        profile = self.user_profiles_[user]  # 1D array of shape (n_terms,)
        if np.all(profile == 0.0):
            # Fallback to user mean
            u_mean = self.user_means_.get(user, self.global_mean_)
            cands = [it for it in self.item_ids_ if it not in self.user_seen_set_.get(user, set())]
            return [(it, u_mean) for it in cands[:n]]

        # Cosine similarity between profile and all items: (n_items, n_terms) @ (n_terms,)
        scores = np.array(self.item_tfidf_ @ profile).ravel()

        if exclude_seen:
            seen_items = self.user_seen_set_.get(user, set())
            for it in seen_items:
                if it in self.item2idx_:
                    scores[self.item2idx_[it]] = -1.0

        top_indices = np.argsort(-scores)[:n]
        return [
            (self.idx2item_[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -0.5
        ]

    def get_top_terms(self, user: Any, item: Any, n: int = 5) -> List[Tuple[str, float]]:
        """Return the TF-IDF terms with the highest elementwise product profile[term] * item_vector[term]."""
        if user not in self.user_profiles_ or item not in self.item2idx_:
            return []

        profile = self.user_profiles_[user]
        i_idx = self.item2idx_[item]
        item_row = self.item_tfidf_[i_idx].tocoo()

        # Compute term-level overlap products
        term_scores = []
        for term_col, item_val in zip(item_row.col, item_row.data):
            p_val = profile[term_col]
            prod = float(p_val * item_val)
            if prod > 0.0:
                term_scores.append((self.idx2term_[term_col], prod))

        term_scores.sort(key=lambda x: -x[1])
        return term_scores[:n]

    def get_similar_rated_items(
        self, user: Any, item: Any, n: int = 3
    ) -> List[Tuple[Any, float]]:
        """Return the user's rated items that are most content-similar to the target item."""
        if user not in self.user_ratings_ or item not in self.item2idx_:
            return []

        target_idx = self.item2idx_[item]
        target_vec = self.item_tfidf_[target_idx]

        user_rated = [it for it in self.user_ratings_[user].keys() if it in self.item2idx_]
        if not user_rated:
            return []

        rated_indices = [self.item2idx_[it] for it in user_rated]
        rated_tfidf = self.item_tfidf_[rated_indices, :]
        sims = (rated_tfidf @ target_vec.T).toarray().ravel()

        results = [(user_rated[i], float(sims[i])) for i in range(len(user_rated))]
        results.sort(key=lambda x: -x[1])
        return results[:n]

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Transparent, human-readable explanation matching content profile."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation matching content profile."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)


    def save(self, path: str | Path) -> None:
        """Serialize model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ContentBased":
        """Deserialize model from disk using joblib."""
        return joblib.load(path)
