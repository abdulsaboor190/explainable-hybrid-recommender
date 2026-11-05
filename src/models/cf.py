"""Collaborative Filtering models: UserBasedCF and ItemBasedCF.

Implements memory-based collaborative filtering using scipy.sparse matrices,
vectorized operations, and strict [1, 5] rating bounds with fallback chains.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.base import BaseRecommender


def _clip_rating(val: float) -> float:
    return float(np.clip(val, 1.0, 5.0))


class UserBasedCF(BaseRecommender):
    """User-based Collaborative Filtering with mean-centering and top-k neighbors.

    Parameters
    ----------
    similarity : str, default="cosine"
        Similarity metric: "cosine" (cosine on user-mean-centered vectors) or
        "pearson" (Pearson correlation computed strictly over co-rated items).
    k : int, default=30
        Neighborhood size (maximum number of similar users).
    min_overlap : int, default=5
        Minimum number of co-rated items required for a valid similarity.
    shrinkage : float, default=0.0
        Shrinkage parameter: sim_shrunk = sim * (overlap / (overlap + shrinkage)).
    """

    def __init__(
        self,
        similarity: str = "cosine",
        k: int = 30,
        min_overlap: int = 5,
        shrinkage: float = 0.0,
    ) -> None:
        if similarity not in ("cosine", "pearson"):
            raise ValueError(f"similarity must be 'cosine' or 'pearson', got {similarity}")
        self.similarity = similarity
        self.k = k
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage

        # Model state
        self.global_mean_: float = 3.0
        self.user2idx_: Dict[Any, int] = {}
        self.idx2user_: Dict[int, Any] = {}
        self.item2idx_: Dict[Any, int] = {}
        self.idx2item_: Dict[int, Any] = {}
        
        self.user_means_: np.ndarray = np.array([])
        self.user_mean_dict_: Dict[Any, float] = {}
        self.sim_matrix_: Optional[np.ndarray] = None  # (n_users, n_users)
        self.R_sparse_: Optional[sp.csr_matrix] = None  # (n_users, n_items)
        self.R_csc_: Optional[sp.csc_matrix] = None     # (n_users, n_items)
        self.R_centered_: Optional[sp.csr_matrix] = None  # (n_users, n_items)
        self.R_centered_csc_: Optional[sp.csc_matrix] = None
        self.user_seen_set_: Dict[Any, Set[Any]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit UserBasedCF model on training interactions."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))

        users = sorted(train_df["user_id"].unique())
        items = sorted(train_df["item_id"].unique())

        self.user2idx_ = {u: idx for idx, u in enumerate(users)}
        self.idx2user_ = {idx: u for idx, u in enumerate(users)}
        self.item2idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx2item_ = {idx: i for idx, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        u_idx = train_df["user_id"].map(self.user2idx_).to_numpy(dtype=np.int32)
        i_idx = train_df["item_id"].map(self.item2idx_).to_numpy(dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float64)

        # Build sparse matrix
        self.R_sparse_ = sp.csr_matrix((ratings, (u_idx, i_idx)), shape=(n_users, n_items))
        self.R_csc_ = self.R_sparse_.tocsc()

        # Compute user means
        user_sums = np.array(self.R_sparse_.sum(axis=1)).ravel()
        user_counts = np.diff(self.R_sparse_.indptr)
        user_counts_nonzero = np.maximum(user_counts, 1)
        self.user_means_ = user_sums / user_counts_nonzero
        self.user_mean_dict_ = {
            self.idx2user_[u]: _clip_rating(self.user_means_[u])
            for u in range(n_users) if user_counts[u] > 0
        }

        # Build centered sparse matrix: r_ui - mean(u)
        centered_data = np.empty_like(ratings)
        for u in range(n_users):
            start = self.R_sparse_.indptr[u]
            end = self.R_sparse_.indptr[u + 1]
            if start < end:
                centered_data[start:end] = ratings[start:end] - self.user_means_[u]
        self.R_centered_ = sp.csr_matrix((centered_data, (u_idx, i_idx)), shape=(n_users, n_items))
        self.R_centered_csc_ = self.R_centered_.tocsc()

        # Track seen items
        self.user_seen_set_ = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        # Compute user similarity matrix
        self._compute_similarity()

    def _compute_similarity(self) -> None:
        """Compute user-user similarity matrix."""
        n_users = len(self.user2idx_)
        # Indicator matrix for overlap computation
        R_bin = self.R_sparse_.copy()
        R_bin.data = np.ones_like(R_bin.data)
        # Overlap matrix: O[u, v] = number of co-rated items
        overlap_matrix = (R_bin @ R_bin.T).toarray()

        if self.similarity == "cosine":
            # Cosine on mean-centered vectors: (R_c @ R_c.T) / (norm_u * norm_v)
            # Dot products of centered vectors
            dot_matrix = (self.R_centered_ @ self.R_centered_.T).toarray()
            # Vector norms
            norms = np.sqrt(np.array(self.R_centered_.power(2).sum(axis=1)).ravel())
            norm_outer = np.outer(norms, norms)
            norm_outer[norm_outer == 0.0] = 1.0
            sim = dot_matrix / norm_outer

            # Apply min_overlap
            sim[overlap_matrix < self.min_overlap] = 0.0

            # Apply shrinkage if requested
            if self.shrinkage > 0.0:
                shrunk = overlap_matrix / (overlap_matrix + self.shrinkage)
                sim = sim * shrunk

            # Zero out diagonal
            np.fill_diagonal(sim, 0.0)
            self.sim_matrix_ = sim

        elif self.similarity == "pearson":
            # Exact Pearson over co-rated items:
            # For each pair (u, v) with co-rated items I_uv:
            # sum_{i in I_uv} (r_ui - mu_u_uv)*(r_vi - mu_v_uv) / (sqrt(sum (r_ui - mu_u_uv)^2) * sqrt(sum (r_vi - mu_v_uv)^2))
            sim = np.zeros((n_users, n_users), dtype=np.float64)
            
            # Efficient vectorized calculation across candidate pairs with overlap >= min_overlap
            # Find non-zero overlaps
            valid_u, valid_v = np.where(overlap_matrix >= self.min_overlap)
            # Keep upper triangle to avoid redundant work
            mask = valid_u < valid_v
            u_pairs = valid_u[mask]
            v_pairs = valid_v[mask]

            # Vectorized pairwise computation on valid pairs
            R_dense = self.R_sparse_.tocsr()
            for u, v in zip(u_pairs, v_pairs):
                items_u = R_dense.indices[R_dense.indptr[u]:R_dense.indptr[u+1]]
                items_v = R_dense.indices[R_dense.indptr[v]:R_dense.indptr[v+1]]
                common_items = np.intersect1d(items_u, items_v, assume_unique=True)
                n_co = len(common_items)
                if n_co < self.min_overlap:
                    continue
                
                # Fetch ratings on co-rated items
                u_pos = np.searchsorted(items_u, common_items)
                v_pos = np.searchsorted(items_v, common_items)
                ru = R_dense.data[R_dense.indptr[u] + u_pos]
                rv = R_dense.data[R_dense.indptr[v] + v_pos]
                
                mu_u = np.mean(ru)
                mu_v = np.mean(rv)
                du = ru - mu_u
                dv = rv - mu_v
                den = np.sqrt(np.sum(du**2)) * np.sqrt(np.sum(dv**2))
                if den > 1e-9:
                    val = float(np.sum(du * dv) / den)
                    if self.shrinkage > 0.0:
                        val *= (n_co / (n_co + self.shrinkage))
                    sim[u, v] = val
                    sim[v, u] = val

            self.sim_matrix_ = sim

    def get_contributions(self, user: Any, item: Any) -> List[Tuple[Any, float, float]]:
        """Return contributing neighbors sorted by absolute contribution.
        
        Returns:
            List of (neighbor_user, similarity, weighted_contribution)
        """
        if user not in self.user2idx_ or item not in self.item2idx_:
            return []

        u_idx = self.user2idx_[user]
        i_idx = self.item2idx_[item]

        # Find users who rated this item
        raters = self.R_csc_.indices[self.R_csc_.indptr[i_idx]:self.R_csc_.indptr[i_idx+1]]

        if len(raters) == 0:
            return []

        # Similarities to raters
        sims = self.sim_matrix_[u_idx, raters]
        
        # Consider only positive similarities or valid similarities
        # Filter top-k by similarity value
        valid_mask = np.abs(sims) > 1e-9
        if not np.any(valid_mask):
            return []

        raters = raters[valid_mask]
        sims = sims[valid_mask]

        # Top-k neighbors by similarity
        top_k_indices = np.argsort(-sims)[:self.k]
        top_raters = raters[top_k_indices]
        top_sims = sims[top_k_indices]

        denom = np.sum(np.abs(top_sims))
        if denom == 0.0:
            return []

        # Centered ratings: r_vi - mean(v)
        centered_ratings = np.array([
            self.R_centered_[v, i_idx] for v in top_raters
        ])

        contributions = (top_sims * centered_ratings) / denom

        results = [
            (self.idx2user_[v], float(s), float(c))
            for v, s, c in zip(top_raters, top_sims, contributions)
        ]
        results.sort(key=lambda x: -abs(x[2]))
        return results

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating bounded in [1.0, 5.0] with user mean and global mean fallback."""
        if user not in self.user2idx_:
            return self.global_mean_

        u_idx = self.user2idx_[user]
        u_mean = self.user_means_[u_idx]

        if item not in self.item2idx_:
            return _clip_rating(u_mean)

        i_idx = self.item2idx_[item]

        # Get users who rated item
        raters = self.R_csc_.indices[self.R_csc_.indptr[i_idx]:self.R_csc_.indptr[i_idx+1]]
        if len(raters) == 0:
            return _clip_rating(u_mean)

        sims = self.sim_matrix_[u_idx, raters]
        valid_mask = np.abs(sims) > 1e-9
        if not np.any(valid_mask):
            return _clip_rating(u_mean)

        raters = raters[valid_mask]
        sims = sims[valid_mask]

        # Top-k most similar users
        top_k_indices = np.argsort(-sims)[:self.k]
        top_raters = raters[top_k_indices]
        top_sims = sims[top_k_indices]

        denom = np.sum(np.abs(top_sims))
        if denom == 0.0:
            return _clip_rating(u_mean)

        centered_ratings = np.array([
            self.R_centered_[v, i_idx] for v in top_raters
        ])

        pred = u_mean + (np.sum(top_sims * centered_ratings) / denom)
        return _clip_rating(pred)

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Vectorized top-n recommendations for user."""
        if user not in self.user2idx_:
            # Unknown user: predict global mean
            seen = set()
            cands = [self.idx2item_[i] for i in range(len(self.item2idx_))]
            return [(it, self.global_mean_) for it in cands[:n]]

        u_idx = self.user2idx_[user]
        u_mean = self.user_means_[u_idx]

        # Similarities of all users to user u
        sims = self.sim_matrix_[u_idx]  # (n_users,)

        # Top-k neighbors overall
        top_user_indices = np.argsort(-sims)[:self.k]
        top_sims = sims[top_user_indices]
        valid = top_sims > 1e-9
        top_user_indices = top_user_indices[valid]
        top_sims = top_sims[valid]

        n_items = len(self.item2idx_)
        if len(top_user_indices) == 0:
            scores = np.full(n_items, u_mean, dtype=np.float64)
        else:
            # Submatrix of centered ratings for top neighbors: (k_valid, n_items)
            sub_R = self.R_centered_[top_user_indices, :].tocsc()
            # Binary indicator of who rated what among neighbors
            sub_bin = (sub_R != 0).astype(np.float64)
            # Denominators per item: sum_v |sim(u, v)| over raters of item i
            denoms = np.array(sub_bin.T @ np.abs(top_sims)).ravel()
            # Numerators per item: sum_v sim(u, v) * (r_vi - mean(v))
            numerators = np.array(sub_R.T @ top_sims).ravel()

            valid_items = denoms > 1e-9
            scores = np.full(n_items, u_mean, dtype=np.float64)
            scores[valid_items] = u_mean + (numerators[valid_items] / denoms[valid_items])

        scores = np.clip(scores, 1.0, 5.0)

        # Exclude seen items
        if exclude_seen:
            seen_items = self.user_seen_set_.get(user, set())
            for it in seen_items:
                if it in self.item2idx_:
                    scores[self.item2idx_[it]] = -1.0

        top_indices = np.argsort(-scores)[:n]
        return [
            (self.idx2item_[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] >= 1.0
        ]

    def explain(self, user: Any, item: Any) -> str:
        """Placeholder explanation based on neighbor consensus."""
        contribs = self.get_contributions(user, item)
        if contribs:
            top_neighbor, sim_val, _ = contribs[0]
            return f"Recommended by UserBasedCF ({self.similarity}) based on consensus from {len(contribs)} similar users, notably user {top_neighbor} (similarity {sim_val:.2f})."
        return f"Recommended using user average rating ({self.predict(user, item):.2f}) as no similar neighbors rated this item."

    def save(self, path: str | Path) -> None:
        """Serialize model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "UserBasedCF":
        """Deserialize model from disk using joblib."""
        return joblib.load(path)


class ItemBasedCF(BaseRecommender):
    """Item-based Collaborative Filtering with adjusted cosine or Pearson similarity.

    Parameters
    ----------
    similarity : str, default="cosine"
        "cosine" (Adjusted Cosine: centered by subtracting user mean across ratings) or
        "pearson" (Pearson correlation centered strictly by item means over co-rated users).
    k : int, default=30
        Neighborhood size (maximum number of similar items).
    min_overlap : int, default=5
        Minimum number of co-raters required for similarity to count.
    shrinkage : float, default=0.0
        Shrinkage parameter: sim_shrunk = sim * (overlap / (overlap + shrinkage)).
    """

    def __init__(
        self,
        similarity: str = "cosine",
        k: int = 30,
        min_overlap: int = 5,
        shrinkage: float = 0.0,
    ) -> None:
        if similarity not in ("cosine", "pearson"):
            raise ValueError(f"similarity must be 'cosine' or 'pearson', got {similarity}")
        self.similarity = similarity
        self.k = k
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage

        # Model state
        self.global_mean_: float = 3.0
        self.user2idx_: Dict[Any, int] = {}
        self.idx2user_: Dict[int, Any] = {}
        self.item2idx_: Dict[Any, int] = {}
        self.idx2item_: Dict[int, Any] = {}

        self.user_means_: np.ndarray = np.array([])
        self.item_means_: np.ndarray = np.array([])
        self.sim_matrix_: Optional[np.ndarray] = None  # (n_items, n_items)
        self.R_sparse_: Optional[sp.csr_matrix] = None  # (n_users, n_items)
        self.R_centered_: Optional[sp.csc_matrix] = None
        self.user_seen_set_: Dict[Any, Set[Any]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit ItemBasedCF model on training interactions."""
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))

        users = sorted(train_df["user_id"].unique())
        items = sorted(train_df["item_id"].unique())

        self.user2idx_ = {u: idx for idx, u in enumerate(users)}
        self.idx2user_ = {idx: u for idx, u in enumerate(users)}
        self.item2idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx2item_ = {idx: i for idx, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        u_idx = train_df["user_id"].map(self.user2idx_).to_numpy(dtype=np.int32)
        i_idx = train_df["item_id"].map(self.item2idx_).to_numpy(dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float64)

        self.R_sparse_ = sp.csr_matrix((ratings, (u_idx, i_idx)), shape=(n_users, n_items))

        # Compute user means and item means
        user_sums = np.array(self.R_sparse_.sum(axis=1)).ravel()
        user_counts = np.maximum(np.diff(self.R_sparse_.indptr), 1)
        self.user_means_ = user_sums / user_counts

        item_sums = np.array(self.R_sparse_.sum(axis=0)).ravel()
        R_csc = self.R_sparse_.tocsc()
        item_counts = np.maximum(np.diff(R_csc.indptr), 1)
        self.item_means_ = item_sums / item_counts

        # Build user-mean centered matrix for Adjusted Cosine: (r_ui - mean(u))
        centered_data = np.empty_like(ratings)
        for u in range(n_users):
            start = self.R_sparse_.indptr[u]
            end = self.R_sparse_.indptr[u + 1]
            if start < end:
                centered_data[start:end] = ratings[start:end] - self.user_means_[u]
        self.R_centered_ = sp.csc_matrix((centered_data, (u_idx, i_idx)), shape=(n_users, n_items))

        self.user_seen_set_ = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        self._compute_similarity()

    def _compute_similarity(self) -> None:
        """Compute item-item similarity matrix."""
        n_items = len(self.item2idx_)

        # Overlap matrix between items (co-rating users)
        R_bin = self.R_sparse_.tocsc().copy()
        R_bin.data = np.ones_like(R_bin.data)
        overlap_matrix = (R_bin.T @ R_bin).toarray()

        if self.similarity == "cosine":
            # Adjusted Cosine Similarity:
            # sim(i, j) = (R_c[:, i] . R_c[:, j]) / (||R_c[:, i]|| * ||R_c[:, j]||)
            # where ratings are centered by subtracting user mean
            dot_matrix = (self.R_centered_.T @ self.R_centered_).toarray()
            norms = np.sqrt(np.array(self.R_centered_.power(2).sum(axis=0)).ravel())
            norm_outer = np.outer(norms, norms)
            norm_outer[norm_outer == 0.0] = 1.0
            sim = dot_matrix / norm_outer

            sim[overlap_matrix < self.min_overlap] = 0.0
            if self.shrinkage > 0.0:
                shrunk = overlap_matrix / (overlap_matrix + self.shrinkage)
                sim = sim * shrunk

            np.fill_diagonal(sim, 0.0)
            self.sim_matrix_ = sim

        elif self.similarity == "pearson":
            # Exact Pearson correlation between item rating vectors over co-raters
            sim = np.zeros((n_items, n_items), dtype=np.float64)
            valid_i, valid_j = np.where(overlap_matrix >= self.min_overlap)
            mask = valid_i < valid_j
            i_pairs = valid_i[mask]
            j_pairs = valid_j[mask]

            R_csc = self.R_sparse_.tocsc()
            for i, j in zip(i_pairs, j_pairs):
                users_i = R_csc.indices[R_csc.indptr[i]:R_csc.indptr[i+1]]
                users_j = R_csc.indices[R_csc.indptr[j]:R_csc.indptr[j+1]]
                common_users = np.intersect1d(users_i, users_j, assume_unique=True)
                n_co = len(common_users)
                if n_co < self.min_overlap:
                    continue

                i_pos = np.searchsorted(users_i, common_users)
                j_pos = np.searchsorted(users_j, common_users)
                ri = R_csc.data[R_csc.indptr[i] + i_pos]
                rj = R_csc.data[R_csc.indptr[j] + j_pos]

                mu_i = np.mean(ri)
                mu_j = np.mean(rj)
                di = ri - mu_i
                dj = rj - mu_j
                den = np.sqrt(np.sum(di**2)) * np.sqrt(np.sum(dj**2))
                if den > 1e-9:
                    val = float(np.sum(di * dj) / den)
                    if self.shrinkage > 0.0:
                        val *= (n_co / (n_co + self.shrinkage))
                    sim[i, j] = val
                    sim[j, i] = val

            self.sim_matrix_ = sim

    def get_contributions(self, user: Any, item: Any) -> List[Tuple[Any, float, float]]:
        """Return contributing rated items sorted by absolute contribution.
        
        Returns:
            List of (neighbor_item, similarity, weighted_contribution)
        """
        if user not in self.user2idx_ or item not in self.item2idx_:
            return []

        u_idx = self.user2idx_[user]
        i_idx = self.item2idx_[item]

        # Items rated by user u
        u_row = self.R_sparse_.getrow(u_idx)
        rated_items = u_row.indices
        if len(rated_items) == 0:
            return []

        sims = self.sim_matrix_[i_idx, rated_items]
        valid_mask = np.abs(sims) > 1e-9
        if not np.any(valid_mask):
            return []

        rated_items = rated_items[valid_mask]
        sims = sims[valid_mask]

        top_k_indices = np.argsort(-sims)[:self.k]
        top_items = rated_items[top_k_indices]
        top_sims = sims[top_k_indices]

        denom = np.sum(np.abs(top_sims))
        if denom == 0.0:
            return []

        # Centered rating of user on item j: (r_uj - mean(j))
        centered_ratings = np.array([
            self.R_sparse_[u_idx, j] - self.item_means_[j] for j in top_items
        ])

        contributions = (top_sims * centered_ratings) / denom
        results = [
            (self.idx2item_[j], float(s), float(c))
            for j, s, c in zip(top_items, top_sims, contributions)
        ]
        results.sort(key=lambda x: -abs(x[2]))
        return results

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating using item means and neighbor deviations."""
        if item not in self.item2idx_:
            # Unknown item: fallback to user mean, then global mean
            if user in self.user2idx_:
                return _clip_rating(self.user_means_[self.user2idx_[user]])
            return self.global_mean_

        i_idx = self.item2idx_[item]
        i_mean = self.item_means_[i_idx]

        if user not in self.user2idx_:
            return _clip_rating(i_mean)

        u_idx = self.user2idx_[user]
        u_row = self.R_sparse_.getrow(u_idx)
        rated_items = u_row.indices
        if len(rated_items) == 0:
            return _clip_rating(i_mean)

        sims = self.sim_matrix_[i_idx, rated_items]
        valid_mask = np.abs(sims) > 1e-9
        if not np.any(valid_mask):
            return _clip_rating(i_mean)

        rated_items = rated_items[valid_mask]
        sims = sims[valid_mask]

        top_k_indices = np.argsort(-sims)[:self.k]
        top_items = rated_items[top_k_indices]
        top_sims = sims[top_k_indices]

        denom = np.sum(np.abs(top_sims))
        if denom == 0.0:
            return _clip_rating(i_mean)

        centered_ratings = np.array([
            self.R_sparse_[u_idx, j] - self.item_means_[j] for j in top_items
        ])

        pred = i_mean + (np.sum(top_sims * centered_ratings) / denom)
        return _clip_rating(pred)

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Vectorized item scoring across all candidate items."""
        if user not in self.user2idx_:
            # Fallback for unknown user: rank items by item mean
            sorted_items = np.argsort(-self.item_means_)[:n]
            return [(self.idx2item_[i], _clip_rating(self.item_means_[i])) for i in sorted_items]

        u_idx = self.user2idx_[user]
        u_row = self.R_sparse_.getrow(u_idx)
        rated_items = u_row.indices

        n_items = len(self.item2idx_)
        if len(rated_items) == 0:
            scores = np.copy(self.item_means_)
        else:
            # Centered ratings on user's rated items: (r_uj - mean(j))
            user_ratings = u_row.data
            centered_ratings = user_ratings - self.item_means_[rated_items]  # (n_rated,)

            # Submatrix of similarity between ALL items i and rated items j: (n_items, n_rated)
            sim_sub = self.sim_matrix_[:, rated_items]

            # Vectorized top-k neighbor selection per item row
            # If n_rated <= k, take all
            if len(rated_items) <= self.k:
                denoms = np.sum(np.abs(sim_sub), axis=1)
                numerators = sim_sub @ centered_ratings
            else:
                # Partition to keep only top-k similarities per row
                part_idx = np.argpartition(-sim_sub, self.k, axis=1)[:, :self.k]
                row_indices = np.arange(n_items)[:, None]
                top_k_sims = sim_sub[row_indices, part_idx]
                top_k_ratings = centered_ratings[part_idx]

                denoms = np.sum(np.abs(top_k_sims), axis=1)
                numerators = np.sum(top_k_sims * top_k_ratings, axis=1)

            valid = denoms > 1e-9
            scores = np.copy(self.item_means_)
            scores[valid] = self.item_means_[valid] + (numerators[valid] / denoms[valid])

        scores = np.clip(scores, 1.0, 5.0)

        if exclude_seen:
            seen_items = self.user_seen_set_.get(user, set())
            for it in seen_items:
                if it in self.item2idx_:
                    scores[self.item2idx_[it]] = -1.0

        top_indices = np.argsort(-scores)[:n]
        return [
            (self.idx2item_[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] >= 1.0
        ]

    def explain(self, user: Any, item: Any) -> str:
        """Placeholder explanation based on item similarities."""
        contribs = self.get_contributions(user, item)
        if contribs:
            top_item, sim_val, _ = contribs[0]
            return f"Recommended by ItemBasedCF ({self.similarity}) because it is similar to item {top_item} you rated (similarity {sim_val:.2f})."
        return f"Recommended based on item average ({self.predict(user, item):.2f})."

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ItemBasedCF":
        """Deserialize model from disk."""
        return joblib.load(path)
