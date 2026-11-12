"""Hybrid Recommender combining Collaborative Filtering and Content-Based models.

Supports modes:
  - "adaptive": alpha_u = n_u / (n_u + k0)
  - "fixed": alpha_u = 0.5
  - "cf_only": alpha_u = 1.0
  - "content_only": alpha_u = 0.0
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import joblib
import numpy as np
import pandas as pd

from src.base import BaseRecommender
from src.models.cf import UserBasedCF, ItemBasedCF
from src.models.content import ContentBased
from src.evaluation import rmse, ranking_metrics


def _clip_rating(val: float) -> float:
    return float(np.clip(val, 1.0, 5.0))


def _min_max_scale(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max scale array values over valid elements to [0, 1]."""
    out = np.zeros_like(arr, dtype=np.float64)
    if mask is None:
        valid_vals = arr
    else:
        valid_vals = arr[mask]

    if len(valid_vals) == 0:
        return out

    min_val = np.min(valid_vals)
    max_val = np.max(valid_vals)
    rng = max_val - min_val

    if rng > 1e-9:
        if mask is None:
            out = (arr - min_val) / rng
        else:
            out[mask] = (arr[mask] - min_val) / rng
    else:
        if mask is None:
            out.fill(0.5)
        else:
            out[mask] = 0.5
    return out


class HybridRecommender(BaseRecommender):
    """Hybrid Recommender combining Collaborative Filtering and Content-Based models.

    Parameters
    ----------
    cf_model : BaseRecommender
        Fitted collaborative filtering model (e.g. ItemBasedCF or UserBasedCF).
    content_model : ContentBased
        Fitted content-based model.
    mode : str, default="adaptive"
        Hybrid weighting mode: "adaptive", "fixed", "cf_only", or "content_only".
    k0 : float, default=10.0
        Half-saturation threshold for adaptive weighting: alpha_u = n_u / (n_u + k0).
    """

    def __init__(
        self,
        cf_model: BaseRecommender,
        content_model: ContentBased,
        mode: str = "adaptive",
        k0: float = 10.0,
    ) -> None:
        valid_modes = ("adaptive", "fixed", "cf_only", "content_only")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        self.cf_model = cf_model
        self.content_model = content_model
        self.mode = mode
        self.k0 = float(k0)

        # User training rating counts: user_id -> count
        self.user_counts_: Dict[Any, int] = {}
        self.global_mean_: float = 3.0
        self.user_seen_set_: Dict[Any, Set[Any]] = {}

        # Explanation context caching from last prediction/recommendation
        self.last_explanation_info_: Dict[str, Any] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Store interaction counts and reference state from train_df.

        (cf_model and content_model are expected to be fitted or are fitted here if uninitialized).
        """
        self.global_mean_ = _clip_rating(float(train_df["rating"].mean()))
        self.user_counts_ = train_df.groupby("user_id")["rating"].count().to_dict()
        self.user_seen_set_ = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    def get_alpha(self, user: Any) -> float:
        """Compute user-specific CF weight alpha_u in [0.0, 1.0]."""
        if self.mode == "cf_only":
            return 1.0
        if self.mode == "content_only":
            return 0.0
        if self.mode == "fixed":
            return 0.5
        if self.mode == "adaptive":
            n_u = self.user_counts_.get(user, 0)
            if n_u == 0:
                # Unknown user has 0 training ratings -> alpha = 0.0 (content/fallback only)
                return 0.0
            return float(n_u / (n_u + self.k0))
        return 0.5

    def predict(self, user: Any, item: Any) -> float:
        """Predict rating bounded in [1.0, 5.0] by blending CF and Content predictions."""
        alpha = self.get_alpha(user)

        if alpha >= 1.0:
            pred = self.cf_model.predict(user, item)
            dominant = "cf"
        elif alpha <= 0.0:
            pred = self.content_model.predict(user, item)
            dominant = "content"
        else:
            r_cf = self.cf_model.predict(user, item)
            r_content = self.content_model.predict(user, item)
            pred = alpha * r_cf + (1.0 - alpha) * r_content
            dominant = "cf" if alpha >= 0.5 else "content"

        self.last_explanation_info_ = {
            "user": user,
            "item": item,
            "alpha": alpha,
            "dominant": dominant,
        }
        return _clip_rating(pred)

    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate top-N hybrid recommendations for a user.

        Because CF scores (predicted ratings) and Content scores (cosine similarities)
        reside on different scales, for each user we min-max normalize each component's
        candidate item scores to [0, 1] before combining them via:
            score = alpha_u * norm(score_cf) + (1 - alpha_u) * norm(score_content)
        """
        alpha = self.get_alpha(user)
        seen_items = self.user_seen_set_.get(user, set()) if exclude_seen else set()

        # If pure ablation mode, bypass normalization
        if alpha >= 1.0:
            return self.cf_model.recommend(user, n=n, exclude_seen=exclude_seen)
        if alpha <= 0.0:
            return self.content_model.recommend(user, n=n, exclude_seen=exclude_seen)

        # Get all candidate items in catalogue order
        item_list = self.content_model.item_ids_
        n_items = len(item_list)
        unseen_mask = np.ones(n_items, dtype=bool)

        if exclude_seen and seen_items:
            for it in seen_items:
                if it in self.content_model.item2idx_:
                    unseen_mask[self.content_model.item2idx_[it]] = False

        if not np.any(unseen_mask):
            return []

        # 1. CF scores across candidates, aligned to content catalogue order.
        #
        # We always use vectorised paths for ItemBasedCF / UserBasedCF even when the CF
        # item catalogue is a subset of the content catalogue (e.g. CF has no cold-start
        # items).  We build cf_raw over all n_items slots (content catalogue) and fill
        # only the positions corresponding to CF-known items; the rest fall back to the
        # CF model's global/item/user mean.  This avoids the O(n_items) predict() loop
        # that caused ~240 s eval times.
        if isinstance(self.cf_model, ItemBasedCF):
            # Default: item mean for every catalogue position
            # Map content-catalogue index → CF item index (None if CF doesn't know it)
            cf_item_indices = np.array(
                [self.cf_model.item2idx_.get(it, -1) for it in item_list], dtype=np.int64
            )
            # Build cf_raw filled with item means where known, global mean elsewhere
            cf_raw = np.full(n_items, self.global_mean_, dtype=np.float64)
            known_mask = cf_item_indices >= 0
            cf_raw[known_mask] = self.cf_model.item_means_[cf_item_indices[known_mask]]

            if user in self.cf_model.user2idx_:
                u_idx = self.cf_model.user2idx_[user]
                u_row = self.cf_model.R_sparse_.getrow(u_idx)
                rated_cf_indices = u_row.indices   # indices in CF's item space
                if len(rated_cf_indices) > 0:
                    centered_ratings = u_row.data - self.cf_model.item_means_[rated_cf_indices]
                    # sim_matrix_ rows/cols are in CF item space
                    sim_sub = self.cf_model.sim_matrix_[:, rated_cf_indices]  # (n_cf_items, n_rated)
                    n_cf_items = len(self.cf_model.item2idx_)
                    if len(rated_cf_indices) <= self.cf_model.k:
                        denoms_cf = np.sum(np.abs(sim_sub), axis=1)
                        numerators_cf = sim_sub @ centered_ratings
                    else:
                        part_idx = np.argpartition(-sim_sub, self.cf_model.k, axis=1)[:, :self.cf_model.k]
                        row_indices = np.arange(n_cf_items)[:, None]
                        top_k_sims = sim_sub[row_indices, part_idx]
                        top_k_ratings = centered_ratings[part_idx]
                        denoms_cf = np.sum(np.abs(top_k_sims), axis=1)
                        numerators_cf = np.sum(top_k_sims * top_k_ratings, axis=1)
                    valid_cf = denoms_cf > 1e-9
                    # Write predictions back into content-catalogue positions
                    cf_known_positions = np.where(known_mask)[0]  # positions in content catalogue
                    cf_known_cf_idx = cf_item_indices[known_mask]  # corresponding CF indices
                    pred_cf = np.copy(self.cf_model.item_means_)
                    pred_cf[valid_cf] = self.cf_model.item_means_[valid_cf] + (
                        numerators_cf[valid_cf] / denoms_cf[valid_cf]
                    )
                    cf_raw[cf_known_positions] = pred_cf[cf_known_cf_idx]

        elif isinstance(self.cf_model, UserBasedCF):
            cf_raw = np.full(n_items, self.global_mean_, dtype=np.float64)
            if user in self.cf_model.user2idx_:
                u_idx = self.cf_model.user2idx_[user]
                u_mean = self.cf_model.user_means_[u_idx]
                sims = self.cf_model.sim_matrix_[u_idx]
                top_u = np.argsort(-sims)[:self.cf_model.k]
                top_u = top_u[sims[top_u] > 1e-9]
                if len(top_u) == 0:
                    cf_raw_cf_space = np.full(len(self.cf_model.item2idx_), u_mean)
                else:
                    sub_R = self.cf_model.R_centered_[top_u, :].tocsc()
                    sub_bin = (sub_R != 0).astype(np.float64)
                    denoms_cf = np.array(sub_bin.T @ np.abs(sims[top_u])).ravel()
                    numerators_cf = np.array(sub_R.T @ sims[top_u]).ravel()
                    valid_items_cf = denoms_cf > 1e-9
                    cf_raw_cf_space = np.full(len(self.cf_model.item2idx_), u_mean)
                    cf_raw_cf_space[valid_items_cf] = u_mean + (
                        numerators_cf[valid_items_cf] / denoms_cf[valid_items_cf]
                    )
                # Map CF item space → content catalogue positions
                for cat_pos, it in enumerate(item_list):
                    ci = self.cf_model.item2idx_.get(it, -1)
                    if ci >= 0:
                        cf_raw[cat_pos] = cf_raw_cf_space[ci]
            # else: unknown user → global mean already filled

        else:
            # Generic CF model: fall back to per-item predict() calls
            cf_raw = np.array([self.cf_model.predict(user, it) for it in item_list])

        # 2. Content scores across candidates
        if user in self.content_model.user_profiles_:
            profile = self.content_model.user_profiles_[user]
            content_raw = np.array(self.content_model.item_tfidf_ @ profile).ravel()
        else:
            content_raw = np.zeros(n_items, dtype=np.float64)

        # 3. Min-Max normalize over unseen candidates
        cf_norm = _min_max_scale(cf_raw, mask=unseen_mask)
        content_norm = _min_max_scale(content_raw, mask=unseen_mask)

        # 4. Blend
        hybrid_scores = alpha * cf_norm + (1.0 - alpha) * content_norm
        hybrid_scores[~unseen_mask] = -1.0

        top_indices = np.argsort(-hybrid_scores)[:n]
        return [
            (self.content_model.idx2item_[idx], float(hybrid_scores[idx]))
            for idx in top_indices
            if hybrid_scores[idx] >= 0.0
        ]

    def explain(self, user: Any, item: Any, **kwargs: Any) -> str:
        """Transparent, human-readable explanation combining CF and Content perspectives."""
        from src.explanations import explain as _explain
        return _explain(self, user, item, **kwargs)

    def explain_structured(self, user: Any, item: Any, **kwargs: Any) -> dict:
        """Structured explanation combining CF and Content perspectives."""
        from src.explanations import explain_structured as _explain_structured
        return _explain_structured(self, user, item, **kwargs)


    def alpha_distribution(self) -> Dict[str, Any]:
        """Compute summary statistics for alpha values across all known training users."""
        alphas = [self.get_alpha(u) for u in self.user_counts_.keys()]
        if not alphas:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "share_below_0_5": 0.0}
        arr = np.array(alphas)
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "share_below_0_5": float(np.mean(arr < 0.5)),
        }

    def tune_k0(
        self,
        val_df: pd.DataFrame,
        grid: Sequence[float] = (1, 2, 5, 10, 20, 50, 100),
        dataset_name: str = "dataset",
        save_results: bool = True,
    ) -> float:
        """Tune parameter k0 on validation split by lowest RMSE.

        Note: We select k0 strictly by lowest validation RMSE to preserve an objective,
        unbiased tuning criterion without fitting to test outcomes.
        """
        records = []
        original_k0 = self.k0
        original_mode = self.mode
        self.mode = "adaptive"

        best_rmse = float("inf")
        best_k0 = grid[0]

        train_df_dummy = pd.DataFrame(
            [(u, list(items)[0], 5.0) for u, items in self.user_seen_set_.items() if len(items) > 0],
            columns=["user_id", "item_id", "rating"]
        )

        for k0_val in grid:
            self.k0 = float(k0_val)
            val_rmse = rmse(self, val_df)
            rank_mets, _ = ranking_metrics(
                self, train_df=train_df_dummy, test_df=val_df, ks=(10,), relevant_threshold=4.0
            )
            ndcg_10 = rank_mets.get("ndcg@10", 0.0)

            records.append({
                "k0": k0_val,
                "val_rmse": val_rmse,
                "val_ndcg@10": ndcg_10,
            })
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_k0 = float(k0_val)

        df_grid = pd.DataFrame(records)
        if save_results:
            out_path = Path("results") / f"k0_tuning_{dataset_name}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_grid.to_csv(out_path, index=False)

        # Set best k0 and restore mode
        self.k0 = best_k0
        self.mode = original_mode
        return best_k0

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        """Deserialize model from disk."""
        return joblib.load(path)
