"""Evaluation module for explainable recommender systems.

Provides rating prediction metrics (RMSE, MAE, cold-item RMSE) and
ranking metrics (Precision@K, Recall@K, NDCG@K, Catalogue Coverage, Intra-List Diversity).
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from src.base import BaseRecommender


def rmse(model: BaseRecommender, test_df: pd.DataFrame) -> float:
    """Calculate Root Mean Squared Error (RMSE) with predictions clipped to [1, 5]."""
    if len(test_df) == 0:
        return 0.0
    preds = np.array([
        np.clip(model.predict(u, i), 1.0, 5.0)
        for u, i in zip(test_df["user_id"], test_df["item_id"])
    ])
    actuals = test_df["rating"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def mae(model: BaseRecommender, test_df: pd.DataFrame) -> float:
    """Calculate Mean Absolute Error (MAE) with predictions clipped to [1, 5]."""
    if len(test_df) == 0:
        return 0.0
    preds = np.array([
        np.clip(model.predict(u, i), 1.0, 5.0)
        for u, i in zip(test_df["user_id"], test_df["item_id"])
    ])
    actuals = test_df["rating"].to_numpy(dtype=float)
    return float(np.mean(np.abs(preds - actuals)))


def compute_ndcg_at_k(recommended_items: Sequence[Any], relevant_set: Set[Any], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K for binary relevance."""
    top_k = recommended_items[:k]
    # Binary hits
    hits = np.array([1.0 if item in relevant_set else 0.0 for item in top_k])
    if hits.sum() == 0.0:
        return 0.0
    
    # DCG = sum_{i=1}^K hit_i / log2(i + 1)
    discounts = np.log2(np.arange(2, len(hits) + 2))
    dcg = float(np.sum(hits / discounts))
    
    # Ideal DCG: min(|relevant|, k) ones at top
    ideal_hits = min(len(relevant_set), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def catalogue_coverage(
    all_recommendations: Sequence[Sequence[Any]], n_items: int
) -> float:
    """Compute catalogue coverage: fraction of catalogue items appearing in any top-K recommendation list."""
    if n_items <= 0:
        return 0.0
    unique_rec_items: Set[Any] = set()
    for rec_list in all_recommendations:
        for item in rec_list:
            unique_rec_items.add(item)
    return float(len(unique_rec_items) / n_items)


def intra_list_diversity(
    all_recommendations: Sequence[Sequence[Any]],
    item_vectors: Dict[Any, np.ndarray],
) -> float:
    """Compute Intra-List Diversity (ILD) as mean pairwise cosine distance within each top-K list."""
    ilds = []
    for rec_list in all_recommendations:
        # Collect feature vectors for recommended items
        vecs = [item_vectors[it] for it in rec_list if it in item_vectors]
        if len(vecs) < 2:
            continue
        matrix = np.vstack(vecs)
        # Check if vectors are zero
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_matrix = matrix / norms
        
        # Pairwise cosine distance = 1.0 - cosine_similarity
        distances = pdist(normalized_matrix, metric="cosine")
        # Handle small floating point artifacts
        distances = np.nan_to_num(distances, nan=0.0)
        ilds.append(float(np.mean(distances)))
        
    return float(np.mean(ilds)) if ilds else 0.0


def ranking_metrics(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ks: Sequence[int] = (5, 10),
    relevant_threshold: float = 4.0,
    seen_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], List[List[Any]]]:
    """Compute Precision@K, Recall@K, NDCG@K for candidates not in seen_df (train + optional val).
    
    Returns:
        metrics: Dict with averages and count of skipped users.
        top_k_recs_max: List of recommended item lists (at max(ks)) for users evaluated.
    """
    max_k = max(ks)
    
    # Combine seen items per user
    seen_source = train_df if seen_df is None else pd.concat([train_df, seen_df], ignore_index=True)
    seen_per_user: Dict[Any, Set[Any]] = (
        seen_source.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    
    # Ground truth relevant items per user in test
    relevant_test = test_df[test_df["rating"] >= relevant_threshold]
    relevant_per_user: Dict[Any, Set[Any]] = (
        relevant_test.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    
    test_users = test_df["user_id"].unique()
    
    precisions: Dict[int, List[float]] = {k: [] for k in ks}
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
    
    skipped_users = 0
    evaluated_recs: List[List[Any]] = []
    
    for u in test_users:
        rel_items = relevant_per_user.get(u, set())
        if len(rel_items) == 0:
            skipped_users += 1
            continue
        
        # User recommendations excluding seen items
        recs_with_scores = model.recommend(u, n=max_k, exclude_seen=True)
        rec_items = [it for it, _ in recs_with_scores]
        evaluated_recs.append(rec_items)
        
        for k in ks:
            top_k = rec_items[:k]
            hits = sum(1 for it in top_k if it in rel_items)
            precisions[k].append(hits / k)
            recalls[k].append(hits / len(rel_items))
            ndcgs[k].append(compute_ndcg_at_k(rec_items, rel_items, k))
            
    res: Dict[str, float] = {}
    for k in ks:
        res[f"precision@{k}"] = float(np.mean(precisions[k])) if precisions[k] else 0.0
        res[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        res[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        
    res["users_evaluated"] = float(len(test_users) - skipped_users)
    res["users_skipped_no_relevant"] = float(skipped_users)
    
    return res, evaluated_recs


def evaluate_model(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ks: Sequence[int] = (5, 10),
    relevant_threshold: float = 4.0,
    seen_df: Optional[pd.DataFrame] = None,
    total_catalogue_items: Optional[int] = None,
    item_vectors: Optional[Dict[Any, np.ndarray]] = None,
) -> Dict[str, float]:
    """Execute complete standardized evaluation on test set."""
    # 1. Rating prediction metrics
    results: Dict[str, float] = {
        "rmse": rmse(model, test_df),
        "mae": mae(model, test_df),
    }
    
    # 2. Cold-start item evaluation
    train_items = set(train_df["item_id"].unique())
    cold_test = test_df[~test_df["item_id"].isin(train_items)]
    results["cold_items_count"] = float(len(cold_test))
    if len(cold_test) > 0:
        results["cold_rmse"] = rmse(model, cold_test)
    else:
        results["cold_rmse"] = 0.0
        
    # 3. Ranking metrics
    rank_metrics, rec_lists = ranking_metrics(
        model=model,
        train_df=train_df,
        test_df=test_df,
        ks=ks,
        relevant_threshold=relevant_threshold,
        seen_df=seen_df,
    )
    results.update(rank_metrics)
    
    # 4. Coverage & Diversity
    if total_catalogue_items is not None:
        for k in ks:
            recs_at_k = [r[:k] for r in rec_lists]
            results[f"coverage@{k}"] = catalogue_coverage(recs_at_k, total_catalogue_items)
            
    if item_vectors is not None:
        for k in ks:
            recs_at_k = [r[:k] for r in rec_lists]
            results[f"diversity@{k}"] = intra_list_diversity(recs_at_k, item_vectors)
            
    return results
