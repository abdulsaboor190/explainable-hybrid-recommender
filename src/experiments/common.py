"""Common experimental utilities, evaluation harness, and bootstrap confidence intervals.

Provides:
  - run_final_evaluation: builds models via pipeline.build_models(dataset, fit_on='train_val'),
    evaluates each on TEST using evaluate_model_per_user, and returns aggregated and per-user data.
  - save_tidy_results: records results to results/<experiment>_<dataset>.csv with metadata.
  - bootstrap_ci: resamples users with replacement to compute 95% confidence intervals.
  - build_item_vectors: helper to extract feature vectors for diversity calculation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.base import BaseRecommender
from src.config import DATASET_CONFIGS, DatasetConfig
from src.evaluation import evaluate_model_per_user
from src.pipeline import build_models, compute_config_hash, load_dataset_data


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    statistic_fn: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval resampling unit observations (e.g. users).

    Parameters
    ----------
    values : Sequence[float]
        1D array or list of per-unit metric values (e.g., per-user NDCG@10 or per-user squared error).
    n_boot : int, default=1000
        Number of bootstrap iterations.
    ci : float, default=0.95
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int, default=42
        Random seed for reproducibility.
    statistic_fn : callable, optional
        Function to compute the point estimate and bootstrap estimates.
        Defaults to np.mean. For RMSE from per-user squared errors, pass `lambda arr: np.sqrt(np.mean(arr))`.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    if statistic_fn is None:
        statistic_fn = np.mean

    point_estimate = float(statistic_fn(arr))

    rng = np.random.RandomState(seed)
    n = len(arr)
    # Generate bootstrap sample indices of shape (n_boot, n)
    boot_indices = rng.randint(0, n, size=(n_boot, n))
    boot_samples = arr[boot_indices]
    boot_stats = np.array([statistic_fn(sample) for sample in boot_samples])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_stats, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))

    return point_estimate, ci_lower, ci_upper


def build_item_vectors(dataset_name: str, items_df: pd.DataFrame) -> Dict[Any, np.ndarray]:
    """Build item feature vectors for intra-list diversity calculation."""
    ds = dataset_name.lower().strip()
    if ds == "movielens":
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        text_series = items_df["genre_tokens"].fillna("unknown") if "genre_tokens" in items_df.columns else items_df["title"]
        matrix = tfidf.fit_transform(text_series).toarray()
    else:
        tfidf = TfidfVectorizer(max_features=1000, token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b")
        text_series = items_df["text"].fillna("") if "text" in items_df.columns else items_df["name"]
        matrix = tfidf.fit_transform(text_series).toarray()

    id_col = "item_id" if "item_id" in items_df.columns else items_df.columns[0]
    return {item_id: matrix[idx] for idx, item_id in enumerate(items_df[id_col])}


def save_tidy_results(
    df: pd.DataFrame,
    experiment_name: str,
    dataset_name: str,
    config_hash: Optional[str] = None,
    results_dir: Union[str, Path] = "results",
) -> Path:
    """Save or append tidy experimental results to CSV with metadata."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_name}_{dataset_name}.csv"

    df_out = df.copy()
    if "timestamp" not in df_out.columns:
        df_out["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if "config_hash" not in df_out.columns and config_hash is not None:
        df_out["config_hash"] = config_hash

    df_out.to_csv(out_path, index=False)
    return out_path


def run_final_evaluation(
    dataset: Union[str, DatasetConfig],
    model_names: Optional[Sequence[str]] = None,
    ks: Sequence[int] = (5, 10),
    force_rebuild: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], str]:
    """Build models on train_val and evaluate on the TEST split.

    Returns:
        (aggregated_metrics_by_model, per_user_details_by_model, config_hash)
    """
    if isinstance(dataset, str):
        key = dataset.lower().strip()
        if key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS.keys())}")
        cfg = DATASET_CONFIGS[key]
    else:
        cfg = dataset

    train_val_df, items_df, _, test_df = load_dataset_data(cfg, fit_on="train_val")
    if test_df is None or len(test_df) == 0:
        raise ValueError(f"No test set found for dataset '{cfg.name}'")

    cfg_hash = compute_config_hash(cfg, fit_on="train_val", fit_df=train_val_df)
    models = build_models(cfg, fit_on="train_val", force_rebuild=force_rebuild)

    if model_names is None:
        target_keys = [
            "global_mean", "user_mean", "item_mean", "popularity",
            "user_cf", "item_cf", "content", "hybrid_fixed", "hybrid_adaptive"
        ]
    else:
        target_keys = list(model_names)

    item_vectors = build_item_vectors(cfg.name, items_df)
    total_catalogue_items = len(items_df)

    agg_results: Dict[str, Dict[str, float]] = {}
    per_user_results: Dict[str, Dict[str, Any]] = {}

    for key in target_keys:
        if key not in models:
            continue
        model = models[key]
        res, user_det = evaluate_model_per_user(
            model=model,
            train_df=train_val_df,
            test_df=test_df,
            ks=ks,
            relevant_threshold=cfg.liked_threshold,
            total_catalogue_items=total_catalogue_items,
            item_vectors=item_vectors,
        )
        agg_results[key] = res
        per_user_results[key] = user_det

    return agg_results, per_user_results, cfg_hash
