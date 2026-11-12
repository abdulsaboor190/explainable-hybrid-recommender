"""Pipeline and Model Orchestration for Explainable Hybrid Recommender.

Provides reproducible dataset loading, model fitting, and caching:
  - build_models(dataset, fit_on="train" | "train_val"): builds all baselines,
    chosen CF model, ContentBased, and all HybridRecommender variants.
  - Caches fitted model objects under data/cache/<dataset>/ keyed by a deterministic
    hash of the dataset configuration and training split.
  - Enables instant startup for demo applications and final evaluation pipelines.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import pandas as pd

from src.base import BaseRecommender
from src.config import DATASET_CONFIGS, DatasetConfig
from src.models.baselines import GlobalMean, ItemMean, Popularity, UserMean
from src.models.cf import ItemBasedCF, UserBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender


def _resolve_data_dir(cfg: DatasetConfig) -> Path:
    """Resolve data directory handling case variations (data vs Data)."""
    if cfg.data_dir.exists():
        return cfg.data_dir
    alt_path = Path(str(cfg.data_dir).replace("data/", "Data/").replace("data\\", "Data\\"))
    if alt_path.exists():
        return alt_path
    return cfg.data_dir


def load_dataset_data(
    dataset: Union[str, DatasetConfig],
    fit_on: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load interaction splits and items catalogue for a dataset.

    Parameters
    ----------
    dataset : Union[str, DatasetConfig]
        Dataset name ("movielens", "foodcom") or DatasetConfig object.
    fit_on : str, default="train"
        Which split(s) to prepare for model fitting:
        - "train": returns train_df as the training dataset.
        - "train_val": concatenates train_df and val_df to refit on train+val
          for final test evaluation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
        (fit_df, items_df, val_df, test_df)
    """
    if isinstance(dataset, str):
        key = dataset.lower().strip()
        if key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS.keys())}")
        cfg = DATASET_CONFIGS[key]
    else:
        cfg = dataset

    data_dir = _resolve_data_dir(cfg)
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"
    items_path = data_dir / "items.csv"

    if not train_path.exists() or not items_path.exists():
        raise FileNotFoundError(f"Missing required data files in {data_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    items_df = pd.read_csv(items_path)

    if fit_on == "train":
        fit_df = train_df
    elif fit_on == "train_val":
        if val_df is None:
            raise FileNotFoundError(f"Cannot fit on 'train_val': {val_path} not found.")
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        raise ValueError(f"Unsupported fit_on='{fit_on}'. Must be 'train' or 'train_val'.")

    return fit_df, items_df, val_df, test_df


def compute_config_hash(
    cfg: DatasetConfig,
    fit_on: str,
    fit_df: pd.DataFrame,
) -> str:
    """Generate a deterministic SHA-256 hash of dataset config and interaction data."""
    config_dict = {
        "dataset_name": cfg.name,
        "fit_on": fit_on,
        "text_column": cfg.text_column,
        "liked_threshold": cfg.liked_threshold,
        "cf_type": cfg.cf_type,
        "cf_similarity": cfg.cf_similarity,
        "cf_k": cfg.cf_k,
        "min_overlap": cfg.min_overlap,
        "shrinkage": cfg.shrinkage,
        "k_content": cfg.k_content,
        "k0": cfg.k0,
        "data_len": len(fit_df),
        "data_ratings_sum": float(fit_df["rating"].sum()),
    }
    dumped = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def build_models(
    dataset: Union[str, DatasetConfig],
    fit_on: str = "train",
    force_rebuild: bool = False,
    cache_dir: Optional[Path] = None,
) -> Dict[str, BaseRecommender]:
    """Build and fit all recommender models (or load from cache if config matches).

    Parameters
    ----------
    dataset : Union[str, DatasetConfig]
        Dataset identifier ("movielens" or "foodcom") or DatasetConfig object.
    fit_on : str, default="train"
        Training interaction split convention:
        - "train": fit on training set only (for validation tuning and evaluation).
        - "train_val": fit on train + validation sets combined (for final test evaluation).
    force_rebuild : bool, default=False
        If True, ignores existing cache files and refits all models.
    cache_dir : Optional[Path], default=None
        Custom cache directory path. If None, defaults to data/cache/<dataset>/.

    Returns
    -------
    Dict[str, BaseRecommender]
        Dictionary mapping model names to fitted recommender instances:
          - 'global_mean': GlobalMean
          - 'user_mean': UserMean
          - 'item_mean': ItemMean
          - 'popularity': Popularity
          - 'cf': Best Collaborative Filtering model (e.g. ItemBasedCF)
          - 'content': ContentBased TF-IDF model
          - 'hybrid_cf_only': HybridRecommender (mode="cf_only")
          - 'hybrid_content_only': HybridRecommender (mode="content_only")
          - 'hybrid_fixed': HybridRecommender (mode="fixed", alpha=0.5)
          - 'hybrid_adaptive': HybridRecommender (mode="adaptive", tuned k0)
    """
    if isinstance(dataset, str):
        key = dataset.lower().strip()
        if key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS.keys())}")
        cfg = DATASET_CONFIGS[key]
    else:
        cfg = dataset

    if fit_on not in ("train", "train_val"):
        raise ValueError(f"fit_on must be 'train' or 'train_val', got '{fit_on}'")

    # 1. Load data
    fit_df, items_df, _, _ = load_dataset_data(cfg, fit_on=fit_on)

    # 2. Compute configuration hash
    cfg_hash = compute_config_hash(cfg, fit_on=fit_on, fit_df=fit_df)

    # 3. Determine cache path
    if cache_dir is None:
        cache_base = Path("data/cache") / cfg.name
    else:
        cache_base = Path(cache_dir) / cfg.name
    cache_base.mkdir(parents=True, exist_ok=True)
    cache_file = cache_base / f"models_{fit_on}.joblib"
    meta_file = cache_base / f"meta_{fit_on}.json"

    # 4. Check if valid cache exists
    if not force_rebuild and cache_file.exists() and meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("config_hash") == cfg_hash:
                loaded_models = joblib.load(cache_file)
                if isinstance(loaded_models, dict) and "hybrid_adaptive" in loaded_models:
                    return loaded_models
        except Exception:
            # Corrupted cache -> proceed to rebuild
            pass

    # 5. Fit all models from scratch
    # A. Baselines
    gm = GlobalMean()
    gm.fit(fit_df)

    um = UserMean()
    um.fit(fit_df)

    im = ItemMean()
    im.fit(fit_df)

    pop = Popularity()
    pop.fit(fit_df)

    # B. Collaborative Filtering
    if cfg.cf_type == "user_based":
        cf_model: BaseRecommender = UserBasedCF(
            similarity=cfg.cf_similarity,
            k=cfg.cf_k,
            min_overlap=cfg.min_overlap,
            shrinkage=cfg.shrinkage,
        )
    else:
        cf_model = ItemBasedCF(
            similarity=cfg.cf_similarity,
            k=cfg.cf_k,
            min_overlap=cfg.min_overlap,
            shrinkage=cfg.shrinkage,
        )
    cf_model.fit(fit_df)

    # C. Content-Based
    content_model = ContentBased(
        items_df=items_df,
        text_column=cfg.text_column,
        liked_threshold=cfg.liked_threshold,
        k_content=cfg.k_content,
    )
    content_model.fit(fit_df)

    # D. Hybrid Recommender Variants
    hyb_cf = HybridRecommender(cf_model, content_model, mode="cf_only")
    hyb_cf.fit(fit_df)

    hyb_content = HybridRecommender(cf_model, content_model, mode="content_only")
    hyb_content.fit(fit_df)

    hyb_fixed = HybridRecommender(cf_model, content_model, mode="fixed")
    hyb_fixed.fit(fit_df)

    hyb_adaptive = HybridRecommender(cf_model, content_model, mode="adaptive", k0=cfg.k0)
    hyb_adaptive.fit(fit_df)

    models_dict: Dict[str, BaseRecommender] = {
        "global_mean": gm,
        "user_mean": um,
        "item_mean": im,
        "popularity": pop,
        "cf": cf_model,
        "content": content_model,
        "hybrid_cf_only": hyb_cf,
        "hybrid_content_only": hyb_content,
        "hybrid_fixed": hyb_fixed,
        "hybrid_adaptive": hyb_adaptive,
    }

    # 6. Save models and metadata to cache
    try:
        joblib.dump(models_dict, cache_file, compress=3)
        meta_data = {
            "dataset": cfg.name,
            "fit_on": fit_on,
            "config_hash": cfg_hash,
            "timestamp": time.time(),
            "models": list(models_dict.keys()),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save models cache to {cache_file}: {e}")

    return models_dict
