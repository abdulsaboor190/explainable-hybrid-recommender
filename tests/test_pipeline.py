"""Unit tests for pipeline loading, model building, caching, and convention splits."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.base import BaseRecommender
from src.config import DATASET_CONFIGS, DatasetConfig
from src.pipeline import build_models, compute_config_hash, load_dataset_data


def test_load_dataset_data_train_and_train_val():
    # Test on MovieLens
    train_df, items_df, val_df, test_df = load_dataset_data("movielens", fit_on="train")
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(items_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) > 0
    assert len(items_df) > 0

    # Test train_val combines train and val splits
    train_val_df, _, _, _ = load_dataset_data("movielens", fit_on="train_val")
    assert len(train_val_df) == len(train_df) + len(val_df)


def test_compute_config_hash_deterministic():
    cfg = DATASET_CONFIGS["movielens"]
    train_df, _, _, _ = load_dataset_data(cfg, fit_on="train")

    hash1 = compute_config_hash(cfg, "train", train_df)
    hash2 = compute_config_hash(cfg, "train", train_df)
    assert hash1 == hash2, "Config hash must be deterministic"

    # Different fit_on should yield a different hash
    hash_tv = compute_config_hash(cfg, "train_val", train_df)
    assert hash1 != hash_tv


def test_build_models_returns_all_expected_models():
    # Test build_models on MovieLens using temporary cache dir
    with tempfile.TemporaryDirectory() as tmpdir:
        models = build_models("movielens", fit_on="train", cache_dir=Path(tmpdir))
        expected_keys = [
            "global_mean",
            "user_mean",
            "item_mean",
            "popularity",
            "user_cf",
            "item_cf",
            "cf",
            "content",
            "hybrid_cf_only",
            "hybrid_content_only",
            "hybrid_fixed",
            "hybrid_adaptive",
        ]
        for key in expected_keys:
            assert key in models, f"Model key '{key}' missing from build_models dictionary"
            assert isinstance(models[key], BaseRecommender), f"Model '{key}' must be BaseRecommender"

        # Verify predictions work
        u = 1
        it = 1
        pred = models["hybrid_adaptive"].predict(u, it)
        assert 1.0 <= pred <= 5.0

        # Verify caching: second call with same cache_dir loads from cache
        models_cached = build_models("movielens", fit_on="train", cache_dir=Path(tmpdir))
        assert set(models_cached.keys()) == set(expected_keys)


def test_build_models_invalid_inputs():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_models("non_existent_dataset")

    with pytest.raises(ValueError, match="fit_on must be"):
        build_models("movielens", fit_on="invalid_split")

