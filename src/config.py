"""Global dataset configurations and column schemas."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    text_column: str
    id_column: str = "item_id"
    title_column: str = "name"
    liked_threshold: float = 4.0

    # Best Collaborative Filtering configuration found on validation set in Task 2
    cf_type: str = "item_based"        # Source: Task 2 benchmarks (ItemBasedCF outperformed UserBasedCF)
    cf_similarity: str = "cosine"      # Source: Task 2 tuning (cosine outperformed Pearson)
    cf_k: int = 30                     # Source: Task 2 tuning (k=30 achieved lowest validation RMSE)
    min_overlap: int = 2               # Source: Task 2 tuning (min_overlap=2 balanced coverage and accuracy)
    shrinkage: float = 0.0             # Source: Task 2 shrinkage grid search

    # Best Content-Based and Adaptive Hybrid configuration found in Tasks 3 & 4
    k_content: int = 30                # Source: Task 3 content neighborhood size
    k0: float = 10.0                   # Source: Task 4 adaptive weighting half-saturation threshold


# MovieLens 100K: Benchmark dataset (Collaborative Filtering strongly dominates)
MOVIELENS_CONFIG = DatasetConfig(
    name="movielens",
    data_dir=Path("data/processed/movielens"),
    text_column="genre_tokens",
    title_column="title",
    liked_threshold=4.0,
    # Best CF Config: ItemBasedCF(cosine, k=30, min_overlap=2) -> Val RMSE: 0.9723 (Task 2)
    cf_type="item_based",
    cf_similarity="cosine",
    cf_k=30,
    min_overlap=2,
    # Tuned Adaptive Hybrid: k0=20.0 -> Val RMSE: 0.9625 (Task 4)
    # User alpha distribution: mean 0.6938, median 0.7059 (83.5% of users have alpha >= 0.5)
    k0=20.0,
)

# Food.com: Primary dataset (Content-Based strongly dominates due to extreme positivity bias & sparsity)
FOODCOM_CONFIG = DatasetConfig(
    name="foodcom",
    data_dir=Path("data/processed/foodcom"),
    text_column="text",
    title_column="name",
    liked_threshold=4.0,
    # Best CF Config: ItemBasedCF(cosine, k=30, min_overlap=2) -> Val RMSE: 0.6067 (Task 2)
    cf_type="item_based",
    cf_similarity="cosine",
    cf_k=30,
    min_overlap=2,
    # Tuned Adaptive Hybrid: k0=100.0 -> Val RMSE: 0.5377 (Task 4)
    # User alpha distribution: mean 0.2516, median 0.2063 (93.5% of users have alpha < 0.5)
    k0=100.0,
)

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "movielens": MOVIELENS_CONFIG,
    "foodcom": FOODCOM_CONFIG,
}


def get_item_name_map(dataset_name: str = "movielens"):
    """Helper mapping item_id -> display name for each dataset."""
    from src.items import get_item_name_map as _get_map
    return _get_map(dataset_name)
