"""Global dataset configurations and column schemas."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    text_column: str
    id_column: str = "item_id"
    title_column: str = "name"
    liked_threshold: float = 4.0
    min_overlap: int = 5


MOVIELENS_CONFIG = DatasetConfig(
    name="movielens",
    data_dir=Path("data/processed/movielens"),
    text_column="genre_tokens",
    title_column="title",
    liked_threshold=4.0,
    min_overlap=5,
)

FOODCOM_CONFIG = DatasetConfig(
    name="foodcom",
    data_dir=Path("data/processed/foodcom"),
    text_column="text",
    title_column="name",
    liked_threshold=4.0,
    min_overlap=5,
)

DATASET_CONFIGS = {
    "movielens": MOVIELENS_CONFIG,
    "foodcom": FOODCOM_CONFIG,
}
