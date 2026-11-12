"""Item metadata and display name utilities for recommendation explanations."""

import html
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from src.config import DATASET_CONFIGS, DatasetConfig, MOVIELENS_CONFIG, FOODCOM_CONFIG

_ITEM_NAME_CACHE: Dict[str, Dict[Any, str]] = {}


def _clean_name(name: Any) -> str:
    """Clean display name by stripping and unescaping HTML entities."""
    s = html.unescape(str(name)).strip()
    return s



def get_item_name_map(
    dataset_or_items: Union[str, DatasetConfig, pd.DataFrame, Path] = "movielens",
) -> Dict[Any, str]:
    """Return a dictionary mapping item_id -> display name (title for movies, recipe name for Food.com).

    Parameters
    ----------
    dataset_or_items : str, DatasetConfig, pd.DataFrame, or Path
        Dataset identifier ("movielens", "foodcom"), a DatasetConfig, an items DataFrame,
        or a Path to items.csv.

    Returns
    -------
    Dict[Any, str]
        Dictionary mapping item_id to its human-readable display name.
    """
    if isinstance(dataset_or_items, str):
        dataset_key = dataset_or_items.lower()
        if dataset_key in _ITEM_NAME_CACHE:
            return _ITEM_NAME_CACHE[dataset_key]

        if dataset_key in DATASET_CONFIGS:
            cfg = DATASET_CONFIGS[dataset_key]
            items_path = cfg.data_dir / "items.csv"
            if items_path.exists():
                df = pd.read_csv(items_path)
                id_col = cfg.id_column
                title_col = cfg.title_column
                mapping = {row[id_col]: _clean_name(row[title_col]) for _, row in df.iterrows()}
                _ITEM_NAME_CACHE[dataset_key] = mapping
                return mapping
            raise FileNotFoundError(f"Items file not found at {items_path}")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_or_items}")

    elif isinstance(dataset_or_items, DatasetConfig):
        return get_item_name_map(dataset_or_items.name)

    elif isinstance(dataset_or_items, Path):
        df = pd.read_csv(dataset_or_items)
        id_col = "item_id" if "item_id" in df.columns else df.columns[0]
        title_col = "title" if "title" in df.columns else ("name" if "name" in df.columns else df.columns[1])
        return {row[id_col]: _clean_name(row[title_col]) for _, row in df.iterrows()}

    elif isinstance(dataset_or_items, pd.DataFrame):
        df = dataset_or_items
        id_col = "item_id" if "item_id" in df.columns else df.columns[0]
        title_col = "title" if "title" in df.columns else ("name" if "name" in df.columns else df.columns[1])
        return {row[id_col]: _clean_name(row[title_col]) for _, row in df.iterrows()}

    else:
        raise TypeError(f"Unsupported type for dataset_or_items: {type(dataset_or_items)}")


def get_item_name(
    item_id: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    dataset: Optional[str] = None,
) -> str:
    """Retrieve human-readable name for an item_id.

    Parameters
    ----------
    item_id : Any
        Identifier of the item.
    item_name_map : Dict[Any, str], optional
        Pre-computed mapping of item_id -> display name.
    dataset : str, optional
        Dataset name to look up in cache if item_name_map is not provided.

    Returns
    -------
    str
        Display name or fallback formatted string.
    """
    if item_name_map is not None:
        if item_id in item_name_map:
            return str(item_name_map[item_id])

    if dataset is not None:
        try:
            m = get_item_name_map(dataset)
            if item_id in m:
                return str(m[item_id])
        except Exception:
            pass

    return f"Item {item_id}"
