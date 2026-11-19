"""Helper utilities and cached loaders for the Streamlit demo application.

Encapsulates data loading, model fetching, profile extraction,
and robust recipe filtering / kitchen recommendation execution.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.base import BaseRecommender
from src.config import DATASET_CONFIGS, DatasetConfig
from src.explanations import explain_structured
from src.items import get_item_name, get_item_name_map
from src.pipeline import build_models, load_dataset_data
from src.recipe_features import (
    DEFAULT_PANTRY_STAPLES,
    filter_recipes,
    kitchen_recommend,
    recommend_with_filters,
)


@st.cache_resource(show_spinner=False)
def get_cached_models(dataset_name: str) -> Dict[str, BaseRecommender]:
    """Load or fit all recommender models on train_val split using pipeline cache."""
    cfg = DATASET_CONFIGS[dataset_name.lower().strip()]
    return build_models(cfg, fit_on="train_val")


@st.cache_data(show_spinner=False)
def get_cached_data(dataset_name: str) -> Dict[str, Any]:
    """Load items, interaction history, item name map, and metadata."""
    cfg = DATASET_CONFIGS[dataset_name.lower().strip()]
    train_val_df, items_df, _, _ = load_dataset_data(cfg, fit_on="train_val")

    # Name map
    name_map = get_item_name_map(items_df)

    # Filter info if available
    filter_info_path = cfg.data_dir / "filter_info.json"
    filter_info = {}
    if filter_info_path.exists():
        try:
            with open(filter_info_path, "r", encoding="utf-8") as f:
                filter_info = json.load(f)
        except Exception:
            filter_info = {}

    # User activity counts in training split
    user_counts = train_val_df.groupby("user_id").size()

    # Dynamic archetypes computed from quantiles
    u_min = int(user_counts.idxmin())
    c_min = int(user_counts.min())

    sorted_users = user_counts.sort_values()
    u_med = int(sorted_users.index[len(sorted_users) // 2])
    c_med = int(user_counts.loc[u_med])

    q95_val = user_counts.quantile(0.95)
    high_candidates = user_counts[user_counts >= q95_val]
    u_heavy = int(high_candidates.index[0]) if len(high_candidates) > 0 else int(user_counts.idxmax())
    c_heavy = int(user_counts.loc[u_heavy])

    archetypes = {
        f"Low-activity user (ID {u_min}, {c_min} ratings)": u_min,
        f"Mid-activity user (ID {u_med}, {c_med} ratings)": u_med,
        f"Heavy user (ID {u_heavy}, {c_heavy} ratings)": u_heavy,
    }

    all_users = sorted(user_counts.index.tolist())

    return {
        "cfg": cfg,
        "items_df": items_df,
        "train_val_df": train_val_df,
        "name_map": name_map,
        "filter_info": filter_info,
        "user_counts": user_counts,
        "archetypes": archetypes,
        "all_users": all_users,
    }


def get_user_profile(user_id: int, train_val_df: pd.DataFrame, items_df: pd.DataFrame, name_map: Dict[Any, str]) -> pd.DataFrame:
    """Extract and format user history sorted descending by rating and timestamp."""
    user_rows = train_val_df[train_val_df["user_id"] == user_id].copy()
    if len(user_rows) == 0:
        return pd.DataFrame(columns=["item_id", "Title / Name", "Rating", "Date"])

    # Merge or map names
    user_rows["Title / Name"] = user_rows["item_id"].map(lambda x: get_item_name(x, name_map))

    if "timestamp" in user_rows.columns:
        user_rows["Date"] = pd.to_datetime(user_rows["timestamp"], unit="s", errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        user_rows["Date"] = "N/A"

    user_rows = user_rows.rename(columns={"rating": "Rating"})
    user_rows = user_rows.sort_values(by=["Rating", "Date", "item_id"], ascending=[False, False, True])
    return user_rows[["item_id", "Title / Name", "Rating", "Date"]].reset_index(drop=True)


def get_adaptive_alpha_explanation(user_id: int, user_counts: pd.Series, k0: float) -> Tuple[float, str]:
    """Compute alpha_u and produce plain-language explanation."""
    n_u = int(user_counts.get(user_id, 0))
    if n_u == 0:
        return 0.0, "This is a new/unseen user (0 training ratings), so the system relies 100% on item content features and global baselines."
    alpha = float(n_u / (n_u + k0))
    pct_cf = int(round(alpha * 100))
    pct_cnt = 100 - pct_cf
    text = (
        f"This user has **{n_u} ratings** in training history (k₀ = {k0:.1f}). "
        f"The adaptive blend assigns **{alpha:.2f} ({pct_cf}%) to collaborative filtering** "
        f"and **{1.0 - alpha:.2f} ({pct_cnt}%) to item content features**."
    )
    return alpha, text


def get_recommendations_for_model(
    model: BaseRecommender,
    dataset_name: str,
    user_id: int,
    n: int,
    items_df: pd.DataFrame,
    name_map: Dict[Any, str],
    filters: Optional[Dict[str, Any]] = None,
    kitchen_ingredients: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Produce top-N recommendations with explanations, metadata, and predicted ratings."""
    ds = dataset_name.lower().strip()
    warning_message: Optional[str] = None
    results: List[Dict[str, Any]] = []

    # 1. Kitchen Mode (Food.com recipes only)
    if ds == "foodcom" and kitchen_ingredients and kitchen_ingredients.strip():
        recs = kitchen_recommend(
            model=model,
            user=user_id,
            available_ingredients=kitchen_ingredients,
            n=n,
            min_overlap=0.1,  # lenient to allow partial matches
            alpha_kitchen=0.5,
            filters=filters,
            items_df=items_df,
        )
        if not recs:
            warning_message = "No recipes met the kitchen ingredient overlap and filter criteria."
            return [], warning_message

        for r in recs:
            item_id = r["item_id"]
            # Predicted star rating from model
            pred_rating = model.predict(user_id, item_id)
            meta = items_df[items_df["item_id"] == item_id].iloc[0] if len(items_df[items_df["item_id"] == item_id]) > 0 else None

            # Get structured explanation
            try:
                struct_exp = explain_structured(model, user_id, item_id, item_name_map=name_map)
            except Exception:
                struct_exp = {"summary": r.get("explanation", ""), "reasons": []}

            results.append({
                "item_id": item_id,
                "name": r["name"],
                "predicted_rating": pred_rating,
                "kitchen_overlap": r.get("overlap", 0.0),
                "missing_ingredients": r.get("missing_ingredients", []),
                "kitchen_explanation": r.get("explanation", ""),
                "structured_explanation": struct_exp,
                "category": meta["category"] if meta is not None and "category" in meta else "Recipe",
                "calories": meta["calories"] if meta is not None and "calories" in meta else "N/A",
                "total_minutes": meta["total_minutes"] if meta is not None and "total_minutes" in meta else "N/A",
            })
        return results, None

    # 2. Recipe Filters Mode (Food.com)
    if ds == "foodcom" and filters:
        # Check how many pass the filter
        matching_ids = filter_recipes(items_df, **filters)
        if len(matching_ids) == 0:
            return [], "Your chosen filters excluded all 3,889 recipes. Please relax the cooking time, calorie, or dietary constraints."
        if len(matching_ids) < n:
            warning_message = f"Only {len(matching_ids)} recipes in the entire catalog matched your strict filters (requested {n})."

        raw_recs = recommend_with_filters(
            model=model,
            user=user_id,
            n=n,
            filters=filters,
            items_df=items_df,
            exclude_seen=True,
        )
    else:
        # Standard unconstrained recommendation
        raw_recs = model.recommend(user_id, n=n, exclude_seen=True)

    # Deterministic tie-breaker: sort primarily by score descending, secondarily by item_id ascending
    raw_recs = sorted(raw_recs, key=lambda x: (-float(x[1]), x[0]))[:n]

    for item_id, rank_score in raw_recs:
        pred_rating = model.predict(user_id, item_id)
        name = get_item_name(item_id, name_map)
        meta = items_df[items_df["item_id"] == item_id].iloc[0] if len(items_df[items_df["item_id"] == item_id]) > 0 else None

        try:
            struct_exp = explain_structured(model, user_id, item_id, item_name_map=name_map)
        except Exception:
            struct_exp = {"summary": f"Recommended for User {user_id} based on model prediction.", "reasons": []}

        entry: Dict[str, Any] = {
            "item_id": item_id,
            "name": name,
            "predicted_rating": pred_rating,
            "rank_score": rank_score,
            "structured_explanation": struct_exp,
        }
        if meta is not None:
            if "category" in meta:
                entry["category"] = meta["category"]
            if "calories" in meta:
                entry["calories"] = meta["calories"]
            if "total_minutes" in meta:
                entry["total_minutes"] = meta["total_minutes"]
            if "genre_tokens" in meta:
                entry["genres"] = meta["genre_tokens"]

        results.append(entry)

    return results, warning_message
