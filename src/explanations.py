"""Explanation Engine for Recommender Models.

Generates transparent, human-readable, and structured explanations
for all recommender models (CF, Content, Hybrid, Baselines).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from src.items import get_item_name, get_item_name_map


def _format_terms_list(terms: Sequence[str]) -> str:
    """Format a list of terms into English prose: 'A, B and C' or 'A and B'."""
    clean_terms = [t.replace("_", " ").strip() for t in terms if t and t.strip()]
    if not clean_terms:
        return ""
    if len(clean_terms) == 1:
        return clean_terms[0]
    if len(clean_terms) == 2:
        return f"{clean_terms[0]} and {clean_terms[1]}"
    return f"{', '.join(clean_terms[:-1])} and {clean_terms[-1]}"


def _format_rating_stars(val: float) -> str:
    """Format rating number into stars string, e.g. '5 stars' or '4.5 stars'."""
    val = round(float(val), 1)
    if val.is_integer():
        return f"{int(val)} stars"
    return f"{val:.1f} stars"


# ---------------------------------------------------------------------------
# Collaborative Filtering Explanations
# ---------------------------------------------------------------------------

def explain_user_cf(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Explain prediction for UserBasedCF."""
    contribs = model.get_contributions(user, item)
    item_display = get_item_name(item, item_name_map)

    if not contribs or user not in getattr(model, "user2idx_", {}):
        fallback_score = model.predict(user, item)
        text = (
            f"Not enough rating history for a personalised reason; "
            f"recommended because it has an average rating of {fallback_score:.1f} stars."
        )
        structured = {
            "component": "cf_user",
            "alpha": 1.0,
            "reasons": [{
                "type": "fallback",
                "weight": 1.0,
                "items": [],
                "terms": [],
                "text": text,
            }],
        }
        return text, structured

    u_idx = model.user2idx_[user]
    i_idx = model.item2idx_.get(item, -1)

    # Filter positive contributing neighbours
    pos_neighbors = [c for c in contribs if c[1] > 0]
    if not pos_neighbors:
        pos_neighbors = contribs[:3]

    # Calculate average rating among contributing neighbours for this item
    neighbor_ratings = []
    if i_idx >= 0 and hasattr(model, "R_sparse_") and model.R_sparse_ is not None:
        for v, _, _ in pos_neighbors:
            if v in model.user2idx_:
                v_idx = model.user2idx_[v]
                r_val = model.R_sparse_[v_idx, i_idx]
                if r_val > 0:
                    neighbor_ratings.append(r_val)

    n_similar = len(neighbor_ratings) if neighbor_ratings else len(pos_neighbors)
    avg_stars = float(np.mean(neighbor_ratings)) if neighbor_ratings else model.global_mean_

    # Find shared items user and top neighbours both rated highly (>= 4.0 or highest)
    shared_items: List[Any] = []
    if hasattr(model, "R_sparse_") and model.R_sparse_ is not None:
        u_row = model.R_sparse_.getrow(u_idx)
        u_rated = dict(zip(u_row.indices, u_row.data))

        for v, _, _ in pos_neighbors[:3]:
            if v in model.user2idx_:
                v_idx = model.user2idx_[v]
                v_row = model.R_sparse_.getrow(v_idx)
                v_rated = dict(zip(v_row.indices, v_row.data))

                common = set(u_rated.keys()).intersection(v_rated.keys())
                liked_common = [
                    idx for idx in common
                    if u_rated[idx] >= 3.5 and v_rated[idx] >= 3.5
                ]
                liked_common.sort(key=lambda idx: -(u_rated[idx] + v_rated[idx]))
                for idx in liked_common:
                    item_id = model.idx2item_[idx]
                    if item_id not in shared_items and item_id != item:
                        shared_items.append(item_id)
                    if len(shared_items) >= 2:
                        break
            if len(shared_items) >= 2:
                break

    shared_names = [get_item_name(it, item_name_map) for it in shared_items[:2]]
    shared_struct_items = [{"item_id": it, "name": name} for it, name in zip(shared_items[:2], shared_names)]

    avg_stars_str = _format_rating_stars(avg_stars)
    if shared_names:
        shared_str = _format_terms_list(shared_names)
        text = (
            f"Users similar to you also rated this highly (e.g. {n_similar} similar users averaged {avg_stars_str}), "
            f"and you both liked {shared_str}."
        )
    else:
        text = f"Users similar to you also rated this highly (e.g. {n_similar} similar users averaged {avg_stars_str})."

    structured = {
        "component": "cf_user",
        "alpha": 1.0,
        "reasons": [{
            "type": "cf_user",
            "weight": 1.0,
            "similar_users_count": n_similar,
            "similar_users_avg_rating": avg_stars,
            "items": shared_struct_items,
            "terms": [],
            "text": text,
        }],
    }
    return text, structured


def explain_item_cf(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Explain prediction for ItemBasedCF."""
    contribs = model.get_contributions(user, item)

    if not contribs or user not in getattr(model, "user2idx_", {}):
        fallback_score = model.predict(user, item)
        text = (
            f"Not enough rating history for a personalised reason; "
            f"recommended based on its historical average rating of {fallback_score:.1f} stars."
        )
        structured = {
            "component": "cf_item",
            "alpha": 1.0,
            "reasons": [{
                "type": "fallback",
                "weight": 1.0,
                "items": [],
                "terms": [],
                "text": text,
            }],
        }
        return text, structured

    u_idx = model.user2idx_[user]

    # Select top positive contributing items rated by user
    pos_contribs = [c for c in contribs if c[1] > 0 and c[2] > 0]
    if not pos_contribs:
        pos_contribs = [c for c in contribs if c[1] > 0]
    if not pos_contribs:
        pos_contribs = contribs[:2]

    top_items = pos_contribs[:2]
    item_entries = []
    parts = []

    for rated_item, sim_val, weight_val in top_items:
        name = get_item_name(rated_item, item_name_map)
        j_idx = model.item2idx_.get(rated_item)
        r_uj = float(model.R_sparse_[u_idx, j_idx]) if j_idx is not None else 4.0
        stars_str = _format_rating_stars(r_uj)
        parts.append(f"{name} ({stars_str})")
        item_entries.append({
            "item_id": rated_item,
            "name": name,
            "rating": r_uj,
            "similarity": float(sim_val),
            "weight": float(weight_val),
        })

    if len(parts) == 1:
        text = f"Because you rated {parts[0]}, which is often rated similarly to this item."
    else:
        text = f"Because you rated {parts[0]} and {parts[1]}, which are often rated similarly to this item."

    structured = {
        "component": "cf_item",
        "alpha": 1.0,
        "reasons": [{
            "type": "cf_item",
            "weight": 1.0,
            "items": item_entries,
            "terms": [],
            "text": text,
        }],
    }
    return text, structured


# ---------------------------------------------------------------------------
# Content-Based Explanations
# ---------------------------------------------------------------------------

def explain_content(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    item_type: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Explain prediction for ContentBased."""
    top_terms = model.get_top_terms(user, item, n=3)
    sim_items = model.get_similar_rated_items(user, item, n=2)

    # Determine domain context (movies vs recipes)
    if item_type is None:
        text_col = getattr(model, "text_column", "")
        if "genre" in text_col.lower():
            item_type = "movie"
        elif "text" in text_col.lower() or "ingredient" in text_col.lower():
            item_type = "recipe"
        else:
            item_type = "item"

    if item_type == "movie":
        noun = "movies"
        verb = "liked"
    elif item_type == "recipe":
        noun = "recipes"
        verb = "rated highly"
    else:
        noun = "items"
        verb = "liked"

    if not top_terms and not sim_items:
        fallback_score = model.predict(user, item)
        text = (
            f"Not enough rating history for a personalised reason; "
            f"recommended because it matches popular attributes ({fallback_score:.1f} stars)."
        )
        structured = {
            "component": "content",
            "alpha": 0.0,
            "reasons": [{
                "type": "fallback",
                "weight": 1.0,
                "items": [],
                "terms": [],
                "text": text,
            }],
        }
        return text, structured

    terms = [term for term, _ in top_terms]
    terms_str = _format_terms_list(terms)

    item_entries = []
    if sim_items:
        top_sim_item_id, top_sim_score = sim_items[0]
        top_sim_name = get_item_name(top_sim_item_id, item_name_map)
        item_entries.append({
            "item_id": top_sim_item_id,
            "name": top_sim_name,
            "similarity": float(top_sim_score),
        })

    if terms_str and item_entries:
        example_name = item_entries[0]["name"]
        text = f"Shares {terms_str} with {noun} you {verb} (e.g. {example_name})."
    elif terms_str:
        text = f"Shares {terms_str} with {noun} matching your taste profile."
    elif item_entries:
        example_name = item_entries[0]["name"]
        text = f"Shares key attributes with {noun} you {verb} (e.g. {example_name})."
    else:
        text = f"Recommended based on your content preferences."

    structured = {
        "component": "content",
        "alpha": 0.0,
        "reasons": [{
            "type": "content",
            "weight": 1.0,
            "terms": terms,
            "items": item_entries,
            "text": text,
        }],
    }
    return text, structured


# ---------------------------------------------------------------------------
# Hybrid Explanations
# ---------------------------------------------------------------------------

def explain_hybrid(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    item_type: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Explain prediction for HybridRecommender."""
    alpha = model.get_alpha(user)
    n_u = model.user_counts_.get(user, 0)
    mode = getattr(model, "mode", "adaptive")

    # Generate constituent explanations
    cf_text, cf_struct = explain_model(model.cf_model, user, item, item_name_map=item_name_map, item_type=item_type)
    content_text, content_struct = explain_model(model.content_model, user, item, item_name_map=item_name_map, item_type=item_type)

    reasons = []

    if mode == "cf_only" or alpha >= 1.0:
        intro = "This recommendation is based entirely on collaborative filtering (weight 1.00)."
        text = f"{intro} {cf_text}"
        reasons.extend(cf_struct.get("reasons", []))

    elif mode == "content_only" or alpha <= 0.0:
        intro = "This recommendation is based entirely on item content (weight 1.00)."
        text = f"{intro} {content_text}"
        reasons.extend(content_struct.get("reasons", []))

    elif n_u == 0:
        intro = "You have no rating history, so we leaned entirely on item content and catalogue averages."
        text = f"{intro} {content_text}"
        reasons.extend(content_struct.get("reasons", []))

    elif alpha >= 0.5:
        # CF dominates
        weight_cf = alpha
        weight_cnt = 1.0 - alpha
        intro = f"You have rated {n_u} items, so we leaned mostly on collaborative filtering (weight {weight_cf:.2f} vs {weight_cnt:.2f})."
        
        # Primary reason from CF
        text = f"{intro} {cf_text}"
        reasons.extend(cf_struct.get("reasons", []))

        # Secondary reason from Content if informative
        top_terms = model.content_model.get_top_terms(user, item, n=2)
        if top_terms:
            terms = [t for t, _ in top_terms]
            terms_str = _format_terms_list(terms)
            sec_text = f"It also matches your preference for {terms_str}."
            text = f"{text} {sec_text}"
            reasons.append({
                "type": "content_secondary",
                "weight": float(weight_cnt),
                "terms": terms,
                "items": [],
                "text": sec_text,
            })

    else:
        # Content dominates (alpha < 0.5)
        weight_cnt = 1.0 - alpha
        weight_cf = alpha
        intro = f"You have rated {n_u} items, so we leaned mostly on item content (weight {weight_cnt:.2f} vs {weight_cf:.2f})."

        # Primary reason from Content
        text = f"{intro} {content_text}"
        reasons.extend(content_struct.get("reasons", []))

        # Secondary reason from CF if informative
        if hasattr(model.cf_model, "get_contributions"):
            contribs = model.cf_model.get_contributions(user, item)
            pos_contribs = [c for c in contribs if c[1] > 0]
            if pos_contribs:
                top_neighbor_or_item = pos_contribs[0][0]
                name = get_item_name(top_neighbor_or_item, item_name_map)
                sec_text = f"Additionally, rating patterns connect this to {name}."
                text = f"{text} {sec_text}"
                reasons.append({
                    "type": "cf_secondary",
                    "weight": float(weight_cf),
                    "terms": [],
                    "items": [{"item_id": top_neighbor_or_item, "name": name}],
                    "text": sec_text,
                })

    structured = {
        "component": "hybrid",
        "alpha": float(alpha),
        "n_ratings": int(n_u),
        "summary": text,
        "reasons": reasons,
    }
    return text, structured


# ---------------------------------------------------------------------------
# Baseline Explanations
# ---------------------------------------------------------------------------

def explain_baseline(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Explain prediction for baseline models (Popularity, ItemMean, UserMean, GlobalMean)."""
    class_name = model.__class__.__name__

    if class_name == "Popularity":
        count = getattr(model, "item_counts_", {}).get(item, 0)
        mean_val = getattr(model, "item_means_", {}).get(item, getattr(model, "global_mean_", 3.0))
        if count > 0:
            text = f"Recommended because it is popular, with {count:,} ratings (average {mean_val:.1f} stars)."
        else:
            text = f"Recommended based on catalog popularity and overall average rating ({getattr(model, 'global_mean_', 3.0):.1f} stars)."
        comp = "popularity"

    elif class_name == "ItemMean":
        if item in getattr(model, "item_means_", {}):
            mean_val = model.item_means_[item]
            text = f"Recommended based on its historical average rating of {mean_val:.1f} stars."
        else:
            text = f"Not enough rating history for this item; recommended using catalog average ({getattr(model, 'global_mean_', 3.0):.1f} stars)."
        comp = "item_mean"

    elif class_name == "UserMean":
        if user in getattr(model, "user_means_", {}):
            mean_val = model.user_means_[user]
            text = f"Recommended based on your personal average rating tendency of {mean_val:.1f} stars."
        else:
            text = f"Not enough user rating history; recommended using catalog average ({getattr(model, 'global_mean_', 3.0):.1f} stars)."
        comp = "user_mean"

    else:  # GlobalMean or generic
        g_mean = getattr(model, "global_mean_", 3.0)
        text = f"Recommended based on the overall catalog average rating of {g_mean:.1f} stars."
        comp = "global_mean"

    structured = {
        "component": comp,
        "alpha": None,
        "reasons": [{
            "type": comp,
            "weight": 1.0,
            "items": [],
            "terms": [],
            "text": text,
        }],
    }
    return text, structured


# ---------------------------------------------------------------------------
# Universal Dispatchers
# ---------------------------------------------------------------------------

def explain_model(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    item_type: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Dispatch explanation call to the appropriate model-specific handler."""
    # Check if model has custom item_name_map attribute
    if item_name_map is None:
        item_name_map = getattr(model, "item_name_map_", None)

    class_name = model.__class__.__name__

    if class_name == "HybridRecommender":
        return explain_hybrid(model, user, item, item_name_map=item_name_map, item_type=item_type)
    elif class_name == "UserBasedCF":
        return explain_user_cf(model, user, item, item_name_map=item_name_map)
    elif class_name == "ItemBasedCF":
        return explain_item_cf(model, user, item, item_name_map=item_name_map)
    elif class_name == "ContentBased":
        return explain_content(model, user, item, item_name_map=item_name_map, item_type=item_type)
    elif class_name in ("GlobalMean", "UserMean", "ItemMean", "Popularity"):
        return explain_baseline(model, user, item, item_name_map=item_name_map)
    else:
        # Fallback generic explanation
        score = model.predict(user, item)
        text = f"Recommended based on model prediction ({score:.1f} stars)."
        structured = {
            "component": class_name.lower(),
            "alpha": None,
            "reasons": [{
                "type": "generic",
                "weight": 1.0,
                "items": [],
                "terms": [],
                "text": text,
            }],
        }
        return text, structured


def explain(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    item_type: Optional[str] = None,
) -> str:
    """Return human-readable explanation string for a user-item prediction."""
    text, _ = explain_model(model, user, item, item_name_map=item_name_map, item_type=item_type)
    return text


def explain_structured(
    model: Any,
    user: Any,
    item: Any,
    item_name_map: Optional[Dict[Any, str]] = None,
    item_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return structured explanation dictionary for a user-item prediction."""
    _, struct = explain_model(model, user, item, item_name_map=item_name_map, item_type=item_type)
    return struct

