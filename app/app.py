"""Streamlit Web Application: Explainable Hybrid Recommender Demo.

Provides interactive recommendation exploration, user rating histories,
transparent structured explanations, model comparisons, dietary & allergen filtering,
and pantry kitchen mode.

Run via:
    streamlit run app/app.py
    or:
    python -m src.run_app
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (Must be very first Streamlit command)
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Explainable Hybrid Recommender",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

import pandas as pd

try:
    from app.helpers import (
        get_adaptive_alpha_explanation,
        get_cached_data,
        get_cached_models,
        get_recommendations_for_model,
        get_user_profile,
    )
except (ImportError, ModuleNotFoundError):
    from helpers import (  # type: ignore
        get_adaptive_alpha_explanation,
        get_cached_data,
        get_cached_models,
        get_recommendations_for_model,
        get_user_profile,
    )

# Custom light styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e3f2fd;
        color: #0d47a1;
        margin-right: 6px;
    }
    .explanation-box {
        background-color: #f1f8e9;
        border-left: 4px solid #7cb342;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# ---------------------------------------------------------------------------
st.sidebar.title("🎛️ Recommender Settings")

# 1. Dataset Picker
dataset_choice = st.sidebar.selectbox(
    "Select Dataset Domain:",
    options=["Food.com Recipes (Primary)", "MovieLens 100K (Benchmark)"],
    index=0,
)
dataset_key = "foodcom" if "Food.com" in dataset_choice else "movielens"

# Show spinner on initial model loading
with st.spinner("Loading models and catalog data from pipeline cache..."):
    data = get_cached_data(dataset_key)
    models = get_cached_models(dataset_key)

cfg = data["cfg"]
items_df = data["items_df"]
train_val_df = data["train_val_df"]
name_map = data["name_map"]
user_counts = data["user_counts"]
archetypes = data["archetypes"]
all_users = data["all_users"]

st.sidebar.markdown("---")

# 2. User Selector
st.sidebar.subheader("👤 User Selection")
archetype_options = ["Custom User ID..."] + list(archetypes.keys())
selected_archetype = st.sidebar.selectbox(
    "Preset User Archetypes:",
    options=archetype_options,
    index=1,  # Default to low-activity user
    help="Archetypes chosen deterministically based on empirical activity quantiles.",
)

if selected_archetype == "Custom User ID...":
    default_uid = all_users[0] if all_users else 1
    user_id = int(st.sidebar.number_input("Enter User ID:", min_value=1, max_value=max(all_users) if all_users else 1000000, value=default_uid, step=1))
else:
    user_id = archetypes[selected_archetype]
    st.sidebar.caption(f"Selected User ID: **{user_id}**")

# 3. Model Selector
st.sidebar.subheader("🤖 Recommender Model")
model_options = {
    "Adaptive Hybrid (Recommended)": "hybrid_adaptive",
    "Fixed Hybrid (50/50 Blend)": "hybrid_fixed",
    "Collaborative Filtering Only": "hybrid_cf_only",
    "Content-Based Only": "hybrid_content_only",
    "Popularity Baseline": "popularity",
}
selected_model_name = st.sidebar.selectbox(
    "Active Model:",
    options=list(model_options.keys()),
    index=0,
)
active_model_key = model_options[selected_model_name]
active_model = models[active_model_key]

# 4. Number of recommendations
top_n = st.sidebar.slider("Number of Recommendations (N):", min_value=5, max_value=20, value=10, step=1)

# 5. Recipe Options (Food.com only)
filters_dict = {}
kitchen_input = ""

if dataset_key == "foodcom":
    st.sidebar.markdown("---")
    with st.sidebar.expander("🥗 Recipe Filters & Kitchen Mode", expanded=False):
        diet_choice = st.selectbox(
            "Dietary Restriction:",
            options=["None", "Vegetarian", "Vegan", "Gluten-Free", "Nut-Free", "Dairy-Free"],
            index=0,
        )
        if diet_choice != "None":
            # Map to backend parameter format
            diet_clean = diet_choice.lower().replace("-", "_")
            filters_dict["diet"] = diet_clean

        # Max cooking time slider with "no limit" checkbox
        col_t1, col_t2 = st.columns([2, 1])
        limit_time = col_t2.checkbox("Limit time", value=False)
        max_time_val = col_t1.slider("Max Time (mins):", min_value=15, max_value=240, value=60, step=15, disabled=not limit_time)
        if limit_time:
            filters_dict["max_minutes"] = max_time_val

        # Max calories slider with "no limit" checkbox
        col_c1, col_c2 = st.columns([2, 1])
        limit_cal = col_c2.checkbox("Limit cal", value=False)
        max_cal_val = col_c1.slider("Max Calories (kcal):", min_value=100, max_value=1500, value=600, step=50, disabled=not limit_cal)
        if limit_cal:
            filters_dict["max_calories"] = max_cal_val

        # Disliked ingredients
        disliked_raw = st.text_input("Disliked Ingredients (comma-separated):", placeholder="e.g. mushrooms, cilantro")
        if disliked_raw.strip():
            disliked_list = [x.strip() for x in disliked_raw.split(",") if x.strip()]
            if disliked_list:
                filters_dict["exclude_ingredients"] = disliked_list

        st.markdown("**🍳 Kitchen Mode (Pantry Match)**")
        kitchen_input = st.text_area(
            "Ingredients I have on hand:",
            placeholder="e.g. chicken breast, garlic, olive oil, onion, rice",
            help="Matches available ingredients against recipe lists, calculating ingredient coverage % and surfacing missing items.",
        )

# ---------------------------------------------------------------------------
# Main Panel: Header & About
# ---------------------------------------------------------------------------
domain_title = "Food.com Recipes Recommender" if dataset_key == "foodcom" else "MovieLens 100K Benchmark"
st.title(f"🍽️ Explainable Hybrid Recommender — {domain_title}")

with st.expander("ℹ️ About this System (Methodology & Dataset Statistics)", expanded=False):
    st.markdown(
        """
        ### Methodology
        - **Collaborative Filtering:** Item-Based Collaborative Filtering using Adjusted Cosine similarity over user-mean centered interaction vectors.
        - **Content-Based Filtering:** TF-IDF vectorization over textual features (recipe names, ingredients, and preparation steps for Food.com; genre tokens for MovieLens).
        - **Adaptive Score Fusion:** Blends normalized CF and Content scores via user profile length:
        $$\\alpha_u = \\frac{n_u}{n_u + k_0}$$
        where $n_u$ is the number of ratings by user $u$, and $k_0$ is tuned on validation data ($k_0 = 100.0$ for Food.com, $k_0 = 20.0$ for MovieLens). Users with few ratings rely primarily on content features; dense users scale up collaborative weight.
        """
    )
    if data["filter_info"]:
        fi = data["filter_info"].get("counts", {}).get("after_sample", {})
        st.markdown(
            f"**Dataset Catalog Sizes:** {fi.get('users', 3000):,} Active Users | {fi.get('recipes', 3889):,} Filtered Recipes | {fi.get('ratings', 164278):,} Total Ratings."
        )

# ---------------------------------------------------------------------------
# Main Panel: Tabs
# ---------------------------------------------------------------------------
tab_profile, tab_recs, tab_compare = st.tabs(["📋 User Profile", "⭐ Recommendations & Explanations", "⚖️ Compare Models"])

# ---------------------------------------------------------------------------
# Tab 1: User Profile
# ---------------------------------------------------------------------------
with tab_profile:
    st.subheader(f"User {user_id} Rating History")
    alpha_val, alpha_expl = get_adaptive_alpha_explanation(user_id, user_counts, cfg.k0)
    st.info(alpha_expl)

    profile_df = get_user_profile(user_id, train_val_df, items_df, name_map)
    if len(profile_df) == 0:
        st.warning(f"User {user_id} has no training interactions recorded in this dataset (Unknown User).")
    else:
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Total Rated Items", f"{len(profile_df)}")
        m_c2.metric("Mean Rating Given", f"{profile_df['Rating'].mean():.2f} / 5.0")
        m_c3.metric("Adaptive CF Weight (α_u)", f"{alpha_val:.2f}")

        st.dataframe(profile_df, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# Tab 2: Recommendations
# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader(f"Top-{top_n} Recommendations using {selected_model_name}")

    filters_to_pass = filters_dict if filters_dict else None
    kitchen_to_pass = kitchen_input if kitchen_input.strip() else None

    recs, warn = get_recommendations_for_model(
        model=active_model,
        dataset_name=dataset_key,
        user_id=user_id,
        n=top_n,
        items_df=items_df,
        name_map=name_map,
        filters=filters_to_pass,
        kitchen_ingredients=kitchen_to_pass,
    )

    if warn:
        st.warning(warn)

    if not recs:
        st.error("No items could be recommended under the current settings. Try relaxing constraints.")
    else:
        for idx, item in enumerate(recs, 1):
            with st.container():
                st.markdown(f"#### {idx}. {item['name']}")

                # Badges row
                b_cols = st.columns([3, 1])
                pred_star = item["predicted_rating"]
                b_cols[0].markdown(
                    f"**Predicted Rating:** ⭐ **{pred_star:.2f} / 5.0**"
                )

                badges_html = ""
                if "category" in item and pd.notna(item["category"]):
                    badges_html += f'<span class="badge">📂 {item["category"]}</span>'
                if "total_minutes" in item and item["total_minutes"] != "N/A" and pd.notna(item["total_minutes"]):
                    try:
                        badges_html += f'<span class="badge">⏱️ {int(float(item["total_minutes"]))} mins</span>'
                    except (ValueError, TypeError):
                        pass
                if "calories" in item and item["calories"] != "N/A" and pd.notna(item["calories"]):
                    try:
                        badges_html += f'<span class="badge">🔥 {int(float(item["calories"]))} kcal</span>'
                    except (ValueError, TypeError):
                        pass
                if "genres" in item and pd.notna(item["genres"]):
                    badges_html += f'<span class="badge">🎭 {item["genres"]}</span>'

                if badges_html:
                    b_cols[0].markdown(badges_html, unsafe_allow_html=True)

                # Kitchen match info
                if "kitchen_overlap" in item:
                    overlap_pct = int(round(item["kitchen_overlap"] * 100))
                    missing_str = ", ".join(item["missing_ingredients"]) if item["missing_ingredients"] else "None (Fully stocked!)"
                    st.caption(f"🍳 **Kitchen Match:** {overlap_pct}% of ingredients on hand. **Missing:** {missing_str}")

                # Explanation Expander
                with st.expander("🔍 Why was this recommended?", expanded=False):
                    struct = item.get("structured_explanation", {})
                    summary_text = struct.get("summary", "")
                    if summary_text:
                        st.markdown(f"*{summary_text}*")
                    reasons = struct.get("reasons", [])
                    if reasons:
                        for r in reasons:
                            r_text = r.get("text", "")
                            if r_text:
                                st.markdown(f"- {r_text}")
                st.markdown("---")

# ---------------------------------------------------------------------------
# Tab 3: Compare Models
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader(f"Model Comparison (Top-5 side-by-side for User {user_id})")
    st.caption("Inspect how Collaborative Filtering, Content-Based, and Hybrid blends rank items differently under the same active filters.")

    compare_keys = [
        ("Adaptive Hybrid", models.get("hybrid_adaptive")),
        ("Fixed Hybrid", models.get("hybrid_fixed")),
        ("CF Only", models.get("hybrid_cf_only")),
        ("Content Only", models.get("hybrid_content_only")),
        ("Popularity", models.get("popularity")),
    ]

    cols = st.columns(len(compare_keys))

    for col, (c_label, c_mod) in zip(cols, compare_keys):
        with col:
            st.markdown(f"**{c_label}**")
            if c_mod is None:
                st.write("Model unavailable")
                continue

            c_recs, _ = get_recommendations_for_model(
                model=c_mod,
                dataset_name=dataset_key,
                user_id=user_id,
                n=5,
                items_df=items_df,
                name_map=name_map,
                filters=filters_to_pass,
                kitchen_ingredients=kitchen_to_pass,
            )

            if not c_recs:
                st.caption("No recommendations")
            else:
                for i, r in enumerate(c_recs, 1):
                    p_star = r["predicted_rating"]
                    st.markdown(f"**{i}.** {r['name']}")
                    st.caption(f"⭐ {p_star:.2f} / 5.0")
