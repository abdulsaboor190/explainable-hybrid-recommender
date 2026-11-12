"""Recipe-Specific Features and Filtering for Food.com Recommender.

Provides transparent rule-based wrappers around fitted recommendation models:
  - Dietary, calorie, time, and allergen filters.
  - Kitchen mode: matching available ingredients to recipes with overlap scoring.
  - Filtered recommendation ranking: applying constraints to candidate sets
    before ranking to guarantee a full top-N list.

Note: These are rule-based wrappers that operate on top of fitted recommender models;
they do not modify internal recommender model classes.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.base import BaseRecommender
from src.config import FOODCOM_CONFIG
from src.items import get_item_name


# ---------------------------------------------------------------------------
# Default Configurable Pantry Staples
# ---------------------------------------------------------------------------
# Common kitchen staples assumed available unless overridden by the user.
DEFAULT_PANTRY_STAPLES: List[str] = [
    "salt",
    "black pepper",
    "pepper",
    "water",
    "oil",
    "olive oil",
    "vegetable oil",
    "cooking spray",
]


# ---------------------------------------------------------------------------
# Explicit Dietary Exclusion Lists & Known Limitations
# ---------------------------------------------------------------------------
# Transparent keyword rules over ingredient lists.
#
# KNOWN LIMITATIONS & EDGE CASES:
# 1. Dataset parsing noise: Certain user-submitted recipes in Food.com have
#    unusual or omitted ingredient lists (e.g. recipe 536 "Corn Dog Casserole"
#    omitted hot dogs from its parsed ingredient list and only listed batter/spices).
# 2. Hidden non-vegetarian ingredients: Generic terms like "natural flavors",
#    "marshmallows" (gelatin), or "rennet" in cheese may not always be declared
#    in raw ingredient lists.
# 3. Cross-contamination: Commercial gluten/nut certifications cannot be verified
#    purely from ingredient text.
# 4. Homonyms & Culinary Nuance:
#    - "peanut butter" contains the word "butter" but is dairy-free;
#    - "coconut milk" contains "milk" but is plant-based;
#    - "butternut squash" contains "nut" but is a vegetable;
#    - "water chestnut" contains "chestnut" but is an aquatic tuber;
#    - "nutmeg" is a spice, not a tree nut.
#    These exceptions are explicitly handled via DIET_EXCEPTIONS to avoid
#    false-positive disqualifications.

DIET_EXCLUSIONS: Dict[str, List[str]] = {
    "vegetarian": [
        # Red meat & game
        "meat", "beef", "pork", "veal", "lamb", "mutton", "venison", "bison", "rabbit",
        "ham", "bacon", "prosciutto", "pancetta", "pork belly", "lard", "suet", "tallow",
        "gelatin", "gelatine",
        # Poultry
        "chicken", "turkey", "duck", "goose", "quail", "pheasant", "poultry",
        # Processed meats
        "sausage", "pepperoni", "salami", "chorizo", "hot dog", "frankfurter",
        "bratwurst", "kielbasa", "pastrami", "corned beef",
        # Seafood & Fish
        "fish", "salmon", "tuna", "cod", "halibut", "trout", "tilapia", "catfish",
        "mackerel", "sardine", "sardines", "anchovy", "anchovies", "snapper",
        "swordfish", "bass", "haddock", "pollock", "flounder", "sole", "mahi",
        "shrimp", "prawn", "prawns", "crab", "lobster", "crawfish", "crayfish",
        "clam", "clams", "mussel", "mussels", "oyster", "oysters", "scallop",
        "scallops", "squid", "calamari", "octopus", "caviar", "roe",
        # Meat stocks and broths
        "beef broth", "beef bouillon", "beef stock", "chicken broth",
        "chicken bouillon", "chicken stock", "bone broth", "fish stock",
        "clam juice", "dashi", "worcestershire", "worcestershire sauce",
    ],
    "vegan": [
        # Inherits all vegetarian meat/seafood exclusions, plus:
        # Red meat & seafood
        "meat", "beef", "pork", "veal", "lamb", "mutton", "venison", "bison", "rabbit",
        "ham", "bacon", "prosciutto", "pancetta", "pork belly", "lard", "suet", "tallow",
        "gelatin", "gelatine", "chicken", "turkey", "duck", "goose", "quail", "pheasant",
        "poultry", "sausage", "pepperoni", "salami", "chorizo", "hot dog", "frankfurter",
        "bratwurst", "kielbasa", "pastrami", "corned beef", "fish", "salmon", "tuna",
        "cod", "halibut", "trout", "tilapia", "catfish", "mackerel", "sardine",
        "sardines", "anchovy", "anchovies", "snapper", "swordfish", "bass", "haddock",
        "pollock", "flounder", "sole", "mahi", "shrimp", "prawn", "prawns", "crab",
        "lobster", "crawfish", "crayfish", "clam", "clams", "mussel", "mussels",
        "oyster", "oysters", "scallop", "scallops", "squid", "calamari", "octopus",
        "caviar", "roe", "beef broth", "beef bouillon", "beef stock", "chicken broth",
        "chicken bouillon", "chicken stock", "bone broth", "fish stock", "clam juice",
        "dashi", "worcestershire", "worcestershire sauce",
        # Dairy
        "milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "sour cream",
        "buttermilk", "whey", "casein", "ghee", "half-and-half", "custard", "curds",
        "lactose", "mascarpone", "ricotta", "mozzarella", "parmesan", "cheddar",
        "brie", "gouda", "feta", "provolone", "swiss cheese", "cottage cheese",
        "cream cheese", "condensed milk", "evaporated milk",
        # Eggs
        "egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks", "mayonnaise",
        # Bee products
        "honey", "beeswax", "royal jelly",
    ],
    "dairy_free": [
        "milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "sour cream",
        "buttermilk", "whey", "casein", "ghee", "half-and-half", "custard", "curds",
        "lactose", "mascarpone", "ricotta", "mozzarella", "parmesan", "cheddar",
        "brie", "gouda", "feta", "provolone", "swiss cheese", "cottage cheese",
        "cream cheese", "condensed milk", "evaporated milk",
    ],
    "gluten_free": [
        "wheat", "flour", "all-purpose flour", "bread flour", "cake flour", "pastry flour",
        "self-rising flour", "wheat flour", "whole wheat flour", "gluten",
        "vital wheat gluten", "barley", "rye", "spelt", "semolina", "couscous",
        "farro", "bulgur", "graham", "durum", "seitan", "malt", "brewer's yeast",
        "pasta", "macaroni", "spaghetti", "penne", "fettuccine", "lasagna",
        "linguine", "ramen", "udon", "breadcrumbs", "panko", "soy sauce", "beer",
    ],
    "nut_free": [
        "nut", "nuts", "peanut", "peanuts", "peanut butter", "almond", "almonds",
        "walnut", "walnuts", "cashew", "cashews", "pecan", "pecans", "pistachio",
        "pistachios", "hazelnut", "hazelnuts", "macadamia", "macadamias", "pine nut",
        "pine nuts", "chestnut", "chestnuts", "brazil nut", "brazil nuts", "praline",
        "marzipan",
    ],
}

DIET_EXCEPTIONS: Dict[str, List[str]] = {
    "vegetarian": [
        "vegetarian", "vegan", "meatless", "plant-based",
    ],
    "vegan": [
        "vegan", "plant-based", "peanut butter", "apple butter", "cocoa butter",
        "cacao butter", "shea butter", "coconut milk", "coconut cream", "almond milk",
        "soy milk", "oat milk", "rice milk", "egg substitute", "vegan mayonnaise",
    ],
    "dairy_free": [
        "peanut butter", "apple butter", "cocoa butter", "cacao butter", "shea butter",
        "coconut milk", "coconut cream", "almond milk", "soy milk", "oat milk",
        "rice milk", "dairy-free", "dairy free", "non-dairy", "vegan cheese",
    ],
    "gluten_free": [
        "gluten-free", "gluten free", "rice flour", "almond flour", "coconut flour",
        "cornstarch", "corn starch", "corn flour", "tapioca flour", "tapioca starch",
        "potato starch", "potato flour", "tamari", "rice noodles",
    ],
    "nut_free": [
        "water chestnut", "water chestnuts", "butternut squash", "butternut",
        "coconut", "coconuts", "coconut milk", "coconut cream", "coconut oil",
        "coconut flour", "nutmeg",
    ],
}


# Precompile regex word-boundary patterns for performance and whole-word matching
_DIET_EXCLUSION_PATTERNS: Dict[str, List[re.Pattern]] = {
    diet: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]
    for diet, terms in DIET_EXCLUSIONS.items()
}

_DIET_EXCEPTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    diet: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]
    for diet, terms in DIET_EXCEPTIONS.items()
}


# ---------------------------------------------------------------------------
# Ingredient Parsing and Normalisation Helpers
# ---------------------------------------------------------------------------

def parse_ingredients_field(val: Any) -> List[str]:
    """Parse a recipe's ingredient list into a list of clean string tokens.
    
    Handles list/iterable objects, JSON string lists, python string lists,
    or comma-separated string fallbacks.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]

    s = str(val).strip()
    if not s:
        return []

    # Check for JSON or Python list string format: '["butter", "flour"]'
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

    # Comma-delimited fallback
    return [x.strip().strip("'\"") for x in s.split(",") if x.strip().strip("'\"")]


def _simple_singular(word: str) -> str:
    """Apply conservative, English rule-based singularisation to a single word.
    
    Examples:
      - 'cherries' -> 'cherry'
      - 'tomatoes' -> 'tomato', 'potatoes' -> 'potato'
      - 'olives' -> 'olive', 'cloves' -> 'clove'
      - 'onions' -> 'onion', 'carrots' -> 'carrot', 'eggs' -> 'egg'
      - 'squash' -> 'squash', 'dress' -> 'dress'
    """
    w = word.strip().lower()
    if len(w) <= 3:
        return w

    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"

    # Avoid trimming 'ss', 'sh', 'ch', 'x', 'z' stems (e.g. glass, squash, peach)
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]

    if w.endswith("es") and len(w) > 4:
        if w.endswith(("toes", "soes")):
            return w[:-2]
        return w[:-1]

    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]

    return w


def normalize_ingredient(ingredient: str) -> str:
    """Normalise an ingredient string across user inputs and recipe lists.
    
    Performs:
      1. Lowercasing and strip.
      2. Replacement of underscores and hyphens with spaces.
      3. Removal of extraneous punctuation.
      4. Conservative singularisation on individual words.
    
    Parameters
    ----------
    ingredient : str
        Input ingredient text.
        
    Returns
    -------
    str
        Canonical normalised ingredient string.
    """
    if not ingredient:
        return ""
    s = str(ingredient).lower().replace("_", " ").replace("-", " ")
    # Keep alphanumeric characters and whitespace
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [_simple_singular(tok) for tok in s.split() if tok.strip()]
    return " ".join(tokens)


def _matches_exclusion_pattern(
    ingredient: str,
    exclusion_patterns: Sequence[re.Pattern],
    exception_patterns: Sequence[re.Pattern],
) -> bool:
    """Check if an ingredient matches an exclusion rule while checking exceptions."""
    clean = ingredient.lower().replace("_", " ").replace("-", " ")
    # If the ingredient matches an explicit exception, do not exclude
    for ep in exception_patterns:
        if ep.search(clean):
            return False

    for xp in exclusion_patterns:
        if xp.search(clean):
            return True
    return False


def _is_spice_pepper(ingredient_text: str) -> bool:
    """Differentiate seasoning pepper (black/white pepper) from vegetables (bell peppers)."""
    s_clean = ingredient_text.lower().replace("_", " ").replace("-", " ")
    veg_peppers = [
        "bell pepper", "sweet pepper", "green pepper", "red pepper",
        "yellow pepper", "chili pepper", "chilli pepper", "jalapeno",
        "habanero", "serrano", "poblano", "banana pepper", "ghost pepper",
    ]
    for vp in veg_peppers:
        if vp in s_clean:
            return False
    return True


def is_ingredient_covered(
    recipe_ingredient: str,
    available_normalized_set: Set[str],
) -> bool:
    """Determine whether a recipe ingredient is covered by available ingredients.
    
    Uses word-boundary matching and checks base ingredient tokens while
    avoiding deceptive sub-string traps (e.g. 'butter' does not match
    'peanut butter', and spice 'pepper' does not match 'bell pepper').
    """
    r_norm = normalize_ingredient(recipe_ingredient)
    if not r_norm:
        return True

    # 1. Exact canonical string match
    if r_norm in available_normalized_set:
        return True

    r_tokens = set(r_norm.split())

    # 2. Match against available ingredients
    for avail in available_normalized_set:
        if not avail:
            continue

        # Exact match
        if avail == r_norm:
            return True

        # Check special exception: 'butter' should not cover 'peanut butter'
        if avail == "butter" and "peanut butter" in r_norm:
            continue

        # Check special exception: 'pepper' should not cover vegetable peppers
        if avail in ("pepper", "black pepper") and not _is_spice_pepper(r_norm):
            continue

        # Whole phrase match with word boundaries
        if re.search(r"\b" + re.escape(avail) + r"\b", r_norm):
            return True

        # Token match: if single word available item is in recipe tokens
        if " " not in avail and avail in r_tokens:
            return True

    return False


# ---------------------------------------------------------------------------
# Part A: Recipe Filters
# ---------------------------------------------------------------------------

def filter_recipes(
    items_df: pd.DataFrame,
    diet: Optional[str] = None,
    max_calories: Optional[float] = None,
    max_minutes: Optional[float] = None,
    exclude_ingredients: Optional[Sequence[str]] = None,
) -> Set[Any]:
    """Filter recipes based on dietary rules, nutrition, prep time, and allergens.
    
    Parameters
    ----------
    items_df : pd.DataFrame
        DataFrame containing catalogue recipes with required columns:
        'item_id', 'ingredients_list', 'calories', 'total_minutes'.
    diet : str, optional
        Dietary restriction rule to apply. Supported options:
        'vegetarian', 'vegan', 'gluten_free', 'nut_free', 'dairy_free'.
        Implemented as keyword exclusion rules matched on whole ingredient words.
    max_calories : float, optional
        Maximum allowable calories. Recipes with missing (NaN/None) calories
        are excluded only when this filter is active.
    max_minutes : float, optional
        Maximum allowable total preparation/cooking time in minutes. Recipes
        with missing (NaN/None) total_minutes are excluded only when this filter
        is active.
    exclude_ingredients : Sequence[str], optional
        User-supplied list of disliked ingredients or allergens to exclude.
        Matched on whole ingredient words.
        
    Returns
    -------
    Set[Any]
        Set of item_ids that satisfy all active filter criteria.
    """
    if items_df.empty:
        return set()

    allowed_indices = np.ones(len(items_df), dtype=bool)

    # 1. Max Calories Filter
    if max_calories is not None:
        if "calories" in items_df.columns:
            cal_series = pd.to_numeric(items_df["calories"], errors="coerce")
            # Exclude missing values only when filter is active
            valid_cal = cal_series.notna() & (cal_series <= float(max_calories))
            allowed_indices &= valid_cal.values
        else:
            # If column missing entirely while filter is active, exclude all
            allowed_indices &= False

    # 2. Max Preparation Time Filter
    if max_minutes is not None:
        if "total_minutes" in items_df.columns:
            min_series = pd.to_numeric(items_df["total_minutes"], errors="coerce")
            # Exclude missing values only when filter is active
            valid_min = min_series.notna() & (min_series <= float(max_minutes))
            allowed_indices &= valid_min.values
        else:
            allowed_indices &= False

    # 3. Dietary & Ingredient-level Exclusions
    needs_ingredient_inspection = bool(diet or exclude_ingredients)

    if needs_ingredient_inspection and np.any(allowed_indices):
        exclusion_patterns: List[re.Pattern] = []
        exception_patterns: List[re.Pattern] = []

        if diet is not None:
            diet_key = str(diet).lower().strip().replace("-", "_").replace(" ", "_")
            if diet_key not in _DIET_EXCLUSION_PATTERNS:
                valid_options = list(_DIET_EXCLUSION_PATTERNS.keys())
                raise ValueError(
                    f"Unsupported diet '{diet}'. Supported options: {valid_options}"
                )
            exclusion_patterns.extend(_DIET_EXCLUSION_PATTERNS[diet_key])
            exception_patterns.extend(_DIET_EXCEPTION_PATTERNS.get(diet_key, []))

        if exclude_ingredients is not None:
            for item in exclude_ingredients:
                s_item = str(item).strip()
                if s_item:
                    pattern = re.compile(r"\b" + re.escape(s_item) + r"\b", re.IGNORECASE)
                    exclusion_patterns.append(pattern)

        # Iterate over remaining candidate rows
        ing_col = "ingredients_list" if "ingredients_list" in items_df.columns else "ingredients"
        if ing_col in items_df.columns:
            for idx in np.where(allowed_indices)[0]:
                raw_ings = items_df.iloc[idx][ing_col]
                ing_list = parse_ingredients_field(raw_ings)
                # Check each ingredient in the recipe
                blocked = False
                for ing in ing_list:
                    if _matches_exclusion_pattern(ing, exclusion_patterns, exception_patterns):
                        blocked = True
                        break
                if blocked:
                    allowed_indices[idx] = False

    id_col = "item_id" if "item_id" in items_df.columns else items_df.columns[0]
    return set(items_df.loc[allowed_indices, id_col].tolist())


# ---------------------------------------------------------------------------
# Part B: Kitchen Mode Recommender
# ---------------------------------------------------------------------------

def _resolve_items_df(model: Any, items_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Resolve catalogue DataFrame from passed argument, model attribute, or disk."""
    if items_df is not None:
        return items_df

    # Check model attributes
    if hasattr(model, "items_df_") and isinstance(model.items_df_, pd.DataFrame):
        return model.items_df_
    content_mod = getattr(model, "content_model", None)
    if content_mod is not None and hasattr(content_mod, "items_df_"):
        if isinstance(content_mod.items_df_, pd.DataFrame):
            return content_mod.items_df_

    # Default to Food.com processed items.csv
    items_path = FOODCOM_CONFIG.data_dir / "items.csv"
    if items_path.exists():
        return pd.read_csv(items_path)

    alt_path = Path("Data/processed/foodcom/items.csv")
    if alt_path.exists():
        return pd.read_csv(alt_path)

    raise FileNotFoundError(
        "Could not automatically resolve items_df. Please pass items_df explicitly."
    )


def kitchen_recommend(
    model: BaseRecommender,
    user: Any,
    available_ingredients: Union[str, Sequence[str]],
    n: int = 10,
    min_overlap: float = 0.5,
    alpha_kitchen: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
    items_df: Optional[pd.DataFrame] = None,
    pantry_staples: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Recommend recipes matching user-available kitchen ingredients and preference scores.
    
    Parameters
    ----------
    model : BaseRecommender
        Fitted recommender model (HybridRecommender, ContentBased, etc.).
    user : Any
        Identifier of the target user.
    available_ingredients : Union[str, Sequence[str]]
        Ingredients the user has on hand, as a comma-separated string or list.
    n : int, default=10
        Number of recommendations to return.
    min_overlap : float, default=0.5
        Minimum fraction of recipe ingredients the user must have to consider
        the recipe viable.
    alpha_kitchen : float, default=0.5
        Weight parameter in [0.0, 1.0] balancing ingredient overlap against
        the user's hybrid preference score:
            final_score = alpha_kitchen * overlap + (1 - alpha_kitchen) * norm_hybrid_score.
        For a user not present in training data, final_score = overlap.
    filters : Dict[str, Any], optional
        Additional constraints (e.g. {'diet': 'vegetarian', 'max_minutes': 45})
        passed to filter_recipes().
    items_df : pd.DataFrame, optional
        Catalogue items DataFrame. If None, resolved from model or disk.
    pantry_staples : Sequence[str], optional
        Common staples assumed available. Defaults to DEFAULT_PANTRY_STAPLES.
        Pass an empty list [] to disable staple assumptions.
        
    Returns
    -------
    List[Dict[str, Any]]
        Ranked list of up to n dictionaries containing:
          - 'item_id': Unique recipe ID
          - 'name': Human-readable recipe name
          - 'overlap': Fraction of ingredients covered in [0.0, 1.0]
          - 'missing_ingredients': List of ingredients the user is missing
          - 'hybrid_score': Min-max normalised preference score (or None for unknown user)
          - 'final_score': Blended kitchen score in [0.0, 1.0]
          - 'explanation': Human-readable explanation of ingredient coverage & missing items
    """
    df = _resolve_items_df(model, items_df)

    # 1. Parse available ingredients and pantry staples
    if isinstance(available_ingredients, str):
        user_ing_list = [s.strip() for s in available_ingredients.split(",") if s.strip()]
    elif available_ingredients is None:
        user_ing_list = []
    else:
        user_ing_list = [str(s).strip() for s in available_ingredients if str(s).strip()]

    staples = DEFAULT_PANTRY_STAPLES if pantry_staples is None else list(pantry_staples)

    # Normalise user items and staples into canonical set
    available_set: Set[str] = set()
    for item in user_ing_list:
        norm = normalize_ingredient(item)
        if norm:
            available_set.add(norm)
    for staple in staples:
        norm = normalize_ingredient(staple)
        if norm:
            available_set.add(norm)

    # 2. Apply additional filters if provided
    allowed_ids: Optional[Set[Any]] = None
    if filters:
        allowed_ids = filter_recipes(df, **filters)

    # 3. Compute overlap and missing ingredients for candidate recipes
    id_col = "item_id" if "item_id" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else ("title" if "title" in df.columns else df.columns[1])
    ing_col = "ingredients_list" if "ingredients_list" in df.columns else "ingredients"

    candidates: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        item_id = row[id_col]
        if allowed_ids is not None and item_id not in allowed_ids:
            continue

        raw_ings = row[ing_col] if ing_col in df.columns else []
        ing_list = parse_ingredients_field(raw_ings)

        if not ing_list:
            overlap = 0.0
            missing = []
            covered_count = 0
            total_count = 0
        else:
            covered: List[str] = []
            missing: List[str] = []
            for ing in ing_list:
                if is_ingredient_covered(ing, available_set):
                    covered.append(ing)
                else:
                    missing.append(ing)
            total_count = len(ing_list)
            covered_count = len(covered)
            overlap = float(covered_count / total_count) if total_count > 0 else 0.0

        if overlap >= float(min_overlap):
            recipe_name = str(row[name_col]) if name_col in df.columns else get_item_name(item_id)
            candidates.append({
                "item_id": item_id,
                "name": recipe_name,
                "overlap": round(overlap, 4),
                "covered_count": covered_count,
                "total_count": total_count,
                "missing_ingredients": missing,
            })

    if not candidates:
        return []

    # 4. Check if user is known in training data
    user_is_known = False
    if hasattr(model, "user_counts_"):
        user_is_known = user in getattr(model, "user_counts_")
    elif hasattr(model, "user2idx_"):
        user_is_known = user in getattr(model, "user2idx_")
    elif hasattr(model, "user_profiles_"):
        user_is_known = user in getattr(model, "user_profiles_")

    # 5. Hybrid Preference Scoring
    if not user_is_known:
        # User not in data: rank by overlap only
        for c in candidates:
            c["hybrid_score"] = None
            c["final_score"] = c["overlap"]
            # Generate explanation
            cov = c["covered_count"]
            tot = c["total_count"]
            pct = int(round(c["overlap"] * 100))
            missing_str = ", ".join(c["missing_ingredients"]) if c["missing_ingredients"] else "none"
            c["explanation"] = (
                f"Matches {cov} of {tot} ingredients ({pct}% overlap). "
                f"Missing: {missing_str}."
            )
    else:
        # User is known: predict scores and min-max normalise across candidate set
        raw_scores: List[float] = []
        for c in candidates:
            try:
                score = float(model.predict(user, c["item_id"]))
            except Exception:
                score = 3.0
            raw_scores.append(score)

        s_arr = np.array(raw_scores, dtype=np.float64)
        min_s = float(np.min(s_arr))
        max_s = float(np.max(s_arr))
        rng_s = max_s - min_s

        if rng_s > 1e-9:
            norm_scores = (s_arr - min_s) / rng_s
        else:
            norm_scores = np.full_like(s_arr, 0.5)

        for i, c in enumerate(candidates):
            h_norm = float(norm_scores[i])
            f_score = float(alpha_kitchen * c["overlap"] + (1.0 - alpha_kitchen) * h_norm)
            c["hybrid_score"] = round(h_norm, 4)
            c["final_score"] = round(f_score, 4)

            # Generate transparent explanation
            cov = c["covered_count"]
            tot = c["total_count"]
            pct = int(round(c["overlap"] * 100))
            missing_str = ", ".join(c["missing_ingredients"]) if c["missing_ingredients"] else "none"
            c["explanation"] = (
                f"Matches {cov} of {tot} ingredients ({pct}% overlap). "
                f"Missing: {missing_str} (Hybrid preference score: {c['hybrid_score']:.2f})."
            )

    # Clean internal temporary keys
    for c in candidates:
        c.pop("covered_count", None)
        c.pop("total_count", None)

    # Sort descending by final_score, then overlap
    candidates.sort(key=lambda x: (x["final_score"], x["overlap"]), reverse=True)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Part C: Filtered Recommendations
# ---------------------------------------------------------------------------

def recommend_with_filters(
    model: BaseRecommender,
    user: Any,
    n: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    items_df: Optional[pd.DataFrame] = None,
    exclude_seen: bool = True,
) -> List[Tuple[Any, float]]:
    """Generate top-N recommendations with pre-ranking constraint filtering.
    
    Filters are applied directly to the candidate set before ranking to ensure
    that the top-N list is always full if enough eligible recipes exist.
    
    Parameters
    ----------
    model : BaseRecommender
        Fitted recommender model.
    user : Any
        Identifier of the target user.
    n : int, default=10
        Number of recommendations to return.
    filters : Dict[str, Any], optional
        Dictionary of filter arguments to pass to filter_recipes()
        (e.g. {'diet': 'vegetarian', 'max_minutes': 45, 'max_calories': 600}).
    items_df : pd.DataFrame, optional
        Catalogue items DataFrame. If None, resolved from model or disk.
    exclude_seen : bool, default=True
        Whether to exclude items the user has already interacted with.
        
    Returns
    -------
    List[Tuple[Any, float]]
        List of (item_id, predicted_score) tuples sorted descending by score.
    """
    df = _resolve_items_df(model, items_df)
    id_col = "item_id" if "item_id" in df.columns else df.columns[0]
    all_catalogue_ids = set(df[id_col].tolist())

    # 1. Determine allowed candidate item_ids
    if filters:
        allowed_ids = filter_recipes(df, **filters)
    else:
        allowed_ids = all_catalogue_ids

    if not allowed_ids:
        return []

    # 2. Obtain recommendations from model across catalogue
    # Rank candidates so that the top-N slice after filtering remains full
    if hasattr(model, "recommend"):
        # Request all items from model.recommend to preserve full catalogue ranking
        all_recs = model.recommend(user, n=len(all_catalogue_ids), exclude_seen=exclude_seen)
        filtered_recs = [
            (item_id, float(score))
            for item_id, score in all_recs
            if item_id in allowed_ids
        ]
        if len(filtered_recs) >= n or len(filtered_recs) == len(allowed_ids):
            return filtered_recs[:n]

    # Fallback if model.recommend returned fewer candidates than allowed_ids:
    # Score remaining candidates using model.predict
    seen_set: Set[Any] = set()
    if exclude_seen:
        if hasattr(model, "user_seen_set_") and user in model.user_seen_set_:
            seen_set = model.user_seen_set_[user]

    candidate_ids = [it for it in allowed_ids if it not in seen_set]
    scored = [(it, float(model.predict(user, it))) for it in candidate_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]

