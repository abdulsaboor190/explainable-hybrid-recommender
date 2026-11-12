"""Demonstration script showcasing explanations for 3 users on MovieLens and Food.com."""

from typing import List, Tuple
import pandas as pd

from src.models.cf import ItemBasedCF
from src.models.content import ContentBased
from src.models.hybrid import HybridRecommender
from src.items import get_item_name_map, get_item_name


def run_dataset_demo(
    name: str,
    train_path: str,
    items_path: str,
    text_col: str,
    item_type: str,
    k0: float,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"EXPLANATION DEMO: {name.upper()}")
    print(f"{'=' * 80}")

    train_df = pd.read_csv(train_path)
    items_df = pd.read_csv(items_path)
    name_map = get_item_name_map(name)

    print("Fitting CF and Content models...")
    cf = ItemBasedCF(similarity="cosine", k=30, min_overlap=2)
    cf.fit(train_df)

    content = ContentBased(items_df, text_column=text_col)
    content.fit(train_df)

    hybrid = HybridRecommender(cf, content, mode="adaptive", k0=k0)
    hybrid.fit(train_df)

    # Pick 3 users: few ratings, middle (median), many ratings
    user_counts = train_df.groupby("user_id").size()
    min_val = user_counts.min()
    med_val = int(user_counts.median())
    max_val = user_counts.max()

    # Find specific user_ids
    user_few = user_counts[user_counts == min_val].index[0]
    user_mid = user_counts[user_counts == med_val].index[0]
    user_many = user_counts[user_counts == max_val].index[0]

    selected_users = [
        ("Few ratings", user_few, min_val),
        ("Middle ratings", user_mid, med_val),
        ("Many ratings", user_many, max_val),
    ]

    for label, u, cnt in selected_users:
        alpha = hybrid.get_alpha(u)
        print(f"\n--- User {u} ({label}: {cnt} training ratings, alpha={alpha:.2f}) ---")
        recs = hybrid.recommend(u, n=3, exclude_seen=True)
        for rank, (item_id, score) in enumerate(recs, 1):
            title = get_item_name(item_id, name_map)
            explanation = hybrid.explain(u, item_id, item_name_map=name_map, item_type=item_type)
            print(f"  {rank}. [{title}] (Hybrid Score: {score:.3f})")
            print(f"     Reason: {explanation}\n")


if __name__ == "__main__":
    # 1. MovieLens (k0 = 20.0)
    run_dataset_demo(
        name="movielens",
        train_path="data/processed/movielens/train.csv",
        items_path="data/processed/movielens/items.csv",
        text_col="genre_tokens",
        item_type="movie",
        k0=20.0,
    )

    # 2. Food.com (k0 = 100.0)
    run_dataset_demo(
        name="foodcom",
        train_path="data/processed/foodcom/train.csv",
        items_path="data/processed/foodcom/items.csv",
        text_col="text",
        item_type="recipe",
        k0=100.0,
    )

