import pandas as pd
for ds in ["movielens","foodcom"]:
    r  = pd.read_csv(f"data/processed/{ds}/ratings.csv")
    tr = pd.read_csv(f"data/processed/{ds}/train.csv")
    va = pd.read_csv(f"data/processed/{ds}/val.csv")
    te = pd.read_csv(f"data/processed/{ds}/test.csv")
    print(f"{ds}: {len(r):,} ratings | {r.user_id.nunique()} users | {r.item_id.nunique()} items | train {len(tr):,} / val {len(va):,} / test {len(te):,}")

items_ml = pd.read_csv("data/processed/movielens/items.csv")
items_fc = pd.read_csv("data/processed/foodcom/items.csv")
print("ML items cols:", list(items_ml.columns))
print("FC items cols:", list(items_fc.columns))

