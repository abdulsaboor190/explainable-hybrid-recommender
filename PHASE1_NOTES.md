# Phase 1 — Implementation Notes

## Final Dataset Counts

### MovieLens 100K
| Split | Rows |
|-------|-----:|
| Full ratings | 100,000 |
| Train | 72,505 |
| Val   |  8,057 |
| Test  | 19,438 |
| Items | 1,682 |
| Users |   943 |
| Sparsity | 0.9370 |
| Cold-start test items | 83 unique (114 rows) |

### Food.com (3,000-user activity-filtered sample)
| Split | Rows |
|-------|-----:|
| Raw ratings | 1,325,734 |
| After iterative filter (≥20/≥20) | 168,460 |
| After sampling 3,000 users | 164,278 |
| Train | 119,916 |
| Val   |  13,313 |
| Test  |  31,049 |
| Items | 3,889 |
| Users | 3,000 |
| Sparsity | 0.9859 |
| Cold-start test items | 0 |

---

## Design Decisions

### 1 · Shared `BaseRecommender` ABC (`src/base.py`)
All models inherit `fit / predict / recommend / explain`. `predict` is contractually bounded to [1, 5] with an explicit fallback chain (user mean → global mean) for unknown users/items. This makes the cold-start policy uniform across every model and forces each one to document its explanation logic.

### 2 · Temporal split with per-user floors (`src/split.py`)
We sort each user's ratings by `(timestamp, item_id)` for full determinism, then allocate the newest 20 % to test and the newest 10 % of the remaining pool to validation. `math.floor` is used for the test cut; `round` for the val cut. Both are clamped so every user always has ≥ 1 row in train and ≥ 1 in test, preventing empty groups that would crash evaluation. This avoids data leakage: the tuning (val) set is strictly older than test for every user.

### 3 · Food.com iterative core filter
A single-pass filter would leave users who barely qualify because neighbours were removed. The iterative approach (repeat until convergence) guarantees the final bipartite graph is truly ≥20-core — every user and every recipe has at least 20 mutual connections inside the filtered set. This takes only 2–3 iterations in practice.

### 4 · Food.com user sampling after filtering
Sampling is applied **after** the core filter, not before. This ensures the sample is drawn from already-active users, so the ≥20-rating guarantee holds post-sample too. `numpy.random.default_rng(42)` is used instead of the legacy `np.random.seed` for reproducibility without global state side-effects.

### 5 · Chunked I/O for Food.com (`src/data_foodcom.py`)
The raw CSVs are ~700 MB (recipes) and ~500 MB (reviews). Loading both at once would require >2 GB RAM. We use `pd.read_csv(..., chunksize=..., usecols=...)` to stream only the columns we need, discarding invalid ratings inside each chunk before concatenation. Peak memory stays under ~600 MB.

### 6 · R-style vector parsing
Food.com stores keyword and ingredient lists as R character vectors: `c("item1", "item2", ...)`. A regex-based parser extracts quoted tokens, handles edge cases (`character(0)`, `NA`, bare strings, single-quoted values). Multi-word ingredients are underscore-tokenised (`"olive oil"` → `"olive_oil"`) so TF-IDF treats them as single features.

### 7 · ISO 8601 duration parsing
`TotalTime` values like `PT1H30M` are parsed with two separate regex groups (hours, minutes), so fractional hours and pure-minute durations both work. Items with unparseable durations get `NaN` (not dropped — they can still be recommended by CF).

### 8 · `matplotlib` added to `requirements.txt`
`matplotlib==3.10.1` was not in the original pinned list. It was added after first use and pinned immediately to maintain reproducibility.

---

## Problems / Deviations

| Issue | Resolution |
|-------|-----------|
| `data/raw/foodcom/` path didn't exist on first run (files were in `Data/food/`) | Copied files to the canonical path; documented correct location in README |
| `data/raw/ml-100k/` likewise missing | Copied from pre-existing `Data/movies/` directory |
| `.gitignore` blocked all of `results/` including CSVs and figures | Narrowed rule to only block `*.pkl`, `*.npz`, `.cache/`; committed outputs |
| `matplotlib` missing from `requirements.txt` | Added `matplotlib==3.10.1`; re-pinned |
| Git author identity not set | Set repo-local `user.name` / `user.email` on first commit |
| Food.com `rpi_min` = 17 after sampling (not 20) | Expected: sampling removes some item ratings, so a few items dip below 20 in the sampled subset even though they passed the filter. The filter guarantee is on the pre-sample full set. This is acceptable — the minimum threshold applies during filtering, not to the final sample. |

---

## Reproducibility Checklist

| Check | Result |
|-------|--------|
| `python -m src.run_phase1` from clean state produces identical counts | ✅ Verified (clean delete + re-run) |
| `pytest -q` passes all tests | ✅ 27/27 passed in 1.31 s |
| Random seed = 42 used in all stochastic steps | ✅ |
| No raw data committed to git | ✅ `data/` in `.gitignore` |

