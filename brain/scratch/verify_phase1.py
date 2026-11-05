"""Exit criteria verification script for Phase 1."""
import json
from pathlib import Path

checks = []

def chk(label, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if evidence:
        print(f"         {evidence}")
    checks.append((label, cond))

print()
print("=== Exit Criteria Verification ===")
print()

# 1. Split files exist for both datasets
for ds in ["movielens", "foodcom"]:
    for split in ["train", "val", "test"]:
        p = Path(f"data/processed/{ds}/{split}.csv")
        if p.exists():
            rows = sum(1 for _ in open(p)) - 1
            chk(f"{ds}/{split}.csv exists", True, f"{rows:,} rows")
        else:
            chk(f"{ds}/{split}.csv exists", False, "MISSING")

# 2. Statistics table complete
for ds in ["movielens", "foodcom"]:
    p = Path(f"results/dataset_stats_{ds}.csv")
    if p.exists():
        lines = sum(1 for _ in open(p)) - 1
        chk(f"dataset_stats_{ds}.csv complete (>=25 metrics)", lines >= 25, f"{lines} metrics")
    else:
        chk(f"dataset_stats_{ds}.csv complete", False, "MISSING")

md = Path("results/dataset_stats_summary.md")
chk("dataset_stats_summary.md exists", md.exists())

# 3. BaseRecommender documented
bp = Path("src/base.py")
content = bp.read_text()
n_abstract = content.count("@abstractmethod")
n_docstrings = content.count('"""')
chk("BaseRecommender ABC with full docstrings",
    "ABC" in content and "abstractmethod" in content and n_docstrings >= 8,
    f"@abstractmethod x{n_abstract}, docstring blocks x{n_docstrings//2}")

# 4. filter_info.json records thresholds and counts
fi = Path("data/processed/foodcom/filter_info.json")
if fi.exists():
    info = json.load(open(fi))
    ok_thresh = info["config"]["min_ratings_per_user"] == 20
    ok_counts = all(k in info["counts"] for k in ["before_filter", "after_filter", "after_sample"])
    chk("filter_info.json thresholds + all count stages", ok_thresh and ok_counts,
        str(info["config"]))
else:
    chk("filter_info.json exists", False, "MISSING")

# 5. .gitignore excludes data/
gi = Path(".gitignore").read_text()
chk(".gitignore excludes data/", "data/" in gi)

print()
passed = sum(1 for _, c in checks if c)
total = len(checks)
print(f"  Result: {passed}/{total} checks passed")

