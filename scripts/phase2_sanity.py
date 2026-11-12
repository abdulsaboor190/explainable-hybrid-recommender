"""Phase 2 sanity check runner script.

Executes test set evaluation for both MovieLens and Food.com across:
  - Global Mean, User Mean, Item Mean, Popularity
  - User-CF, Item-CF, Content
  - Fixed Hybrid, Adaptive Hybrid

Saves outputs to results/phase2_sanity_<dataset>.csv.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_phase2 import run_all


if __name__ == "__main__":
    run_all()

