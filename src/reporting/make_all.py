"""Reporting master runner.

Regenerates every figure and table from results/*.csv files alone (no model refitting).
Usage:
    python -m src.reporting.make_all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reporting.plots import generate_all_plots
from src.reporting.tables import generate_all_tables


def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print("REGENERATING ALL EXPERIMENTAL FIGURES AND TABLES FROM CSVs")
    print("=" * 75)

    plots = generate_all_plots()
    print(f"\nSuccessfully generated {len(plots)} figures in results/figures/.")

    tables = generate_all_tables()
    print(f"\nSuccessfully generated {len(tables)} tables (Markdown & LaTeX) in results/tables/.")

    print(f"\nComplete pipeline finished in {time.time()-t0:.2f}s.")
    print("=" * 75)


if __name__ == "__main__":
    main()

