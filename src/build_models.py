"""Command-line utility to build and cache all recommender models.

Usage:
    python -m src.build_models --dataset movielens
    python -m src.build_models --dataset foodcom
    python -m src.build_models --dataset movielens --fit-on train_val
    python -m src.build_models --dataset foodcom --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, fit, and cache recommender models for MovieLens or Food.com."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["movielens", "foodcom"],
        help="Target dataset to build models for ('movielens' or 'foodcom').",
    )
    parser.add_argument(
        "--fit-on",
        type=str,
        default="train",
        choices=["train", "train_val"],
        help="Training interaction split: 'train' (for tuning/validation) or 'train_val' (for final evaluation).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild from scratch, ignoring existing cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print(f"BUILD MODELS PIPELINE: {args.dataset.upper()} (fit_on={args.fit_on}, force={args.force})")
    print("=" * 80)

    t0 = time.perf_counter()
    models = build_models(
        dataset=args.dataset,
        fit_on=args.fit_on,
        force_rebuild=args.force,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nSuccessfully ready {len(models)} models in {elapsed:.3f} seconds:")
    for name, model in models.items():
        print(f"  - {name:<22}: {model.__class__.__name__}")
    print("\nModels are cached and immediately available for demo and evaluation.")


if __name__ == "__main__":
    main()

