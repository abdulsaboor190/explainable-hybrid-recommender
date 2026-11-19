"""CLI launcher for the Streamlit demo application.

Usage:
    python -m src.run_app [--port 8501]

Checks that pipeline model caches exist on disk (building them if not),
verifies streamlit is installed, and spawns the Streamlit server.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_models


def ensure_caches() -> None:
    """Check that train_val cache files exist for both datasets, building if needed."""
    print("Checking model caches for demo application...")
    for ds in ["foodcom", "movielens"]:
        cache_file = PROJECT_ROOT / "data" / "cache" / ds / "models_train_val.joblib"
        if not cache_file.exists():
            print(f"  Building cached models for {ds} (fit_on='train_val')...")
            build_models(ds, fit_on="train_val")
        else:
            print(f"  Cache verified for {ds}: {cache_file.name}")
    print("All model caches ready.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Streamlit Recommender Demo App.")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically.")
    args = parser.parse_args()

    ensure_caches()

    app_path = PROJECT_ROOT / "app" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Missing Streamlit entry file at {app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    if args.no_browser:
        cmd.append("--server.headless=true")

    print(f"Starting Streamlit: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit demo stopped.")


if __name__ == "__main__":
    main()
