"""
Runner script to compare Fama-French vs XGBoost predictive performance.

Usage:
    python -m src.run_model_comparison --test-size 0.2 --out results.json

Ensure that a Fama-French factors CSV exists in src/data_collection/data.
You can generate it with: python -m src.run_data_gathering
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.compare_models import compare_models


def main():
    parser = argparse.ArgumentParser(description="Compare Fama-French vs XGBoost models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (0-1)")
    parser.add_argument("--out", type=str, default="", help="Optional JSON file path to save results")
    args = parser.parse_args()

    results = compare_models(test_size=args.test_size)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved comparison results to {out_path}")


if __name__ == "__main__":
    main()
