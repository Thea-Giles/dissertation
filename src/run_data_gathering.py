"""
Runner script to gather market data and Fama-French factors, merge them,
and save a CSV for downstream model comparison.

Usage:
    python -m src.run_data_gathering --start 2014-01-01 --end 2024-12-31 --ticker ^GSPC
    python -m src.run_data_gathering --outdir src/data_collection/data
"""
from __future__ import annotations

import argparse
from datetime import datetime, date
from pathlib import Path

import pandas as pd

from data_collection.market_data import (
    get_stock_data,
    get_factor_data,
    calculate_fama_french_factors,
)


def _default_outdir() -> Path:
    # Place under src/data_collection/data relative to this file
    here = Path(__file__).resolve().parent
    return here / "data_collection" / "data"


def run(start: str, end: str, ticker: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    # Fetch market data and Fama-French factors
    market_df = get_stock_data(ticker=ticker, start_date=start, end_date=end)
    if market_df.empty:
        raise RuntimeError("No market data retrieved. Check ticker and date range.")

    ff_df = get_factor_data(start_date=start, end_date=end)
    if ff_df.empty:
        raise RuntimeError("No Fama-French factor data retrieved. Check date range or network connectivity.")

    merged = calculate_fama_french_factors(market_data=market_df, factor_data=ff_df)
    # Ensure index name and cleanliness
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "Date"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"fama_french_factors_{timestamp}.csv"
    merged.to_csv(outfile, index=True)
    print(f"Saved merged Fama-French + market data to: {outfile}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Gather market and Fama-French data and save CSV")
    parser.add_argument("--start", type=str, default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=date.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Market index ticker (default ^GSPC)")
    parser.add_argument("--outdir", type=str, default=str(_default_outdir()), help="Output directory for CSV")

    args = parser.parse_args()
    run(start=args.start, end=args.end, ticker=args.ticker, outdir=Path(args.outdir))


if __name__ == "__main__":
    main()
