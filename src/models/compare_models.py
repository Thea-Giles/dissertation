"""
Compare Fama-French vs XGBoost for predicting market returns.

This script loads the latest Fama-French factors dataset produced by the data
pipeline, prepares a consistent train/test split, trains both models, and
compares their out-of-sample performance using the same test horizon.

Usage:
    python -m src.models.compare_models

Requirements:
    - src/data_collection/data/fama_french_factors_*.csv exists with columns:
      Date, Daily_Return (or similar), MRP, RF, SMB, HML
    - xgboost installed for the XGBoost model
"""
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .fama_french import FamaFrenchModel
from .xgboost_model import FinancialXGBoostModel


def _find_latest_factors_csv() -> Optional[str]:
    """Locate the most recent fama_french_factors_*.csv in the data folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # FIX: only go up one level to reach .../src
    src_dir = os.path.dirname(script_dir)  # .../src
    data_dir = os.path.join(src_dir, "data_collection", "data")
    print(f"Looking in: {data_dir}")
    if not os.path.isdir(data_dir):
        print("Data directory does not exist")
        return None
    # FIX: remove leading slash in the pattern so data_dir is not discarded
    pattern = os.path.join(data_dir, "fama_french_factors_*.csv")
    files = glob.glob(pattern)
    print(f"Matched files: {files[:3]}{'...' if len(files) > 3 else ''}")
    if not files:
        return None
    return max(files, key=os.path.getmtime)




def _load_factors_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the factors CSV robustly, handling both standard and pipeline-specific formats."""
    # First try a standard CSV where Date is a column or an unnamed index column
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) == 0:
            raise ValueError("Empty CSV")
        first_col = df.columns[0]
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            return df
        # Unnamed index column
        if first_col and ('date' in str(first_col).lower() or 'unnamed' in str(first_col).lower()):
            df = df.rename(columns={first_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            return df
    except Exception:
        pass

    # Fallback: previous pipeline format with extra header rows
    header_df = pd.read_csv(csv_path, nrows=1)
    column_names = header_df.columns.tolist()
    df = pd.read_csv(csv_path, skiprows=3, header=None, names=column_names)
    date_col = column_names[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def _detect_return_column(df: pd.DataFrame) -> str:
    if 'Daily_Return' in df.columns:
        return 'Daily_Return'
    # Fallback: first column containing 'return'
    candidates = [c for c in df.columns if 'return' in str(c).lower()]
    if candidates:
        return candidates[0]
    raise ValueError("No return column found in the factors data")


def _compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def compare_models(test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate Fama-French and XGBoost models on the same test horizon.

    Returns a dictionary with metrics for both models.
    """
    csv_path = _find_latest_factors_csv()
    if not csv_path:
        raise FileNotFoundError(
            "No Fama-French factors data found. Please generate it first by running: "
            "python -m src.run_data_gathering (which saves to src/data_collection/data)."
        )

    print(f"Using data from: {csv_path}")
    df = _load_factors_dataframe(csv_path)

    # Identify needed columns
    ret_col = _detect_return_column(df)
    required = ['MRP', 'RF']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in factors data: {missing}")

    # Optional FF columns
    has_smb = 'SMB' in df.columns
    has_hml = 'HML' in df.columns

    # Prepare market returns and ensure viability
    market_returns = df['MRP'] + df['RF']  # Market return = MRP + RF

    # Build a minimal market_data DataFrame for the XGBoost model
    market_data = pd.DataFrame({
        'Date': df.index.tz_localize(None) if getattr(df.index, 'tz', None) is not None else df.index,
        'Daily_Return': df[ret_col].values,
    })

    # 1) Prepare XGBoost features on the full series (it will drop initial NaNs)
    xgb_model = FinancialXGBoostModel(model_type='regression', random_state=42)
    features_df = xgb_model.create_features(market_data=market_data, sentiment_data=None, target_column='Daily_Return')

    # Determine a common cutoff date based on XGBoost's feature-ready dataset
    features_df = features_df.sort_values('Date')
    cutoff_idx = int(len(features_df) * (1 - test_size))
    if cutoff_idx <= 0 or cutoff_idx >= len(features_df):
        raise ValueError("Not enough data to perform the requested train/test split.")
    cutoff_date = pd.to_datetime(features_df['Date'].iloc[cutoff_idx])

    # 2) Fama-French: train on dates before cutoff_date, test on/after cutoff_date
    stock_returns_series = df[ret_col]
    smb_series = df['SMB'] if has_smb else None
    hml_series = df['HML'] if has_hml else None

    train_mask_ff = df.index < cutoff_date
    test_mask_ff = df.index >= cutoff_date

    ff_model = FamaFrenchModel(risk_free_rate=df['RF'].mean() * 252)  # Annualized RF
    ff_model.fit(
        stock_returns=stock_returns_series.loc[train_mask_ff],
        market_returns=market_returns.loc[train_mask_ff],
        smb=smb_series.loc[train_mask_ff] if has_smb else None,
        hml=hml_series.loc[train_mask_ff] if has_hml else None,
    )

    ff_pred = ff_model.predict(
        market_returns=market_returns.loc[test_mask_ff],
        smb=smb_series.loc[test_mask_ff] if has_smb else None,
        hml=hml_series.loc[test_mask_ff] if has_hml else None,
    )

    ff_actual = stock_returns_series.loc[ff_pred.index]

    # Evaluate FF
    ff_eval = ff_model.evaluate(ff_actual, ff_pred)
    # Compute R2 on aligned, non-NaN observations to avoid sklearn errors
    _ff_df = pd.DataFrame({'actual': ff_actual, 'predicted': ff_pred}).dropna()
    if len(_ff_df) >= 2:
        ff_r2 = r2_score(_ff_df['actual'].values, _ff_df['predicted'].values)
    else:
        ff_r2 = float('nan')

    # 3) XGBoost: use its temporal split machinery then override test horizon to align by cutoff_date
    # Prepare scaled features with temporal split
    X_train, X_test, y_train, y_test = xgb_model.prepare_train_test_data(features_df, test_size=test_size, time_series_split=True)

    # Train with early stopping
    _ = xgb_model.train(X_train, y_train, X_test=X_test, y_test=y_test, tune_hyperparams=False)

    # Evaluate on XGBoost's test split (which corresponds to dates >= cutoff_date by construction)
    xgb_eval = xgb_model.evaluate(X_test, y_test)
    xgb_dir_acc = _compute_directional_accuracy(y_test, xgb_eval['predictions'])

    results = {
        'fama_french': {
            'mse': float(ff_eval['mse']),
            'rmse': float(ff_eval['rmse']),
            'mae': float(ff_eval['mae']),
            'r2_score': float(ff_r2),
            'directional_accuracy': float(ff_eval['directional_accuracy']),
        },
        'xgboost': {
            'mse': float(xgb_eval['mse']),
            'rmse': float(xgb_eval['rmse']),
            'mae': float(xgb_eval['mae']),
            'r2_score': float(xgb_eval['r2_score']),
            'directional_accuracy': float(xgb_dir_acc),
        }
    }

    # Pretty print
    print("\nModel Comparison (same test horizon):")
    print("-----------------------------------")
    for model_name, metrics in results.items():
        print(f"{model_name.title()}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        print()

    return results


def main():
    compare_models(test_size=0.2)


if __name__ == "__main__":
    main()
