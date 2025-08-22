"""
XGBoost Model for Financial Market Prediction

This module implements an XGBoost model for predicting financial market direction
using both market data and sentiment features.

Adapted to read real market and sentiment data from CSV files:
- Market: src/data_collection/data/sp500_data_*.csv (multi-row header)
- Sentiment: src/sentiment_analysis/data/combined_cleaned_tweets_with_distilbert_sentiment.csv

Note: The main() function trains on the prepared dataset and saves model artifacts to the models/ directory.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pickle
from datetime import datetime

# -----------------------------
# Data loading helper functions
# -----------------------------

def load_sp500_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load S&P 500 market data from a CSV with a multi-row header.
    Expects the format observed in sp500_data_20250803_202443.csv where:
      - Row 0 contains column labels for numeric fields (Adj Close, Close, High, Low, Open, ...)
      - Row 2 contains 'Date' in the first column
      - Data starts from row 3
    Returns a DataFrame with a proper 'Date' column and numeric types.
    """
    df_raw = pd.read_csv(csv_path, header=None)
    if df_raw.shape[0] < 4:
        raise ValueError(f"Unexpected SP500 CSV format: {csv_path}")

    first_header = df_raw.iloc[0].tolist()
    date_label = str(df_raw.iloc[2, 0]) if pd.notna(df_raw.iloc[2, 0]) else 'Date'

    # Determine how many columns the data section has
    data_df = df_raw.iloc[3:].copy()
    n_cols = data_df.shape[1]

    # Build columns dynamically to match the data
    if n_cols == len(first_header) + 1:
        # Format: Date + all metric labels (total n_cols columns)
        columns = [date_label] + [str(c) for c in first_header]
    elif n_cols == len(first_header):
        # Format: Date shares the first header cell; drop the first label (e.g., 'Price')
        columns = [date_label] + [str(c) for c in first_header[1:]]
    else:
        # Fallback: prefer dropping first header cell and then pad/truncate to match
        columns = [date_label] + [str(c) for c in first_header[1:]]
        if len(columns) < n_cols:
            # Pad with generic names
            columns += [f"col_{i}" for i in range(len(columns), n_cols)]
        elif len(columns) > n_cols:
            # Truncate extras
            columns = columns[:n_cols]

    df = data_df
    df.columns = columns

    # Normalize date column name to 'Date'
    df['Date'] = pd.to_datetime(df[date_label])
    if date_label != 'Date':
        df.drop(columns=[date_label], inplace=True)

    # Coerce numerics where applicable
    for col in df.columns:
        if col not in ('Date', 'Ticker'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('Date').reset_index(drop=True)
    return df


def load_distilbert_sentiment(
    csv_path: Union[str, Path],
    filter_tickers: Optional[List[str]] = None,
    *,
    aggregate: bool = True,
    group_by_ticker: bool = False,
    freq: Optional[str] = 'D',
    verbose: bool = False,
) -> pd.DataFrame:
    """Load combined DistilBERT tweet sentiment and optionally aggregate.
    - Handles columns as found in combined_cleaned_tweets_with_distilbert_sentiment*.csv
    - Optionally filters by tickers in a 'stock'/'ticker' column (e.g., ['SPX', 'SPY']).
    - By default aggregates to daily level (one row per unique date). This can drastically reduce the
      number of rows (e.g., millions of tweets -> ~thousands of days). Set aggregate=False to keep
      row-level data, or group_by_ticker=True to aggregate per date and ticker.
    - Returns a DataFrame with a 'date' column (datetime). If aggregated, includes aggregated metrics.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names
    cols_lower = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)

    # Map possible date/ticker column names
    date_col = 'date' if 'date' in df.columns else (
        'created_at' if 'created_at' in df.columns else None
    )
    if date_col is None:
        raise ValueError("Sentiment CSV must contain a 'date' or 'created_at' column")

    ticker_col = 'stock' if 'stock' in df.columns else ('ticker' if 'ticker' in df.columns else None)
    if ticker_col is None:
        # Create empty ticker column if absent
        df['stock'] = ''
        ticker_col = 'stock'

    # Ensure sentiment score columns exist; compute if necessary
    if 'sentiment_score' not in df.columns:
        if {'positive_score', 'negative_score'}.issubset(df.columns):
            df['sentiment_score'] = df['positive_score'] - df['negative_score']
        else:
            raise ValueError("Sentiment CSV must contain 'sentiment_score' or both 'positive_score' and 'negative_score'")

    # Optional filter by tickers present in ticker column
    if filter_tickers:
        pattern = '|'.join([f"\\b{t}\\b" for t in filter_tickers])
        df = df[df[ticker_col].fillna('').astype(str).str.contains(pattern, case=False, regex=True)]

    # Parse datetime (handle mixed tz-aware/naive strings)
    try:
        # pandas >= 2.0 supports format='mixed'
        df[date_col] = pd.to_datetime(df[date_col], utc=True, format='mixed', errors='coerce')
    except TypeError:
        # Fallback for older pandas versions
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

    # Drop rows where datetime could not be parsed
    df = df[pd.notna(df[date_col])]

    # Create/floor date column for grouping
    if freq:
        try:
            df['date'] = df[date_col].dt.floor(freq)
        except Exception:
            # If invalid freq provided, default to daily
            df['date'] = df[date_col].dt.floor('D')
            if verbose:
                print(f"Invalid freq '{freq}' provided. Defaulted to 'D'.")
    else:
        df['date'] = df[date_col]

    if not aggregate:
        # Return row-level data with parsed 'date' column, tz-naive for merging
        try:
            df['date'] = df['date'].dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        if verbose:
            print(f"Rows (no aggregation): {len(df.index)}")
        return df

    # Aggregate
    agg_dict = {
        'sentiment_score': ['mean', 'std', 'count']
    }
    for c in ('positive_score', 'negative_score', 'neutral_score'):
        if c in df.columns:
            agg_dict[c] = ['mean', 'sum']

    group_keys = ['date', ticker_col] if group_by_ticker else ['date']
    if verbose:
        print(f"Aggregating by: {group_keys}")
        print(f"Input rows: {len(df.index)}")

    agg = df.groupby(group_keys).agg(agg_dict)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    if verbose:
        print(f"Output rows after aggregation: {len(agg.index)}")

    # Additional derived features
    if 'positive_score_sum' in agg.columns and 'negative_score_sum' in agg.columns:
        agg['bullish_bearish_ratio'] = agg['positive_score_sum'] / agg['negative_score_sum']
        agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure 'date' is tz-naive for merging with market data
    if 'date' in agg.columns:
        try:
            agg['date'] = agg['date'].dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass

    return agg


def get_prepared_dataset(market_csv_path: Union[str, Path],
                         sentiment_csv_path: Union[str, Path],
                         model_type: str = 'classification',
                         filter_tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Load market and sentiment CSVs and build a feature dataset ready for modeling."""
    market_df = load_sp500_data(market_csv_path)
    sentiment_df = load_distilbert_sentiment(sentiment_csv_path, filter_tickers=filter_tickers)

    model = FinancialXGBoostModel(model_type=model_type)
    features_df = model.create_features(market_data=market_df,
                                        sentiment_data=sentiment_df,
                                        target_column='Daily_Return')
    return features_df


class FinancialXGBoostModel:
    def __init__(self, model_type: str = 'classification', random_state: int = 42):
        """
        Initialize the XGBoost model for financial prediction.
        
        Args:
            model_type: Type of model ('classification' for direction prediction or 'regression' for return prediction)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = None
        
    def create_features(self, 
                      market_data: pd.DataFrame, 
                      sentiment_data: Optional[pd.DataFrame] = None,
                      target_column: str = 'Daily_Return',
                      window_sizes: List[int] = [5, 10, 20, 50],
                      include_day_of_week: bool = True) -> pd.DataFrame:
        """
        Create features for the XGBoost model from market and sentiment data.
        
        Args:
            market_data: DataFrame containing market data
            sentiment_data: DataFrame containing sentiment data (optional)
            target_column: Column to use as target variable
            window_sizes: List of window sizes for rolling features
            include_day_of_week: Whether to include day of week features
            
        Returns:
            DataFrame with features and target variable
        """
        # Make a copy to avoid modifying the original
        data = market_data.copy()
        
        # Ensure data is sorted by date
        if 'Date' in data.columns:
            data = data.sort_values('Date')
        
        # Create target variable
        if self.model_type == 'classification':
            # Direction of price movement (1 for up, 0 for down)
            data['Target'] = (data[target_column] > 0).astype(int)
        else:
            # Actual return value
            data['Target'] = data[target_column]
            
        # Create lagged returns
        for lag in range(1, 6):
            data[f'Return_Lag_{lag}'] = data[target_column].shift(lag)
            
        # Create rolling window features
        for window in window_sizes:
            # Rolling mean of returns
            data[f'Return_Mean_{window}d'] = data[target_column].rolling(window=window).mean()
            
            # Rolling standard deviation (volatility)
            data[f'Return_Std_{window}d'] = data[target_column].rolling(window=window).std()
            
            # Rolling min and max
            data[f'Return_Min_{window}d'] = data[target_column].rolling(window=window).min()
            data[f'Return_Max_{window}d'] = data[target_column].rolling(window=window).max()
            
            # Rolling skewness and kurtosis
            data[f'Return_Skew_{window}d'] = data[target_column].rolling(window=window).skew()
            data[f'Return_Kurt_{window}d'] = data[target_column].rolling(window=window).kurt()
            
            # Volume features
            if 'Volume' in data.columns:
                data[f'Volume_Mean_{window}d'] = data['Volume'].rolling(window=window).mean()
                data[f'Volume_Std_{window}d'] = data['Volume'].rolling(window=window).std()
                data[f'Volume_Change_{window}d'] = data['Volume'].pct_change(periods=window)
        
        # Technical indicators
        # Moving Average Convergence Divergence (MACD)
        if 'Close' in data.columns:
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            
            # Relative Strength Index (RSI)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
        # Day of week features (one-hot encoded)
        if include_day_of_week and 'Date' in data.columns:
            data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.dayofweek
            day_dummies = pd.get_dummies(data['DayOfWeek'], prefix='Day')
            data = pd.concat([data, day_dummies], axis=1)
            
        # Add sentiment features if provided
        if sentiment_data is not None:
            # Ensure sentiment data has a date column
            if 'date' not in sentiment_data.columns and 'Date' not in sentiment_data.columns:
                raise ValueError("Sentiment data must have a 'date' or 'Date' column")
                
            date_col = 'date' if 'date' in sentiment_data.columns else 'Date'
            
            # Merge sentiment data with market data
            sentiment_data = sentiment_data.copy()
            sentiment_data[date_col] = pd.to_datetime(sentiment_data[date_col])
            
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                merged_data = pd.merge(data, sentiment_data, left_on='Date', right_on=date_col, how='left')
            else:
                # If no Date column, use index
                merged_data = pd.merge(data, sentiment_data, left_index=True, right_on=date_col, how='left')
                
            data = merged_data
            
            # Fill missing sentiment values
            sentiment_cols = [col for col in sentiment_data.columns if col != date_col]
            data[sentiment_cols] = data[sentiment_cols].ffill()
            
        # Drop rows with NaN values (due to rolling windows)
        data = data.dropna()
        
        return data
    
    def prepare_train_test_data(self, 
                              data: pd.DataFrame, 
                              test_size: float = 0.2,
                              time_series_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and testing data.
        
        Args:
            data: DataFrame with features and target
            test_size: Proportion of data to use for testing
            time_series_split: Whether to use time series split (vs random split)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Define features and target
        target = data['Target']
        
        # Exclude non-feature columns
        exclude_cols = ['Target', 'Date', 'Ticker', 'date', 'time_period', 'DayOfWeek']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        features = data[feature_cols]
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Split data
        if time_series_split:
            # Use the last test_size portion of data for testing
            split_idx = int(len(data) * (1 - test_size))
            X_train = features_scaled[:split_idx]
            X_test = features_scaled[split_idx:]
            y_train = target.iloc[:split_idx].values
            y_test = target.iloc[split_idx:].values
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, target.values, test_size=test_size, random_state=self.random_state
            )
            
        return X_train, X_test, y_train, y_test
    
    def train(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_test: np.ndarray = None,
            y_test: np.ndarray = None,
            tune_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features (for early stopping)
            y_test: Testing target (for early stopping)
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        # Set up evaluation set for early stopping if provided
        eval_set = [(X_train, y_train)]
        if X_test is not None and y_test is not None:
            eval_set.append((X_test, y_test))
            
        # Set up model parameters
        if self.model_type == 'classification':
            objective = 'binary:logistic'
            eval_metric = ['logloss', 'error']
        else:
            objective = 'reg:squarederror'
            eval_metric = ['rmse', 'mae']
            
        # Default parameters
        params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state
        }
        
        # Hyperparameter tuning
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            
            # Set up model for tuning
            if self.model_type == 'classification':
                model = xgb.XGBClassifier(objective=objective, random_state=self.random_state)
            else:
                model = xgb.XGBRegressor(objective=objective, random_state=self.random_state)
                
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Update parameters
            params.update(best_params)
            
            # Set model to best estimator
            self.model = grid_search.best_estimator_
            
        else:
            # Train model without tuning
            if self.model_type == 'classification':
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
                
            # Robust training: try callbacks, then early_stopping_rounds, then plain fit
            try:
                # Attempt with callbacks if available
                from xgboost.callback import EarlyStopping
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=True,
                        callbacks=[EarlyStopping(rounds=50, save_best=True, maximize=False)]
                    )
                except TypeError:
                    # Fallback to early_stopping_rounds
                    try:
                        self.model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            verbose=True,
                            early_stopping_rounds=50
                        )
                    except TypeError:
                        # Final fallback: no early stopping
                        self.model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            verbose=True
                        )
            except Exception:
                # If callbacks import fails, try early_stopping_rounds then plain fit
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=True,
                        early_stopping_rounds=50
                    )
                except TypeError:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=True
                    )
            
        # Return training results
        return {
            'model': self.model,
            'params': params,
            'feature_names': self.feature_names
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for classification only).
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        if self.model_type != 'classification':
            raise ValueError("predict_proba() is only available for classification models")
            
        return self.model.predict_proba(X)
    
    def evaluate(self, 
               X_test: np.ndarray, 
               y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        if self.model_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Get prediction probabilities
            y_prob = self.predict_proba(X_test)[:, 1]
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_prob
            }
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'predictions': y_pred
            }
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def compare_with_baseline(self, 
                            X_test: np.ndarray, 
                            y_test: np.ndarray, 
                            baseline_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Compare model performance with a baseline model.
        
        Args:
            X_test: Test features
            y_test: Test target
            baseline_predictions: Predictions from baseline model
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate model
        model_eval = self.evaluate(X_test, y_test)
        
        # Evaluate baseline
        if self.model_type == 'classification':
            baseline_accuracy = accuracy_score(y_test, baseline_predictions)
            baseline_precision = precision_score(y_test, baseline_predictions)
            baseline_recall = recall_score(y_test, baseline_predictions)
            baseline_f1 = f1_score(y_test, baseline_predictions)
            
            baseline_eval = {
                'accuracy': baseline_accuracy,
                'precision': baseline_precision,
                'recall': baseline_recall,
                'f1_score': baseline_f1
            }
            
            # Calculate improvement
            improvement = {
                'accuracy': model_eval['accuracy'] - baseline_accuracy,
                'precision': model_eval['precision'] - baseline_precision,
                'recall': model_eval['recall'] - baseline_recall,
                'f1_score': model_eval['f1_score'] - baseline_f1
            }
        else:
            baseline_mse = mean_squared_error(y_test, baseline_predictions)
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_mae = mean_absolute_error(y_test, baseline_predictions)
            baseline_r2 = r2_score(y_test, baseline_predictions)
            
            baseline_eval = {
                'mse': baseline_mse,
                'rmse': baseline_rmse,
                'mae': baseline_mae,
                'r2_score': baseline_r2
            }
            
            # Calculate improvement (lower is better for error metrics)
            improvement = {
                'mse': baseline_mse - model_eval['mse'],
                'rmse': baseline_rmse - model_eval['rmse'],
                'mae': baseline_mae - model_eval['mae'],
                'r2_score': model_eval['r2_score'] - baseline_r2
            }
            
        return {
            'model_eval': model_eval,
            'baseline_eval': baseline_eval,
            'improvement': improvement
        }


def main():
    """
    Train an XGBoost model on real market and sentiment data and save artifacts.
    """
    repo_root = Path(__file__).resolve().parents[2]
    market_csv = repo_root / "src" / "data_collection" / "data" / "sp500_data_20250820_203144.csv"
    sentiment_csv = repo_root / "src" / "sentiment_analysis" / "data" / "combined_cleaned_tweets_with_distilbert_sentiment.csv"

    print(f"Loading market data from: {market_csv}")
    print(f"Loading sentiment data from: {sentiment_csv}")

    # Load datasets
    try:
        market_df = load_sp500_data(market_csv)
    except Exception as e:
        print(f"Failed to load market data: {e}")
        return

    try:
        sentiment_df = load_distilbert_sentiment(sentiment_csv)  # optional filter
    except Exception as e:
        print(f"Failed to load sentiment data: {e}")
        return

    print('rows:')
    print(len(sentiment_df.index))
    print(len(market_df.index))
    # Create features
    model = FinancialXGBoostModel(model_type='classification')
    features_df = model.create_features(
        market_data=market_df,
        sentiment_data=sentiment_df,
        target_column='Daily_Return'
    )

    print("\nPrepared feature dataset:")
    print(f"Rows: {features_df.shape[0]}, Columns: {features_df.shape[1]}")

    # Prepare train/test data
    X_train, X_test, y_train, y_test = model.prepare_train_test_data(
        data=features_df,
        test_size=0.2,
        time_series_split=True
    )

    # Train model
    print("\nTraining XGBoost model...")
    training_results = model.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparams=False
    )

    # Evaluate model
    print("\nEvaluating model...")
    evaluation = model.evaluate(X_test, y_test)

    # Prepare output directory
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"xgb_{model.model_type}_{ts}"

    # Save model
    model_path = models_dir / f"{prefix}.json"
    try:
        model.model.save_model(str(model_path))
    except Exception as e:
        print(f"Warning: Failed to save XGBoost model JSON: {e}")

    # Save scaler
    scaler_path = models_dir / f"{prefix}_scaler.pkl"
    try:
        with open(scaler_path, "wb") as f:
            pickle.dump(model.feature_scaler, f)
    except Exception as e:
        print(f"Warning: Failed to save scaler: {e}")

    # Save feature names
    features_path = models_dir / f"{prefix}_features.json"
    try:
        with open(features_path, "w") as f:
            json.dump(model.feature_names, f)
    except Exception as e:
        print(f"Warning: Failed to save feature names: {e}")

    # Save metrics
    metrics_path = models_dir / f"{prefix}_metrics.json"
    try:
        # Convert arrays to lists for JSON
        eval_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in evaluation.items()}
        # If probabilities/predictions exist, ensure lists
        if 'predictions' in eval_to_save and isinstance(eval_to_save['predictions'], (np.ndarray, list)):
            eval_to_save['predictions'] = list(eval_to_save['predictions'])
        if 'probabilities' in eval_to_save and isinstance(eval_to_save['probabilities'], (np.ndarray, list)):
            eval_to_save['probabilities'] = list(eval_to_save['probabilities'])
        with open(metrics_path, "w") as f:
            json.dump(eval_to_save, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save metrics: {e}")

    # Save training params
    params_path = models_dir / f"{prefix}_params.json"
    try:
        with open(params_path, "w") as f:
            json.dump(training_results.get('params', {}), f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save params: {e}")

    print("\nTraining complete. Artifacts saved:")
    print(f" - Model: {model_path}")
    print(f" - Scaler: {scaler_path}")
    print(f" - Feature names: {features_path}")
    print(f" - Metrics: {metrics_path}")
    print(f" - Params: {params_path}")

    if model.model_type == 'classification':
        print("\nTest metrics:")
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
        print(f"F1 Score: {evaluation['f1_score']:.4f}")
    else:
        print("\nTest metrics:")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"R² Score: {evaluation['r2_score']:.4f}")


if __name__ == "__main__":
    main()