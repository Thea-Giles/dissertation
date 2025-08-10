"""
XGBoost Model for Financial Market Prediction
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pickle
from datetime import datetime


def load_sp500_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, header=None)
    if df_raw.shape[0] < 4:
        raise ValueError(f"Unexpected SP500 CSV format: {csv_path}")

    first_header = df_raw.iloc[0].tolist()
    date_label = str(df_raw.iloc[2, 0]) if pd.notna(df_raw.iloc[2, 0]) else 'Date'

    # Determine how many columns the data section has
    data_df = df_raw.iloc[3:].copy()
    n_cols = data_df.shape[1]

    if n_cols == len(first_header) + 1:
        # Date + all metric labels
        columns = [date_label] + [str(c) for c in first_header]
    elif n_cols == len(first_header):
        # Date shares the first header cell; drop the first label (e.g., 'Price')
        columns = [date_label] + [str(c) for c in first_header[1:]]
    else:
        # prefer dropping first header cell and then pad/truncate to match
        columns = [date_label] + [str(c) for c in first_header[1:]]
        if len(columns) < n_cols:
            # Pad with generic names
            columns += [f"col_{i}" for i in range(len(columns), n_cols)]
        elif len(columns) > n_cols:
            columns = columns[:n_cols]

    df = data_df
    df.columns = columns

    # Normalize date column name to 'Date'
    df['Date'] = pd.to_datetime(df[date_label])
    if date_label != 'Date':
        df.drop(columns=[date_label], inplace=True)

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
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names
    cols_lower = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)

    # Map possible date/ticker column names
    date_col = 'date' if 'date' in df.columns else (
        'created_at' if 'created_at' in df.columns else None
    )

    ticker_col = 'stock' if 'stock' in df.columns else ('ticker' if 'ticker' in df.columns else None)
    if ticker_col is None:
        df['stock'] = ''
        ticker_col = 'stock'

    # Ensure sentiment score columns exist; compute if necessary
    if 'sentiment_score' not in df.columns:
        if {'positive_score', 'negative_score'}.issubset(df.columns):
            df['sentiment_score'] = df['positive_score'] - df['negative_score']
        else:
            raise ValueError("Sentiment CSV missing")

    if filter_tickers:
        pattern = '|'.join([f"\\b{t}\\b" for t in filter_tickers])
        df = df[df[ticker_col].fillna('').astype(str).str.contains(pattern, case=False, regex=True)]

    # Parse datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col], utc=True, format='mixed', errors='coerce')
    except TypeError:
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

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
    market_df = load_sp500_data(market_csv_path)
    sentiment_df = load_distilbert_sentiment(sentiment_csv_path, filter_tickers=filter_tickers)

    model = FinancialXGBoostModel(model_type=model_type)
    features_df = model.create_features(market_data=market_df,
                                        sentiment_data=sentiment_df,
                                        target_column='Daily_Return')
    return features_df


class FinancialXGBoostModel:
    def __init__(self, model_type: str = 'classification', random_state: int = 42):
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
        data = market_data.copy()
        
        if 'Date' in data.columns:
            data = data.sort_values('Date')
        if self.model_type == 'classification':
            # Direction of price movement (1 for up, 0 for down)
            data['Target'] = (data[target_column] > 0).astype(int)
        else:
            # Actual return value
            data['Target'] = data[target_column]
            
        for lag in range(1, 6):
            data[f'Return_Lag_{lag}'] = data[target_column].shift(lag)
            
        for window in window_sizes:
            data[f'Return_Mean_{window}d'] = data[target_column].rolling(window=window).mean()
            data[f'Return_Std_{window}d'] = data[target_column].rolling(window=window).std()
            data[f'Return_Min_{window}d'] = data[target_column].rolling(window=window).min()
            data[f'Return_Max_{window}d'] = data[target_column].rolling(window=window).max()
            data[f'Return_Skew_{window}d'] = data[target_column].rolling(window=window).skew()
            data[f'Return_Kurt_{window}d'] = data[target_column].rolling(window=window).kurt()
            if 'Volume' in data.columns:
                data[f'Volume_Mean_{window}d'] = data['Volume'].rolling(window=window).mean()
                data[f'Volume_Std_{window}d'] = data['Volume'].rolling(window=window).std()
                data[f'Volume_Change_{window}d'] = data['Volume'].pct_change(periods=window)
        
        if 'Close' in data.columns:
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
        if include_day_of_week and 'Date' in data.columns:
            data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.dayofweek
            day_dummies = pd.get_dummies(data['DayOfWeek'], prefix='Day')
            data = pd.concat([data, day_dummies], axis=1)
            
        # Add sentiment features if provided
        if sentiment_data is not None:
            if 'date' not in sentiment_data.columns and 'Date' not in sentiment_data.columns:
                raise ValueError("Sentiment data must have a 'date' or 'Date' column")
                
            date_col = 'date' if 'date' in sentiment_data.columns else 'Date'
            
            sentiment_data = sentiment_data.copy()
            sentiment_data[date_col] = pd.to_datetime(sentiment_data[date_col])
            
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                merged_data = pd.merge(data, sentiment_data, left_on='Date', right_on=date_col, how='left')
            else:
                merged_data = pd.merge(data, sentiment_data, left_index=True, right_on=date_col, how='left')
                
            data = merged_data
            
            sentiment_cols = [col for col in sentiment_data.columns if col != date_col]
            data[sentiment_cols] = data[sentiment_cols].ffill()

            if 'date' in data.columns and 'Date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'])
                except Exception:
                    pass
                data['date'] = data['date'].fillna(pd.to_datetime(data['Date'], errors='coerce'))

            data[sentiment_cols] = data[sentiment_cols].fillna(0.0)
            
        data = data.dropna()
        
        return data
    
    def prepare_train_test_data(self, 
                              data: pd.DataFrame, 
                              test_size: float = 0.2,
                              time_series_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = data['Target']
        exclude_cols = ['Target', 'Date', 'Ticker', 'date', 'time_period', 'DayOfWeek']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        features = data[feature_cols]
        
        if time_series_split:
            split_idx = int(len(data) * (1 - test_size))
            X_train_raw = features.iloc[:split_idx]
            X_test_raw = features.iloc[split_idx:]
            y_train = target.iloc[:split_idx].values
            y_test = target.iloc[split_idx:].values
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                features, target.values, test_size=test_size, random_state=self.random_state
            )
        
        X_train = self.feature_scaler.fit_transform(X_train_raw)
        X_test = self.feature_scaler.transform(X_test_raw)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_test: np.ndarray = None,
            y_test: np.ndarray = None,
            tune_hyperparams: bool = False) -> Dict[str, Any]:
        eval_set = [(X_train, y_train)]
        if X_test is not None and y_test is not None:
            eval_set.append((X_test, y_test))
            
        if self.model_type == 'classification':
            objective = 'binary:logistic'
            eval_metric = ['logloss', 'error']
        else:
            objective = 'reg:squarederror'
            eval_metric = ['rmse', 'mae']
            
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
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            
            if self.model_type == 'classification':
                model = xgb.XGBClassifier(objective=objective, random_state=self.random_state)
            else:
                model = xgb.XGBRegressor(objective=objective, random_state=self.random_state)
                
            tscv = TimeSeriesSplit(n_splits=5)
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            params.update(best_params)
            self.model = grid_search.best_estimator_
            
        else:
            if self.model_type == 'classification':
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
                
            try:
                from xgboost.callback import EarlyStopping
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=True,
                        callbacks=[EarlyStopping(rounds=50, save_best=True, maximize=False)]
                    )
                except TypeError:
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
            except Exception:
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
            
        return {
            'model': self.model,
            'params': params,
            'feature_names': self.feature_names
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def evaluate(self, 
               X_test: np.ndarray, 
               y_test: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        y_pred = self.predict(X_test)
        
        # Calculate metrics
        if self.model_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # prediction probabilities
            y_prob = self.predict_proba(X_test)[:, 1]

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            # ROC-AUC and PR-AUC (handle edge cases where only one class in y_test)
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_test, y_prob))
            except ValueError:
                pr_auc = None

            # Class balance
            n_pos = int(np.sum(y_test == 1))
            n_neg = int(np.sum(y_test == 0))
            test_class_balance = {
                'positives': n_pos,
                'negatives': n_neg,
                'positive_rate': float(n_pos / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else None
            }

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'threshold': 0.5,
                'positive_label': 1,
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                },
                'test_class_balance': test_class_balance,
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


from typing import Optional as _Optional

def load_saved_financial_xgb(model_path: Union[str, Path], model_type: _Optional[str] = None) -> FinancialXGBoostModel:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    inferred_type = None
    if model_type is None:
        name = path.stem.lower()
        if 'classification' in name:
            inferred_type = 'classification'
        elif 'regression' in name:
            inferred_type = 'regression'
    mt = model_type or inferred_type or 'classification'

    model = FinancialXGBoostModel(model_type=mt)
    # Create underlying XGB model and load booster
    if mt == 'classification':
        xb = xgb.XGBClassifier()
    else:
        xb = xgb.XGBRegressor()
    xb.load_model(str(path))
    model.model = xb

    # Load sidecar artifacts
    prefix = path.stem
    dir_path = path.parent
    scaler_path = dir_path / f"{prefix}_scaler.pkl"
    features_path = dir_path / f"{prefix}_features.json"

    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                model.feature_scaler = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load scaler from {scaler_path}: {e}")
    else:
        print(f"Warning: scaler file not found: {scaler_path}")

    if features_path.exists():
        try:
            with open(features_path, 'r') as f:
                model.feature_names = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load feature names from {features_path}: {e}")
    else:
        print(f"Warning: feature names file not found: {features_path}")

    return model


def main():
    repo_root = Path(__file__).resolve().parents[2]
    market_csv = repo_root / "src" / "data_collection" / "sp500_data_20250825_094341.csv"
    sentiment_csv = repo_root / "src" / "sentiment_analysis" / "data" / "combined_cleaned_tweets_with_distilbert_sentiment.csv"

    print(f"Loading market data from: {market_csv}")
    print(f"Loading sentiment data from: {sentiment_csv}")

    try:
        market_df = load_sp500_data(market_csv)
    except Exception as e:
        print(f"Failed to load market data: {e}")
        return

    try:
        sentiment_df = load_distilbert_sentiment(sentiment_csv)
    except Exception as e:
        print(f"Failed to load sentiment data: {e}")
        return

    print('rows:')
    print(len(sentiment_df.index))
    print(len(market_df.index))
    model = FinancialXGBoostModel(model_type='classification')
    features_df = model.create_features(
        market_data=market_df,
        sentiment_data=sentiment_df,
        target_column='Daily_Return'
    )

    print("\nPrepared feature dataset:")
    print(f"Rows: {features_df.shape[0]}, Columns: {features_df.shape[1]}")

    X_train, X_test, y_train, y_test = model.prepare_train_test_data(
        data=features_df,
        test_size=0.2,
        time_series_split=True
    )

    print("\nTraining XGBoost model...")
    training_results = model.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparams=False
    )

    print("\nEvaluating model...")
    evaluation = model.evaluate(X_test, y_test)

    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"xgb_{model.model_type}_{ts}"

    model_path = models_dir / f"{prefix}.json"
    try:
        model.model.save_model(str(model_path))
    except Exception as e:
        print(f"Warning: Failed to save XGBoost model JSON: {e}")

    scaler_path = models_dir / f"{prefix}_scaler.pkl"
    try:
        with open(scaler_path, "wb") as f:
            pickle.dump(model.feature_scaler, f)
    except Exception as e:
        print(f"Warning: Failed to save scaler: {e}")

    features_path = models_dir / f"{prefix}_features.json"
    try:
        with open(features_path, "w") as f:
            json.dump(model.feature_names, f)
    except Exception as e:
        print(f"Warning: Failed to save feature names: {e}")

    metrics_path = models_dir / f"{prefix}_metrics.json"
    try:
        eval_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in evaluation.items()}
        if 'predictions' in eval_to_save and isinstance(eval_to_save['predictions'], (np.ndarray, list)):
            eval_to_save['predictions'] = list(eval_to_save['predictions'])
        if 'probabilities' in eval_to_save and isinstance(eval_to_save['probabilities'], (np.ndarray, list)):
            eval_to_save['probabilities'] = list(eval_to_save['probabilities'])
        with open(metrics_path, "w") as f:
            json.dump(eval_to_save, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save metrics: {e}")

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
        if 'roc_auc' in evaluation and evaluation['roc_auc'] is not None:
            print(f"ROC-AUC: {evaluation['roc_auc']:.4f}")
        if 'pr_auc' in evaluation and evaluation['pr_auc'] is not None:
            print(f"PR-AUC: {evaluation['pr_auc']:.4f}")
        cm = evaluation.get('confusion_matrix', {})
        if cm:
            print("Confusion Matrix (tn, fp, fn, tp):", cm.get('tn'), cm.get('fp'), cm.get('fn'), cm.get('tp'))
        train_pos = int(np.sum(y_train == 1))
        train_neg = int(np.sum(y_train == 0))
        test_pos = evaluation.get('test_class_balance', {}).get('positives')
        test_neg = evaluation.get('test_class_balance', {}).get('negatives')
        print(f"Train class balance: pos={train_pos} ({(train_pos/(train_pos+train_neg)):.2%}), neg={train_neg} ({(train_neg/(train_pos+train_neg)):.2%})")
        print(f"Test class balance: pos={test_pos}, neg={test_neg}")
        print(f"Decision threshold: {evaluation.get('threshold', 0.5)}  (positive_label={evaluation.get('positive_label', 1)})")
    else:
        print("\nTest metrics:")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"R² Score: {evaluation['r2_score']:.4f}")


if __name__ == "__main__":
    main()