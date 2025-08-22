"""
Fama-French Three-Factor Model
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional, Union


class FamaFrenchModel:
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the Fama-French model.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
        """
        self.risk_free_rate = risk_free_rate
        self.model = None
        self.results = None
        
    def calculate_daily_risk_free_rate(self) -> float:
        """
        Convert annual risk-free rate to daily rate.
        
        Returns:
            Daily risk-free rate
        """
        return (1 + self.risk_free_rate) ** (1/252) - 1
    
    def prepare_data(self, 
                   stock_returns: pd.Series, 
                   market_returns: pd.Series, 
                   smb: Optional[pd.Series] = None, 
                   hml: Optional[pd.Series] = None) -> pd.DataFrame:
        # Calculate excess returns
        daily_rf = self.calculate_daily_risk_free_rate()
        excess_stock_returns = stock_returns - daily_rf
        excess_market_returns = market_returns - daily_rf
        
        # Create DataFrame
        data = pd.DataFrame({
            'excess_returns': excess_stock_returns,
            'mkt_excess': excess_market_returns
        })
        
        # Add SMB and HML if provided
        if smb is not None:
            data['smb'] = smb
        if hml is not None:
            data['hml'] = hml
            
        # Drop missing values
        data = data.dropna()
        
        return data
    
    def fit(self, 
          stock_returns: pd.Series, 
          market_returns: pd.Series, 
          smb: Optional[pd.Series] = None, 
          hml: Optional[pd.Series] = None) -> Dict:
        # Prepare data
        data = self.prepare_data(stock_returns, market_returns, smb, hml)
        
        # Define X and y
        y = data['excess_returns']
        
        if smb is not None and hml is not None:
            X = data[['mkt_excess', 'smb', 'hml']]
            model_type = 'three_factor'
        else:
            X = data[['mkt_excess']]
            model_type = 'capm'
            
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        
        # Extract coefficients
        alpha = self.results.params['const']
        beta = self.results.params['mkt_excess']
        
        # Create results dictionary
        results_dict = {
            'alpha': alpha,
            'beta': beta,
            'r_squared': self.results.rsquared,
            'adj_r_squared': self.results.rsquared_adj,
            'model_type': model_type,
            'summary': self.results.summary()
        }
        
        # Add SMB and HML coefficients if available
        if model_type == 'three_factor':
            results_dict['smb_coef'] = self.results.params['smb']
            results_dict['hml_coef'] = self.results.params['hml']
            
        return results_dict
    
    def predict(self, 
              market_returns: pd.Series, 
              smb: Optional[pd.Series] = None, 
              hml: Optional[pd.Series] = None) -> pd.Series:
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Calculate excess market returns
        daily_rf = self.calculate_daily_risk_free_rate()
        excess_market_returns = market_returns - daily_rf
        
        # Create DataFrame for prediction
        pred_data = pd.DataFrame({'mkt_excess': excess_market_returns})
        
        # Check what model type was used during fitting
        model_type = 'capm'  # Default to CAPM
        if 'smb' in self.results.params.index and 'hml' in self.results.params.index:
            model_type = 'three_factor'
        
        # Add SMB and HML if this is a three-factor model
        if model_type == 'three_factor':
            if smb is None or hml is None:
                raise ValueError("SMB and HML must be provided for a three-factor model")
            pred_data['smb'] = smb
            pred_data['hml'] = hml
            
        # Add constant
        pred_data = sm.add_constant(pred_data)
        
        # Ensure pred_data has the same columns as were used in fitting
        required_columns = self.results.params.index.tolist()
        missing_columns = set(required_columns) - set(pred_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in prediction data: {missing_columns}")
        
        # Reorder columns to match the order used in fitting
        pred_data = pred_data[required_columns]
        
        # Predict
        predicted_excess_returns = self.results.predict(pred_data)
        
        # Convert back to actual returns
        predicted_returns = predicted_excess_returns + daily_rf
        
        return predicted_returns
    
    def evaluate(self, 
               actual_returns: pd.Series, 
               predicted_returns: pd.Series) -> Dict:
        # Align data
        data = pd.DataFrame({
            'actual': actual_returns,
            'predicted': predicted_returns
        }).dropna()
        
        # Calculate metrics
        mse = ((data['actual'] - data['predicted']) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(data['actual'] - data['predicted']).mean()
        
        # Calculate directional accuracy
        data['actual_direction'] = np.sign(data['actual'])
        data['predicted_direction'] = np.sign(data['predicted'])
        directional_accuracy = (data['actual_direction'] == data['predicted_direction']).mean()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }


def main():
    """
    Main function to demonstrate the Fama-French model using data from market_data.py.
    
    This function:
    1. Loads the Fama-French factors data from the data_collection/data/ folder
    2. Prepares the data for the model
    3. Creates and fits the Fama-French model
    4. Displays the results
    """
    import os
    import glob
    
    # Find the most recent fama_french_factors CSV file
    # Use an absolute path that works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "src", "data_collection", "data")
    ff_files = glob.glob(os.path.join(data_dir, "fama_french_factors_*.csv"))
    
    if not ff_files:
        print("No Fama-French factors data found. Please run market_data.py first.")
        return
    
    # Sort by modification time (most recent first)
    latest_file = max(ff_files, key=os.path.getmtime)
    print(f"Using data from: {latest_file}")
    
    # Load the CSV file with a more complex approach to handle the unusual format
    # First, read just the header row to get column names
    header_df = pd.read_csv(latest_file, nrows=1)
    column_names = header_df.columns.tolist()
    
    # Now read the actual data, skipping the header and the two unusual rows
    data = pd.read_csv(latest_file, skiprows=3, header=None, names=column_names)
    
    # Convert the first column (which should be 'Date') to datetime and set as index
    data[column_names[0]] = pd.to_datetime(data[column_names[0]])
    data.set_index(column_names[0], inplace=True)
    
    # Print column names for debugging
    print("\nAvailable columns in the DataFrame:")
    for col in data.columns:
        print(f"- {col}")
    
    # Extract the required series
    if 'Daily_Return' in data.columns:
        stock_returns = data['Daily_Return']
        print("Using 'Daily_Return' column for stock returns")
    else:
        # Try to find a column that might contain returns
        return_columns = [col for col in data.columns if 'return' in col.lower()]
        if return_columns:
            stock_returns = data[return_columns[0]]
            print(f"Using '{return_columns[0]}' as stock returns")
        else:
            raise ValueError("No return column found in the data")
    market_returns = data['MRP'] + data['RF']  # Market return = MRP + RF
    smb = data['SMB']
    hml = data['HML']
    rf = data['RF']
    
    # Create and fit the model
    model = FamaFrenchModel(risk_free_rate=rf.mean() * 252)  # Annualized RF
    
    # Fit the model
    results = model.fit(stock_returns, market_returns, smb, hml)
    
    # Print results
    print("\nFama-French Model Results:")
    print(f"Alpha: {results['alpha']:.6f}")
    print(f"Beta: {results['beta']:.6f}")
    print(f"SMB coefficient: {results.get('smb_coef', 'N/A')}")
    print(f"HML coefficient: {results.get('hml_coef', 'N/A')}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"Adjusted R-squared: {results['adj_r_squared']:.6f}")
    
    # Predict returns
    # Make sure we're using the same data for prediction as we used for fitting
    # Create a subset of data for the same time period
    prediction_data = pd.DataFrame({
        'market_returns': market_returns,
        'smb': smb,
        'hml': hml
    })
    
    # Drop any rows with missing values
    prediction_data = prediction_data.dropna()
    
    # Get the corresponding stock returns for the same dates
    actual_returns = stock_returns.loc[prediction_data.index]
    
    # Predict returns
    predicted_returns = model.predict(
        prediction_data['market_returns'], 
        prediction_data['smb'], 
        prediction_data['hml']
    )
    
    # Evaluate the model
    eval_results = model.evaluate(actual_returns, predicted_returns)
    
    print("\nModel Evaluation:")
    print(f"MSE: {eval_results['mse']:.6f}")
    print(f"RMSE: {eval_results['rmse']:.6f}")
    print(f"MAE: {eval_results['mae']:.6f}")
    print(f"Directional Accuracy: {eval_results['directional_accuracy']:.6f}")


if __name__ == "__main__":
    main()