"""
Market Data Collection Module
"""
import datetime
import io
import zipfile
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def get_stock_data(ticker: str,
                   start_date: str,
                   end_date: str,
                   interval: str = "1d") -> pd.DataFrame:

    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        if data.empty:
            print(f"No data found for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame()

        # Add ticker column
        data['Ticker'] = ticker

        # Calculate returns - use Close if Adj Close is not available
        if 'Adj Close' in data.columns:
            price_col = 'Adj Close'
        else:
            price_col = 'Close'

        data['Daily_Return'] = data[price_col].pct_change()

        # Calculate log returns
        data['Log_Return'] = np.log(data[price_col] / data[price_col].shift(1))

        # Calculate volatility (20-day rolling standard deviation of returns)
        data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()

        print(f"Retrieved {len(data)} records for {ticker}")
        return data

    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return pd.DataFrame()


def get_multiple_stocks(tickers: List[str],
                        start_date: str,
                        end_date: str,
                        interval: str = "1d") -> Dict[str, pd.DataFrame]:

    stock_data = {}
    for ticker in tickers:
        data = get_stock_data(ticker, start_date, end_date, interval)
        if not data.empty:
            stock_data[ticker] = data

    return stock_data


def get_factor_data(start_date: str, end_date: str) -> pd.DataFrame:
    # URL for the Fama-French 3 Factors (daily)
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    
    try:
        # Download the zip file
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download data: HTTP {response.status_code}")
            return pd.DataFrame()
            
        # Extract the CSV file from the zip archive
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                content = f.read().decode('utf-8')
                
                # Find the line where the data starts and ends
                lines = content.split('\n')
                start_idx = 0
                end_idx = len(lines)
                
                # Find where the actual data begins (after headers)
                for i, line in enumerate(lines):
                    if line.strip() and line[0].isdigit():
                        start_idx = i
                        break
                
                # Find where the data ends
                for i in range(start_idx, len(lines)):
                    line = lines[i].strip()
                    if not line or not line[0].isdigit():
                        end_idx = i
                        break
                
                # Extract just the data portion
                data_lines = lines[start_idx:end_idx]
                
                # Parse the data
                data = pd.read_csv(io.StringIO('\n'.join(data_lines)), 
                                  sep=',', 
                                  header=None,
                                  names=['Date', 'MRP', 'SMB', 'HML', 'RF'])

                data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
                data = data.set_index('Date')
                
                # Filter by date range
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                data = data[(data.index >= start) & (data.index <= end)]
                
                # Convert from percentage to decimal
                for col in ['MRP', 'SMB', 'HML', 'RF']:
                    data[col] = data[col] / 100.0
                
                print(f"Retrieved {len(data)} days of Fama-French factor data")
                return data
                
    except Exception as e:
        print(f"Error retrieving Fama-French data: {e}")
        return pd.DataFrame(columns=['MRP', 'SMB', 'HML', 'RF'])


def calculate_fama_french_factors(market_data: pd.DataFrame,
                                  factor_data: Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> pd.DataFrame:

    # If factor_data
    if factor_data is None:
        if start_date is None or end_date is None:
            raise ValueError("If factor_data is not provided, start_date and end_date must be provided")

        factor_data = get_factor_data(start_date, end_date)

    # Align French data with market data
    aligned_factors = factor_data.reindex(market_data.index)

    # Add factors to market data
    market_data['MRP'] = aligned_factors['MRP']
    market_data['SMB'] = aligned_factors['SMB']
    market_data['HML'] = aligned_factors['HML']

    # Add risk-free rate if available
    if 'RF' in aligned_factors.columns:
        market_data['RF'] = aligned_factors['RF']

    return market_data


def align_market_with_sentiment(market_data: pd.DataFrame,
                                sentiment_data: pd.DataFrame,
                                date_column: str = 'created_at') -> pd.DataFrame:

    # Convert sentiment dates to datetime if they're not already
    if sentiment_data[date_column].dtype != 'datetime64[ns]':
        sentiment_data[date_column] = pd.to_datetime(sentiment_data[date_column])

    # Extract just the date part (no time)
    sentiment_data['date'] = sentiment_data[date_column].dt.date

    # Group sentiment by date
    daily_sentiment = sentiment_data.groupby('date').mean(numeric_only=True).reset_index()

    # Convert market data index to date
    market_data = market_data.reset_index()
    market_data['date'] = market_data['Date'].dt.date

    # Merge the datasets
    aligned_data = pd.merge(market_data, daily_sentiment, on='date', how='left')

    return aligned_data


def save_market_data(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str):
    if isinstance(data, pd.DataFrame):
        if not data.empty:
            data.to_csv(filename, index=True)
            print(f"Saved market data to {filename}")
        else:
            print("No market data to save")
    elif isinstance(data, dict):
        # For multiple stocks, save each to a separate file
        for ticker, df in data.items():
            if not df.empty:
                ticker_filename = f"{ticker}_{filename}"
                df.to_csv(ticker_filename, index=True)
                print(f"Saved {ticker} data to {ticker_filename}")
    else:
        print("Unsupported data type for saving")


def main():
    # Set date range (2020-2022)
    start_date = "2015-01-01"
    end_date = "2022-12-31"

    # Get market index data (S&P 500)
    sp500_data = get_stock_data("^GSPC", start_date, end_date)

    print(sp500_data.values[45])

    # Get factor data from Kenneth French's data library
    french_data = get_factor_data(start_date, end_date)
    
    # Calculate Fama-French factors using French data
    ff_factors = calculate_fama_french_factors(
        market_data=sp500_data,
        factor_data=french_data
    )

    # Save data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_market_data(sp500_data, f"sp500_data_{timestamp}.csv")
    save_market_data(ff_factors, f"fama_french_factors_{timestamp}.csv")
    
    # Also save the raw French data
    save_market_data(french_data, f"french_factors_raw_{timestamp}.csv")
    
    # Get and save ETF factor data
    factor_data = get_factor_data(start_date, end_date)
    factor_etfs = {k: v for k, v in factor_data.items()}
    save_market_data(factor_etfs, f"factor_etfs_{timestamp}.csv")


if __name__ == "__main__":
    main()