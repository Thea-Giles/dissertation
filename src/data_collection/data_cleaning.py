"""
Data Cleaning Module for External Datasets
"""
import re
from typing import Optional

import pandas as pd

def clean_tweet_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Standardize column names
    df = df.rename(columns={
        'tweet_id': 'tweet_id',
        'post_date': 'date',
        'body': 'text',
        'writer': 'user'
    })
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], unit='s')
    
    # Extract stock symbols from text
    def extract_stock_symbols(text):
        if not isinstance(text, str):
            return None
        symbols = re.findall(r'\$([A-Z][A-Z0-9_]*)', text)
        return ','.join(symbols) if symbols else None
    
    df['stock'] = df['text'].apply(extract_stock_symbols)
    columns = ['tweet_id', 'date', 'text', 'stock', 'user']
    return df[columns]


def clean_stockerbot_export(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, quoting=1, escapechar='\\', on_bad_lines='warn')
    
    # Standardize column names
    df = df.rename(columns={
        'id': 'tweet_id',
        'timestamp': 'date',
        'text': 'text',
        'symbols': 'stock'
    })
    
    # Convert date string to datetime with error handling
    try:
        df['date'] = pd.to_datetime(df['date'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion error - {e}")
        # Try a more flexible parsing approach
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Select and reorder columns, handling missing columns
    columns = ['tweet_id', 'date', 'text', 'stock']
    for col in ['company_names', 'verified']:
        if col in df.columns:
            columns.append(col)
    
    return df[columns]


def clean_tweets_remaining(file_path: str) -> pd.DataFrame:
    # Read the file as text to handle multi-line tweets
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Get the header line (first line)
    header = lines[0].strip()
    
    # Process lines to handle multi-line tweets
    processed_lines = [header]  # Start with the header
    current_tweet = None
    
    for line in lines[1:]:  # Skip the header line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this is a new tweet (starts with an ID)
        if re.match(r'^\d+;', line):
            # If we have a previous tweet, add it to processed lines
            if current_tweet:
                processed_lines.append(current_tweet)
            
            # Start a new tweet
            current_tweet = line
        else:
            # This is a continuation of the current tweet
            if current_tweet:
                current_tweet += " " + line
    
    # Add the last tweet if there is one
    if current_tweet:
        processed_lines.append(current_tweet)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        temp_file.write('\n'.join(processed_lines))
        temp_path = temp_file.name

    # Read the processed CSV file
    try:
        df = pd.read_csv(temp_path, sep=';')

    finally:
        import os
        os.unlink(temp_path)  # Delete the temporary file
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'id':
            column_mapping[col] = 'tweet_id'
        elif col.lower() == 'created_at':
            column_mapping[col] = 'date'
        elif col.lower() == 'full_text':
            column_mapping[col] = 'text'
    
    df = df.rename(columns=column_mapping)
    
    # Extract stock symbols from text
    def extract_stock_symbols(text):
        if not isinstance(text, str):
            return None
        symbols = re.findall(r'\$([A-Z][A-Z0-9_]*)', text)
        return ','.join(symbols) if symbols else None
    
    # Make sure we're using the correct column name for text
    text_column = 'text' if 'text' in df.columns else 'full_text'
    df['stock'] = df[text_column].apply(extract_stock_symbols)

    df['source'] = 'tweets_remaining'
    
    # Select and reorder columns, handling missing columns
    columns = []
    if 'tweet_id' in df.columns:
        columns.append('tweet_id')
    elif 'id' in df.columns:
        columns.append('id')
        
    if 'date' in df.columns:
        columns.append('date')
    elif 'created_at' in df.columns:
        columns.append('created_at')
        
    if 'text' in df.columns:
        columns.append('text')
    elif 'full_text' in df.columns:
        columns.append('full_text')
        
    columns.extend(['stock', 'source'])
    
    return df[columns]


def load_and_clean_all_datasets(
    tweets_remaining_path: Optional[str] = None,
    tweet_csv_path: Optional[str] = None,
    stockerbot_path: Optional[str] = None
) -> dict:
    result = {}
    
    if tweets_remaining_path:
        result['tweets_remaining'] = clean_tweets_remaining(tweets_remaining_path)
        
    if tweet_csv_path:
        result['tweet_csv'] = clean_tweet_csv(tweet_csv_path)
        
    if stockerbot_path:
        result['stockerbot'] = clean_stockerbot_export(stockerbot_path)
        
    return result


def combine_datasets(datasets: dict) -> pd.DataFrame:
    dfs = []
    
    # Add a source column to each dataframe and append to the list
    for source, df in datasets.items():
        df_copy = df.copy()
        df_copy['source'] = source
        dfs.append(df_copy)
    
    # Concatenate all dataframes
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def main():
    # Define paths to the datasets
    tweet_csv_path = "../../external_data/Tweet.csv"
    stockerbot_path = "../../external_data/stockerbot-export.csv"
    tweets_remaining_path = "../../external_data/tweets_remaining_09042020_16072020.csv"
    
    # Clean individual datasets
    tweet_csv_df = clean_tweet_csv(tweet_csv_path)
    stockerbot_df = clean_stockerbot_export(stockerbot_path)
    tweets_remaining_df = clean_tweets_remaining(tweets_remaining_path)

    print(f"Tweet CSV: {len(tweet_csv_df)} rows")
    print(f"Stockerbot: {len(stockerbot_df)} rows")
    print(f"Tweets Remaining: {len(tweets_remaining_df)} rows")
    
    # Alternatively, use the combined function
    datasets = load_and_clean_all_datasets(
        tweets_remaining_path=tweets_remaining_path,
        tweet_csv_path=tweet_csv_path,
        stockerbot_path=stockerbot_path
    )
    
    # Combine all datasets
    combined_df = combine_datasets(datasets)
    print(f"Combined dataset: {len(combined_df)} rows")
    
    # Save the combined dataset
    combined_df.to_csv("/data/combined_cleaned_tweets2.csv", index=False)
    print("Combined dataset saved to combined_cleaned_tweets.csv")

if __name__ == "__main__":
    main()