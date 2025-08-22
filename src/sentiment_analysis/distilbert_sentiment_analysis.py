"""
Sentiment Analysis Module using DistilBERT

This module handles sentiment analysis of text data using DistilBERT,
a distilled version of BERT that's smaller and faster while retaining most of BERT's performance.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Union, Tuple, Optional
import concurrent.futures
import time
from tqdm import tqdm


class DistilBERTSentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the DistilBERT sentiment analyzer.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        # DistilBERT SST-2 label mapping (binary sentiment)
        self.labels = ["negative", "positive"]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary containing sentiment scores and label
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get predicted label
        predicted_class_id = int(np.argmax(probs))
        predicted_label = self.labels[predicted_class_id]

        # Create result dictionary with the same format as FinBERT
        # For compatibility, we'll map the binary sentiment to the three-class format
        # by setting neutral_score to 0
        negative_score = float(probs[0])
        positive_score = float(probs[1])
        
        result = {
            "text": text,
            "sentiment": predicted_label,
            "negative_score": negative_score,
            "neutral_score": 0.0,  # DistilBERT SST-2 doesn't have neutral class
            "positive_score": positive_score,
            "sentiment_score": float(positive_score - negative_score)  # positive - negative
        }

        return result

    def analyze_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts in batches.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing (default: 32)
            show_progress: Whether to show a progress bar (default: True)

        Returns:
            List of dictionaries containing sentiment scores and labels
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create iterator with or without progress bar
        if show_progress:
            batch_iterator = tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Processing batches")
        else:
            batch_iterator = range(0, len(texts), batch_size)

        # Process in batches
        for i in batch_iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()

            # Process each item in batch
            for j, text in enumerate(batch_texts):
                predicted_class_id = int(np.argmax(probs[j]))
                predicted_label = self.labels[predicted_class_id]

                negative_score = float(probs[j][0])
                positive_score = float(probs[j][1])

                result = {
                    "text": text,
                    "sentiment": predicted_label,
                    "negative_score": negative_score,
                    "neutral_score": 0.0,  # DistilBERT SST-2 doesn't have neutral class
                    "positive_score": positive_score,
                    "sentiment_score": float(positive_score - negative_score)  # positive - negative
                }

                results.append(result)

        return results
        
    def _process_chunk(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Process a chunk of texts (used for parallel processing).
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries containing sentiment scores and labels
        """
        return self.analyze_texts(texts, batch_size, show_progress=False)
        
    def analyze_texts_parallel(self, 
                              texts: List[str], 
                              batch_size: int = 32, 
                              num_workers: int = 4,
                              show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts in parallel using multiple workers.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing (default: 32)
            num_workers: Number of parallel workers (default: 4)
            show_progress: Whether to show a progress bar (default: True)
            
        Returns:
            List of dictionaries containing sentiment scores and labels
        """
        # Adjust num_workers if there are too few texts
        num_workers = min(num_workers, max(1, len(texts) // batch_size))
        
        # Split texts into chunks for parallel processing
        chunk_size = len(texts) // num_workers
        if chunk_size == 0:
            # If texts is smaller than num_workers, just process it directly
            return self.analyze_texts(texts, batch_size, show_progress)
            
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts) - chunk_size + 1, chunk_size)]
        # Add any remaining texts to the last chunk
        if len(texts) % chunk_size != 0:
            chunks[-1].extend(texts[-(len(texts) % chunk_size):])
            
        start_time = time.time()
        all_results = []
        
        if show_progress:
            print(f"Processing {len(texts)} texts using {num_workers} workers...")
            
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk, batch_size) for chunk in chunks]
            
            if show_progress:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
                    all_results.extend(future.result())
            else:
                for future in concurrent.futures.as_completed(futures):
                    all_results.extend(future.result())
                    
        end_time = time.time()
        if show_progress:
            print(f"Processed {len(texts)} texts in {end_time - start_time:.2f} seconds")
            
        return all_results

    def analyze_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str,
                          batch_size: int = 32,
                          use_parallel: bool = True,
                          num_workers: int = 4,
                          show_progress: bool = True,
                          chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze sentiment of texts in a DataFrame.

        Args:
            df: DataFrame containing texts
            text_column: Name of column containing texts
            batch_size: Batch size for processing (default: 32)
            use_parallel: Whether to use parallel processing (default: True)
            num_workers: Number of parallel workers (default: 4)
            show_progress: Whether to show a progress bar (default: True)
            chunk_size: Size of chunks to process at once for memory efficiency (default: None, process all at once)

        Returns:
            DataFrame with sentiment analysis results
        """
        start_time = time.time()
        
        # Process in chunks if specified
        if chunk_size is not None and len(df) > chunk_size:
            if show_progress:
                print(f"Processing DataFrame in chunks of {chunk_size} rows...")
            
            # Initialize an empty DataFrame for results
            df_with_sentiment = pd.DataFrame()
            
            # Process in chunks
            for i in tqdm(range(0, len(df), chunk_size), desc="Processing chunks") if show_progress else range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i + chunk_size].copy()
                
                # Process this chunk
                chunk_result = self._process_dataframe_chunk(
                    chunk_df, 
                    text_column, 
                    batch_size, 
                    use_parallel, 
                    num_workers, 
                    show_progress=False  # Don't show progress for individual chunks
                )
                
                # Append to results
                df_with_sentiment = pd.concat([df_with_sentiment, chunk_result], ignore_index=True)
                
            end_time = time.time()
            if show_progress:
                print(f"Processed {len(df)} rows in {end_time - start_time:.2f} seconds")
                
            return df_with_sentiment
        else:
            # Process the entire DataFrame at once
            return self._process_dataframe_chunk(df, text_column, batch_size, use_parallel, num_workers, show_progress)
    
    def _process_dataframe_chunk(self,
                               df: pd.DataFrame,
                               text_column: str,
                               batch_size: int = 32,
                               use_parallel: bool = True,
                               num_workers: int = 4,
                               show_progress: bool = True) -> pd.DataFrame:
        """
        Process a chunk of the DataFrame (helper method for analyze_dataframe).
        
        Args:
            df: DataFrame chunk containing texts
            text_column: Name of column containing texts
            batch_size: Batch size for processing
            use_parallel: Whether to use parallel processing
            num_workers: Number of parallel workers
            show_progress: Whether to show a progress bar
            
        Returns:
            DataFrame chunk with sentiment analysis results
        """
        # Extract texts
        texts = df[text_column].tolist()

        # Analyze texts
        if use_parallel and len(texts) > batch_size:
            results = self.analyze_texts_parallel(texts, batch_size, num_workers, show_progress)
        else:
            results = self.analyze_texts(texts, batch_size, show_progress)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Merge with original DataFrame
        df_with_sentiment = pd.concat([df.reset_index(drop=True),
                                      results_df.drop('text', axis=1).reset_index(drop=True)],
                                     axis=1)

        return df_with_sentiment

    def aggregate_sentiment(self,
                            df: pd.DataFrame,
                            groupby_column: str = None,
                            time_column: str = None,
                            freq: str = 'D') -> pd.DataFrame:
        """
        Aggregate sentiment scores by group or time period.

        Args:
            df: DataFrame containing sentiment scores
            groupby_column: Column to group by (e.g., 'ticker')
            time_column: Column containing timestamps
            freq: Frequency for time-based aggregation ('D' for daily, 'H' for hourly, etc.)

        Returns:
            DataFrame with aggregated sentiment scores
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # If time column is provided, convert to datetime
        if time_column is not None:
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])

        # Define aggregation functions
        agg_funcs = {
            'sentiment_score': ['mean', 'median', 'std', 'count'],
            'positive_score': ['mean', 'sum'],
            'negative_score': ['mean', 'sum'],
            'neutral_score': ['mean', 'sum']
        }

        # Group by time period
        if time_column is not None and groupby_column is not None:
            # Group by both time and another column
            df_copy['time_period'] = df_copy[time_column].dt.floor(freq)
            aggregated = df_copy.groupby([groupby_column, 'time_period']).agg(agg_funcs)
        elif time_column is not None:
            # Group by time only
            df_copy['time_period'] = df_copy[time_column].dt.floor(freq)
            aggregated = df_copy.groupby('time_period').agg(agg_funcs)
        elif groupby_column is not None:
            # Group by column only
            aggregated = df_copy.groupby(groupby_column).agg(agg_funcs)
        else:
            # No grouping, just aggregate all
            return pd.DataFrame({
                'sentiment_score_mean': [df_copy['sentiment_score'].mean()],
                'sentiment_score_median': [df_copy['sentiment_score'].median()],
                'sentiment_score_std': [df_copy['sentiment_score'].std()],
                'sentiment_score_count': [df_copy['sentiment_score'].count()],
                'positive_score_mean': [df_copy['positive_score'].mean()],
                'positive_score_sum': [df_copy['positive_score'].sum()],
                'negative_score_mean': [df_copy['negative_score'].mean()],
                'negative_score_sum': [df_copy['negative_score'].sum()],
                'neutral_score_mean': [df_copy['neutral_score'].mean()],
                'neutral_score_sum': [df_copy['neutral_score'].sum()]
            })

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        # Reset index
        aggregated = aggregated.reset_index()

        # Calculate bullish-to-bearish ratio
        aggregated['bullish_bearish_ratio'] = aggregated['positive_score_sum'] / aggregated['negative_score_sum']

        # Calculate sentiment volatility (if we have enough data points)
        if 'sentiment_score_std' in aggregated.columns:
            aggregated['sentiment_volatility'] = aggregated['sentiment_score_std']

        return aggregated


def main():
    """
    Analyze sentiment of tweets in the combined_cleaned_tweets2.csv file and
    save the results to a new CSV file with sentiment scores added.
    
    This function supports command-line arguments for controlling the optimization parameters:
    --batch-size: Batch size for processing (default: 32)
    --no-parallel: Disable parallel processing
    --workers: Number of parallel workers (default: 4)
    --chunk-size: Size of chunks to process at once for memory efficiency (default: 1000)
    --no-progress: Disable progress bars
    --test-only: Only run a test on a small subset of data
    """
    import os
    import argparse
    import sys
    from pathlib import Path
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze sentiment of tweets using DistilBERT')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of chunks to process at once')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    parser.add_argument('--test-only', action='store_true', help='Only run a test on a small subset of data')
    
    # Check if running as script or imported
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Default arguments when imported as a module
        args = parser.parse_args([])
    
    # Define file paths
    # Use absolute paths based on the repository root
    repo_root = Path(__file__).parent.parent.parent  # Go up three levels from this file
    input_file = repo_root / "src" / "data_collection" / "data" / "combined_cleaned_tweets2_part8.csv"
    output_dir = Path(__file__).parent / "data"  # Create data dir in the same directory as this script
    output_file = output_dir / "combined_cleaned_tweets_with_distilbert_sentiment_part8.csv"
    
    print(f"Repository root: {repo_root}")
    print(f"Looking for input file at: {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading tweets from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Print basic info about the dataset
    print(f"Dataset contains {len(df)} tweets")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize analyzer
    analyzer = DistilBERTSentimentAnalyzer()
    
    # First, test with a small subset
    print("\nTesting with 5 tweets...")
    test_df = df.head(5).copy()
    test_results = analyzer.analyze_dataframe(
        test_df, 
        text_column='text',
        batch_size=args.batch_size,
        use_parallel=not args.no_parallel,
        num_workers=args.workers,
        show_progress=not args.no_progress
    )
    print(test_results[['text', 'sentiment', 'sentiment_score']].head())
    
    # If test only, return here
    if args.test_only:
        print("Test completed successfully. Exiting as requested.")
        return test_results
    
    # Process the entire dataset
    print(f"\nAnalyzing sentiment for all {len(df)} tweets...")
    print(f"Using configuration: batch_size={args.batch_size}, parallel={not args.no_parallel}, workers={args.workers}, chunk_size={args.chunk_size}")
    
    start_time = time.time()
    
    df_with_sentiment = analyzer.analyze_dataframe(
        df, 
        text_column='text',
        batch_size=args.batch_size,
        use_parallel=not args.no_parallel,
        num_workers=args.workers,
        show_progress=not args.no_progress,
        chunk_size=args.chunk_size
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save the results
    print(f"Saving results to {output_file}...")
    df_with_sentiment.to_csv(output_file, index=False)
    
    print(f"Sentiment analysis complete. Results saved to {output_file}")
    print(f"Total processing time: {processing_time:.2f} seconds for {len(df)} tweets")
    print(f"Processing speed: {len(df) / processing_time:.2f} tweets per second")
    
    # Print some statistics
    print("\nSentiment distribution:")
    print(df_with_sentiment['sentiment'].value_counts())
    
    print("\nAverage sentiment score:", df_with_sentiment['sentiment_score'].mean())
    
    # Return the DataFrame for testing purposes
    return df_with_sentiment


if __name__ == "__main__":
    main()