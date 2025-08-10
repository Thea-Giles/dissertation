"""
Sentiment Analysis Module using DistilBERT
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
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        self.labels = ["negative", "positive"]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get predicted label
        predicted_class_id = int(np.argmax(probs))
        predicted_label = self.labels[predicted_class_id]

        negative_score = float(probs[0])
        positive_score = float(probs[1])
        
        result = {
            "text": text,
            "sentiment": predicted_label,
            "negative_score": negative_score,
            "neutral_score": 0.0,
            "positive_score": positive_score,
            "sentiment_score": float(positive_score - negative_score)
        }

        return result

    def analyze_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[Dict[str, Any]]:
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        

        batch_iterator = range(0, len(texts), batch_size)

        # Process in batches
        for i in batch_iterator:
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()

            for j, text in enumerate(batch_texts):
                predicted_class_id = int(np.argmax(probs[j]))
                predicted_label = self.labels[predicted_class_id]

                negative_score = float(probs[j][0])
                positive_score = float(probs[j][1])

                result = {
                    "text": text,
                    "sentiment": predicted_label,
                    "negative_score": negative_score,
                    "neutral_score": 0.0,
                    "positive_score": positive_score,
                    "sentiment_score": float(positive_score - negative_score)
                }

                results.append(result)

        return results
        
    def _process_chunk(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        return self.analyze_texts(texts, batch_size, show_progress=False)
        
    def analyze_texts_parallel(self, 
                              texts: List[str], 
                              batch_size: int = 32, 
                              num_workers: int = 4,
                              show_progress: bool = True) -> List[Dict[str, Any]]:
        num_workers = min(num_workers, max(1, len(texts) // batch_size))

        chunk_size = len(texts) // num_workers
        if chunk_size == 0:
            return self.analyze_texts(texts, batch_size, show_progress)
            
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts) - chunk_size + 1, chunk_size)]

        if len(texts) % chunk_size != 0:
            chunks[-1].extend(texts[-(len(texts) % chunk_size):])
            
        start_time = time.time()
        all_results = []
        
        if show_progress:
            print(f"Processing {len(texts)} texts using {num_workers} workers...")

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
        start_time = time.time()

        if chunk_size is not None and len(df) > chunk_size:
            if show_progress:
                print(f"Processing DataFrame in chunks of {chunk_size} rows...")

            df_with_sentiment = pd.DataFrame()

            for i in tqdm(range(0, len(df), chunk_size), desc="Processing chunks") if show_progress else range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i + chunk_size].copy()

                chunk_result = self._process_dataframe_chunk(
                    chunk_df, 
                    text_column, 
                    batch_size, 
                    use_parallel, 
                    num_workers, 
                    show_progress=False
                )

                df_with_sentiment = pd.concat([df_with_sentiment, chunk_result], ignore_index=True)
                
            end_time = time.time()
            if show_progress:
                print(f"Processed {len(df)} rows in {end_time - start_time:.2f} seconds")
                
            return df_with_sentiment
        else:
            return self._process_dataframe_chunk(df, text_column, batch_size, use_parallel, num_workers, show_progress)
    
    def _process_dataframe_chunk(self,
                               df: pd.DataFrame,
                               text_column: str,
                               batch_size: int = 32,
                               use_parallel: bool = True,
                               num_workers: int = 4,
                               show_progress: bool = True) -> pd.DataFrame:
        texts = df[text_column].tolist()

        if use_parallel and len(texts) > batch_size:
            results = self.analyze_texts_parallel(texts, batch_size, num_workers, show_progress)
        else:
            results = self.analyze_texts(texts, batch_size, show_progress)

        results_df = pd.DataFrame(results)

        df_with_sentiment = pd.concat([df.reset_index(drop=True),
                                      results_df.drop('text', axis=1).reset_index(drop=True)],
                                     axis=1)

        return df_with_sentiment

    def aggregate_sentiment(self,
                            df: pd.DataFrame,
                            groupby_column: str = None,
                            time_column: str = None,
                            freq: str = 'D') -> pd.DataFrame:
        df_copy = df.copy()

        if time_column is not None:
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])

        agg_funcs = {
            'sentiment_score': ['mean', 'median', 'std', 'count'],
            'positive_score': ['mean', 'sum'],
            'negative_score': ['mean', 'sum'],
            'neutral_score': ['mean', 'sum']
        }

        if time_column is not None and groupby_column is not None:
            df_copy['time_period'] = df_copy[time_column].dt.floor(freq)
            aggregated = df_copy.groupby([groupby_column, 'time_period']).agg(agg_funcs)
        elif time_column is not None:
            df_copy['time_period'] = df_copy[time_column].dt.floor(freq)
            aggregated = df_copy.groupby('time_period').agg(agg_funcs)
        elif groupby_column is not None:
            aggregated = df_copy.groupby(groupby_column).agg(agg_funcs)
        else:
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

        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated = aggregated.reset_index()

        aggregated['bullish_bearish_ratio'] = aggregated['positive_score_sum'] / aggregated['negative_score_sum']

        if 'sentiment_score_std' in aggregated.columns:
            aggregated['sentiment_volatility'] = aggregated['sentiment_score_std']

        return aggregated


def main():
    import os
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Analyze sentiment of tweets using DistilBERT')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of chunks to process at once')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    parser.add_argument('--test-only', action='store_true', help='Only run a test on a small subset of data')

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    repo_root = Path(__file__).parent.parent.parent
    input_file = repo_root / "src" / "data_collection" / "data" / "combined_cleaned_tweets2.csv"
    output_dir = Path(__file__).parent / "data"
    output_file = output_dir / "combined_cleaned_tweets_with_distilbert_sentiment.csv"
    
    print(f"Repository root: {repo_root}")
    print(f"Looking for input file at: {input_file}")

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading tweets from {input_file}...")

    df = pd.read_csv(input_file)
    
    # Print basic info about the dataset
    print(f"Dataset contains {len(df)} tweets")
    print(f"Columns: {df.columns.tolist()}")

    analyzer = DistilBERTSentimentAnalyzer()

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

    if args.test_only:
        print("Test completed successfully. Exiting as requested.")
        return test_results
    
    # ntire dataset
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

    print(f"Saving results to {output_file}...")
    df_with_sentiment.to_csv(output_file, index=False)
    
    print(f"Sentiment analysis complete. Results saved to {output_file}")
    print(f"Total processing time: {processing_time:.2f} seconds for {len(df)} tweets")
    print(f"Processing speed: {len(df) / processing_time:.2f} tweets per second")

    print("\nSentiment distribution:")
    print(df_with_sentiment['sentiment'].value_counts())
    
    print("\nAverage sentiment score:", df_with_sentiment['sentiment_score'].mean())

    return df_with_sentiment


if __name__ == "__main__":
    main()