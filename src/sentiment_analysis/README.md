# Sentiment Analysis Module

This module handles sentiment analysis of financial text data using FinBERT, a pre-trained NLP model specifically designed for financial text.

## Features

- Sentiment analysis of individual texts or large datasets
- Parallel processing for improved performance
- Memory-efficient processing of large datasets
- Batch processing for optimal GPU/CPU utilization
- Progress tracking with tqdm
- Aggregation of sentiment scores by time period or other groupings

## Usage

### Basic Usage

```python
from src.sentiment_analysis.sentiment_analysis import FinBERTSentimentAnalyzer

# Initialize the analyzer
analyzer = FinBERTSentimentAnalyzer()

# Analyze a single text
result = analyzer.analyze_text("The company reported strong earnings, beating analyst expectations.")
print(f"Sentiment: {result['sentiment']}")
print(f"Sentiment Score: {result['sentiment_score']:.4f}")

# Analyze multiple texts
texts = [
    "The company reported strong earnings, beating analyst expectations.",
    "The stock price plummeted after the CEO resigned.",
    "Investors are cautious about the company's future prospects."
]
results = analyzer.analyze_texts(texts)

# Analyze texts in a DataFrame
import pandas as pd
df = pd.DataFrame({"text": texts})
df_with_sentiment = analyzer.analyze_dataframe(df, text_column="text")
print(df_with_sentiment[["text", "sentiment", "sentiment_score"]])
```

### Command-line Usage

The module can be run as a script to analyze tweets in the `combined_cleaned_tweets2.csv` file:

```
python -m src.sentiment_analysis.sentiment_analysis [options]
```

Options:
- `--batch-size`: Batch size for processing (default: 32)
- `--no-parallel`: Disable parallel processing
- `--workers`: Number of parallel workers (default: 4)
- `--chunk-size`: Size of chunks to process at once for memory efficiency (default: 1000)
- `--no-progress`: Disable progress bars
- `--test-only`: Only run a test on a small subset of data

### Performance Testing

To find the optimal configuration for your hardware, use the `test_performance.py` script:

```
python -m src.sentiment_analysis.test_performance [options]
```

Options:
- `--sample-size`: Number of tweets to sample for testing (default: 100)
- `--quick`: Run only a quick test with default configuration

## Performance Optimization Tips

1. **Batch Size**: 
   - Larger batch sizes generally improve performance but require more memory
   - For GPU processing, try batch sizes of 32, 64, or 128
   - For CPU processing, smaller batch sizes (16-32) might be more efficient

2. **Parallel Processing**:
   - Enable parallel processing for significant speedups on multi-core systems
   - Adjust the number of workers based on your CPU cores (typically 4-8 workers)
   - Disable parallel processing for very small datasets or when memory is limited

3. **Chunk Size**:
   - For large datasets, processing in chunks reduces memory usage
   - Smaller chunks (500-1000) are more memory-efficient but might be slightly slower
   - Set to None (or 0 via command line) to process the entire dataset at once if memory allows

4. **GPU Acceleration**:
   - The module automatically uses GPU if available
   - For best GPU performance, use larger batch sizes and disable chunking

## Memory Usage Considerations

For very large datasets, consider the following:

1. Process the data in chunks by setting an appropriate `chunk_size`
2. Reduce the batch size if you encounter memory issues
3. Save intermediate results to disk if processing millions of texts

## Example: Processing a Large Dataset

```python
from src.sentiment_analysis.sentiment_analysis import FinBERTSentimentAnalyzer
import pandas as pd

# Load a large dataset
df = pd.read_csv("large_dataset.csv")

# Initialize analyzer
analyzer = FinBERTSentimentAnalyzer()

# Process in memory-efficient chunks with parallel processing
df_with_sentiment = analyzer.analyze_dataframe(
    df,
    text_column="text",
    batch_size=32,
    use_parallel=True,
    num_workers=4,
    show_progress=True,
    chunk_size=1000
)

# Save results
df_with_sentiment.to_csv("large_dataset_with_sentiment.csv", index=False)
```

## Performance Benchmarks

Performance varies based on hardware, but here are some typical improvements:

- **Original implementation**: ~10-20 texts per second
- **Optimized with parallel processing**: ~40-100 texts per second (4x-5x speedup)
- **GPU acceleration**: ~200-500 texts per second (10x-25x speedup)

Run the `test_performance.py` script to benchmark on your specific hardware.