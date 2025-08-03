# Financial Market Prediction using Sentiment Analysis and Machine Learning

This project implements a research framework for predicting a financial market direction using sentiment analysis from 
social media alongside traditional financial models.


## Research Hypotheses

1. **H1**: Financial news headlines and social media sentiment can be used to accurately predict a financial market direction in the short term.
2. **H2**: Machine learning models that incorporate sentiment data will outperform traditional econometric models in predicting market trends.
3. **H3**: During periods of financial crises, social media sentiment will be a more reliable predictor of market volatility compared to conventional sentiment indicators.


## Project Structure
```
├── src/                                            # Source code
│   ├── data_collection/                            # Data collection
│   │   ├── data/                   
│   │   │   └── combined_cleaned_tweets2.csv        # Cleaned tweet data (can't commit to git)
│   │   ├── twitter_data.py                         # Twitter data collection
│   │   └── market_data.py                          # Financial market data collection  
└── README.md                                       # Project documentation
```

## Components

### Data Collection

- [**Market Data Collection**](/src/data_collection/market_data.py): Fetches historical stock price data and financial factors:
  - Stock price data from Yahoo Finance
  - Fama-French three-factor model data directly from Kenneth French's data library
- [**Twitter Data Collection**](/src/data_collection/twitter_data.py): Collects tweets related to specific stock tickers and financial terms:
  - Direct collection via Twitter API
  - Access to pre-existing stock tweet datasets for S&P 500 companies (alternative to API limits)
- [**Pre-existing Data Cleaning**](/src/data_collection/data_cleaning.py): Cleans data collected from external sources. 
  - Due to Twitter API rate limits, the following pre-existing datasets will be used:
    1. **Stock Market Tweets**: 5.8M tweets about major S&P 500 companies from 2015-2020
       - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020)
       - Coverage: AAPL, AMZN, FB, GOOG, MSFT, TSLA

    2. **Twitter Financial Sentiment**: 10K+ labeled tweets about financial markets and stocks
       - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/davidwallach/financial-tweets)
       - Coverage: Various stocks

    3. **Stock Market Tweets Data**: 900k+ tweets using the S&P 500 tag from 2020
       - Source: [ieee Dataport](https://ieee-dataport.org/open-access/stock-market-tweets-data)
       - Coverage: S&P500
       
    4. **Tweets Remaining**: 900k+ tweets about financial markets from April to July 2020
       - Source: External dataset (tweets_remaining_09042020_16072020.csv)
       - Coverage: Various stocks and financial markets
       - Note: This dataset contains multi-line tweets that are cleaned using the `clean_tweets_remaining` function