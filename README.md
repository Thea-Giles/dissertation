# Financial Market Prediction using Sentiment Analysis and Machine Learning

This project implements a research framework for predicting a financial market direction using sentiment analysis from 
social media alongside traditional financial models.


## Research Hypotheses

1. **H1**: Financial news headlines and social media sentiment can be used to accurately predict a financial market direction in the short term.
2. **H2**: Machine learning models that incorporate sentiment data will outperform traditional econometric models in predicting market trends.
3. **H3**: During periods of financial crises, social media sentiment will be a more reliable predictor of market volatility compared to conventional sentiment indicators.


## Project Structure

## Components

### Data Collection

- [**Market Data Collection**](/src/data_collection/market_data.py): Fetches historical stock price data and financial factors:
  - Stock price data from Yahoo Finance
  - Fama-French three-factor model data directly from Kenneth French's data library
