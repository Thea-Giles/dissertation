"""
Twitter Data Collection Module
"""

import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import tweepy


class TwitterDataCollector:
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        self.auth = tweepy.OAuthHandler(api_key, api_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        
    def search_tweets(self, 
                     query: str, 
                     count: int = 100, 
                     lang: str = "en", 
                     since_id: Optional[int] = None,
                     until: Optional[str] = None) -> List[Dict[str, Any]]:
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.api.search_tweets, 
                                      q=query,
                                      lang=lang,
                                      count=count,
                                      tweet_mode='extended',
                                      since_id=since_id,
                                      until=until).items(count):
                tweet_dict = {
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'user_followers': tweet.user.followers_count,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                }
                tweets.append(tweet_dict)
                
            print(f"Retrieved {len(tweets)} tweets for query: {query}")
            return tweets
        except tweepy.TweepyException as e:
            print(f"Error retrieving tweets: {e}")
            return []
    
    def collect_financial_tweets(self, 
                               tickers: List[str], 
                               additional_keywords: List[str] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               days_back: int = 7,
                               tweets_per_query: int = 100) -> pd.DataFrame:
            
        all_tweets = []
        
        # Create queries for each ticker
        for ticker in tickers:
            query = f"${ticker} OR #{ticker} OR {ticker} AND ({' OR '.join(additional_keywords)})"
            
            # Calculate dates
            if end_date is None:
                until_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                until_date = end_date
                
            if start_date is None:
                since_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
            else:
                since_date = start_date
            
            print(f"Collecting tweets for {ticker} from {since_date} to {until_date}")
            
            # Search for tweets
            ticker_tweets = self.search_tweets(
                query=query,
                count=tweets_per_query,
                until=until_date
            )
            
            # Filter tweets by date if start_date is provided
            if start_date is not None:
                since_date_obj = datetime.datetime.strptime(since_date, '%Y-%m-%d')
                ticker_tweets = [tweet for tweet in ticker_tweets 
                                if tweet['created_at'].date() >= since_date_obj.date()]
            
            # Add ticker information to each tweet
            for tweet in ticker_tweets:
                tweet['ticker'] = ticker
                
            all_tweets.extend(ticker_tweets)
        
        # Convert to DataFrame
        if all_tweets:
            df = pd.DataFrame(all_tweets)
            return df
        else:
            return pd.DataFrame()
    
    def save_tweets(self, tweets_df: pd.DataFrame, filename: str):
        if not tweets_df.empty:
            tweets_df.to_csv(filename, index=False)
            print(f"Saved {len(tweets_df)} tweets to {filename}")
        else:
            print("No tweets to save")


def main():
    api_key = "XXXXXXXXXXXXXXXXXXX"
    api_secret = "XXXXXXXXXX"
    access_token = "XXXXXXXXXXXXXX"
    access_token_secret = "XXXXXXXXXXXXXX"
    
    # Initialize collector
    collector = TwitterDataCollector(api_key, api_secret, access_token, access_token_secret)

    # Not a ticker, but what people might tweet, should probably use both.
    tickers = ["S&P500"]
    additional_keywords = ["earnings", "stock", "price", "market", "trading", "investor"]
    
    # Using days_back
    print("Collecting tweets from the last 7 days")
    tweets_df1 = collector.collect_financial_tweets(
        tickers=tickers,
        additional_keywords=additional_keywords,
        days_back=7,
        tweets_per_query=100
    )
    
    # Using specific date range
    print("\nCollecting tweets from a specific date range (2020-01-01 to 2022-12-31)")
    tweets_df2 = collector.collect_financial_tweets(
        tickers=tickers,
        additional_keywords=additional_keywords,
        start_date="2020-01-01",
        end_date="2022-12-31",
        tweets_per_query=100
    )
    
    # Use the most recent collection for saving
    tweets_df = tweets_df2 if not tweets_df2.empty else tweets_df1
    
    # Save tweets
    if not tweets_df.empty:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        collector.save_tweets(tweets_df, f"financial_tweets_{timestamp}.csv")


if __name__ == "__main__":
    main()