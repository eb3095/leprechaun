"""Sentiment data source integrations."""

from src.core.sentiment.sources.base import SentimentSource
from src.core.sentiment.sources.news import NewsSource
from src.core.sentiment.sources.reddit import RedditSource
from src.core.sentiment.sources.stocktwits import StockTwitsSource

__all__ = [
    "NewsSource",
    "RedditSource",
    "SentimentSource",
    "StockTwitsSource",
]
