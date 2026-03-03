"""Sentiment analysis module for Leprechaun trading bot."""

from src.core.sentiment.analyzer import (
    FINANCIAL_LEXICON,
    FinBERTAnalyzer,
    SentimentAnalyzer,
)
from src.core.sentiment.manipulation import ManipulationDetector
from src.core.sentiment.sources import (
    NewsSource,
    RedditSource,
    SentimentSource,
    StockTwitsSource,
)

__all__ = [
    "FINANCIAL_LEXICON",
    "FinBERTAnalyzer",
    "ManipulationDetector",
    "NewsSource",
    "RedditSource",
    "SentimentAnalyzer",
    "SentimentSource",
    "StockTwitsSource",
]
