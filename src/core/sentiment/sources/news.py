"""
News sentiment source using Finnhub API.

Finnhub provides company news and general market news with built-in sentiment
scores for some articles.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from src.core.sentiment.sources.base import SentimentSource

logger = logging.getLogger(__name__)


class NewsSource(SentimentSource):
    """
    News data source using Finnhub API.

    Fetches company-specific news and general market news for sentiment analysis.
    Requires Finnhub API key for access.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub news source.

        Args:
            api_key: Finnhub API key. Free tier allows 60 requests/minute.
        """
        super().__init__("news", rate_limit=60)
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

        if api_key:
            self._initialized = True
            logger.info("Finnhub news source initialized")
        else:
            logger.info(
                "Finnhub API key not provided, news source will return empty results"
            )

    def fetch_mentions(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch news articles for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL').
            since: Only fetch articles after this timestamp.
            limit: Maximum number of articles to fetch.

        Returns:
            List of normalized article dictionaries.
        """
        if not self._initialized or not self.api_key:
            return []

        self._respect_rate_limit()

        symbol_upper = symbol.upper()

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=7)

        from_date = since.strftime("%Y-%m-%d")
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        url = f"{self.base_url}/company-news"
        params = {
            "symbol": symbol_upper,
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 401:
                logger.warning("Finnhub API key invalid or expired")
                return []

            if response.status_code == 429:
                logger.warning("Finnhub rate limit exceeded")
                return []

            if response.status_code != 200:
                logger.warning(f"Finnhub API error: {response.status_code}")
                return []

            articles = response.json()

            if not isinstance(articles, list):
                return []

            posts = []
            for article in articles[:limit]:
                post = self._normalize_article(article, symbol_upper)
                if post:
                    if post["timestamp"] > since:
                        posts.append(post)

            return posts

        except requests.exceptions.RequestException as e:
            return self._handle_api_error(e, f"fetch_mentions({symbol})")
        except Exception as e:
            return self._handle_api_error(e, f"fetch_mentions({symbol})")

    def fetch_trending(self, limit: int = 10) -> list[dict]:
        """
        Get trending news topics.

        Note: Finnhub doesn't have a direct trending endpoint, so this
        returns general market news topics instead.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of trending news topics.
        """
        news = self.fetch_market_news(category="general", limit=limit)

        trending = []
        symbol_counts: dict[str, int] = {}

        for article in news:
            symbol = article.get("metadata", {}).get("symbol")
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        for symbol, count in sorted(
            symbol_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]:
            trending.append(
                {
                    "symbol": symbol,
                    "mentions": count,
                    "source": "news",
                }
            )

        return trending

    def fetch_market_news(
        self,
        category: str = "general",
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch general market news.

        Args:
            category: News category - 'general', 'forex', 'crypto', 'merger'.
            limit: Maximum number of articles to fetch.

        Returns:
            List of normalized article dictionaries.
        """
        if not self._initialized or not self.api_key:
            return []

        self._respect_rate_limit()

        url = f"{self.base_url}/news"
        params = {
            "category": category,
            "token": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Finnhub market news error: {response.status_code}")
                return []

            articles = response.json()

            if not isinstance(articles, list):
                return []

            posts = []
            for article in articles[:limit]:
                post = self._normalize_article(article)
                if post:
                    posts.append(post)

            return posts

        except requests.exceptions.RequestException as e:
            return self._handle_api_error(e, f"fetch_market_news({category})")
        except Exception as e:
            return self._handle_api_error(e, f"fetch_market_news({category})")

    def has_recent_news(
        self,
        symbol: str,
        hours: int = 24,
    ) -> bool:
        """
        Check if there's recent material news for a symbol.

        Useful for determining if a price move has a news catalyst.

        Args:
            symbol: Stock ticker symbol.
            hours: Look back period in hours.

        Returns:
            True if recent news exists for the symbol.
        """
        if not self._initialized:
            return False

        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        news = self.fetch_mentions(symbol, since=since, limit=5)

        return len(news) > 0

    def get_news_sentiment(
        self,
        symbol: str,
    ) -> Optional[dict]:
        """
        Get Finnhub's built-in sentiment score for a symbol.

        Note: This endpoint may require a premium Finnhub subscription.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Sentiment data dictionary or None if unavailable.
        """
        if not self._initialized or not self.api_key:
            return None

        self._respect_rate_limit()

        url = f"{self.base_url}/news-sentiment"
        params = {
            "symbol": symbol.upper(),
            "token": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            if not data or "sentiment" not in data:
                return None

            sentiment = data.get("sentiment", {})
            buzz = data.get("buzz", {})

            return {
                "symbol": symbol.upper(),
                "sentiment_score": sentiment.get("bearishPercent", 0) * -1
                + sentiment.get("bullishPercent", 0),
                "bullish_percent": sentiment.get("bullishPercent", 0),
                "bearish_percent": sentiment.get("bearishPercent", 0),
                "articles_in_week": buzz.get("articlesInLastWeek", 0),
                "buzz_score": buzz.get("buzz", 0),
                "weekly_average": buzz.get("weeklyAverage", 0),
            }

        except Exception as e:
            logger.debug(f"Failed to get news sentiment for {symbol}: {e}")
            return None

    def _normalize_article(
        self,
        article: dict,
        symbol: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Normalize Finnhub article to standard format.

        Args:
            article: Raw Finnhub article data.
            symbol: Optional symbol this article is associated with.

        Returns:
            Normalized post dictionary or None if invalid.
        """
        try:
            datetime_val = article.get("datetime")
            if datetime_val:
                timestamp = datetime.fromtimestamp(datetime_val, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline}\n{summary}" if summary else headline

            return {
                "id": str(article.get("id", hash(headline))),
                "text": text,
                "author": article.get("source", "unknown"),
                "author_data": {
                    "is_verified": True,
                    "account_age_days": 365 * 10,
                },
                "timestamp": timestamp,
                "source": "news",
                "metadata": {
                    "headline": headline,
                    "summary": summary,
                    "source_name": article.get("source"),
                    "url": article.get("url"),
                    "image": article.get("image"),
                    "category": article.get("category"),
                    "symbol": symbol or article.get("related"),
                },
            }

        except Exception as e:
            logger.debug(f"Failed to normalize article: {e}")
            return None
