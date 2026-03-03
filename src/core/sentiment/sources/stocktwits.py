"""
StockTwits sentiment source.

StockTwits is a social network for traders and investors. Messages ("twits")
are tagged with stock symbols and often include author sentiment.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from src.core.sentiment.sources.base import SentimentSource

logger = logging.getLogger(__name__)


class StockTwitsSource(SentimentSource):
    """
    StockTwits data source via API or RapidAPI.

    Fetches messages and trending symbols from StockTwits for sentiment analysis.
    Works without API key for basic access, with key for extended features.
    """

    def __init__(self, api_key: Optional[str] = None, use_rapidapi: bool = False):
        """
        Initialize StockTwits source.

        Args:
            api_key: API key for StockTwits or RapidAPI.
            use_rapidapi: If True, use RapidAPI endpoint instead of direct API.
        """
        super().__init__("stocktwits", rate_limit=200)
        self.api_key = api_key
        self.use_rapidapi = use_rapidapi

        if use_rapidapi:
            self.base_url = "https://stocktwits.p.rapidapi.com"
            self.headers = {
                "X-RapidAPI-Key": api_key or "",
                "X-RapidAPI-Host": "stocktwits.p.rapidapi.com",
            }
        else:
            self.base_url = "https://api.stocktwits.com/api/2"
            self.headers = {}
            if api_key:
                self.headers["Authorization"] = f"Bearer {api_key}"

        self._initialized = True
        logger.info(f"StockTwits source initialized (RapidAPI: {use_rapidapi})")

    def fetch_mentions(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 30,
    ) -> list[dict]:
        """
        Fetch StockTwits messages for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL').
            since: Only fetch messages after this timestamp.
            limit: Maximum number of messages to fetch.

        Returns:
            List of normalized message dictionaries.
        """
        self._respect_rate_limit()

        symbol_upper = symbol.upper()
        url = f"{self.base_url}/streams/symbol/{symbol_upper}.json"

        params = {"limit": min(limit, 30)}

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 404:
                logger.debug(f"Symbol not found on StockTwits: {symbol}")
                return []

            if response.status_code == 429:
                logger.warning("StockTwits rate limit exceeded")
                return []

            if response.status_code != 200:
                logger.warning(
                    f"StockTwits API error: {response.status_code} - {response.text[:200]}"
                )
                return []

            data = response.json()

            if data.get("response", {}).get("status") != 200:
                return []

            messages = data.get("messages", [])
            posts = []

            for msg in messages:
                post = self._normalize_message(msg)
                if post:
                    if since is None or post["timestamp"] > since:
                        posts.append(post)

            return posts

        except requests.exceptions.RequestException as e:
            return self._handle_api_error(e, f"fetch_mentions({symbol})")
        except Exception as e:
            return self._handle_api_error(e, f"fetch_mentions({symbol})")

    def fetch_trending(self, limit: int = 10) -> list[dict]:
        """
        Get trending symbols from StockTwits.

        Args:
            limit: Maximum number of trending items to return.

        Returns:
            List of trending ticker dictionaries.
        """
        self._respect_rate_limit()

        url = f"{self.base_url}/trending/symbols.json"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"StockTwits trending API error: {response.status_code}")
                return []

            data = response.json()

            if data.get("response", {}).get("status") != 200:
                return []

            symbols = data.get("symbols", [])
            trending = []

            for item in symbols[:limit]:
                trending.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "title": item.get("title", ""),
                        "watchlist_count": item.get("watchlist_count", 0),
                        "source": "stocktwits",
                    }
                )

            return trending

        except requests.exceptions.RequestException as e:
            return self._handle_api_error(e, "fetch_trending")
        except Exception as e:
            return self._handle_api_error(e, "fetch_trending")

    def fetch_streams_home(self, limit: int = 30) -> list[dict]:
        """
        Fetch general stream of popular messages.

        Args:
            limit: Maximum number of messages to fetch.

        Returns:
            List of normalized message dictionaries.
        """
        self._respect_rate_limit()

        url = f"{self.base_url}/streams/home.json"
        params = {"limit": min(limit, 30)}

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code != 200:
                return []

            data = response.json()
            messages = data.get("messages", [])

            return [
                post
                for msg in messages
                if (post := self._normalize_message(msg)) is not None
            ]

        except Exception as e:
            return self._handle_api_error(e, "fetch_streams_home")

    def _normalize_message(self, msg: dict) -> Optional[dict]:
        """
        Normalize StockTwits message to standard format.

        Args:
            msg: Raw StockTwits message data.

        Returns:
            Normalized post dictionary or None if invalid.
        """
        try:
            created_at = msg.get("created_at")
            timestamp = self._parse_timestamp(created_at)

            user = msg.get("user", {})
            author_name = user.get("username", "unknown")

            join_date = user.get("join_date")
            account_age_days = 365
            if join_date:
                try:
                    join_dt = self._parse_timestamp(join_date)
                    account_age_days = (datetime.now(timezone.utc) - join_dt).days
                except Exception:
                    pass

            author_data = {
                "username": author_name,
                "account_age_days": account_age_days,
                "followers": user.get("followers", 0),
                "following": user.get("following", 0),
                "ideas": user.get("ideas", 0),
                "is_verified": user.get("official", False),
            }

            if author_data["ideas"] > 0:
                author_data["posts_per_day"] = author_data["ideas"] / max(
                    account_age_days, 1
                )

            symbols = msg.get("symbols", [])
            symbol_names = [s.get("symbol", "") for s in symbols if s.get("symbol")]

            sentiment = msg.get("entities", {}).get("sentiment", {})
            author_sentiment = sentiment.get("basic") if sentiment else None

            return {
                "id": str(msg.get("id", "")),
                "text": msg.get("body", ""),
                "author": author_name,
                "author_data": author_data,
                "timestamp": timestamp,
                "source": "stocktwits",
                "metadata": {
                    "symbols": symbol_names,
                    "author_sentiment": author_sentiment,
                    "likes": msg.get("likes", {}).get("total", 0),
                    "reshares": msg.get("reshares", {}).get("total", 0),
                    "url": f"https://stocktwits.com/{author_name}/message/{msg.get('id', '')}",
                },
            }

        except Exception as e:
            logger.debug(f"Failed to normalize StockTwits message: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get detailed information about a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Symbol info dictionary or None if not found.
        """
        self._respect_rate_limit()

        url = f"{self.base_url}/symbols/{symbol.upper()}.json"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            symbol_data = data.get("symbol", {})

            return {
                "symbol": symbol_data.get("symbol"),
                "title": symbol_data.get("title"),
                "exchange": symbol_data.get("exchange"),
                "watchlist_count": symbol_data.get("watchlist_count", 0),
            }

        except Exception as e:
            logger.debug(f"Failed to get symbol info for {symbol}: {e}")
            return None
