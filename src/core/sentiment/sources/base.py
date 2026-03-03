"""
Base class for sentiment data sources.

Provides common functionality for rate limiting, normalization, and error handling
that all sentiment sources should implement.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SentimentSource(ABC):
    """
    Abstract base class for sentiment data sources.

    All sentiment sources (Reddit, StockTwits, News, etc.) should inherit from
    this class and implement the abstract methods.

    Provides:
    - Rate limiting to respect API limits
    - Common post normalization format
    - Error handling patterns
    """

    def __init__(self, name: str, rate_limit: int = 60):
        """
        Initialize sentiment source.

        Args:
            name: Source identifier (e.g., 'reddit', 'stocktwits').
            rate_limit: Maximum requests per minute.
        """
        self.name = name
        self.rate_limit = rate_limit
        self.last_request: Optional[float] = None
        self._request_count = 0
        self._window_start: Optional[float] = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if source is properly initialized and ready to use."""
        return self._initialized

    @abstractmethod
    def fetch_mentions(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch mentions for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL').
            since: Only fetch mentions after this timestamp.

        Returns:
            List of normalized post dictionaries.
        """
        pass

    @abstractmethod
    def fetch_trending(self, limit: int = 10) -> list[dict]:
        """
        Fetch trending tickers/topics.

        Args:
            limit: Maximum number of trending items to return.

        Returns:
            List of trending item dictionaries with at minimum:
                - symbol: Ticker symbol
                - mentions: Number of recent mentions
                - sentiment: Aggregate sentiment if available
        """
        pass

    def _respect_rate_limit(self) -> None:
        """
        Sleep if needed to respect rate limit.

        Implements a sliding window rate limiter that tracks requests
        per minute and sleeps when limit is reached.
        """
        current_time = time.time()

        if self._window_start is None:
            self._window_start = current_time
            self._request_count = 0

        if current_time - self._window_start >= 60:
            self._window_start = current_time
            self._request_count = 0

        if self._request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self._window_start)
            if sleep_time > 0:
                logger.debug(
                    f"{self.name}: Rate limit reached, sleeping {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                self._window_start = time.time()
                self._request_count = 0

        self._request_count += 1
        self.last_request = time.time()

    def _normalize_post(self, raw: dict) -> dict:
        """
        Normalize raw post data to common format.

        Override in subclasses to handle source-specific fields.

        Args:
            raw: Raw post data from the source API.

        Returns:
            Normalized post dictionary with keys:
                - id: Unique identifier for the post
                - text: Post content/body
                - author: Author username or identifier
                - timestamp: Post creation time (datetime)
                - source: Source name (e.g., 'reddit')
                - metadata: Additional source-specific data
        """
        return {
            "id": str(raw.get("id", "")),
            "text": raw.get("text", raw.get("body", raw.get("content", ""))),
            "author": raw.get(
                "author", raw.get("user", raw.get("username", "unknown"))
            ),
            "timestamp": self._parse_timestamp(
                raw.get("timestamp", raw.get("created_at"))
            ),
            "source": self.name,
            "metadata": raw.get("metadata", {}),
        }

    def _parse_timestamp(self, ts: Any) -> datetime:
        """
        Parse timestamp from various formats to datetime.

        Args:
            ts: Timestamp in various formats (str, int, float, datetime).

        Returns:
            Datetime object in UTC.
        """
        if ts is None:
            return datetime.now(timezone.utc)

        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)

        if isinstance(ts, str):
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    dt = datetime.strptime(ts, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

            try:
                from dateutil import parser

                dt = parser.parse(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ImportError, ValueError):
                pass

        logger.warning(f"{self.name}: Could not parse timestamp: {ts}")
        return datetime.now(timezone.utc)

    def _handle_api_error(self, error: Exception, context: str = "") -> list[dict]:
        """
        Handle API errors gracefully.

        Args:
            error: The exception that occurred.
            context: Additional context about what operation failed.

        Returns:
            Empty list (allows graceful degradation).
        """
        logger.warning(
            f"{self.name}: API error{' (' + context + ')' if context else ''}: {error}"
        )
        return []
