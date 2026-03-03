"""
Reddit sentiment source using PRAW.

Fetches posts and comments from finance-related subreddits including:
- r/wallstreetbets
- r/stocks
- r/investing
- r/stockmarket
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.core.sentiment.sources.base import SentimentSource

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "stockmarket",
    "options",
    "thetagang",
    "pennystocks",
]

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
TICKER_PATTERN_NO_DOLLAR = re.compile(r"\b([A-Z]{2,5})\b")


class RedditSource(SentimentSource):
    """
    Reddit data source using PRAW (Python Reddit API Wrapper).

    Fetches posts and comments from financial subreddits for sentiment analysis.
    Gracefully handles missing credentials by returning empty results.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        subreddits: Optional[list[str]] = None,
    ):
        """
        Initialize Reddit source.

        Args:
            client_id: Reddit OAuth client ID.
            client_secret: Reddit OAuth client secret.
            user_agent: User agent string for API requests.
            subreddits: List of subreddit names to monitor.
        """
        super().__init__("reddit", rate_limit=10)
        self.reddit = None
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self._init_client(client_id, client_secret, user_agent)

    def _init_client(
        self,
        client_id: Optional[str],
        client_secret: Optional[str],
        user_agent: Optional[str],
    ) -> None:
        """Initialize PRAW client if credentials provided."""
        if not client_id or not client_secret:
            logger.info(
                "Reddit credentials not provided, source will return empty results"
            )
            return

        try:
            import praw

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent or "leprechaun:v1.0 (sentiment analysis bot)",
            )

            self.reddit.user.me()
            self._initialized = True
            logger.info("Reddit client initialized successfully")

        except praw.exceptions.PRAWException as e:
            logger.warning(f"Reddit authentication failed: {e}")
        except ImportError:
            logger.warning("PRAW not installed, Reddit source unavailable")
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit client: {e}")

    def fetch_mentions(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch Reddit posts mentioning a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL').
            since: Only fetch posts after this timestamp.
            limit: Maximum number of posts to fetch per subreddit.

        Returns:
            List of normalized post dictionaries.
        """
        if not self._initialized or self.reddit is None:
            return []

        self._respect_rate_limit()

        posts = []
        symbol_upper = symbol.upper()
        search_terms = [f"${symbol_upper}", symbol_upper]

        try:
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for search_term in search_terms:
                        for submission in subreddit.search(
                            search_term,
                            sort="new",
                            time_filter="week",
                            limit=limit // len(self.subreddits),
                        ):
                            post = self._normalize_submission(submission, symbol_upper)
                            if post and self._is_relevant(post, symbol_upper):
                                if since is None or post["timestamp"] > since:
                                    posts.append(post)

                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue

        except Exception as e:
            return self._handle_api_error(e, f"fetch_mentions({symbol})")

        seen_ids = set()
        unique_posts = []
        for post in posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                unique_posts.append(post)

        return unique_posts

    def fetch_trending(self, limit: int = 10) -> list[dict]:
        """
        Get trending tickers from WSB and other subreddits.

        Analyzes hot posts to find most mentioned symbols.

        Args:
            limit: Maximum number of trending items to return.

        Returns:
            List of trending ticker dictionaries.
        """
        if not self._initialized or self.reddit is None:
            return []

        self._respect_rate_limit()

        ticker_mentions: dict[str, int] = {}
        ticker_sentiment: dict[str, list[float]] = {}

        try:
            for subreddit_name in self.subreddits[:3]:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for submission in subreddit.hot(limit=50):
                        text = f"{submission.title} {submission.selftext or ''}"
                        tickers = self._extract_tickers(text)

                        for ticker in tickers:
                            ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1

                except Exception as e:
                    logger.warning(
                        f"Error fetching trending from r/{subreddit_name}: {e}"
                    )
                    continue

        except Exception as e:
            return self._handle_api_error(e, "fetch_trending")

        trending = [
            {"symbol": symbol, "mentions": count, "source": "reddit"}
            for symbol, count in sorted(
                ticker_mentions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:limit]
        ]

        return trending

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 100,
        sort: str = "new",
    ) -> list[dict]:
        """
        Fetch recent posts from a specific subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            limit: Maximum number of posts to fetch.
            sort: Sort method ('new', 'hot', 'top', 'rising').

        Returns:
            List of normalized post dictionaries.
        """
        if not self._initialized or self.reddit is None:
            return []

        self._respect_rate_limit()

        posts = []

        try:
            sub = self.reddit.subreddit(subreddit)

            if sort == "new":
                submissions = sub.new(limit=limit)
            elif sort == "hot":
                submissions = sub.hot(limit=limit)
            elif sort == "top":
                submissions = sub.top(limit=limit, time_filter="day")
            elif sort == "rising":
                submissions = sub.rising(limit=limit)
            else:
                submissions = sub.new(limit=limit)

            for submission in submissions:
                post = self._normalize_submission(submission)
                if post:
                    posts.append(post)

        except Exception as e:
            return self._handle_api_error(e, f"fetch_subreddit_posts({subreddit})")

        return posts

    def _normalize_submission(
        self,
        submission: Any,
        target_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Normalize Reddit submission to standard format.

        Args:
            submission: PRAW Submission object.
            target_symbol: Optional target symbol for relevance filtering.

        Returns:
            Normalized post dictionary or None if invalid.
        """
        try:
            created_utc = getattr(submission, "created_utc", None)
            if created_utc:
                timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            author = submission.author
            author_name = str(author) if author else "[deleted]"

            author_data = {}
            if author and hasattr(author, "created_utc"):
                try:
                    account_created = datetime.fromtimestamp(
                        author.created_utc, tz=timezone.utc
                    )
                    account_age = (datetime.now(timezone.utc) - account_created).days
                    author_data = {
                        "account_age_days": account_age,
                        "karma": getattr(author, "link_karma", 0)
                        + getattr(author, "comment_karma", 0),
                        "username": author_name,
                    }
                except Exception:
                    pass

            text = f"{submission.title}\n{submission.selftext or ''}"
            tickers = self._extract_tickers(text)

            return {
                "id": submission.id,
                "text": text,
                "author": author_name,
                "author_data": author_data,
                "timestamp": timestamp,
                "source": "reddit",
                "metadata": {
                    "subreddit": str(submission.subreddit),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": getattr(submission, "upvote_ratio", None),
                    "url": f"https://reddit.com{submission.permalink}",
                    "tickers_found": tickers,
                },
            }

        except Exception as e:
            logger.debug(f"Failed to normalize submission: {e}")
            return None

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock tickers from text."""
        if not text:
            return []

        tickers = set()

        dollar_matches = TICKER_PATTERN.findall(text)
        tickers.update(dollar_matches)

        common_words = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "HAD",
            "HER",
            "WAS",
            "ONE",
            "OUR",
            "OUT",
            "HAS",
            "CEO",
            "CFO",
            "IPO",
            "ETF",
            "NYSE",
            "USA",
            "USD",
            "GDP",
            "IMO",
            "LOL",
            "OMG",
            "WTF",
            "BTW",
            "FYI",
            "TBH",
            "SMH",
            "LMAO",
            "EOD",
            "EOM",
            "WSB",
            "DD",
            "YOLO",
            "FOMO",
            "FUD",
            "ATH",
            "ATL",
            "PT",
            "SP",
            "IV",
            "ITM",
            "OTM",
            "DTE",
            "EDIT",
            "POST",
            "THIS",
            "THAT",
            "WITH",
            "FROM",
            "WHAT",
            "JUST",
            "LIKE",
            "MORE",
            "VERY",
            "WHEN",
            "WILL",
            "BEEN",
        }

        plain_matches = TICKER_PATTERN_NO_DOLLAR.findall(text)
        for match in plain_matches:
            if match not in common_words and len(match) >= 2:
                tickers.add(match)

        return sorted(tickers)

    def _is_relevant(self, post: dict, symbol: str) -> bool:
        """Check if post is relevant to the given symbol."""
        text = post.get("text", "").upper()
        tickers = post.get("metadata", {}).get("tickers_found", [])

        if symbol in tickers:
            return True

        if f"${symbol}" in text or f" {symbol} " in f" {text} ":
            return True

        return False
