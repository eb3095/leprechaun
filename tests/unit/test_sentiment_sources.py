"""Unit tests for sentiment data sources."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.sentiment.sources.base import SentimentSource
from src.core.sentiment.sources.news import NewsSource
from src.core.sentiment.sources.reddit import RedditSource
from src.core.sentiment.sources.stocktwits import StockTwitsSource


class ConcreteSentimentSource(SentimentSource):
    """Concrete implementation for testing abstract base class."""

    def fetch_mentions(self, symbol, since=None):
        return []

    def fetch_trending(self, limit=10):
        return []


class TestSentimentSourceBase:
    """Tests for SentimentSource base class."""

    @pytest.fixture
    def source(self):
        """Create concrete source instance."""
        return ConcreteSentimentSource("test", rate_limit=60)

    def test_initialization(self, source):
        """Test base class initialization."""
        assert source.name == "test"
        assert source.rate_limit == 60
        assert source.last_request is None

    def test_is_available_default(self, source):
        """Test is_available returns False by default."""
        assert source.is_available is False

    def test_parse_timestamp_datetime(self, source):
        """Test parsing datetime objects."""
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = source._parse_timestamp(dt)

        assert result == dt

    def test_parse_timestamp_unix(self, source):
        """Test parsing Unix timestamps."""
        ts = 1705320000
        result = source._parse_timestamp(ts)

        assert isinstance(result, datetime)

    def test_parse_timestamp_iso_string(self, source):
        """Test parsing ISO format strings."""
        ts = "2024-01-15T12:00:00Z"
        result = source._parse_timestamp(ts)

        assert isinstance(result, datetime)

    def test_parse_timestamp_none(self, source):
        """Test parsing None returns current time."""
        result = source._parse_timestamp(None)

        assert isinstance(result, datetime)

    def test_normalize_post(self, source):
        """Test post normalization."""
        raw = {
            "id": "123",
            "text": "Test post content",
            "author": "testuser",
            "timestamp": "2024-01-15T12:00:00Z",
        }

        result = source._normalize_post(raw)

        assert result["id"] == "123"
        assert result["text"] == "Test post content"
        assert result["author"] == "testuser"
        assert result["source"] == "test"
        assert isinstance(result["timestamp"], datetime)

    def test_handle_api_error_returns_empty_list(self, source):
        """Test API error handler returns empty list."""
        result = source._handle_api_error(Exception("Test error"), "test context")

        assert result == []


class TestRedditSource:
    """Tests for Reddit sentiment source."""

    @pytest.fixture
    def source_no_credentials(self):
        """Create Reddit source without credentials."""
        return RedditSource()

    @pytest.fixture
    def source_with_mock_client(self):
        """Create Reddit source with mocked PRAW client."""
        with patch("src.core.sentiment.sources.reddit.praw") as mock_praw:
            mock_reddit = MagicMock()
            mock_praw.Reddit.return_value = mock_reddit
            mock_reddit.user.me.return_value = MagicMock()

            source = RedditSource(
                client_id="test_id",
                client_secret="test_secret",
                user_agent="test_agent",
            )
            source.reddit = mock_reddit
            source._initialized = True
            return source

    def test_initialization_without_credentials(self, source_no_credentials):
        """Test initialization without credentials."""
        assert source_no_credentials.reddit is None
        assert source_no_credentials._initialized is False
        assert source_no_credentials.name == "reddit"

    def test_default_subreddits(self, source_no_credentials):
        """Test default subreddits are set."""
        assert "wallstreetbets" in source_no_credentials.subreddits
        assert "stocks" in source_no_credentials.subreddits

    def test_fetch_mentions_without_credentials(self, source_no_credentials):
        """Test fetch_mentions returns empty without credentials."""
        result = source_no_credentials.fetch_mentions("AAPL")

        assert result == []

    def test_fetch_trending_without_credentials(self, source_no_credentials):
        """Test fetch_trending returns empty without credentials."""
        result = source_no_credentials.fetch_trending()

        assert result == []

    def test_fetch_subreddit_posts_without_credentials(self, source_no_credentials):
        """Test fetch_subreddit_posts returns empty without credentials."""
        result = source_no_credentials.fetch_subreddit_posts("wallstreetbets")

        assert result == []

    def test_extract_tickers_with_dollar_sign(self, source_no_credentials):
        """Test ticker extraction with $ prefix."""
        tickers = source_no_credentials._extract_tickers("Buying $AAPL and $TSLA")

        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_extract_tickers_filters_common_words(self, source_no_credentials):
        """Test common words are filtered."""
        tickers = source_no_credentials._extract_tickers("THE market AND stocks")

        assert "THE" not in tickers
        assert "AND" not in tickers

    def test_is_relevant_with_ticker_in_text(self, source_no_credentials):
        """Test relevance check with ticker in text."""
        post = {
            "text": "Looking at $AAPL today",
            "metadata": {"tickers_found": ["AAPL"]},
        }

        assert source_no_credentials._is_relevant(post, "AAPL") is True

    def test_is_relevant_without_ticker(self, source_no_credentials):
        """Test relevance check without ticker."""
        post = {
            "text": "General market discussion",
            "metadata": {"tickers_found": []},
        }

        assert source_no_credentials._is_relevant(post, "AAPL") is False


class TestStockTwitsSource:
    """Tests for StockTwits sentiment source."""

    @pytest.fixture
    def source(self):
        """Create StockTwits source."""
        return StockTwitsSource()

    @pytest.fixture
    def source_with_key(self):
        """Create StockTwits source with API key."""
        return StockTwitsSource(api_key="test_key")

    def test_initialization(self, source):
        """Test initialization."""
        assert source.name == "stocktwits"
        assert source.rate_limit == 200
        assert source._initialized is True

    def test_initialization_with_rapidapi(self):
        """Test initialization with RapidAPI."""
        source = StockTwitsSource(api_key="test_key", use_rapidapi=True)

        assert "rapidapi" in source.base_url.lower()

    @patch("src.core.sentiment.sources.stocktwits.requests.get")
    def test_fetch_mentions_success(self, mock_get, source):
        """Test successful fetch_mentions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {"status": 200},
            "messages": [
                {
                    "id": 123,
                    "body": "Test message",
                    "created_at": "2024-01-15T12:00:00Z",
                    "user": {"username": "testuser"},
                },
            ],
        }
        mock_get.return_value = mock_response

        result = source.fetch_mentions("AAPL")

        assert len(result) == 1
        assert result[0]["text"] == "Test message"

    @patch("src.core.sentiment.sources.stocktwits.requests.get")
    def test_fetch_mentions_404(self, mock_get, source):
        """Test fetch_mentions with 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = source.fetch_mentions("INVALID")

        assert result == []

    @patch("src.core.sentiment.sources.stocktwits.requests.get")
    def test_fetch_mentions_rate_limit(self, mock_get, source):
        """Test fetch_mentions with rate limit response."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = source.fetch_mentions("AAPL")

        assert result == []

    @patch("src.core.sentiment.sources.stocktwits.requests.get")
    def test_fetch_trending_success(self, mock_get, source):
        """Test successful fetch_trending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {"status": 200},
            "symbols": [
                {"symbol": "AAPL", "title": "Apple Inc.", "watchlist_count": 1000},
                {"symbol": "TSLA", "title": "Tesla Inc.", "watchlist_count": 900},
            ],
        }
        mock_get.return_value = mock_response

        result = source.fetch_trending(limit=2)

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"

    def test_normalize_message(self, source):
        """Test message normalization."""
        msg = {
            "id": 123,
            "body": "Test message body",
            "created_at": "2024-01-15T12:00:00Z",
            "user": {
                "username": "testuser",
                "followers": 100,
                "following": 50,
            },
            "symbols": [{"symbol": "AAPL"}],
        }

        result = source._normalize_message(msg)

        assert result["id"] == "123"
        assert result["text"] == "Test message body"
        assert result["author"] == "testuser"
        assert result["source"] == "stocktwits"
        assert "AAPL" in result["metadata"]["symbols"]


class TestNewsSource:
    """Tests for Finnhub news sentiment source."""

    @pytest.fixture
    def source_no_key(self):
        """Create news source without API key."""
        return NewsSource()

    @pytest.fixture
    def source_with_key(self):
        """Create news source with API key."""
        return NewsSource(api_key="test_key")

    def test_initialization_without_key(self, source_no_key):
        """Test initialization without API key."""
        assert source_no_key._initialized is False
        assert source_no_key.name == "news"

    def test_initialization_with_key(self, source_with_key):
        """Test initialization with API key."""
        assert source_with_key._initialized is True
        assert source_with_key.api_key == "test_key"

    def test_fetch_mentions_without_key(self, source_no_key):
        """Test fetch_mentions returns empty without API key."""
        result = source_no_key.fetch_mentions("AAPL")

        assert result == []

    def test_fetch_trending_without_key(self, source_no_key):
        """Test fetch_trending returns empty without API key."""
        result = source_no_key.fetch_trending()

        assert result == []

    @patch("src.core.sentiment.sources.news.requests.get")
    def test_fetch_mentions_success(self, mock_get, source_with_key):
        """Test successful fetch_mentions."""
        import time

        current_ts = int(time.time())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 123,
                "headline": "Apple announces new product",
                "summary": "Apple Inc. today announced...",
                "source": "Reuters",
                "url": "https://example.com/news/1",
                "datetime": current_ts,
            },
        ]
        mock_get.return_value = mock_response

        result = source_with_key.fetch_mentions("AAPL")

        assert len(result) == 1
        assert "Apple announces" in result[0]["text"]

    @patch("src.core.sentiment.sources.news.requests.get")
    def test_fetch_mentions_auth_error(self, mock_get, source_with_key):
        """Test fetch_mentions with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = source_with_key.fetch_mentions("AAPL")

        assert result == []

    @patch("src.core.sentiment.sources.news.requests.get")
    def test_fetch_market_news_success(self, mock_get, source_with_key):
        """Test successful fetch_market_news."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "headline": "Market update",
                "source": "MarketWatch",
                "datetime": 1705320000,
            },
        ]
        mock_get.return_value = mock_response

        result = source_with_key.fetch_market_news(category="general")

        assert len(result) == 1

    @patch("src.core.sentiment.sources.news.requests.get")
    def test_has_recent_news_true(self, mock_get, source_with_key):
        """Test has_recent_news returns True when news exists."""
        import time

        current_ts = int(time.time())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"headline": "Recent news", "datetime": current_ts},
        ]
        mock_get.return_value = mock_response

        result = source_with_key.has_recent_news("AAPL", hours=24)

        assert result is True

    @patch("src.core.sentiment.sources.news.requests.get")
    def test_has_recent_news_false(self, mock_get, source_with_key):
        """Test has_recent_news returns False when no news."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = source_with_key.has_recent_news("AAPL", hours=24)

        assert result is False

    def test_normalize_article(self, source_with_key):
        """Test article normalization."""
        article = {
            "id": 123,
            "headline": "Test Headline",
            "summary": "Test summary content",
            "source": "Reuters",
            "url": "https://example.com",
            "datetime": 1705320000,
        }

        result = source_with_key._normalize_article(article, "AAPL")

        assert "Test Headline" in result["text"]
        assert result["author"] == "Reuters"
        assert result["source"] == "news"
        assert result["metadata"]["symbol"] == "AAPL"
