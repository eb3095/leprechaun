"""Unit tests for market data provider."""

from datetime import datetime, time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.data.market import EASTERN_TZ, MarketDataProvider


class TestMarketDataProviderInit:
    """Test MarketDataProvider initialization."""

    def test_init_without_client(self):
        provider = MarketDataProvider()
        assert provider.client is None
        assert provider._alpaca_available is None
        assert provider._yf_available is None

    def test_init_with_client(self):
        mock_client = MagicMock()
        provider = MarketDataProvider(alpaca_client=mock_client)
        assert provider.client == mock_client


class TestGetQuote:
    """Test quote retrieval."""

    def test_get_quote_from_alpaca(self):
        provider = MarketDataProvider()
        provider._alpaca_available = True
        provider.client = MagicMock()

        mock_quote = MagicMock()
        mock_quote.bid_price = 150.0
        mock_quote.bid_size = 100
        mock_quote.ask_price = 150.05
        mock_quote.ask_size = 200
        mock_quote.timestamp = datetime.now(pytz.UTC)

        provider.client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        with patch.object(provider, "_get_quote_alpaca") as mock_method:
            mock_method.return_value = {
                "symbol": "AAPL",
                "bid": 150.0,
                "ask": 150.05,
                "source": "alpaca",
            }
            result = provider.get_quote("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["bid"] == 150.0
        assert result["ask"] == 150.05
        assert result["source"] == "alpaca"

    def test_get_quote_fallback_to_yfinance(self):
        provider = MarketDataProvider()

        with patch.object(provider, "_get_quote_alpaca", return_value=None):
            with patch.object(provider, "_get_quote_yfinance") as mock_yf:
                mock_yf.return_value = {
                    "symbol": "AAPL",
                    "bid": 150.0,
                    "ask": 150.05,
                    "source": "yfinance",
                }
                result = provider.get_quote("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["bid"] == 150.0
        assert result["source"] == "yfinance"

    def test_get_quote_empty_when_unavailable(self):
        provider = MarketDataProvider()
        provider._alpaca_available = False
        provider._yf_available = False

        result = provider.get_quote("AAPL")
        assert result == {}

    def test_get_quote_uppercase(self):
        provider = MarketDataProvider()

        with patch.object(provider, "_get_quote_alpaca", return_value=None):
            with patch.object(provider, "_get_quote_yfinance") as mock_yf:
                mock_yf.return_value = {
                    "symbol": "AAPL",
                    "bid": 150.0,
                    "source": "yfinance",
                }
                result = provider.get_quote("aapl")

        assert result["symbol"] == "AAPL"


class TestGetQuotes:
    """Test bulk quote retrieval."""

    def test_get_quotes_from_alpaca(self):
        provider = MarketDataProvider()
        provider._alpaca_available = False

        with patch.object(provider, "_check_alpaca", return_value=False):
            with patch.object(provider, "_get_quote_yfinance") as mock_yf:
                mock_yf.return_value = {"symbol": "AAPL", "source": "yfinance"}
                result = provider.get_quotes(["AAPL", "MSFT"])

        assert "AAPL" in result and "MSFT" in result

    def test_get_quotes_fallback_for_missing(self):
        provider = MarketDataProvider()
        provider._alpaca_available = False
        provider._yf_available = True

        with patch.object(provider, "_check_alpaca", return_value=False):
            with patch.object(provider, "_get_quote_yfinance") as mock_yf:
                mock_yf.return_value = {"symbol": "AAPL", "source": "yfinance"}
                result = provider.get_quotes(["AAPL"])

        assert result["AAPL"]["source"] == "yfinance"


class TestGetBars:
    """Test bar data retrieval."""

    def test_get_bars_from_alpaca(self):
        provider = MarketDataProvider()

        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [150.0],
            "high": [152.0],
            "low": [149.0],
            "close": [151.0],
            "volume": [1000000],
        })

        with patch.object(provider, "_get_bars_alpaca", return_value=mock_df):
            result = provider.get_bars("AAPL", "1Day", 10)

        assert not result.empty
        assert "close" in result.columns

    def test_get_bars_fallback_to_yfinance(self):
        provider = MarketDataProvider()

        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [150.0],
            "high": [152.0],
            "low": [149.0],
            "close": [151.0],
            "volume": [1000000],
        })

        with patch.object(provider, "_get_bars_alpaca", return_value=None):
            with patch.object(provider, "_get_bars_yfinance", return_value=mock_df):
                result = provider.get_bars("AAPL", "1Day", 10)

        assert not result.empty

    def test_get_bars_empty_when_unavailable(self):
        provider = MarketDataProvider()
        provider._alpaca_available = False
        provider._yf_available = False

        result = provider.get_bars("AAPL")
        assert result.empty
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(result.columns) == expected_cols


class TestMarketStatus:
    """Test market status checks."""

    def test_is_market_open_during_hours(self):
        provider = MarketDataProvider()

        market_time = datetime(2024, 1, 15, 10, 30, 0)
        market_time = EASTERN_TZ.localize(market_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = market_time
            result = provider.is_market_open()

        assert result is True

    def test_is_market_open_before_hours(self):
        provider = MarketDataProvider()

        market_time = datetime(2024, 1, 15, 8, 0, 0)
        market_time = EASTERN_TZ.localize(market_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = market_time
            result = provider.is_market_open()

        assert result is False

    def test_is_market_open_after_hours(self):
        provider = MarketDataProvider()

        market_time = datetime(2024, 1, 15, 17, 0, 0)
        market_time = EASTERN_TZ.localize(market_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = market_time
            result = provider.is_market_open()

        assert result is False

    def test_is_market_open_weekend(self):
        provider = MarketDataProvider()

        saturday = datetime(2024, 1, 13, 12, 0, 0)
        saturday = EASTERN_TZ.localize(saturday)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            result = provider.is_market_open()

        assert result is False

    def test_is_extended_hours_premarket(self):
        provider = MarketDataProvider()

        premarket_time = datetime(2024, 1, 15, 7, 0, 0)
        premarket_time = EASTERN_TZ.localize(premarket_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = premarket_time
            result = provider.is_extended_hours()

        assert result is True

    def test_is_extended_hours_afterhours(self):
        provider = MarketDataProvider()

        afterhours_time = datetime(2024, 1, 15, 18, 0, 0)
        afterhours_time = EASTERN_TZ.localize(afterhours_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = afterhours_time
            result = provider.is_extended_hours()

        assert result is True

    def test_is_extended_hours_regular(self):
        provider = MarketDataProvider()

        regular_time = datetime(2024, 1, 15, 12, 0, 0)
        regular_time = EASTERN_TZ.localize(regular_time)

        with patch("src.data.market.datetime") as mock_dt:
            mock_dt.now.return_value = regular_time
            result = provider.is_extended_hours()

        assert result is False


class TestGetMarketStatus:
    """Test detailed market status."""

    def test_get_market_status_open(self):
        provider = MarketDataProvider()

        with patch.object(provider, "is_market_open", return_value=True):
            with patch.object(provider, "is_extended_hours", return_value=False):
                with patch("src.data.market.datetime") as mock_dt:
                    mock_now = datetime(2024, 1, 15, 12, 0, 0)
                    mock_now = EASTERN_TZ.localize(mock_now)
                    mock_dt.now.return_value = mock_now

                    result = provider.get_market_status()

        assert result["status"] == "open"
        assert result["session"] == "regular"
        assert result["is_regular_hours"] is True

    def test_get_market_status_extended(self):
        provider = MarketDataProvider()

        with patch.object(provider, "is_market_open", return_value=False):
            with patch.object(provider, "is_extended_hours", return_value=True):
                with patch("src.data.market.datetime") as mock_dt:
                    mock_now = datetime(2024, 1, 15, 7, 0, 0)
                    mock_now = EASTERN_TZ.localize(mock_now)
                    mock_dt.now.return_value = mock_now

                    result = provider.get_market_status()

        assert result["status"] == "open"
        assert result["session"] == "pre_market"
        assert result["is_extended_hours"] is True

    def test_get_market_status_closed(self):
        provider = MarketDataProvider()

        with patch.object(provider, "is_market_open", return_value=False):
            with patch.object(provider, "is_extended_hours", return_value=False):
                with patch("src.data.market.datetime") as mock_dt:
                    mock_now = datetime(2024, 1, 15, 22, 0, 0)
                    mock_now = EASTERN_TZ.localize(mock_now)
                    mock_dt.now.return_value = mock_now
                    mock_dt.min.time.return_value = time(0, 0)

                    result = provider.get_market_status()

        assert result["status"] == "closed"
        assert result["session"] == "closed"


class TestGetSnapshot:
    """Test market snapshot retrieval."""

    def test_get_snapshot_with_data(self):
        provider = MarketDataProvider()

        mock_quote = {"symbol": "AAPL", "bid": 150.0, "ask": 150.05}
        mock_bars = pd.DataFrame({
            "timestamp": [datetime.now(), datetime.now()],
            "open": [149.0, 150.0],
            "high": [151.0, 152.0],
            "low": [148.0, 149.0],
            "close": [150.0, 151.0],
            "volume": [1000000, 1100000],
        })
        mock_status = {"status": "open", "session": "regular"}

        with patch.object(provider, "get_quote", return_value=mock_quote):
            with patch.object(provider, "get_bars", return_value=mock_bars):
                with patch.object(provider, "get_market_status", return_value=mock_status):
                    result = provider.get_snapshot("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["quote"] == mock_quote
        assert "latest_bar" in result
        assert result["latest_bar"]["close"] == 151.0
        assert "daily_change" in result

    def test_get_snapshot_without_bars(self):
        provider = MarketDataProvider()

        mock_quote = {"symbol": "AAPL", "bid": 150.0}
        mock_status = {"status": "closed"}

        with patch.object(provider, "get_quote", return_value=mock_quote):
            with patch.object(provider, "get_bars", return_value=pd.DataFrame()):
                with patch.object(provider, "get_market_status", return_value=mock_status):
                    result = provider.get_snapshot("AAPL")

        assert result["symbol"] == "AAPL"
        assert "latest_bar" not in result
        assert "daily_change" not in result


class TestAlpacaClientCheck:
    """Test Alpaca client availability checking."""

    def test_check_alpaca_with_existing_client(self):
        mock_client = MagicMock()
        provider = MarketDataProvider(alpaca_client=mock_client)

        result = provider._check_alpaca()
        assert result is True

    def test_check_alpaca_caches_true(self):
        provider = MarketDataProvider()
        provider._alpaca_available = True

        result = provider._check_alpaca()
        assert result is True

    def test_check_alpaca_caches_false(self):
        provider = MarketDataProvider()
        provider._alpaca_available = False

        result = provider._check_alpaca()
        assert result is False


class TestYfinanceCheck:
    """Test yfinance availability checking."""

    def test_check_yfinance_caches_true(self):
        provider = MarketDataProvider()
        provider._yf_available = True

        result = provider._check_yfinance()
        assert result is True

    def test_check_yfinance_caches_false(self):
        provider = MarketDataProvider()
        provider._yf_available = False

        result = provider._check_yfinance()
        assert result is False

    def test_check_yfinance_caches_result(self):
        provider = MarketDataProvider()
        provider._yf_available = True

        result = provider._check_yfinance()
        assert result is True
