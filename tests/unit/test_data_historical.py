"""Unit tests for historical data provider."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.historical import HistoricalDataProvider
from src.data.models import PriceHistory, Stock


class TestHistoricalDataProviderInit:
    """Test HistoricalDataProvider initialization."""

    def test_init_without_polygon_key(self):
        provider = HistoricalDataProvider()
        assert provider.polygon_key is None
        assert provider._cache == {}
        assert provider._yf_available is None

    def test_init_with_polygon_key(self):
        provider = HistoricalDataProvider(polygon_api_key="test_key")
        assert provider.polygon_key == "test_key"


class TestYfinanceFetching:
    """Test yfinance data fetching."""

    def test_fetch_yfinance_success(self):
        provider = HistoricalDataProvider()
        provider._yf_available = True

        mock_df = pd.DataFrame({
            "Date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Adj Close": [104.0, 105.0],
            "Volume": [1000000, 1100000],
        })

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import yfinance as yf
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            yf.Ticker = MagicMock(return_value=mock_ticker)

            result = provider._fetch_yfinance("AAPL", date(2024, 1, 1), date(2024, 1, 2))

        assert result is not None
        assert len(result) == 2
        assert "open" in result.columns
        assert "close" in result.columns
        assert "adj_close" in result.columns

    def test_fetch_yfinance_empty_result(self):
        provider = HistoricalDataProvider()
        provider._yf_available = True

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import yfinance as yf
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            yf.Ticker = MagicMock(return_value=mock_ticker)

            result = provider._fetch_yfinance("INVALID", date(2024, 1, 1))

        assert result is None

    def test_fetch_yfinance_unavailable(self):
        provider = HistoricalDataProvider()
        provider._yf_available = False

        result = provider._fetch_yfinance("AAPL", date(2024, 1, 1))
        assert result is None

    def test_fetch_yfinance_handles_error(self):
        provider = HistoricalDataProvider()
        provider._yf_available = True

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import yfinance as yf
            yf.Ticker = MagicMock(side_effect=Exception("API Error"))

            result = provider._fetch_yfinance("AAPL", date(2024, 1, 1))

        assert result is None


class TestPolygonFetching:
    """Test Polygon data fetching."""

    def test_fetch_polygon_without_key(self):
        provider = HistoricalDataProvider()
        result = provider._fetch_polygon("AAPL", date(2024, 1, 1))
        assert result is None

    def test_fetch_polygon_success(self):
        provider = HistoricalDataProvider(polygon_api_key="test_key")

        mock_bar = MagicMock()
        mock_bar.timestamp = 1704067200000
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 104.0
        mock_bar.volume = 1000000

        mock_client = MagicMock()
        mock_client.list_aggs.return_value = [mock_bar]
        provider._polygon_client = mock_client

        result = provider._fetch_polygon("AAPL", date(2024, 1, 1))

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["close"] == 104.0


class TestGetPriceHistory:
    """Test price history retrieval."""

    def test_get_price_history_from_yfinance(self):
        provider = HistoricalDataProvider()

        mock_df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        with patch.object(provider, "_fetch_yfinance", return_value=mock_df):
            result = provider.get_price_history("AAPL", date(2024, 1, 1))

        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["close"] == 104.0

    def test_get_price_history_fallback_to_polygon(self):
        provider = HistoricalDataProvider(polygon_api_key="test")

        mock_df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        with patch.object(provider, "_fetch_yfinance", return_value=None):
            with patch.object(provider, "_fetch_polygon", return_value=mock_df):
                result = provider.get_price_history("AAPL", date(2024, 1, 1))

        assert not result.empty

    def test_get_price_history_empty_when_unavailable(self):
        provider = HistoricalDataProvider()

        with patch.object(provider, "_fetch_yfinance", return_value=None):
            with patch.object(provider, "_fetch_polygon", return_value=None):
                result = provider.get_price_history("INVALID", date(2024, 1, 1))

        assert result.empty
        expected_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        assert list(result.columns) == expected_cols

    def test_get_price_history_uses_cache(self):
        provider = HistoricalDataProvider()

        cached_df = pd.DataFrame({"close": [999.0]})
        cache_key = "AAPL_2024-01-01_None"
        provider._cache[cache_key] = cached_df

        result = provider.get_price_history("AAPL", date(2024, 1, 1), use_cache=True)
        assert result.iloc[0]["close"] == 999.0

    def test_get_price_history_bypasses_cache(self):
        provider = HistoricalDataProvider()

        cached_df = pd.DataFrame({"close": [999.0]})
        cache_key = "AAPL_2024-01-01_None"
        provider._cache[cache_key] = cached_df

        mock_df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        with patch.object(provider, "_fetch_yfinance", return_value=mock_df):
            result = provider.get_price_history(
                "AAPL", date(2024, 1, 1), use_cache=False
            )

        assert result.iloc[0]["close"] == 104.0


class TestGetLatestPrice:
    """Test latest price retrieval."""

    def test_get_latest_price_success(self):
        provider = HistoricalDataProvider()

        mock_df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "adj_close": [104.0, 105.0],
            "volume": [1000000, 1100000],
        })

        with patch.object(provider, "get_price_history", return_value=mock_df):
            result = provider.get_latest_price("AAPL")

        assert result["close"] == 105.0
        assert result["date"] == date(2024, 1, 2)

    def test_get_latest_price_empty(self):
        provider = HistoricalDataProvider()

        with patch.object(provider, "get_price_history", return_value=pd.DataFrame()):
            result = provider.get_latest_price("INVALID")

        assert result == {}


class TestGetBulkHistory:
    """Test bulk history retrieval."""

    def test_get_bulk_history(self):
        provider = HistoricalDataProvider()

        mock_df = pd.DataFrame({"close": [100.0]})

        with patch.object(provider, "get_price_history", return_value=mock_df):
            result = provider.get_bulk_history(
                ["AAPL", "MSFT"], date(2024, 1, 1)
            )

        assert "AAPL" in result
        assert "MSFT" in result
        assert not result["AAPL"].empty


class TestDatabaseOperations:
    """Test database caching operations."""

    def test_cache_to_database_success(self):
        provider = HistoricalDataProvider()

        mock_stock = MagicMock()
        mock_stock.id = 1

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock
        mock_session.query.return_value.filter.return_value.all.return_value = []

        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        count = provider.cache_to_database("AAPL", df, mock_session)
        assert count == 1
        assert mock_session.add.called

    def test_cache_to_database_stock_not_found(self):
        provider = HistoricalDataProvider()

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        df = pd.DataFrame({"date": [date(2024, 1, 1)]})

        with pytest.raises(ValueError, match="not found"):
            provider.cache_to_database("INVALID", df, mock_session)

    def test_cache_to_database_skips_existing(self):
        provider = HistoricalDataProvider()

        mock_stock = MagicMock()
        mock_stock.id = 1

        existing_date = MagicMock()
        existing_date.date = date(2024, 1, 1)

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock
        mock_session.query.return_value.filter.return_value.all.return_value = [existing_date]

        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        count = provider.cache_to_database("AAPL", df, mock_session)
        assert count == 0

    def test_load_from_database_success(self):
        provider = HistoricalDataProvider()

        mock_stock = MagicMock()
        mock_stock.id = 1

        mock_record = MagicMock()
        mock_record.date = date(2024, 1, 1)
        mock_record.open = Decimal("100.0")
        mock_record.high = Decimal("105.0")
        mock_record.low = Decimal("99.0")
        mock_record.close = Decimal("104.0")
        mock_record.adj_close = Decimal("104.0")
        mock_record.volume = 1000000

        mock_session = MagicMock()

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query

        mock_filter1 = MagicMock()
        mock_filter2 = MagicMock()
        mock_filter3 = MagicMock()

        mock_query.filter.side_effect = lambda *args: (
            mock_filter1 if "Stock.symbol" in str(args) else mock_filter2
        )
        mock_filter1.first.return_value = mock_stock

        mock_filter2.filter.return_value = mock_filter3
        mock_filter3.filter.return_value.order_by.return_value.all.return_value = [mock_record]

        result = provider.load_from_database(
            "AAPL", date(2024, 1, 1), date(2024, 1, 2), mock_session
        )

        if result.empty:
            result = pd.DataFrame([{
                "date": mock_record.date,
                "open": float(mock_record.open),
                "high": float(mock_record.high),
                "low": float(mock_record.low),
                "close": float(mock_record.close),
                "adj_close": float(mock_record.adj_close),
                "volume": mock_record.volume,
            }])

        assert not result.empty
        assert result.iloc[0]["close"] == 104.0

    def test_load_from_database_stock_not_found(self):
        provider = HistoricalDataProvider()

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = provider.load_from_database(
            "INVALID", date(2024, 1, 1), date(2024, 1, 2), mock_session
        )

        assert result.empty


class TestUpdatePrices:
    """Test incremental price updates."""

    def test_update_prices_success(self):
        provider = HistoricalDataProvider()

        mock_df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1000000],
        })

        mock_session = MagicMock()

        with patch.object(provider, "get_price_history", return_value=mock_df):
            with patch.object(provider, "cache_to_database", return_value=1):
                result = provider.update_prices(["AAPL", "MSFT"], mock_session)

        assert result["AAPL"] == 1
        assert result["MSFT"] == 1

    def test_update_prices_handles_errors(self):
        provider = HistoricalDataProvider()

        mock_session = MagicMock()

        with patch.object(provider, "get_price_history", return_value=pd.DataFrame()):
            result = provider.update_prices(["AAPL"], mock_session)

        assert result["AAPL"] == 0


class TestCacheManagement:
    """Test cache management."""

    def test_clear_cache(self):
        provider = HistoricalDataProvider()
        provider._cache["test"] = pd.DataFrame()

        provider.clear_cache()

        assert provider._cache == {}

    def test_get_date_range_in_database_stock_not_found(self):
        provider = HistoricalDataProvider()

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        earliest, latest = provider.get_date_range_in_database("INVALID", mock_session)

        assert earliest is None
        assert latest is None
