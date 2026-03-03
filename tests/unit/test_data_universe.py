"""Unit tests for stock universe management."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.models import IndexMembership, Stock
from src.data.universe import (
    NASDAQ100_SYMBOLS,
    SP500_SYMBOLS,
    StockUniverse,
)


class TestStockUniverseSymbolLists:
    """Test fallback symbol lists."""

    def test_sp500_symbols_not_empty(self):
        assert len(SP500_SYMBOLS) >= 100

    def test_nasdaq100_symbols_not_empty(self):
        assert len(NASDAQ100_SYMBOLS) >= 100

    def test_common_symbols_present(self):
        common = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]
        for symbol in common:
            assert symbol in SP500_SYMBOLS
            assert symbol in NASDAQ100_SYMBOLS


class TestStockUniverseInit:
    """Test StockUniverse initialization."""

    def test_init_without_session(self):
        universe = StockUniverse()
        assert universe.session is None
        assert universe._sp500_cache is None
        assert universe._nasdaq100_cache is None

    def test_init_with_session(self):
        mock_session = MagicMock()
        universe = StockUniverse(db_session=mock_session)
        assert universe.session == mock_session


class TestGetSymbols:
    """Test symbol retrieval methods."""

    def test_get_sp500_symbols_fallback(self):
        universe = StockUniverse()
        with patch.object(universe, "_fetch_sp500_from_wikipedia", return_value=[]):
            symbols = universe.get_sp500_symbols()
            assert len(symbols) >= 100
            assert "AAPL" in symbols

    def test_get_nasdaq100_symbols_fallback(self):
        universe = StockUniverse()
        with patch.object(universe, "_fetch_nasdaq100_from_wikipedia", return_value=[]):
            symbols = universe.get_nasdaq100_symbols()
            assert len(symbols) >= 100
            assert "MSFT" in symbols

    def test_get_sp500_symbols_from_wikipedia(self):
        universe = StockUniverse()
        mock_symbols = ["AAPL", "MSFT", "GOOGL"]
        with patch.object(
            universe, "_fetch_sp500_from_wikipedia", return_value=mock_symbols
        ):
            symbols = universe.get_sp500_symbols()
            assert symbols == mock_symbols

    def test_get_nasdaq100_symbols_from_wikipedia(self):
        universe = StockUniverse()
        mock_symbols = ["AAPL", "NVDA", "META"]
        with patch.object(
            universe, "_fetch_nasdaq100_from_wikipedia", return_value=mock_symbols
        ):
            symbols = universe.get_nasdaq100_symbols()
            assert symbols == mock_symbols

    def test_get_all_symbols_combines_indexes(self):
        universe = StockUniverse()
        with patch.object(universe, "get_sp500_symbols", return_value=["AAPL", "XOM"]):
            with patch.object(
                universe, "get_nasdaq100_symbols", return_value=["AAPL", "NVDA"]
            ):
                all_symbols = universe.get_all_symbols()
                assert sorted(all_symbols) == ["AAPL", "NVDA", "XOM"]

    def test_get_symbols_uses_cache(self):
        universe = StockUniverse()
        universe._sp500_cache = ["CACHED"]
        universe._cache_timestamp = datetime.now()

        symbols = universe.get_sp500_symbols(use_cache=True)
        assert symbols == ["CACHED"]

    def test_get_symbols_bypasses_cache(self):
        universe = StockUniverse()
        universe._sp500_cache = ["CACHED"]
        universe._cache_timestamp = datetime.now()

        with patch.object(universe, "_fetch_sp500_from_wikipedia", return_value=[]):
            symbols = universe.get_sp500_symbols(use_cache=False)
            assert symbols != ["CACHED"]

    def test_cache_expiry(self):
        universe = StockUniverse()
        universe._sp500_cache = ["CACHED"]
        universe._cache_timestamp = datetime.now() - timedelta(hours=25)

        with patch.object(universe, "_fetch_sp500_from_wikipedia", return_value=[]):
            symbols = universe.get_sp500_symbols(use_cache=True)
            assert symbols != ["CACHED"]


class TestIsInUniverse:
    """Test universe membership checking."""

    def test_is_in_universe_true(self):
        universe = StockUniverse()
        with patch.object(universe, "get_all_symbols", return_value=["AAPL", "MSFT"]):
            assert universe.is_in_universe("AAPL") is True

    def test_is_in_universe_false(self):
        universe = StockUniverse()
        with patch.object(universe, "get_all_symbols", return_value=["AAPL", "MSFT"]):
            assert universe.is_in_universe("INVALID") is False

    def test_is_in_universe_case_insensitive(self):
        universe = StockUniverse()
        with patch.object(universe, "get_all_symbols", return_value=["AAPL", "MSFT"]):
            assert universe.is_in_universe("aapl") is True

    def test_is_in_universe_handles_dots(self):
        universe = StockUniverse()
        with patch.object(universe, "get_all_symbols", return_value=["BRK-B"]):
            assert universe.is_in_universe("BRK.B") is True


class TestGetIndexMembership:
    """Test index membership determination."""

    def test_sp500_only(self):
        universe = StockUniverse()
        with patch.object(universe, "get_sp500_symbols", return_value=["XOM"]):
            with patch.object(universe, "get_nasdaq100_symbols", return_value=["NVDA"]):
                membership = universe.get_index_membership("XOM")
                assert membership == IndexMembership.SP500

    def test_nasdaq100_only(self):
        universe = StockUniverse()
        with patch.object(universe, "get_sp500_symbols", return_value=["XOM"]):
            with patch.object(universe, "get_nasdaq100_symbols", return_value=["NVDA"]):
                membership = universe.get_index_membership("NVDA")
                assert membership == IndexMembership.NASDAQ100

    def test_both_indexes(self):
        universe = StockUniverse()
        with patch.object(universe, "get_sp500_symbols", return_value=["AAPL"]):
            with patch.object(universe, "get_nasdaq100_symbols", return_value=["AAPL"]):
                membership = universe.get_index_membership("AAPL")
                assert membership == IndexMembership.BOTH

    def test_not_in_universe(self):
        universe = StockUniverse()
        with patch.object(universe, "get_sp500_symbols", return_value=["AAPL"]):
            with patch.object(universe, "get_nasdaq100_symbols", return_value=["NVDA"]):
                membership = universe.get_index_membership("INVALID")
                assert membership is None


class TestRefreshUniverse:
    """Test universe refresh functionality."""

    def test_refresh_clears_cache(self):
        universe = StockUniverse()
        universe._sp500_cache = ["CACHED"]
        universe._nasdaq100_cache = ["CACHED"]
        universe._cache_timestamp = datetime.now()
        universe._stock_info_cache = {"AAPL": {}}

        with patch.object(universe, "get_sp500_symbols", return_value=["A", "B"]):
            with patch.object(
                universe, "get_nasdaq100_symbols", return_value=["B", "C"]
            ):
                result = universe.refresh_universe()

        assert result["sp500"] == 2
        assert result["nasdaq100"] == 2
        assert result["total"] == 3
        assert universe._stock_info_cache == {}


class TestGetStockInfo:
    """Test stock information retrieval."""

    def test_get_stock_info_fallback(self):
        universe = StockUniverse()
        with patch.object(universe, "get_index_membership", return_value=IndexMembership.SP500):
            with patch.dict("sys.modules", {"yfinance": MagicMock()}):
                import yfinance as yf
                mock_ticker = MagicMock()
                mock_ticker.info = {}
                yf.Ticker = MagicMock(return_value=mock_ticker)

                info = universe.get_stock_info("AAPL")
                assert info["symbol"] == "AAPL"
                assert info["index_membership"] == IndexMembership.SP500

    def test_get_stock_info_with_yfinance(self):
        universe = StockUniverse()
        with patch.object(universe, "get_index_membership", return_value=IndexMembership.BOTH):
            with patch.dict("sys.modules", {"yfinance": MagicMock()}):
                import yfinance as yf
                mock_ticker = MagicMock()
                mock_ticker.info = {
                    "longName": "Apple Inc.",
                    "sector": "Technology",
                    "marketCap": 3000000000000,
                }
                yf.Ticker = MagicMock(return_value=mock_ticker)

                info = universe.get_stock_info("AAPL")
                assert info["symbol"] == "AAPL"

    def test_get_stock_info_uses_cache(self):
        universe = StockUniverse()
        cached_info = {"symbol": "AAPL", "name": "Cached"}
        universe._stock_info_cache["AAPL"] = cached_info

        info = universe.get_stock_info("AAPL", use_cache=True)
        assert info["name"] == "Cached"

    def test_get_stock_info_handles_errors(self):
        universe = StockUniverse()
        with patch.object(universe, "get_index_membership", return_value=None):
            with patch.dict("sys.modules", {"yfinance": MagicMock()}):
                import yfinance as yf
                yf.Ticker = MagicMock(side_effect=Exception("API Error"))

                info = universe.get_stock_info("INVALID")
                assert info["symbol"] == "INVALID"
                assert info["index_membership"] is None


class TestSyncToDatabase:
    """Test database sync functionality."""

    def test_sync_requires_session(self):
        universe = StockUniverse()
        with pytest.raises(RuntimeError, match="Database session required"):
            universe.sync_to_database()

    def test_sync_inserts_new_stocks(self):
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = []

        universe = StockUniverse(db_session=mock_session)
        with patch.object(universe, "get_all_symbols", return_value=["AAPL"]):
            with patch.object(universe, "get_index_membership", return_value=IndexMembership.BOTH):
                with patch.object(
                    universe,
                    "get_stock_info",
                    return_value={"name": "Apple", "sector": "Tech", "market_cap": None},
                ):
                    result = universe.sync_to_database()

        assert result["inserted"] == 1
        assert result["updated"] == 0
        assert mock_session.add.called

    def test_sync_updates_existing_stocks(self):
        existing_stock = MagicMock()
        existing_stock.symbol = "AAPL"
        existing_stock.is_active = True

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [existing_stock]

        universe = StockUniverse(db_session=mock_session)
        with patch.object(universe, "get_all_symbols", return_value=["AAPL"]):
            with patch.object(universe, "get_index_membership", return_value=IndexMembership.BOTH):
                with patch.object(
                    universe, "get_stock_info", return_value={"name": "Apple", "sector": "Tech", "market_cap": None}
                ):
                    result = universe.sync_to_database()

        assert result["updated"] == 1
        assert result["inserted"] == 0

    def test_sync_deactivates_removed_stocks(self):
        existing_stock = MagicMock()
        existing_stock.symbol = "REMOVED"
        existing_stock.is_active = True

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [existing_stock]

        universe = StockUniverse(db_session=mock_session)
        with patch.object(universe, "get_all_symbols", return_value=[]):
            result = universe.sync_to_database()

        assert result["deactivated"] == 1
        assert existing_stock.is_active is False


class TestLoadFromDatabase:
    """Test loading symbols from database."""

    def test_load_requires_session(self):
        universe = StockUniverse()
        with pytest.raises(RuntimeError, match="Database session required"):
            universe.load_from_database()

    def test_load_returns_active_symbols(self):
        stock1 = MagicMock()
        stock1.symbol = "AAPL"
        stock2 = MagicMock()
        stock2.symbol = "MSFT"

        mock_session = MagicMock()
        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.all.return_value = [stock1, stock2]

        universe = StockUniverse(db_session=mock_session)
        symbols = universe.load_from_database()

        assert symbols == ["AAPL", "MSFT"]
