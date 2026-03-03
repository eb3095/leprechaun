"""Stock universe management for S&P 500 and NASDAQ 100 stocks."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.models import IndexMembership, Stock

logger = logging.getLogger(__name__)

SP500_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "GOOG", "BRK.B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK",
    "LLY", "ABBV", "PEP", "COST", "KO", "ADBE", "WMT", "MCD", "CSCO", "CRM",
    "BAC", "PFE", "TMO", "ACN", "NFLX", "AMD", "ABT", "ORCL", "DHR", "DIS",
    "CMCSA", "VZ", "INTC", "PM", "WFC", "TXN", "NKE", "COP", "MS", "NEE",
    "RTX", "UNP", "UPS", "BMY", "QCOM", "HON", "ELV", "LOW", "T", "IBM",
    "CAT", "SPGI", "INTU", "GS", "AMGN", "DE", "BA", "PLD", "LMT", "AXP",
    "MDT", "SBUX", "SCHW", "SYK", "ISRG", "BLK", "ADI", "GILD", "BKNG", "MMC",
    "ADP", "MDLZ", "TJX", "CVS", "AMT", "CB", "C", "GE", "MO", "TMUS",
    "LRCX", "CI", "ZTS", "SO", "VRTX", "EOG", "REGN", "PGR", "NOW", "DUK",
]

NASDAQ100_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "PEP", "ADBE", "NFLX", "CSCO", "AMD", "CMCSA", "TMUS", "INTC", "TXN", "QCOM",
    "INTU", "AMGN", "HON", "ISRG", "SBUX", "BKNG", "ADP", "MDLZ", "ADI", "GILD",
    "LRCX", "VRTX", "REGN", "PYPL", "MU", "AMAT", "SNPS", "KLAC", "PANW", "CSX",
    "CDNS", "MELI", "ORLY", "ASML", "CTAS", "MAR", "MNST", "CHTR", "ABNB", "MRVL",
    "PCAR", "KDP", "WDAY", "NXPI", "AEP", "DXCM", "KHC", "PAYX", "FTNT", "AZN",
    "EXC", "CPRT", "LULU", "ROST", "EA", "ODFL", "FAST", "XEL", "VRSK", "BIIB",
    "IDXX", "CTSH", "CSGP", "BKR", "GEHC", "FANG", "ON", "DDOG", "ANSS", "ZS",
    "TEAM", "GFS", "CEG", "DLTR", "MRNA", "ILMN", "WBD", "SIRI", "JD", "PDD",
    "LCID", "RIVN", "ZM", "ALGN", "ENPH", "CRWD", "EBAY", "WBA", "SWKS", "SPLK",
]


def _create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class StockUniverse:
    """Manages the S&P 500 and NASDAQ 100 stock universe."""

    WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    WIKIPEDIA_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
    CACHE_DURATION = timedelta(hours=24)

    def __init__(self, db_session=None):
        """Initialize the stock universe manager.

        Args:
            db_session: Optional SQLAlchemy session for database operations.
        """
        self.session = db_session
        self._sp500_cache: Optional[list[str]] = None
        self._nasdaq100_cache: Optional[list[str]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._stock_info_cache: dict[str, dict] = {}
        self._http_session = _create_session()

    def _is_cache_valid(self) -> bool:
        """Check if the cache is still valid."""
        if self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self.CACHE_DURATION

    def _fetch_sp500_from_wikipedia(self) -> list[str]:
        """Fetch S&P 500 symbols from Wikipedia."""
        try:
            tables = pd.read_html(
                self.WIKIPEDIA_SP500_URL,
                storage_options={"User-Agent": "Mozilla/5.0"},
            )
            if tables and len(tables) > 0:
                df = tables[0]
                if "Symbol" in df.columns:
                    symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
                    return [s for s in symbols if isinstance(s, str) and len(s) <= 10]
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500 from Wikipedia: {e}")
        return []

    def _fetch_nasdaq100_from_wikipedia(self) -> list[str]:
        """Fetch NASDAQ 100 symbols from Wikipedia."""
        try:
            tables = pd.read_html(
                self.WIKIPEDIA_NASDAQ100_URL,
                storage_options={"User-Agent": "Mozilla/5.0"},
            )
            for table in tables:
                if "Ticker" in table.columns:
                    symbols = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                    return [s for s in symbols if isinstance(s, str) and len(s) <= 10]
                elif "Symbol" in table.columns:
                    symbols = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                    return [s for s in symbols if isinstance(s, str) and len(s) <= 10]
        except Exception as e:
            logger.warning(f"Failed to fetch NASDAQ 100 from Wikipedia: {e}")
        return []

    def get_sp500_symbols(self, use_cache: bool = True) -> list[str]:
        """Get current S&P 500 symbols.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            List of S&P 500 stock symbols.
        """
        if use_cache and self._sp500_cache and self._is_cache_valid():
            return self._sp500_cache.copy()

        symbols = self._fetch_sp500_from_wikipedia()
        if not symbols:
            logger.info("Using fallback S&P 500 symbols")
            symbols = SP500_SYMBOLS.copy()

        self._sp500_cache = symbols
        if self._cache_timestamp is None or not self._is_cache_valid():
            self._cache_timestamp = datetime.now()

        return symbols.copy()

    def get_nasdaq100_symbols(self, use_cache: bool = True) -> list[str]:
        """Get current NASDAQ 100 symbols.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            List of NASDAQ 100 stock symbols.
        """
        if use_cache and self._nasdaq100_cache and self._is_cache_valid():
            return self._nasdaq100_cache.copy()

        symbols = self._fetch_nasdaq100_from_wikipedia()
        if not symbols:
            logger.info("Using fallback NASDAQ 100 symbols")
            symbols = NASDAQ100_SYMBOLS.copy()

        self._nasdaq100_cache = symbols
        if self._cache_timestamp is None or not self._is_cache_valid():
            self._cache_timestamp = datetime.now()

        return symbols.copy()

    def get_all_symbols(self) -> list[str]:
        """Get combined unique symbols from both indexes.

        Returns:
            Sorted list of unique symbols from S&P 500 and NASDAQ 100.
        """
        sp500 = set(self.get_sp500_symbols())
        nasdaq100 = set(self.get_nasdaq100_symbols())
        return sorted(sp500 | nasdaq100)

    def refresh_universe(self) -> dict[str, int]:
        """Refresh stock universe from external sources.

        Returns:
            Dictionary with counts: {sp500: count, nasdaq100: count, total: count}
        """
        self._cache_timestamp = None
        self._sp500_cache = None
        self._nasdaq100_cache = None
        self._stock_info_cache.clear()

        sp500 = self.get_sp500_symbols(use_cache=False)
        nasdaq100 = self.get_nasdaq100_symbols(use_cache=False)
        all_symbols = self.get_all_symbols()

        return {
            "sp500": len(sp500),
            "nasdaq100": len(nasdaq100),
            "total": len(all_symbols),
        }

    def is_in_universe(self, symbol: str) -> bool:
        """Check if symbol is in our trading universe.

        Args:
            symbol: Stock symbol to check.

        Returns:
            True if the symbol is in S&P 500 or NASDAQ 100.
        """
        symbol = symbol.upper().replace(".", "-")
        return symbol in self.get_all_symbols()

    def get_index_membership(self, symbol: str) -> Optional[IndexMembership]:
        """Determine which index(es) a symbol belongs to.

        Args:
            symbol: Stock symbol to check.

        Returns:
            IndexMembership enum value or None if not in universe.
        """
        symbol = symbol.upper().replace(".", "-")
        in_sp500 = symbol in self.get_sp500_symbols()
        in_nasdaq100 = symbol in self.get_nasdaq100_symbols()

        if in_sp500 and in_nasdaq100:
            return IndexMembership.BOTH
        elif in_sp500:
            return IndexMembership.SP500
        elif in_nasdaq100:
            return IndexMembership.NASDAQ100
        return None

    def get_stock_info(self, symbol: str, use_cache: bool = True) -> dict:
        """Get stock metadata (name, sector, market cap).

        Uses yfinance to fetch stock information. Gracefully returns
        minimal data if the service is unavailable.

        Args:
            symbol: Stock symbol.
            use_cache: Whether to use cached data if available.

        Returns:
            Dictionary with stock metadata.
        """
        symbol = symbol.upper().replace(".", "-")

        if use_cache and symbol in self._stock_info_cache:
            return self._stock_info_cache[symbol].copy()

        info = {
            "symbol": symbol,
            "name": None,
            "sector": None,
            "market_cap": None,
            "index_membership": self.get_index_membership(symbol),
        }

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            ticker_info = ticker.info
            info["name"] = ticker_info.get("longName") or ticker_info.get("shortName")
            info["sector"] = ticker_info.get("sector")
            info["market_cap"] = ticker_info.get("marketCap")
        except Exception as e:
            logger.warning(f"Failed to fetch info for {symbol}: {e}")

        self._stock_info_cache[symbol] = info
        return info.copy()

    def get_bulk_stock_info(self, symbols: list[str]) -> dict[str, dict]:
        """Get stock metadata for multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dictionary mapping symbols to their metadata.
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_stock_info(symbol)
        return result

    def sync_to_database(self) -> dict[str, int]:
        """Sync universe to database stocks table.

        Updates existing stocks and inserts new ones. Does not delete
        stocks that are no longer in the index (marks them inactive).

        Returns:
            Dictionary with counts: {inserted: count, updated: count, deactivated: count}

        Raises:
            RuntimeError: If no database session is available.
        """
        if self.session is None:
            raise RuntimeError("Database session required for sync_to_database")

        all_symbols = self.get_all_symbols()
        existing_stocks = {
            s.symbol: s for s in self.session.query(Stock).all()
        }

        inserted = 0
        updated = 0
        deactivated = 0

        for symbol in all_symbols:
            membership = self.get_index_membership(symbol)
            info = self.get_stock_info(symbol)

            if symbol in existing_stocks:
                stock = existing_stocks[symbol]
                stock.is_active = True
                stock.index_membership = membership
                if info.get("name"):
                    stock.name = info["name"]
                if info.get("sector"):
                    stock.sector = info["sector"]
                if info.get("market_cap"):
                    stock.market_cap = info["market_cap"]
                updated += 1
            else:
                stock = Stock(
                    symbol=symbol,
                    name=info.get("name"),
                    sector=info.get("sector"),
                    market_cap=info.get("market_cap"),
                    index_membership=membership,
                    is_active=True,
                )
                self.session.add(stock)
                inserted += 1

        for symbol, stock in existing_stocks.items():
            if symbol not in all_symbols and stock.is_active:
                stock.is_active = False
                deactivated += 1

        self.session.flush()

        return {
            "inserted": inserted,
            "updated": updated,
            "deactivated": deactivated,
        }

    def load_from_database(self) -> list[str]:
        """Load active symbols from the database.

        Returns:
            List of active stock symbols from the database.

        Raises:
            RuntimeError: If no database session is available.
        """
        if self.session is None:
            raise RuntimeError("Database session required for load_from_database")

        stocks = (
            self.session.query(Stock)
            .filter(Stock.is_active.is_(True))
            .all()
        )
        return [s.symbol for s in stocks]
