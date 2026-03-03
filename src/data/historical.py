"""Historical price data fetching from yfinance and Polygon."""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.data.models import PriceHistory, Stock

logger = logging.getLogger(__name__)


class HistoricalDataProvider:
    """Fetches historical price data from yfinance and Polygon.

    Primary data source is yfinance (free), with Polygon.io as a backup.
    Supports caching to database for reduced API calls and offline access.
    """

    def __init__(self, polygon_api_key: Optional[str] = None):
        """Initialize the historical data provider.

        Args:
            polygon_api_key: Optional Polygon.io API key for backup data source.
        """
        self.polygon_key = polygon_api_key
        self._cache: dict[str, pd.DataFrame] = {}
        self._yf_available: Optional[bool] = None
        self._polygon_client = None

    def _init_polygon_client(self):
        """Lazily initialize Polygon client if API key is available."""
        if self._polygon_client is not None or not self.polygon_key:
            return

        try:
            from polygon import RESTClient

            self._polygon_client = RESTClient(api_key=self.polygon_key)
        except ImportError:
            logger.warning("polygon-api-client not installed, Polygon backup unavailable")
        except Exception as e:
            logger.warning(f"Failed to initialize Polygon client: {e}")

    def _check_yfinance(self) -> bool:
        """Check if yfinance is available and working."""
        if self._yf_available is not None:
            return self._yf_available

        try:
            import yfinance as yf

            ticker = yf.Ticker("AAPL")
            test = ticker.history(period="1d")
            self._yf_available = not test.empty
        except Exception:
            self._yf_available = False

        return self._yf_available

    def _fetch_yfinance(
        self, symbol: str, start: date, end: Optional[date] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance.

        Args:
            symbol: Stock symbol.
            start: Start date for historical data.
            end: End date (defaults to today).

        Returns:
            DataFrame with OHLCV data or None if fetch failed.
        """
        if not self._check_yfinance():
            return None

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            end_date = end or date.today()

            df = ticker.history(
                start=start.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                auto_adjust=False,
            )

            if df.empty:
                return None

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            elif "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date
                df = df.drop(columns=["datetime"])

            expected_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
            for col in expected_cols:
                if col not in df.columns:
                    if col == "adj_close" and "adj._close" in df.columns:
                        df["adj_close"] = df["adj._close"]
                    elif col == "adj_close":
                        df["adj_close"] = df["close"]

            return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _fetch_polygon(
        self, symbol: str, start: date, end: Optional[date] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io.

        Args:
            symbol: Stock symbol.
            start: Start date for historical data.
            end: End date (defaults to today).

        Returns:
            DataFrame with OHLCV data or None if fetch failed.
        """
        self._init_polygon_client()
        if self._polygon_client is None:
            return None

        try:
            end_date = end or date.today()

            aggs = list(
                self._polygon_client.list_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=start.isoformat(),
                    to=end_date.isoformat(),
                    limit=50000,
                )
            )

            if not aggs:
                return None

            data = []
            for bar in aggs:
                bar_date = datetime.fromtimestamp(bar.timestamp / 1000).date()
                data.append({
                    "date": bar_date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "adj_close": bar.close,
                    "volume": bar.volume,
                })

            return pd.DataFrame(data)

        except Exception as e:
            logger.warning(f"Polygon fetch failed for {symbol}: {e}")
            return None

    def get_price_history(
        self,
        symbol: str,
        start: date,
        end: Optional[date] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get OHLCV data for symbol.

        Tries yfinance first, falls back to Polygon if available.

        Args:
            symbol: Stock symbol.
            start: Start date for historical data.
            end: End date (defaults to today).
            use_cache: Whether to use in-memory cache.

        Returns:
            DataFrame with columns: date, open, high, low, close, adj_close, volume.
            Returns empty DataFrame if data is unavailable.
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}_{start}_{end}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        df = self._fetch_yfinance(symbol, start, end)

        if df is None or df.empty:
            df = self._fetch_polygon(symbol, start, end)

        if df is None:
            logger.warning(f"No historical data available for {symbol}")
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        if use_cache:
            self._cache[cache_key] = df.copy()

        return df

    def get_latest_price(self, symbol: str) -> dict:
        """Get most recent price data.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with latest OHLCV data, or empty dict if unavailable.
        """
        end = date.today()
        start = end - timedelta(days=7)

        df = self.get_price_history(symbol, start, end, use_cache=False)
        if df.empty:
            return {}

        latest = df.iloc[-1]
        return {
            "date": latest["date"],
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "adj_close": float(latest["adj_close"]),
            "volume": int(latest["volume"]),
        }

    def get_bulk_history(
        self,
        symbols: list[str],
        start: date,
        end: Optional[date] = None,
    ) -> dict[str, pd.DataFrame]:
        """Get history for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols.
            start: Start date for historical data.
            end: End date (defaults to today).

        Returns:
            Dictionary mapping symbols to their historical DataFrames.
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_price_history(symbol, start, end)
        return result

    def cache_to_database(
        self, symbol: str, df: pd.DataFrame, db_session
    ) -> int:
        """Cache price history to database.

        Args:
            symbol: Stock symbol.
            df: DataFrame with price history.
            db_session: SQLAlchemy session.

        Returns:
            Number of rows inserted/updated.

        Raises:
            ValueError: If symbol not found in database.
        """
        stock = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if stock is None:
            raise ValueError(f"Stock {symbol} not found in database")

        existing_dates = {
            row.date
            for row in db_session.query(PriceHistory.date)
            .filter(PriceHistory.stock_id == stock.id)
            .all()
        }

        count = 0
        for _, row in df.iterrows():
            row_date = row["date"]
            if isinstance(row_date, datetime):
                row_date = row_date.date()

            if row_date in existing_dates:
                continue

            price_record = PriceHistory(
                stock_id=stock.id,
                date=row_date,
                open=Decimal(str(row["open"])) if pd.notna(row["open"]) else None,
                high=Decimal(str(row["high"])) if pd.notna(row["high"]) else None,
                low=Decimal(str(row["low"])) if pd.notna(row["low"]) else None,
                close=Decimal(str(row["close"])) if pd.notna(row["close"]) else None,
                adj_close=Decimal(str(row["adj_close"])) if pd.notna(row["adj_close"]) else None,
                volume=int(row["volume"]) if pd.notna(row["volume"]) else None,
            )
            db_session.add(price_record)
            count += 1

        db_session.flush()
        return count

    def load_from_database(
        self, symbol: str, start: date, end: date, db_session
    ) -> pd.DataFrame:
        """Load cached data from database.

        Args:
            symbol: Stock symbol.
            start: Start date.
            end: End date.
            db_session: SQLAlchemy session.

        Returns:
            DataFrame with cached price history.
        """
        stock = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if stock is None:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        records = (
            db_session.query(PriceHistory)
            .filter(
                PriceHistory.stock_id == stock.id,
                PriceHistory.date >= start,
                PriceHistory.date <= end,
            )
            .order_by(PriceHistory.date)
            .all()
        )

        if not records:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        data = []
        for r in records:
            data.append({
                "date": r.date,
                "open": float(r.open) if r.open else None,
                "high": float(r.high) if r.high else None,
                "low": float(r.low) if r.low else None,
                "close": float(r.close) if r.close else None,
                "adj_close": float(r.adj_close) if r.adj_close else None,
                "volume": r.volume,
            })

        return pd.DataFrame(data)

    def update_prices(
        self, symbols: list[str], db_session, lookback_days: int = 7
    ) -> dict[str, int]:
        """Update price history for symbols (incremental).

        Fetches only missing recent data and caches to database.

        Args:
            symbols: List of stock symbols.
            db_session: SQLAlchemy session.
            lookback_days: Number of days to look back for updates.

        Returns:
            Dictionary mapping symbols to number of new records inserted.
        """
        result = {}
        end = date.today()
        start = end - timedelta(days=lookback_days)

        for symbol in symbols:
            try:
                df = self.get_price_history(symbol, start, end)
                if not df.empty:
                    count = self.cache_to_database(symbol, df, db_session)
                    result[symbol] = count
                else:
                    result[symbol] = 0
            except ValueError as e:
                logger.warning(f"Skipping {symbol}: {e}")
                result[symbol] = -1
            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                result[symbol] = -1

        return result

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()

    def get_date_range_in_database(
        self, symbol: str, db_session
    ) -> tuple[Optional[date], Optional[date]]:
        """Get the date range of cached data in the database.

        Args:
            symbol: Stock symbol.
            db_session: SQLAlchemy session.

        Returns:
            Tuple of (earliest_date, latest_date) or (None, None) if no data.
        """
        stock = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if stock is None:
            return None, None

        from sqlalchemy import func

        result = (
            db_session.query(
                func.min(PriceHistory.date),
                func.max(PriceHistory.date),
            )
            .filter(PriceHistory.stock_id == stock.id)
            .first()
        )

        return result[0], result[1]
