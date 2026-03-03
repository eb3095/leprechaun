"""Real-time market data provider using Alpaca and yfinance."""

import logging
from datetime import datetime, time, timedelta
from typing import Any, Optional

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

EASTERN_TZ = pytz.timezone("US/Eastern")


class MarketDataProvider:
    """Provides real-time market data from Alpaca and yfinance.

    Uses Alpaca for real-time quotes when available, falls back to
    yfinance for delayed data.
    """

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EXTENDED_OPEN = time(4, 0)
    EXTENDED_CLOSE = time(20, 0)

    def __init__(self, alpaca_client=None):
        """Initialize the market data provider.

        Args:
            alpaca_client: Optional Alpaca trading client. If None, will
                          attempt to create one from config or fall back
                          to yfinance-only mode.
        """
        self.client = alpaca_client
        self._alpaca_available: Optional[bool] = None
        self._yf_available: Optional[bool] = None

    def _check_alpaca(self) -> bool:
        """Check if Alpaca client is available and working."""
        if self._alpaca_available is not None:
            return self._alpaca_available

        if self.client is None:
            try:
                from alpaca.data import StockHistoricalDataClient

                from src.utils.config import get_config

                config = get_config()
                self.client = StockHistoricalDataClient(
                    api_key=config.alpaca.api_key,
                    secret_key=config.alpaca.api_secret,
                )
                self._alpaca_available = True
            except Exception as e:
                logger.info(f"Alpaca client not available: {e}")
                self._alpaca_available = False
        else:
            self._alpaca_available = True

        return self._alpaca_available

    def _check_yfinance(self) -> bool:
        """Check if yfinance is available."""
        if self._yf_available is not None:
            return self._yf_available

        try:
            import yfinance as yf

            ticker = yf.Ticker("AAPL")
            info = ticker.fast_info
            self._yf_available = info is not None
        except Exception:
            self._yf_available = False

        return self._yf_available

    def _get_quote_alpaca(self, symbol: str) -> Optional[dict]:
        """Get quote from Alpaca."""
        if not self._check_alpaca():
            return None

        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price) if quote.bid_price else None,
                    "bid_size": quote.bid_size,
                    "ask": float(quote.ask_price) if quote.ask_price else None,
                    "ask_size": quote.ask_size,
                    "last": None,
                    "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
                    "source": "alpaca",
                }
        except Exception as e:
            logger.warning(f"Alpaca quote failed for {symbol}: {e}")

        return None

    def _get_quote_yfinance(self, symbol: str) -> Optional[dict]:
        """Get quote from yfinance."""
        if not self._check_yfinance():
            return None

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            return {
                "symbol": symbol,
                "bid": getattr(info, "bid", None),
                "bid_size": None,
                "ask": getattr(info, "ask", None),
                "ask_size": None,
                "last": getattr(info, "last_price", None) or getattr(info, "previous_close", None),
                "timestamp": datetime.now(EASTERN_TZ).isoformat(),
                "source": "yfinance",
            }
        except Exception as e:
            logger.warning(f"yfinance quote failed for {symbol}: {e}")

        return None

    def get_quote(self, symbol: str) -> dict:
        """Get current quote (bid, ask, last price).

        Tries Alpaca first for real-time data, falls back to yfinance.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with quote data, or empty dict if unavailable.
        """
        symbol = symbol.upper()

        quote = self._get_quote_alpaca(symbol)
        if quote:
            return quote

        quote = self._get_quote_yfinance(symbol)
        if quote:
            return quote

        logger.warning(f"No quote available for {symbol}")
        return {}

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dictionary mapping symbols to their quote data.
        """
        result = {}

        if self._check_alpaca():
            try:
                from alpaca.data.requests import StockLatestQuoteRequest

                request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes = self.client.get_stock_latest_quote(request)

                for symbol in symbols:
                    if symbol in quotes:
                        quote = quotes[symbol]
                        result[symbol] = {
                            "symbol": symbol,
                            "bid": float(quote.bid_price) if quote.bid_price else None,
                            "bid_size": quote.bid_size,
                            "ask": float(quote.ask_price) if quote.ask_price else None,
                            "ask_size": quote.ask_size,
                            "last": None,
                            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
                            "source": "alpaca",
                        }
            except Exception as e:
                logger.warning(f"Alpaca bulk quotes failed: {e}")

        for symbol in symbols:
            if symbol not in result:
                quote = self._get_quote_yfinance(symbol)
                if quote:
                    result[symbol] = quote

        return result

    def _get_bars_alpaca(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Get bars from Alpaca."""
        if not self._check_alpaca():
            return None

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf_map = {
                "1min": TimeFrame(1, TimeFrameUnit.Minute),
                "5min": TimeFrame(5, TimeFrameUnit.Minute),
                "15min": TimeFrame(15, TimeFrameUnit.Minute),
                "1hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1day": TimeFrame(1, TimeFrameUnit.Day),
            }

            tf = tf_map.get(timeframe.lower().replace(" ", ""), TimeFrame.Day)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit,
            )
            bars = self.client.get_stock_bars(request)

            if symbol not in bars or not bars[symbol]:
                return None

            data = []
            for bar in bars[symbol]:
                data.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": bar.volume,
                })

            return pd.DataFrame(data)

        except Exception as e:
            logger.warning(f"Alpaca bars failed for {symbol}: {e}")
            return None

    def _get_bars_yfinance(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Get bars from yfinance."""
        if not self._check_yfinance():
            return None

        try:
            import yfinance as yf

            interval_map = {
                "1min": "1m",
                "5min": "5m",
                "15min": "15m",
                "1hour": "1h",
                "1day": "1d",
            }

            interval = interval_map.get(timeframe.lower().replace(" ", ""), "1d")

            period_map = {
                "1m": "7d",
                "5m": "60d",
                "15m": "60d",
                "1h": "730d",
                "1d": "max",
            }
            period = period_map.get(interval, "1mo")

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            df = df.tail(limit).reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            else:
                df["timestamp"] = df.index

            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.warning(f"yfinance bars failed for {symbol}: {e}")
            return None

    def get_bars(
        self, symbol: str, timeframe: str = "1Day", limit: int = 100
    ) -> pd.DataFrame:
        """Get recent bars.

        Args:
            symbol: Stock symbol.
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day").
            limit: Maximum number of bars to return.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        symbol = symbol.upper()

        df = self._get_bars_alpaca(symbol, timeframe, limit)
        if df is not None and not df.empty:
            return df

        df = self._get_bars_yfinance(symbol, timeframe, limit)
        if df is not None and not df.empty:
            return df

        logger.warning(f"No bars available for {symbol}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def is_market_open(self) -> bool:
        """Check if the regular US stock market is currently open.

        Returns:
            True if the market is open for regular trading.
        """
        now = datetime.now(EASTERN_TZ)

        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def is_extended_hours(self) -> bool:
        """Check if we're in extended trading hours.

        Returns:
            True if we're in pre-market or after-hours trading.
        """
        now = datetime.now(EASTERN_TZ)

        if now.weekday() >= 5:
            return False

        current_time = now.time()

        if self.EXTENDED_OPEN <= current_time < self.MARKET_OPEN:
            return True

        if self.MARKET_CLOSE <= current_time < self.EXTENDED_CLOSE:
            return True

        return False

    def get_market_status(self) -> dict[str, Any]:
        """Get detailed market status.

        Returns:
            Dictionary with market status information.
        """
        now = datetime.now(EASTERN_TZ)
        is_open = self.is_market_open()
        is_extended = self.is_extended_hours()

        if is_open:
            status = "open"
            session = "regular"
        elif is_extended:
            status = "open"
            session = "pre_market" if now.time() < self.MARKET_OPEN else "after_hours"
        else:
            status = "closed"
            session = "closed"

        if status == "closed":
            if now.weekday() >= 5:
                days_until = 7 - now.weekday()
                next_open = (now + timedelta(days=days_until)).replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
            elif now.time() >= self.EXTENDED_CLOSE:
                next_open = (now + timedelta(days=1)).replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
            else:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            next_open = None

        if is_open:
            close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            time_until_close = close_time - now
        else:
            time_until_close = None

        return {
            "status": status,
            "session": session,
            "is_regular_hours": is_open,
            "is_extended_hours": is_extended,
            "current_time": now.isoformat(),
            "timezone": "US/Eastern",
            "next_open": next_open.isoformat() if next_open else None,
            "time_until_close_seconds": (
                int(time_until_close.total_seconds()) if time_until_close else None
            ),
        }

    def get_snapshot(self, symbol: str) -> dict:
        """Get a complete market snapshot for a symbol.

        Combines quote and recent bar data.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with quote, latest bar, and daily stats.
        """
        symbol = symbol.upper()

        quote = self.get_quote(symbol)
        bars = self.get_bars(symbol, "1Day", 2)

        result = {
            "symbol": symbol,
            "quote": quote,
            "market_status": self.get_market_status(),
        }

        if not bars.empty:
            latest = bars.iloc[-1]
            result["latest_bar"] = {
                "timestamp": str(latest["timestamp"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": int(latest["volume"]),
            }

            if len(bars) >= 2:
                prev = bars.iloc[-2]
                change = float(latest["close"]) - float(prev["close"])
                change_pct = (change / float(prev["close"])) * 100
                result["daily_change"] = {
                    "change": round(change, 2),
                    "change_percent": round(change_pct, 2),
                    "previous_close": float(prev["close"]),
                }

        return result
