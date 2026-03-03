#!/usr/bin/env python3
"""
Backtesting script for Leprechaun trading strategy.

Runs historical backtests using archived or synthetic sentiment data
to evaluate strategy performance.

Usage:
    python scripts/backtest.py --start 2024-01-01 --end 2024-12-31 --initial-cash 100000

Options:
    --start           Start date (YYYY-MM-DD)
    --end             End date (YYYY-MM-DD)
    --initial-cash    Starting capital (default: 100000)
    --use-synthetic   Force synthetic sentiment even if archived data exists
    --technicals-only Ignore sentiment, use only technical indicators
    --symbols         Comma-separated list of symbols (default: all universe)
    --output          Output file for results (JSON)
    --verbose         Enable verbose logging
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core.sentiment.synthetic import (
    BacktestSentimentProvider,
    SyntheticSentimentGenerator,
)
from src.core.technical.indicators import calculate_ema, calculate_rsi
from src.core.trading.strategy import TradingStrategy
from src.data.historical import HistoricalDataProvider
from src.data.sentiment_archive import SentimentArchive
from src.data.universe import StockUniverse
from src.utils.config import BacktestConfig, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """Represents a position during backtesting."""

    symbol: str
    entry_date: date
    entry_price: float
    shares: float
    entry_signal_score: float = 0.0
    sentiment_score: float = 0.0
    manipulation_score: float = 0.0

    def calculate_pnl(self, current_price: float) -> tuple[float, float]:
        """Calculate P&L given current price."""
        pnl = (current_price - self.entry_price) * self.shares
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        return pnl, pnl_pct


@dataclass
class BacktestTrade:
    """Represents a completed trade."""

    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: float
    exit_reason: str
    pnl: float
    pnl_percent: float
    sentiment_score: float = 0.0
    manipulation_score: float = 0.0
    was_synthetic: bool = False


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    start_date: date
    end_date: date
    initial_cash: float
    final_value: float
    total_return: float
    total_return_pct: float
    trades: list[BacktestTrade]
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    daily_returns: list[dict]
    config: dict
    coverage_info: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_cash": self.initial_cash,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat(),
                    "exit_date": t.exit_date.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "sentiment_score": t.sentiment_score,
                    "manipulation_score": t.manipulation_score,
                    "was_synthetic": t.was_synthetic,
                }
                for t in self.trades
            ],
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "daily_returns": self.daily_returns,
            "config": self.config,
            "coverage_info": self.coverage_info,
        }


class SimulatedExecutor:
    """Simulates order execution for backtesting."""

    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0):
        """Initialize simulated executor.

        Args:
            initial_cash: Starting cash balance.
            commission: Per-trade commission (default 0 for Alpaca).
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.daily_values: list[dict] = []

    def buy(
        self,
        symbol: str,
        price: float,
        amount: float,
        current_date: date,
        sentiment_score: float = 0.0,
        manipulation_score: float = 0.0,
        signal_score: float = 0.0,
    ) -> bool:
        """Execute a buy order.

        Args:
            symbol: Stock symbol.
            price: Current price.
            amount: Dollar amount to invest.
            current_date: Trade date.
            sentiment_score: Current sentiment.
            manipulation_score: Current manipulation score.
            signal_score: Entry signal score.

        Returns:
            True if order executed successfully.
        """
        if symbol in self.positions:
            logger.debug("Already have position in %s", symbol)
            return False

        cost = amount + self.commission
        if cost > self.cash:
            logger.debug("Insufficient cash for %s: need $%.2f, have $%.2f", symbol, cost, self.cash)
            return False

        shares = amount / price

        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            entry_date=current_date,
            entry_price=price,
            shares=shares,
            entry_signal_score=signal_score,
            sentiment_score=sentiment_score,
            manipulation_score=manipulation_score,
        )

        self.cash -= cost

        logger.debug(
            "BUY %s: %.2f shares @ $%.2f (total: $%.2f)",
            symbol,
            shares,
            price,
            amount,
        )
        return True

    def sell(
        self,
        symbol: str,
        price: float,
        current_date: date,
        exit_reason: str,
        was_synthetic: bool = False,
    ) -> Optional[BacktestTrade]:
        """Execute a sell order.

        Args:
            symbol: Stock symbol.
            price: Current price.
            current_date: Trade date.
            exit_reason: Reason for exit.
            was_synthetic: Whether sentiment was synthetic.

        Returns:
            BacktestTrade if successful, None otherwise.
        """
        if symbol not in self.positions:
            logger.warning("No position to sell for %s", symbol)
            return None

        position = self.positions[symbol]
        proceeds = position.shares * price - self.commission

        pnl, pnl_pct = position.calculate_pnl(price)

        trade = BacktestTrade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=current_date,
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_pct,
            sentiment_score=position.sentiment_score,
            manipulation_score=position.manipulation_score,
            was_synthetic=was_synthetic,
        )

        self.trades.append(trade)
        self.cash += proceeds
        del self.positions[symbol]

        logger.debug(
            "SELL %s: %.2f shares @ $%.2f (P&L: $%.2f / %.2f%%)",
            symbol,
            position.shares,
            price,
            pnl,
            pnl_pct,
        )
        return trade

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value.

        Args:
            prices: Dict mapping symbols to current prices.

        Returns:
            Total portfolio value.
        """
        positions_value = sum(
            pos.shares * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def record_daily_value(self, current_date: date, prices: dict[str, float]) -> None:
        """Record daily portfolio value.

        Args:
            current_date: Current date.
            prices: Current prices.
        """
        value = self.get_portfolio_value(prices)
        self.daily_values.append({
            "date": current_date.isoformat(),
            "value": value,
            "cash": self.cash,
            "positions_value": value - self.cash,
            "position_count": len(self.positions),
        })


class Backtester:
    """Runs backtests for the Leprechaun trading strategy."""

    def __init__(
        self,
        config: BacktestConfig,
        trading_config: Optional[TradingConfig] = None,
        db_session=None,
    ):
        """Initialize the backtester.

        Args:
            config: Backtest configuration.
            trading_config: Trading strategy configuration.
            db_session: Optional database session for archived sentiment.
        """
        self.config = config
        self.trading_config = trading_config or TradingConfig()

        self.archive = SentimentArchive(db_session)
        self.generator = SyntheticSentimentGenerator(
            seed=42,
            noise_level=config.synthetic_noise_level,
        )
        self.sentiment_provider = BacktestSentimentProvider(
            archive=self.archive,
            generator=self.generator,
            min_coverage=config.min_archive_coverage,
        )

        self.historical = HistoricalDataProvider()
        self.universe = StockUniverse(db_session)
        self.strategy = TradingStrategy(self.trading_config)

        self.executor: Optional[SimulatedExecutor] = None
        self.results: Optional[BacktestResult] = None

    def run(
        self,
        start: date,
        end: date,
        initial_cash: float = 100000.0,
        symbols: Optional[list[str]] = None,
        max_positions: int = 5,
        position_size_pct: float = 0.2,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            start: Start date.
            end: End date.
            initial_cash: Starting capital.
            symbols: List of symbols to trade (default: all universe).
            max_positions: Maximum concurrent positions.
            position_size_pct: Position size as fraction of portfolio.

        Returns:
            BacktestResult with performance metrics.
        """
        self.executor = SimulatedExecutor(initial_cash)

        if symbols is None:
            symbols = self.universe.get_all_symbols()[:50]

        logger.info(
            "Starting backtest from %s to %s with %d symbols",
            start,
            end,
            len(symbols),
        )

        price_cache = self._load_price_data(symbols, start, end)
        sentiment_cache = self._load_sentiment_data(symbols, start, end, price_cache)

        coverage_info = self.sentiment_provider.get_coverage_info(symbols, start, end)

        trading_days = self._get_trading_days(start, end)

        for current_date in trading_days:
            current_prices = self._get_prices_for_date(price_cache, current_date)
            if not current_prices:
                continue

            self._process_exits(current_date, current_prices, sentiment_cache)

            if self._is_monday(current_date) and len(self.executor.positions) < max_positions:
                self._process_entries(
                    current_date,
                    current_prices,
                    price_cache,
                    sentiment_cache,
                    symbols,
                    max_positions,
                    position_size_pct,
                )

            self.executor.record_daily_value(current_date, current_prices)

        final_prices = self._get_prices_for_date(price_cache, end)
        for symbol in self.executor.positions.copy():
            if symbol in final_prices:
                self.executor.sell(symbol, final_prices[symbol], end, "BACKTEST_END")

        self.results = self._generate_results(start, end, initial_cash, coverage_info)
        return self.results

    def _load_price_data(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        """Load historical price data for all symbols."""
        logger.info("Loading price data for %d symbols...", len(symbols))

        lookback_start = start - timedelta(days=60)

        return self.historical.get_bulk_history(symbols, lookback_start, end)

    def _load_sentiment_data(
        self,
        symbols: list[str],
        start: date,
        end: date,
        price_cache: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Load sentiment data (archived or synthetic)."""
        logger.info("Loading sentiment data for %d symbols...", len(symbols))

        sentiment_cache = {}
        for symbol in symbols:
            price_df = price_cache.get(symbol)
            sentiment_cache[symbol] = self.sentiment_provider.get_sentiment(
                symbol,
                start,
                end,
                price_df=price_df,
                force_synthetic=self.config.use_synthetic_sentiment,
            )

        return sentiment_cache

    def _get_trading_days(self, start: date, end: date) -> list[date]:
        """Get list of trading days between start and end."""
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days

    def _is_monday(self, d: date) -> bool:
        """Check if date is Monday (or first trading day of week)."""
        return d.weekday() == 0

    def _is_friday(self, d: date) -> bool:
        """Check if date is Friday (or last trading day of week)."""
        return d.weekday() == 4

    def _get_prices_for_date(
        self, price_cache: dict[str, pd.DataFrame], target_date: date
    ) -> dict[str, float]:
        """Get closing prices for a specific date."""
        prices = {}
        for symbol, df in price_cache.items():
            if df.empty:
                continue
            date_rows = df[df["date"] == target_date]
            if not date_rows.empty:
                prices[symbol] = float(date_rows.iloc[0]["close"])
        return prices

    def _get_sentiment_for_date(
        self, sentiment_cache: dict[str, pd.DataFrame], symbol: str, target_date: date
    ) -> dict[str, Any]:
        """Get sentiment for a symbol on a specific date."""
        df = sentiment_cache.get(symbol)
        if df is None or df.empty:
            return {
                "sentiment_score": 0.0,
                "manipulation_score": 0.0,
                "is_synthetic": True,
            }

        date_rows = df[df["date"] == target_date]
        if date_rows.empty:
            df_before = df[df["date"] < target_date]
            if not df_before.empty:
                return df_before.iloc[-1].to_dict()
            return {
                "sentiment_score": 0.0,
                "manipulation_score": 0.0,
                "is_synthetic": True,
            }

        return date_rows.iloc[0].to_dict()

    def _calculate_indicators(
        self, price_df: pd.DataFrame, target_date: date
    ) -> dict[str, Any]:
        """Calculate technical indicators for a symbol."""
        if price_df.empty:
            return {}

        df_until = price_df[price_df["date"] <= target_date].copy()
        if len(df_until) < 14:
            return {}

        rsi = calculate_rsi(df_until["close"], period=14)
        ema_21 = calculate_ema(df_until["close"], period=21)

        return {
            "rsi": float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None,
            "ema_21": float(ema_21.iloc[-1]) if not ema_21.empty and not pd.isna(ema_21.iloc[-1]) else None,
        }

    def _process_exits(
        self,
        current_date: date,
        current_prices: dict[str, float],
        sentiment_cache: dict[str, pd.DataFrame],
    ) -> None:
        """Process exit signals for existing positions."""
        is_friday = self._is_friday(current_date)

        for symbol in self.executor.positions.copy():
            if symbol not in current_prices:
                continue

            position = self.executor.positions[symbol]
            price = current_prices[symbol]

            sentiment = self._get_sentiment_for_date(sentiment_cache, symbol, current_date)

            exit_signal = self.strategy.evaluate_exit(
                {
                    "symbol": symbol,
                    "entry_price": position.entry_price,
                    "shares": position.shares,
                },
                {
                    "price": price,
                    "is_friday_close": is_friday,
                    "has_breaking_news": False,
                    "rsi": None,
                },
            )

            if exit_signal.should_exit:
                self.executor.sell(
                    symbol,
                    price,
                    current_date,
                    exit_signal.exit_type,
                    was_synthetic=sentiment.get("is_synthetic", True),
                )

    def _process_entries(
        self,
        current_date: date,
        current_prices: dict[str, float],
        price_cache: dict[str, pd.DataFrame],
        sentiment_cache: dict[str, pd.DataFrame],
        symbols: list[str],
        max_positions: int,
        position_size_pct: float,
    ) -> None:
        """Process entry signals for potential new positions."""
        if self.config.technicals_only:
            return

        candidates = []
        available_slots = max_positions - len(self.executor.positions)

        for symbol in symbols:
            if symbol in self.executor.positions:
                continue
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            price_df = price_cache.get(symbol)
            if price_df is None or price_df.empty:
                continue

            indicators = self._calculate_indicators(price_df, current_date)
            sentiment = self._get_sentiment_for_date(sentiment_cache, symbol, current_date)

            stock_data = {
                "symbol": symbol,
                "price": price,
                "rsi": indicators.get("rsi"),
                "ema_21": indicators.get("ema_21"),
                "sentiment_score": sentiment.get("sentiment_score", 0.0),
                "manipulation_score": sentiment.get("manipulation_score", 0.0),
                "has_news": False,
            }

            entry_signal = self.strategy.evaluate_entry(stock_data)

            if entry_signal.should_enter:
                candidates.append({
                    **stock_data,
                    "signal_score": entry_signal.score,
                    "is_synthetic": sentiment.get("is_synthetic", True),
                })

        candidates.sort(key=lambda x: x["signal_score"], reverse=True)

        portfolio_value = self.executor.get_portfolio_value(current_prices)
        position_amount = portfolio_value * position_size_pct

        for candidate in candidates[:available_slots]:
            self.executor.buy(
                candidate["symbol"],
                candidate["price"],
                position_amount,
                current_date,
                sentiment_score=candidate["sentiment_score"],
                manipulation_score=candidate["manipulation_score"],
                signal_score=candidate["signal_score"],
            )

    def _generate_results(
        self,
        start: date,
        end: date,
        initial_cash: float,
        coverage_info: Optional[dict],
    ) -> BacktestResult:
        """Generate backtest results from trades."""
        final_value = self.executor.cash

        trades = self.executor.trades
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / len(trades) if trades else 0.0

        avg_win = sum(t.pnl for t in wins) / win_count if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / loss_count if losses else 0.0

        daily_values = self.executor.daily_values
        max_drawdown = self._calculate_max_drawdown(daily_values)
        sharpe = self._calculate_sharpe_ratio(daily_values)

        total_return = final_value - initial_cash
        total_return_pct = (total_return / initial_cash) * 100

        return BacktestResult(
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=trades,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            daily_returns=daily_values,
            config={
                "use_synthetic_sentiment": self.config.use_synthetic_sentiment,
                "synthetic_noise_level": self.config.synthetic_noise_level,
                "min_archive_coverage": self.config.min_archive_coverage,
                "technicals_only": self.config.technicals_only,
            },
            coverage_info=coverage_info,
        )

    def _calculate_max_drawdown(self, daily_values: list[dict]) -> float:
        """Calculate maximum drawdown from daily values."""
        if not daily_values:
            return 0.0

        values = [d["value"] for d in daily_values]
        peak = values[0]
        max_dd = 0.0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd * 100

    def _calculate_sharpe_ratio(
        self, daily_values: list[dict], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio from daily values."""
        if len(daily_values) < 2:
            return 0.0

        values = [d["value"] for d in daily_values]
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                r = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(r)

        if not returns:
            return 0.0

        import numpy as np

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)

        return float(sharpe)

    def generate_report(self) -> dict[str, Any]:
        """Generate a detailed performance report."""
        if self.results is None:
            raise RuntimeError("No backtest results available. Run backtest first.")

        return self.results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest Leprechaun trading strategy"
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash (default: 100000)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force synthetic sentiment even if archived data exists",
    )
    parser.add_argument(
        "--technicals-only",
        action="store_true",
        help="Ignore sentiment, use only technical indicators",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    symbols = args.symbols.split(",") if args.symbols else None

    config = BacktestConfig(
        use_synthetic_sentiment=args.use_synthetic or True,
        technicals_only=args.technicals_only,
    )

    backtester = Backtester(config)

    print(f"\nRunning backtest from {start} to {end}")
    print(f"Initial cash: ${args.initial_cash:,.2f}")
    print(f"Symbols: {len(symbols) if symbols else 'all universe'}")
    print(f"Mode: {'Technicals only' if args.technicals_only else 'Sentiment + Technicals'}")
    print("-" * 50)

    try:
        results = backtester.run(
            start=start,
            end=end,
            initial_cash=args.initial_cash,
            symbols=symbols,
            max_positions=args.max_positions,
        )
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        return 1

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {results.start_date} to {results.end_date}")
    print(f"Initial Value: ${results.initial_cash:,.2f}")
    print(f"Final Value:   ${results.final_value:,.2f}")
    print(f"Total Return:  ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)")
    print("-" * 50)
    print(f"Total Trades:  {len(results.trades)}")
    print(f"Win Rate:      {results.win_rate * 100:.1f}%")
    print(f"Avg Win:       ${results.avg_win:,.2f}")
    print(f"Avg Loss:      ${results.avg_loss:,.2f}")
    print(f"Max Drawdown:  {results.max_drawdown:.2f}%")
    print(f"Sharpe Ratio:  {results.sharpe_ratio:.2f}")
    print("=" * 50)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
