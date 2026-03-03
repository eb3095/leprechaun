"""Repository pattern for database access in Leprechaun trading bot."""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, TypeVar, Generic

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from src.data.models import (
    Alert,
    AlertChannel,
    AlertType,
    ExitReason,
    IndexMembership,
    PortfolioMetric,
    Position,
    PositionStatus,
    PriceHistory,
    SignalConfidence,
    SignalType,
    Stock,
    TradingSignal,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    model: type[T]

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy database session.
        """
        self.session = session

    def get_by_id(self, id: int) -> Optional[T]:
        """Get entity by primary key.

        Args:
            id: Primary key value.

        Returns:
            Entity instance or None if not found.
        """
        return self.session.query(self.model).get(id)

    def get_all(self, limit: Optional[int] = None) -> list[T]:
        """Get all entities with optional limit.

        Args:
            limit: Maximum number of entities to return.

        Returns:
            List of entity instances.
        """
        query = self.session.query(self.model)
        if limit:
            query = query.limit(limit)
        return query.all()

    def delete(self, entity: T) -> None:
        """Delete an entity.

        Args:
            entity: Entity instance to delete.
        """
        self.session.delete(entity)
        self.session.flush()

    def count(self) -> int:
        """Count total entities.

        Returns:
            Total count of entities.
        """
        return self.session.query(func.count(self.model.id)).scalar()


class StockRepository(BaseRepository[Stock]):
    """Repository for Stock entities."""

    model = Stock

    def get_by_symbol(self, symbol: str) -> Optional[Stock]:
        """Get stock by ticker symbol.

        Args:
            symbol: Stock ticker symbol (case-insensitive).

        Returns:
            Stock instance or None if not found.
        """
        return (
            self.session.query(Stock)
            .filter(Stock.symbol == symbol.upper())
            .first()
        )

    def get_active_stocks(self) -> list[Stock]:
        """Get all active stocks in the trading universe.

        Returns:
            List of active Stock instances.
        """
        return (
            self.session.query(Stock)
            .filter(Stock.is_active.is_(True))
            .order_by(Stock.symbol)
            .all()
        )

    def get_by_index(self, index: IndexMembership) -> list[Stock]:
        """Get stocks by index membership.

        Args:
            index: Index membership (SP500, NASDAQ100, or BOTH).

        Returns:
            List of Stock instances in the specified index.
        """
        if index == IndexMembership.BOTH:
            return (
                self.session.query(Stock)
                .filter(Stock.index_membership == IndexMembership.BOTH)
                .filter(Stock.is_active.is_(True))
                .all()
            )
        else:
            return (
                self.session.query(Stock)
                .filter(
                    Stock.index_membership.in_([index, IndexMembership.BOTH])
                )
                .filter(Stock.is_active.is_(True))
                .all()
            )

    def get_by_sector(self, sector: str) -> list[Stock]:
        """Get active stocks by sector.

        Args:
            sector: Sector name.

        Returns:
            List of Stock instances in the specified sector.
        """
        return (
            self.session.query(Stock)
            .filter(Stock.sector == sector)
            .filter(Stock.is_active.is_(True))
            .order_by(Stock.symbol)
            .all()
        )

    def get_sectors(self) -> list[str]:
        """Get list of unique sectors.

        Returns:
            List of sector names.
        """
        result = (
            self.session.query(Stock.sector)
            .filter(Stock.sector.isnot(None))
            .filter(Stock.is_active.is_(True))
            .distinct()
            .all()
        )
        return [r[0] for r in result]

    def upsert(self, stock_data: dict) -> Stock:
        """Insert or update a stock record.

        Args:
            stock_data: Dictionary with stock fields.
                Required: symbol
                Optional: name, sector, market_cap, index_membership, is_active

        Returns:
            Created or updated Stock instance.
        """
        symbol = stock_data["symbol"].upper()
        stock = self.get_by_symbol(symbol)

        if stock is None:
            stock = Stock(symbol=symbol)
            self.session.add(stock)

        if "name" in stock_data:
            stock.name = stock_data["name"]
        if "sector" in stock_data:
            stock.sector = stock_data["sector"]
        if "market_cap" in stock_data:
            stock.market_cap = stock_data["market_cap"]
        if "index_membership" in stock_data:
            stock.index_membership = stock_data["index_membership"]
        if "is_active" in stock_data:
            stock.is_active = stock_data["is_active"]

        self.session.flush()
        return stock

    def deactivate(self, symbol: str) -> bool:
        """Mark a stock as inactive.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            True if stock was deactivated, False if not found.
        """
        stock = self.get_by_symbol(symbol)
        if stock is None:
            return False
        stock.is_active = False
        self.session.flush()
        return True


class PositionRepository(BaseRepository[Position]):
    """Repository for Position entities."""

    model = Position

    def get_open_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of open Position instances.
        """
        return (
            self.session.query(Position)
            .filter(Position.status == PositionStatus.OPEN)
            .order_by(Position.entry_date)
            .all()
        )

    def get_pending_positions(self) -> list[Position]:
        """Get all pending positions.

        Returns:
            List of pending Position instances.
        """
        return (
            self.session.query(Position)
            .filter(Position.status == PositionStatus.PENDING)
            .order_by(Position.entry_date)
            .all()
        )

    def get_positions_by_stock(
        self, stock_id: int, status: Optional[PositionStatus] = None
    ) -> list[Position]:
        """Get positions for a specific stock.

        Args:
            stock_id: Stock ID.
            status: Optional filter by position status.

        Returns:
            List of Position instances.
        """
        query = self.session.query(Position).filter(Position.stock_id == stock_id)
        if status:
            query = query.filter(Position.status == status)
        return query.order_by(desc(Position.entry_date)).all()

    def get_closed_positions(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> list[Position]:
        """Get closed positions with optional date filter.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of closed Position instances.
        """
        query = self.session.query(Position).filter(
            Position.status == PositionStatus.CLOSED
        )

        if start_date:
            query = query.filter(Position.exit_date >= start_date)
        if end_date:
            query = query.filter(Position.exit_date <= end_date)

        return query.order_by(desc(Position.exit_date)).all()

    def create(self, position_data: dict) -> Position:
        """Create a new position.

        Args:
            position_data: Dictionary with position fields.
                Required: stock_id, entry_date, entry_price, shares, status
                Optional: alpaca_order_id, entry_signal_id

        Returns:
            Created Position instance.
        """
        position = Position(
            stock_id=position_data["stock_id"],
            entry_date=position_data["entry_date"],
            entry_price=Decimal(str(position_data["entry_price"])),
            shares=Decimal(str(position_data["shares"])),
            status=position_data.get("status", PositionStatus.OPEN),
            alpaca_order_id=position_data.get("alpaca_order_id"),
            entry_signal_id=position_data.get("entry_signal_id"),
        )
        self.session.add(position)
        self.session.flush()
        return position

    def close_position(self, position_id: int, exit_data: dict) -> Optional[Position]:
        """Close an open position.

        Args:
            position_id: Position ID to close.
            exit_data: Dictionary with exit details.
                Required: exit_date, exit_price, exit_reason
                Optional: exit_signal_id

        Returns:
            Updated Position instance or None if not found.
        """
        position = self.get_by_id(position_id)
        if position is None:
            return None

        exit_price = Decimal(str(exit_data["exit_price"]))
        position.exit_date = exit_data["exit_date"]
        position.exit_price = exit_price
        position.exit_reason = exit_data["exit_reason"]
        position.exit_signal_id = exit_data.get("exit_signal_id")
        position.status = PositionStatus.CLOSED

        pnl = (exit_price - position.entry_price) * position.shares
        pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100
        position.realized_pnl = pnl
        position.realized_pnl_percent = pnl_percent

        self.session.flush()
        return position

    def get_total_exposure(self) -> Decimal:
        """Calculate total open position value.

        Returns:
            Sum of (entry_price * shares) for all open positions.
        """
        result = (
            self.session.query(
                func.sum(Position.entry_price * Position.shares)
            )
            .filter(Position.status == PositionStatus.OPEN)
            .scalar()
        )
        return result or Decimal("0")

    def get_performance_summary(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> dict[str, Any]:
        """Get performance summary for closed positions.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Dictionary with performance metrics.
        """
        positions = self.get_closed_positions(start_date, end_date)

        if not positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": Decimal("0"),
                "average_pnl": Decimal("0"),
                "average_pnl_percent": Decimal("0"),
            }

        wins = [p for p in positions if p.realized_pnl and p.realized_pnl > 0]
        losses = [p for p in positions if p.realized_pnl and p.realized_pnl <= 0]
        total_pnl = sum(p.realized_pnl or 0 for p in positions)
        avg_pnl = total_pnl / len(positions)
        avg_pnl_pct = sum(p.realized_pnl_percent or 0 for p in positions) / len(positions)

        return {
            "total_trades": len(positions),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(positions) if positions else 0.0,
            "total_pnl": total_pnl,
            "average_pnl": avg_pnl,
            "average_pnl_percent": avg_pnl_pct,
        }


class TradingSignalRepository(BaseRepository[TradingSignal]):
    """Repository for TradingSignal entities."""

    model = TradingSignal

    def get_latest_signals(self, limit: int = 100) -> list[TradingSignal]:
        """Get most recent trading signals.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List of TradingSignal instances ordered by timestamp desc.
        """
        return (
            self.session.query(TradingSignal)
            .order_by(desc(TradingSignal.timestamp))
            .limit(limit)
            .all()
        )

    def get_signals_for_stock(
        self, stock_id: int, limit: int = 50
    ) -> list[TradingSignal]:
        """Get recent signals for a specific stock.

        Args:
            stock_id: Stock ID.
            limit: Maximum number of signals to return.

        Returns:
            List of TradingSignal instances.
        """
        return (
            self.session.query(TradingSignal)
            .filter(TradingSignal.stock_id == stock_id)
            .order_by(desc(TradingSignal.timestamp))
            .limit(limit)
            .all()
        )

    def create(self, signal_data: dict) -> TradingSignal:
        """Create a new trading signal.

        Args:
            signal_data: Dictionary with signal fields.
                Required: stock_id, timestamp, signal_type, confidence
                Optional: manipulation_score, rsi, sentiment_score, triggered_rules

        Returns:
            Created TradingSignal instance.
        """
        signal = TradingSignal(
            stock_id=signal_data["stock_id"],
            timestamp=signal_data["timestamp"],
            signal_type=signal_data["signal_type"],
            confidence=signal_data["confidence"],
            manipulation_score=(
                Decimal(str(signal_data["manipulation_score"]))
                if signal_data.get("manipulation_score") is not None
                else None
            ),
            rsi=(
                Decimal(str(signal_data["rsi"]))
                if signal_data.get("rsi") is not None
                else None
            ),
            sentiment_score=(
                Decimal(str(signal_data["sentiment_score"]))
                if signal_data.get("sentiment_score") is not None
                else None
            ),
            triggered_rules=signal_data.get("triggered_rules"),
            is_executed=signal_data.get("is_executed", False),
        )
        self.session.add(signal)
        self.session.flush()
        return signal

    def get_unexecuted_signals(
        self, signal_type: Optional[SignalType] = None
    ) -> list[TradingSignal]:
        """Get signals that haven't been executed yet.

        Args:
            signal_type: Optional filter by signal type.

        Returns:
            List of unexecuted TradingSignal instances.
        """
        query = self.session.query(TradingSignal).filter(
            TradingSignal.is_executed.is_(False)
        )
        if signal_type:
            query = query.filter(TradingSignal.signal_type == signal_type)
        return query.order_by(TradingSignal.timestamp).all()

    def mark_executed(self, signal_id: int) -> bool:
        """Mark a signal as executed.

        Args:
            signal_id: Signal ID.

        Returns:
            True if signal was updated, False if not found.
        """
        signal = self.get_by_id(signal_id)
        if signal is None:
            return False
        signal.is_executed = True
        self.session.flush()
        return True

    def get_buy_signals_in_range(
        self, start: datetime, end: datetime
    ) -> list[TradingSignal]:
        """Get buy signals within a time range.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of BUY TradingSignal instances.
        """
        return (
            self.session.query(TradingSignal)
            .filter(
                TradingSignal.signal_type == SignalType.BUY,
                TradingSignal.timestamp >= start,
                TradingSignal.timestamp <= end,
            )
            .order_by(desc(TradingSignal.confidence))
            .all()
        )


class PortfolioMetricsRepository(BaseRepository[PortfolioMetric]):
    """Repository for PortfolioMetric entities."""

    model = PortfolioMetric

    def get_latest(self) -> Optional[PortfolioMetric]:
        """Get most recent portfolio metrics.

        Returns:
            Latest PortfolioMetric instance or None if no data.
        """
        return (
            self.session.query(PortfolioMetric)
            .order_by(desc(PortfolioMetric.date))
            .first()
        )

    def get_by_date(self, target_date: date) -> Optional[PortfolioMetric]:
        """Get metrics for a specific date.

        Args:
            target_date: Date to look up.

        Returns:
            PortfolioMetric instance or None if not found.
        """
        return (
            self.session.query(PortfolioMetric)
            .filter(PortfolioMetric.date == target_date)
            .first()
        )

    def get_history(self, days: int = 30) -> list[PortfolioMetric]:
        """Get historical portfolio metrics.

        Args:
            days: Number of days of history to retrieve.

        Returns:
            List of PortfolioMetric instances ordered by date asc.
        """
        cutoff = date.today() - timedelta(days=days)
        return (
            self.session.query(PortfolioMetric)
            .filter(PortfolioMetric.date >= cutoff)
            .order_by(PortfolioMetric.date)
            .all()
        )

    def create_snapshot(self, metrics_data: dict) -> PortfolioMetric:
        """Create a new portfolio metrics snapshot.

        If a snapshot for the date already exists, it will be updated.

        Args:
            metrics_data: Dictionary with metrics fields.
                Required: date
                Optional: All other PortfolioMetric fields

        Returns:
            Created or updated PortfolioMetric instance.
        """
        target_date = metrics_data["date"]
        existing = self.get_by_date(target_date)

        if existing:
            for key, value in metrics_data.items():
                if key != "date" and hasattr(existing, key):
                    if isinstance(value, (int, float)) and key not in ("win_count", "loss_count"):
                        setattr(existing, key, Decimal(str(value)))
                    else:
                        setattr(existing, key, value)
            self.session.flush()
            return existing

        metric = PortfolioMetric(
            date=target_date,
            total_value=(
                Decimal(str(metrics_data["total_value"]))
                if metrics_data.get("total_value") is not None
                else None
            ),
            cash=(
                Decimal(str(metrics_data["cash"]))
                if metrics_data.get("cash") is not None
                else None
            ),
            positions_value=(
                Decimal(str(metrics_data["positions_value"]))
                if metrics_data.get("positions_value") is not None
                else None
            ),
            daily_pnl=(
                Decimal(str(metrics_data["daily_pnl"]))
                if metrics_data.get("daily_pnl") is not None
                else None
            ),
            daily_pnl_percent=(
                Decimal(str(metrics_data["daily_pnl_percent"]))
                if metrics_data.get("daily_pnl_percent") is not None
                else None
            ),
            weekly_pnl=(
                Decimal(str(metrics_data["weekly_pnl"]))
                if metrics_data.get("weekly_pnl") is not None
                else None
            ),
            weekly_pnl_percent=(
                Decimal(str(metrics_data["weekly_pnl_percent"]))
                if metrics_data.get("weekly_pnl_percent") is not None
                else None
            ),
            total_pnl=(
                Decimal(str(metrics_data["total_pnl"]))
                if metrics_data.get("total_pnl") is not None
                else None
            ),
            total_pnl_percent=(
                Decimal(str(metrics_data["total_pnl_percent"]))
                if metrics_data.get("total_pnl_percent") is not None
                else None
            ),
            win_count=metrics_data.get("win_count"),
            loss_count=metrics_data.get("loss_count"),
            win_rate=(
                Decimal(str(metrics_data["win_rate"]))
                if metrics_data.get("win_rate") is not None
                else None
            ),
            sharpe_ratio=(
                Decimal(str(metrics_data["sharpe_ratio"]))
                if metrics_data.get("sharpe_ratio") is not None
                else None
            ),
            max_drawdown=(
                Decimal(str(metrics_data["max_drawdown"]))
                if metrics_data.get("max_drawdown") is not None
                else None
            ),
        )
        self.session.add(metric)
        self.session.flush()
        return metric

    def get_cumulative_pnl(self, days: int = 30) -> list[tuple[date, Decimal]]:
        """Get cumulative P&L over time.

        Args:
            days: Number of days of history.

        Returns:
            List of (date, cumulative_pnl) tuples.
        """
        history = self.get_history(days)
        result = []
        cumulative = Decimal("0")

        for metric in history:
            if metric.daily_pnl:
                cumulative += metric.daily_pnl
            result.append((metric.date, cumulative))

        return result


class AlertRepository(BaseRepository[Alert]):
    """Repository for Alert entities."""

    model = Alert

    def get_recent(self, limit: int = 50) -> list[Alert]:
        """Get most recent alerts.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of Alert instances ordered by timestamp desc.
        """
        return (
            self.session.query(Alert)
            .order_by(desc(Alert.timestamp))
            .limit(limit)
            .all()
        )

    def get_by_type(
        self, alert_type: AlertType, limit: int = 50
    ) -> list[Alert]:
        """Get alerts of a specific type.

        Args:
            alert_type: Type of alert to filter by.
            limit: Maximum number of alerts to return.

        Returns:
            List of Alert instances.
        """
        return (
            self.session.query(Alert)
            .filter(Alert.alert_type == alert_type)
            .order_by(desc(Alert.timestamp))
            .limit(limit)
            .all()
        )

    def get_failed_alerts(self, limit: int = 50) -> list[Alert]:
        """Get alerts that failed to send.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of failed Alert instances.
        """
        return (
            self.session.query(Alert)
            .filter(Alert.sent_successfully.is_(False))
            .order_by(desc(Alert.timestamp))
            .limit(limit)
            .all()
        )

    def create(self, alert_data: dict) -> Alert:
        """Create a new alert record.

        Args:
            alert_data: Dictionary with alert fields.
                Required: timestamp, alert_type, channel
                Optional: title, message, metadata, sent_successfully, error_message

        Returns:
            Created Alert instance.
        """
        alert = Alert(
            timestamp=alert_data["timestamp"],
            alert_type=alert_data["alert_type"],
            channel=alert_data["channel"],
            title=alert_data.get("title"),
            message=alert_data.get("message"),
            alert_metadata=alert_data.get("metadata"),
            sent_successfully=alert_data.get("sent_successfully"),
            error_message=alert_data.get("error_message"),
        )
        self.session.add(alert)
        self.session.flush()
        return alert

    def mark_sent(
        self, alert_id: int, success: bool, error_message: Optional[str] = None
    ) -> bool:
        """Update alert send status.

        Args:
            alert_id: Alert ID.
            success: Whether the send was successful.
            error_message: Optional error message if send failed.

        Returns:
            True if alert was updated, False if not found.
        """
        alert = self.get_by_id(alert_id)
        if alert is None:
            return False

        alert.sent_successfully = success
        alert.error_message = error_message
        self.session.flush()
        return True

    def get_alerts_in_range(
        self, start: datetime, end: datetime
    ) -> list[Alert]:
        """Get alerts within a time range.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of Alert instances.
        """
        return (
            self.session.query(Alert)
            .filter(
                Alert.timestamp >= start,
                Alert.timestamp <= end,
            )
            .order_by(Alert.timestamp)
            .all()
        )

    def count_by_type_today(self) -> dict[AlertType, int]:
        """Count alerts by type for today.

        Returns:
            Dictionary mapping AlertType to count.
        """
        today_start = datetime.combine(date.today(), datetime.min.time())

        result = (
            self.session.query(
                Alert.alert_type, func.count(Alert.id)
            )
            .filter(Alert.timestamp >= today_start)
            .group_by(Alert.alert_type)
            .all()
        )

        return {alert_type: count for alert_type, count in result}


class PriceHistoryRepository(BaseRepository[PriceHistory]):
    """Repository for PriceHistory entities."""

    model = PriceHistory

    def get_for_stock(
        self,
        stock_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> list[PriceHistory]:
        """Get price history for a stock.

        Args:
            stock_id: Stock ID.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            limit: Optional limit on results.

        Returns:
            List of PriceHistory instances ordered by date.
        """
        query = self.session.query(PriceHistory).filter(
            PriceHistory.stock_id == stock_id
        )

        if start_date:
            query = query.filter(PriceHistory.date >= start_date)
        if end_date:
            query = query.filter(PriceHistory.date <= end_date)

        query = query.order_by(PriceHistory.date)

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_latest_price(self, stock_id: int) -> Optional[PriceHistory]:
        """Get most recent price record for a stock.

        Args:
            stock_id: Stock ID.

        Returns:
            Most recent PriceHistory instance or None.
        """
        return (
            self.session.query(PriceHistory)
            .filter(PriceHistory.stock_id == stock_id)
            .order_by(desc(PriceHistory.date))
            .first()
        )

    def bulk_insert(self, records: list[dict]) -> int:
        """Bulk insert price history records.

        Args:
            records: List of dictionaries with price history data.
                Required: stock_id, date
                Optional: open, high, low, close, adj_close, volume

        Returns:
            Number of records inserted.
        """
        count = 0
        for record in records:
            price = PriceHistory(
                stock_id=record["stock_id"],
                date=record["date"],
                open=Decimal(str(record["open"])) if record.get("open") else None,
                high=Decimal(str(record["high"])) if record.get("high") else None,
                low=Decimal(str(record["low"])) if record.get("low") else None,
                close=Decimal(str(record["close"])) if record.get("close") else None,
                adj_close=Decimal(str(record["adj_close"])) if record.get("adj_close") else None,
                volume=record.get("volume"),
            )
            self.session.add(price)
            count += 1

        self.session.flush()
        return count

    def delete_for_stock(self, stock_id: int) -> int:
        """Delete all price history for a stock.

        Args:
            stock_id: Stock ID.

        Returns:
            Number of records deleted.
        """
        count = (
            self.session.query(PriceHistory)
            .filter(PriceHistory.stock_id == stock_id)
            .delete()
        )
        self.session.flush()
        return count
