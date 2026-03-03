"""Unit tests for repository pattern implementations."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

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
from src.data.repository import (
    AlertRepository,
    PortfolioMetricsRepository,
    PositionRepository,
    PriceHistoryRepository,
    StockRepository,
    TradingSignalRepository,
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock()


class TestStockRepository:
    """Test StockRepository."""

    def test_get_by_symbol(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stock = MagicMock(spec=Stock)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        result = repo.get_by_symbol("AAPL")

        assert result == mock_stock
        mock_session.query.assert_called_with(Stock)

    def test_get_by_symbol_case_insensitive(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stock = MagicMock(spec=Stock)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        result = repo.get_by_symbol("aapl")

        assert result == mock_stock

    def test_get_active_stocks(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stocks = [MagicMock(spec=Stock), MagicMock(spec=Stock)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_stocks

        result = repo.get_active_stocks()

        assert result == mock_stocks

    def test_get_by_index(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stocks = [MagicMock(spec=Stock)]
        mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = mock_stocks

        result = repo.get_by_index(IndexMembership.SP500)

        assert result == mock_stocks

    def test_get_by_sector(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stocks = [MagicMock(spec=Stock)]
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_stocks

        result = repo.get_by_sector("Technology")

        assert result == mock_stocks

    def test_get_sectors(self, mock_session):
        repo = StockRepository(mock_session)

        mock_session.query.return_value.filter.return_value.filter.return_value.distinct.return_value.all.return_value = [
            ("Technology",),
            ("Healthcare",),
        ]

        result = repo.get_sectors()

        assert result == ["Technology", "Healthcare"]

    def test_upsert_new_stock(self, mock_session):
        repo = StockRepository(mock_session)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = repo.upsert({
            "symbol": "NEWSTOCK",
            "name": "New Stock Inc",
            "sector": "Technology",
        })

        assert mock_session.add.called
        mock_session.flush.assert_called_once()

    def test_upsert_existing_stock(self, mock_session):
        repo = StockRepository(mock_session)

        existing_stock = MagicMock(spec=Stock)
        mock_session.query.return_value.filter.return_value.first.return_value = existing_stock

        result = repo.upsert({
            "symbol": "AAPL",
            "name": "Apple Inc Updated",
        })

        assert existing_stock.name == "Apple Inc Updated"
        assert not mock_session.add.called

    def test_deactivate(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stock = MagicMock(spec=Stock)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        result = repo.deactivate("AAPL")

        assert result is True
        assert mock_stock.is_active is False

    def test_deactivate_not_found(self, mock_session):
        repo = StockRepository(mock_session)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = repo.deactivate("INVALID")

        assert result is False


class TestPositionRepository:
    """Test PositionRepository."""

    def test_get_open_positions(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_open_positions()

        assert result == mock_positions

    def test_get_pending_positions(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_pending_positions()

        assert result == mock_positions

    def test_get_positions_by_stock(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_positions_by_stock(1)

        assert result == mock_positions

    def test_get_positions_by_stock_with_status(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_positions_by_stock(1, PositionStatus.OPEN)

        assert result == mock_positions

    def test_get_closed_positions(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_closed_positions()

        assert result == mock_positions

    def test_get_closed_positions_with_date_range(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [MagicMock(spec=Position)]
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_positions

        result = repo.get_closed_positions(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert result == mock_positions

    def test_create_position(self, mock_session):
        repo = PositionRepository(mock_session)

        position_data = {
            "stock_id": 1,
            "entry_date": date(2024, 1, 15),
            "entry_price": 150.0,
            "shares": 10,
            "status": PositionStatus.OPEN,
        }

        result = repo.create(position_data)

        assert mock_session.add.called
        mock_session.flush.assert_called_once()

    def test_close_position(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_position = MagicMock(spec=Position)
        mock_position.entry_price = Decimal("150.0")
        mock_position.shares = Decimal("10")

        with patch.object(repo, "get_by_id", return_value=mock_position):
            exit_data = {
                "exit_date": date(2024, 1, 19),
                "exit_price": 155.0,
                "exit_reason": ExitReason.TARGET,
            }

            result = repo.close_position(1, exit_data)

        assert result == mock_position
        assert mock_position.status == PositionStatus.CLOSED
        assert mock_position.exit_price == Decimal("155.0")

    def test_close_position_not_found(self, mock_session):
        repo = PositionRepository(mock_session)

        with patch.object(repo, "get_by_id", return_value=None):
            result = repo.close_position(999, {})

        assert result is None

    def test_get_total_exposure(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_session.query.return_value.filter.return_value.scalar.return_value = Decimal("15000")

        result = repo.get_total_exposure()

        assert result == Decimal("15000")

    def test_get_total_exposure_empty(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_session.query.return_value.filter.return_value.scalar.return_value = None

        result = repo.get_total_exposure()

        assert result == Decimal("0")

    def test_get_performance_summary(self, mock_session):
        repo = PositionRepository(mock_session)

        mock_positions = [
            MagicMock(realized_pnl=Decimal("100"), realized_pnl_percent=Decimal("2.5")),
            MagicMock(realized_pnl=Decimal("-50"), realized_pnl_percent=Decimal("-1.25")),
            MagicMock(realized_pnl=Decimal("75"), realized_pnl_percent=Decimal("1.5")),
        ]

        with patch.object(repo, "get_closed_positions", return_value=mock_positions):
            result = repo.get_performance_summary()

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["total_pnl"] == Decimal("125")

    def test_get_performance_summary_empty(self, mock_session):
        repo = PositionRepository(mock_session)

        with patch.object(repo, "get_closed_positions", return_value=[]):
            result = repo.get_performance_summary()

        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0


class TestTradingSignalRepository:
    """Test TradingSignalRepository."""

    def test_get_latest_signals(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signals = [MagicMock(spec=TradingSignal)]
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = mock_signals

        result = repo.get_latest_signals(limit=10)

        assert result == mock_signals

    def test_get_signals_for_stock(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signals = [MagicMock(spec=TradingSignal)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_signals

        result = repo.get_signals_for_stock(1)

        assert result == mock_signals

    def test_create_signal(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        signal_data = {
            "stock_id": 1,
            "timestamp": datetime.now(),
            "signal_type": SignalType.BUY,
            "confidence": SignalConfidence.HIGH,
            "manipulation_score": 0.75,
            "rsi": 28.5,
        }

        result = repo.create(signal_data)

        assert mock_session.add.called
        mock_session.flush.assert_called_once()

    def test_get_unexecuted_signals(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signals = [MagicMock(spec=TradingSignal)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_signals

        result = repo.get_unexecuted_signals()

        assert result == mock_signals

    def test_get_unexecuted_signals_by_type(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signals = [MagicMock(spec=TradingSignal)]
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_signals

        result = repo.get_unexecuted_signals(signal_type=SignalType.BUY)

        assert result == mock_signals

    def test_mark_executed(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signal = MagicMock(spec=TradingSignal)

        with patch.object(repo, "get_by_id", return_value=mock_signal):
            result = repo.mark_executed(1)

        assert result is True
        assert mock_signal.is_executed is True

    def test_mark_executed_not_found(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        with patch.object(repo, "get_by_id", return_value=None):
            result = repo.mark_executed(999)

        assert result is False

    def test_get_buy_signals_in_range(self, mock_session):
        repo = TradingSignalRepository(mock_session)

        mock_signals = [MagicMock(spec=TradingSignal)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_signals

        result = repo.get_buy_signals_in_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        assert result == mock_signals


class TestPortfolioMetricsRepository:
    """Test PortfolioMetricsRepository."""

    def test_get_latest(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        mock_metric = MagicMock(spec=PortfolioMetric)
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_metric

        result = repo.get_latest()

        assert result == mock_metric

    def test_get_by_date(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        mock_metric = MagicMock(spec=PortfolioMetric)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_metric

        result = repo.get_by_date(date(2024, 1, 15))

        assert result == mock_metric

    def test_get_history(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        mock_metrics = [MagicMock(spec=PortfolioMetric)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_metrics

        result = repo.get_history(days=30)

        assert result == mock_metrics

    def test_create_snapshot_new(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        with patch.object(repo, "get_by_date", return_value=None):
            metrics_data = {
                "date": date(2024, 1, 15),
                "total_value": 100000.0,
                "cash": 50000.0,
                "positions_value": 50000.0,
            }

            result = repo.create_snapshot(metrics_data)

        assert mock_session.add.called

    def test_create_snapshot_update_existing(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        existing_metric = MagicMock(spec=PortfolioMetric)

        with patch.object(repo, "get_by_date", return_value=existing_metric):
            metrics_data = {
                "date": date(2024, 1, 15),
                "total_value": 110000.0,
            }

            result = repo.create_snapshot(metrics_data)

        assert result == existing_metric
        assert not mock_session.add.called

    def test_get_cumulative_pnl(self, mock_session):
        repo = PortfolioMetricsRepository(mock_session)

        mock_metrics = [
            MagicMock(date=date(2024, 1, 1), daily_pnl=Decimal("100")),
            MagicMock(date=date(2024, 1, 2), daily_pnl=Decimal("50")),
            MagicMock(date=date(2024, 1, 3), daily_pnl=Decimal("-25")),
        ]

        with patch.object(repo, "get_history", return_value=mock_metrics):
            result = repo.get_cumulative_pnl(days=30)

        assert len(result) == 3
        assert result[0][1] == Decimal("100")
        assert result[1][1] == Decimal("150")
        assert result[2][1] == Decimal("125")


class TestAlertRepository:
    """Test AlertRepository."""

    def test_get_recent(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alerts = [MagicMock(spec=Alert)]
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = mock_alerts

        result = repo.get_recent(limit=50)

        assert result == mock_alerts

    def test_get_by_type(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alerts = [MagicMock(spec=Alert)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_alerts

        result = repo.get_by_type(AlertType.TRADE)

        assert result == mock_alerts

    def test_get_failed_alerts(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alerts = [MagicMock(spec=Alert)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_alerts

        result = repo.get_failed_alerts()

        assert result == mock_alerts

    def test_create_alert(self, mock_session):
        repo = AlertRepository(mock_session)

        alert_data = {
            "timestamp": datetime.now(),
            "alert_type": AlertType.TRADE,
            "channel": AlertChannel.DISCORD,
            "title": "Test Alert",
            "message": "Test message",
        }

        result = repo.create(alert_data)

        assert mock_session.add.called

    def test_mark_sent_success(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alert = MagicMock(spec=Alert)

        with patch.object(repo, "get_by_id", return_value=mock_alert):
            result = repo.mark_sent(1, True)

        assert result is True
        assert mock_alert.sent_successfully is True

    def test_mark_sent_failure(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alert = MagicMock(spec=Alert)

        with patch.object(repo, "get_by_id", return_value=mock_alert):
            result = repo.mark_sent(1, False, "Network error")

        assert result is True
        assert mock_alert.sent_successfully is False
        assert mock_alert.error_message == "Network error"

    def test_mark_sent_not_found(self, mock_session):
        repo = AlertRepository(mock_session)

        with patch.object(repo, "get_by_id", return_value=None):
            result = repo.mark_sent(999, True)

        assert result is False

    def test_get_alerts_in_range(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_alerts = [MagicMock(spec=Alert)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_alerts

        result = repo.get_alerts_in_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        assert result == mock_alerts

    def test_count_by_type_today(self, mock_session):
        repo = AlertRepository(mock_session)

        mock_session.query.return_value.filter.return_value.group_by.return_value.all.return_value = [
            (AlertType.TRADE, 5),
            (AlertType.SIGNAL, 10),
        ]

        result = repo.count_by_type_today()

        assert result[AlertType.TRADE] == 5
        assert result[AlertType.SIGNAL] == 10


class TestPriceHistoryRepository:
    """Test PriceHistoryRepository."""

    def test_get_for_stock(self, mock_session):
        repo = PriceHistoryRepository(mock_session)

        mock_history = [MagicMock(spec=PriceHistory)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_history

        result = repo.get_for_stock(1)

        assert result == mock_history

    def test_get_for_stock_with_date_range(self, mock_session):
        repo = PriceHistoryRepository(mock_session)

        mock_history = [MagicMock(spec=PriceHistory)]
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_history

        result = repo.get_for_stock(
            1,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert result == mock_history

    def test_get_latest_price(self, mock_session):
        repo = PriceHistoryRepository(mock_session)

        mock_price = MagicMock(spec=PriceHistory)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_price

        result = repo.get_latest_price(1)

        assert result == mock_price

    def test_bulk_insert(self, mock_session):
        repo = PriceHistoryRepository(mock_session)

        records = [
            {
                "stock_id": 1,
                "date": date(2024, 1, 1),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 1000000,
            },
            {
                "stock_id": 1,
                "date": date(2024, 1, 2),
                "open": 104.0,
                "high": 106.0,
                "low": 103.0,
                "close": 105.0,
                "volume": 1100000,
            },
        ]

        count = repo.bulk_insert(records)

        assert count == 2
        assert mock_session.add.call_count == 2

    def test_delete_for_stock(self, mock_session):
        repo = PriceHistoryRepository(mock_session)

        mock_session.query.return_value.filter.return_value.delete.return_value = 100

        count = repo.delete_for_stock(1)

        assert count == 100


class TestBaseRepository:
    """Test BaseRepository generic methods."""

    def test_get_by_id(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stock = MagicMock(spec=Stock)
        mock_session.query.return_value.get.return_value = mock_stock

        result = repo.get_by_id(1)

        assert result == mock_stock

    def test_get_all(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stocks = [MagicMock(spec=Stock)]
        mock_session.query.return_value.all.return_value = mock_stocks

        result = repo.get_all()

        assert result == mock_stocks

    def test_get_all_with_limit(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stocks = [MagicMock(spec=Stock)]
        mock_session.query.return_value.limit.return_value.all.return_value = mock_stocks

        result = repo.get_all(limit=10)

        assert result == mock_stocks

    def test_delete(self, mock_session):
        repo = StockRepository(mock_session)

        mock_stock = MagicMock(spec=Stock)
        repo.delete(mock_stock)

        mock_session.delete.assert_called_once_with(mock_stock)
        mock_session.flush.assert_called_once()

    def test_count(self, mock_session):
        repo = StockRepository(mock_session)

        mock_session.query.return_value.scalar.return_value = 100

        result = repo.count()

        assert result == 100
