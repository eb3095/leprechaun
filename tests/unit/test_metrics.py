"""Unit tests for metrics module."""

import pytest
from prometheus_client import REGISTRY

from src.utils import metrics


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metric values before each test where possible."""
    yield


class TestTradeMetrics:
    def test_record_trade_increments_counter(self):
        initial = metrics.trades_total.labels(action="buy", result="success")._value.get()
        metrics.record_trade("buy", "success")
        assert metrics.trades_total.labels(action="buy", result="success")._value.get() == initial + 1

    def test_record_trade_different_labels(self):
        metrics.record_trade("sell", "failed")
        metrics.record_trade("buy", "partial")
        sell_failed = metrics.trades_total.labels(action="sell", result="failed")._value.get()
        buy_partial = metrics.trades_total.labels(action="buy", result="partial")._value.get()
        assert sell_failed >= 1
        assert buy_partial >= 1


class TestPositionMetrics:
    def test_update_position_value(self):
        metrics.update_position_value("AAPL", 15000.50)
        assert metrics.position_value.labels(symbol="AAPL")._value.get() == 15000.50

    def test_update_position_value_overwrites(self):
        metrics.update_position_value("TSLA", 10000.0)
        metrics.update_position_value("TSLA", 12000.0)
        assert metrics.position_value.labels(symbol="TSLA")._value.get() == 12000.0


class TestClearPosition:
    def test_clear_existing_position(self):
        metrics.update_position_value("NVDA", 5000.0)
        metrics.clear_position("NVDA")
        
    def test_clear_nonexistent_position_no_error(self):
        metrics.clear_position("NONEXISTENT_SYMBOL_XYZ")


class TestPortfolioMetrics:
    def test_update_portfolio(self):
        metrics.update_portfolio(total_value=100000.0, cash=25000.0)
        assert metrics.portfolio_value._value.get() == 100000.0
        assert metrics.cash_balance._value.get() == 25000.0


class TestSentimentMetrics:
    def test_update_manipulation_score(self):
        metrics.update_manipulation_score("GME", 0.75)
        assert metrics.manipulation_score.labels(symbol="GME")._value.get() == 0.75

    def test_update_sentiment_score(self):
        metrics.update_sentiment_score("AMC", -0.5)
        assert metrics.sentiment_score.labels(symbol="AMC")._value.get() == -0.5


class TestTradingStatusMetrics:
    def test_set_trading_halted_true(self):
        metrics.set_trading_halted(True)
        assert metrics.trading_halted._value.get() == 1

    def test_set_trading_halted_false(self):
        metrics.set_trading_halted(False)
        assert metrics.trading_halted._value.get() == 0


class TestPnlMetrics:
    def test_update_pnl(self):
        metrics.update_pnl(
            daily=500.0,
            daily_percent=0.5,
            weekly=1200.0,
            weekly_percent=1.2,
        )
        assert metrics.daily_pnl._value.get() == 500.0
        assert metrics.daily_pnl_percent._value.get() == 0.5
        assert metrics.weekly_pnl._value.get() == 1200.0
        assert metrics.weekly_pnl_percent._value.get() == 1.2

    def test_update_pnl_negative(self):
        metrics.update_pnl(
            daily=-300.0,
            daily_percent=-0.3,
            weekly=-800.0,
            weekly_percent=-0.8,
        )
        assert metrics.daily_pnl._value.get() == -300.0
        assert metrics.weekly_pnl._value.get() == -800.0


class TestSignalMetrics:
    def test_record_signal(self):
        initial = metrics.signals_generated.labels(signal_type="buy", confidence="high")._value.get()
        metrics.record_signal("buy", "high")
        assert (
            metrics.signals_generated.labels(signal_type="buy", confidence="high")._value.get()
            == initial + 1
        )


class TestDataSourceMetrics:
    def test_record_data_source_error(self):
        initial = metrics.data_source_errors.labels(source="reddit")._value.get()
        metrics.record_data_source_error("reddit")
        assert metrics.data_source_errors.labels(source="reddit")._value.get() == initial + 1


class TestAlertMetrics:
    def test_record_alert(self):
        initial = metrics.alerts_sent.labels(channel="discord", alert_type="trade")._value.get()
        metrics.record_alert("discord", "trade")
        assert (
            metrics.alerts_sent.labels(channel="discord", alert_type="trade")._value.get()
            == initial + 1
        )


class TestAppInfo:
    def test_set_app_info(self):
        metrics.set_app_info(version="1.0.0", env="production")
