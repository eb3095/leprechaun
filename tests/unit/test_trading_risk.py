"""Unit tests for risk management module."""

import pytest

from src.core.trading.risk import (
    HaltDecision,
    RiskManager,
    ValidationResult,
)
from src.utils.config import TradingConfig


@pytest.fixture
def default_risk_manager():
    """Create risk manager with default configuration."""
    return RiskManager()


@pytest.fixture
def custom_risk_manager():
    """Create risk manager with custom configuration."""
    config = TradingConfig(
        profit_target_percent=3.0,
        stop_loss_percent=2.0,
        position_risk_percent=2.0,
        daily_loss_limit_percent=3.0,
        weekly_loss_limit_percent=7.0,
    )
    return RiskManager(config)


class TestRiskManagerInit:
    def test_default_config_values(self, default_risk_manager):
        assert default_risk_manager.position_risk_pct == 0.01
        assert default_risk_manager.daily_loss_limit == 0.02
        assert default_risk_manager.weekly_loss_limit == 0.05
        assert default_risk_manager.stop_loss_pct == 0.0125
        assert default_risk_manager.profit_target_pct == 0.025

    def test_custom_config_values(self, custom_risk_manager):
        assert custom_risk_manager.position_risk_pct == 0.02
        assert custom_risk_manager.daily_loss_limit == 0.03
        assert custom_risk_manager.weekly_loss_limit == 0.07
        assert custom_risk_manager.stop_loss_pct == 0.02
        assert custom_risk_manager.profit_target_pct == 0.03


class TestCalculatePositionSize:
    def test_basic_position_sizing(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=100000,
            entry_price=100,
        )
        assert shares > 0
        assert shares <= 100

    def test_respects_max_position_limit(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=100000,
            entry_price=10,
        )
        max_value = 100000 * 0.10
        assert shares * 10 <= max_value

    def test_custom_stop_loss_price(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=100000,
            entry_price=100,
            stop_loss_price=95,
        )
        max_risk = 100000 * 0.01
        risk_per_share = 5
        expected_shares = int(max_risk / risk_per_share)
        assert shares <= expected_shares

    def test_zero_account_value(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=0,
            entry_price=100,
        )
        assert shares == 0

    def test_zero_entry_price(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=100000,
            entry_price=0,
        )
        assert shares == 0

    def test_negative_account_value(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=-10000,
            entry_price=100,
        )
        assert shares == 0

    def test_stop_loss_above_entry(self, default_risk_manager):
        shares = default_risk_manager.calculate_position_size(
            account_value=100000,
            entry_price=100,
            stop_loss_price=105,
        )
        assert shares == 0


class TestCheckDailyLossLimit:
    def test_loss_within_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_daily_loss_limit(
            daily_pnl=-1000,
            account_value=100000,
        )
        assert exceeded is False

    def test_loss_exceeds_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_daily_loss_limit(
            daily_pnl=-2500,
            account_value=100000,
        )
        assert exceeded is True

    def test_loss_at_exact_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_daily_loss_limit(
            daily_pnl=-2000,
            account_value=100000,
        )
        assert exceeded is True

    def test_positive_pnl(self, default_risk_manager):
        exceeded = default_risk_manager.check_daily_loss_limit(
            daily_pnl=5000,
            account_value=100000,
        )
        assert exceeded is False

    def test_zero_account_value(self, default_risk_manager):
        exceeded = default_risk_manager.check_daily_loss_limit(
            daily_pnl=-100,
            account_value=0,
        )
        assert exceeded is True


class TestCheckWeeklyLossLimit:
    def test_loss_within_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_weekly_loss_limit(
            weekly_pnl=-3000,
            account_value=100000,
        )
        assert exceeded is False

    def test_loss_exceeds_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_weekly_loss_limit(
            weekly_pnl=-6000,
            account_value=100000,
        )
        assert exceeded is True

    def test_loss_at_exact_limit(self, default_risk_manager):
        exceeded = default_risk_manager.check_weekly_loss_limit(
            weekly_pnl=-5000,
            account_value=100000,
        )
        assert exceeded is True


class TestShouldHaltTrading:
    def test_no_halt_normal_conditions(self, default_risk_manager):
        metrics = {
            "daily_pnl": -500,
            "weekly_pnl": -2000,
            "account_value": 100000,
            "start_of_day_value": 100000,
            "start_of_week_value": 100000,
            "consecutive_losses": 2,
            "error_count": 0,
        }
        decision = default_risk_manager.should_halt_trading(metrics)
        assert decision.should_halt is False

    def test_halt_on_daily_loss(self, default_risk_manager):
        metrics = {
            "daily_pnl": -2500,
            "weekly_pnl": -2500,
            "account_value": 97500,
            "start_of_day_value": 100000,
            "start_of_week_value": 100000,
        }
        decision = default_risk_manager.should_halt_trading(metrics)
        assert decision.should_halt is True
        assert "daily" in decision.reason.lower()

    def test_halt_on_weekly_loss(self, default_risk_manager):
        metrics = {
            "daily_pnl": -1000,
            "weekly_pnl": -6000,
            "account_value": 94000,
            "start_of_day_value": 95000,
            "start_of_week_value": 100000,
        }
        decision = default_risk_manager.should_halt_trading(metrics)
        assert decision.should_halt is True
        assert "weekly" in decision.reason.lower()

    def test_halt_on_consecutive_losses(self, default_risk_manager):
        metrics = {
            "daily_pnl": -500,
            "weekly_pnl": -2000,
            "account_value": 98000,
            "start_of_day_value": 100000,
            "start_of_week_value": 100000,
            "consecutive_losses": 5,
        }
        decision = default_risk_manager.should_halt_trading(metrics)
        assert decision.should_halt is True
        assert "consecutive" in decision.reason.lower()

    def test_halt_on_errors(self, default_risk_manager):
        metrics = {
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "account_value": 100000,
            "start_of_day_value": 100000,
            "start_of_week_value": 100000,
            "error_count": 10,
        }
        decision = default_risk_manager.should_halt_trading(metrics)
        assert decision.should_halt is True
        assert "error" in decision.reason.lower()

    def test_halt_decision_to_dict(self, default_risk_manager):
        metrics = {"daily_pnl": -2500, "start_of_day_value": 100000}
        decision = default_risk_manager.should_halt_trading(metrics)
        d = decision.to_dict()
        assert "should_halt" in d
        assert "reason" in d


class TestCalculateStopLossPrice:
    def test_stop_loss_calculation(self, default_risk_manager):
        stop = default_risk_manager.calculate_stop_loss_price(100)
        assert stop == 98.75

    def test_custom_stop_loss(self, custom_risk_manager):
        stop = custom_risk_manager.calculate_stop_loss_price(100)
        assert stop == 98.0


class TestCalculateProfitTargetPrice:
    def test_profit_target_calculation(self, default_risk_manager):
        target = default_risk_manager.calculate_profit_target_price(100)
        assert abs(target - 102.5) < 0.0001

    def test_custom_profit_target(self, custom_risk_manager):
        target = custom_risk_manager.calculate_profit_target_price(100)
        assert abs(target - 103.0) < 0.0001


class TestValidateOrder:
    @pytest.fixture
    def base_account(self):
        return {"buying_power": 50000, "equity": 100000}

    @pytest.fixture
    def base_positions(self):
        return [{"symbol": "AAPL", "qty": 50}]

    def test_valid_buy_order(self, default_risk_manager, base_account):
        order = {"symbol": "MSFT", "side": "buy", "qty": 10, "price": 300}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is True

    def test_insufficient_buying_power(self, default_risk_manager, base_account):
        order = {"symbol": "MSFT", "side": "buy", "qty": 200, "price": 300}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "buying power" in result.reason.lower()

    def test_position_too_large(self, default_risk_manager, base_account):
        order = {"symbol": "MSFT", "side": "buy", "qty": 50, "price": 300}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "too large" in result.reason.lower()

    def test_duplicate_position(self, default_risk_manager, base_account, base_positions):
        order = {"symbol": "AAPL", "side": "buy", "qty": 10, "price": 150}
        result = default_risk_manager.validate_order(order, base_account, base_positions)
        assert result.is_valid is False
        assert "already" in result.reason.lower()

    def test_valid_sell_order(self, default_risk_manager, base_account, base_positions):
        order = {"symbol": "AAPL", "side": "sell", "qty": 25, "price": 150}
        result = default_risk_manager.validate_order(order, base_account, base_positions)
        assert result.is_valid is True

    def test_sell_no_position(self, default_risk_manager, base_account):
        order = {"symbol": "GOOG", "side": "sell", "qty": 10, "price": 100}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "no position" in result.reason.lower()

    def test_sell_exceeds_position(self, default_risk_manager, base_account, base_positions):
        order = {"symbol": "AAPL", "side": "sell", "qty": 100, "price": 150}
        result = default_risk_manager.validate_order(order, base_account, base_positions)
        assert result.is_valid is False
        assert "exceeds" in result.reason.lower()

    def test_missing_symbol(self, default_risk_manager, base_account):
        order = {"side": "buy", "qty": 10, "price": 100}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "symbol" in result.reason.lower()

    def test_invalid_side(self, default_risk_manager, base_account):
        order = {"symbol": "AAPL", "side": "hold", "qty": 10, "price": 100}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "side" in result.reason.lower()

    def test_zero_quantity(self, default_risk_manager, base_account):
        order = {"symbol": "AAPL", "side": "buy", "qty": 0, "price": 100}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "quantity" in result.reason.lower()

    def test_zero_price(self, default_risk_manager, base_account):
        order = {"symbol": "AAPL", "side": "buy", "qty": 10, "price": 0}
        result = default_risk_manager.validate_order(order, base_account, [])
        assert result.is_valid is False
        assert "price" in result.reason.lower()

    def test_validation_result_to_dict(self, default_risk_manager, base_account):
        order = {"symbol": "MSFT", "side": "buy", "qty": 10, "price": 300}
        result = default_risk_manager.validate_order(order, base_account, [])
        d = result.to_dict()
        assert "is_valid" in d
        assert "reason" in d


class TestCalculateMaxPositions:
    def test_normal_account(self, default_risk_manager):
        max_pos = default_risk_manager.calculate_max_positions(100000, 5000)
        assert 5 <= max_pos <= 10

    def test_small_account(self, default_risk_manager):
        max_pos = default_risk_manager.calculate_max_positions(10000, 2000)
        assert max_pos >= 5

    def test_zero_account(self, default_risk_manager):
        max_pos = default_risk_manager.calculate_max_positions(0, 5000)
        assert max_pos == 0

    def test_zero_position_size(self, default_risk_manager):
        max_pos = default_risk_manager.calculate_max_positions(100000, 0)
        assert max_pos == 0


class TestCalculateRiskMetrics:
    def test_with_positions(self, default_risk_manager):
        positions = [
            {"entry_price": 100, "current_price": 105, "qty": 50},
            {"entry_price": 200, "current_price": 195, "qty": 25},
        ]
        metrics = default_risk_manager.calculate_risk_metrics(positions, 100000)
        assert metrics["total_exposure"] > 0
        assert "exposure_pct" in metrics
        assert "unrealized_pnl" in metrics
        assert "at_risk" in metrics

    def test_empty_positions(self, default_risk_manager):
        metrics = default_risk_manager.calculate_risk_metrics([], 100000)
        assert metrics["total_exposure"] == 0
        assert metrics["unrealized_pnl"] == 0

    def test_zero_account_value(self, default_risk_manager):
        positions = [{"entry_price": 100, "current_price": 105, "qty": 50}]
        metrics = default_risk_manager.calculate_risk_metrics(positions, 0)
        assert metrics["total_exposure"] == 0

    def test_unrealized_pnl_calculation(self, default_risk_manager):
        positions = [{"entry_price": 100, "current_price": 110, "qty": 10}]
        metrics = default_risk_manager.calculate_risk_metrics(positions, 100000)
        assert metrics["unrealized_pnl"] == 100

    def test_at_risk_calculation(self, default_risk_manager):
        positions = [{"entry_price": 100, "current_price": 100, "qty": 10}]
        metrics = default_risk_manager.calculate_risk_metrics(positions, 100000)
        stop = default_risk_manager.calculate_stop_loss_price(100)
        expected_risk = (100 - stop) * 10
        assert metrics["at_risk"] == expected_risk
