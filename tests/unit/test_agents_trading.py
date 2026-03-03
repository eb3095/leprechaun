"""Unit tests for trading agent."""

from datetime import datetime
from decimal import Decimal

import pytest

from src.agents.decision_log import Decision
from src.agents.trading_agent import RiskManager, TradingAgent, TradingStrategy


class TestTradingStrategy:
    """Tests for TradingStrategy dataclass."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = TradingStrategy()

        assert strategy.profit_target_pct == 2.5
        assert strategy.stop_loss_pct == 1.25
        assert strategy.min_manipulation_score == 0.5
        assert strategy.max_rsi_for_buy == 35.0
        assert strategy.max_positions == 10

    def test_custom_values(self):
        """Test custom strategy values."""
        strategy = TradingStrategy(
            profit_target_pct=3.0,
            stop_loss_pct=1.5,
            max_positions=5,
        )

        assert strategy.profit_target_pct == 3.0
        assert strategy.stop_loss_pct == 1.5
        assert strategy.max_positions == 5


class TestRiskManager:
    """Tests for RiskManager dataclass."""

    def test_default_values(self):
        """Test default risk values."""
        risk = RiskManager()

        assert risk.max_position_risk_pct == 1.0
        assert risk.max_daily_loss_pct == 2.0
        assert risk.max_weekly_loss_pct == 5.0
        assert risk.is_halted is False

    def test_halt_state(self):
        """Test halt state management."""
        risk = RiskManager(is_halted=True, halt_reason="Daily loss exceeded")

        assert risk.is_halted is True
        assert risk.halt_reason == "Daily loss exceeded"


class TestTradingAgent:
    """Tests for TradingAgent."""

    def test_init_default(self):
        """Test initialization with defaults."""
        agent = TradingAgent()

        assert agent.strategy is not None
        assert agent.risk_manager is not None
        assert agent.strategy.profit_target_pct == 2.5

    def test_init_custom_strategy(self):
        """Test initialization with custom strategy."""
        strategy = TradingStrategy(profit_target_pct=3.0)
        agent = TradingAgent(strategy=strategy)

        assert agent.strategy.profit_target_pct == 3.0

    def test_make_decision_when_halted(self):
        """Test decision making when trading is halted."""
        risk = RiskManager(is_halted=True, halt_reason="Test halt")
        agent = TradingAgent(risk_manager=risk)

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={},
            sentiment={},
            manipulation={},
            account={},
        )

        assert decision.decision == "SKIP"
        assert "halted" in decision.reasoning[0].lower()

    def test_make_decision_max_positions_reached(self):
        """Test decision when max positions reached."""
        strategy = TradingStrategy(max_positions=2)
        agent = TradingAgent(strategy=strategy)

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={},
            sentiment={},
            manipulation={},
            account={"positions": [{"symbol": "MSFT"}, {"symbol": "GOOGL"}]},
        )

        assert decision.decision == "SKIP"
        assert "max positions" in decision.reasoning[0].lower()

    def test_make_decision_entry_low_manipulation(self):
        """Test entry decision with low manipulation score."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={"technical_indicators": {"rsi_14": 25.0}},
            sentiment={"composite_score": -0.5},
            manipulation={"manipulation_score": 0.3, "evidence": {}},
            account={"positions": []},
        )

        assert decision.decision == "SKIP"
        assert "threshold" in decision.reasoning[1].lower()

    def test_make_decision_entry_high_rsi(self):
        """Test entry decision with RSI too high."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={"technical_indicators": {"rsi_14": 45.0}},
            sentiment={"composite_score": -0.5},
            manipulation={"manipulation_score": 0.6, "evidence": {}},
            account={"positions": []},
        )

        assert decision.decision == "SKIP"
        assert "rsi" in decision.reasoning[2].lower()

    def test_make_decision_entry_positive_sentiment(self):
        """Test entry decision with positive sentiment."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={"technical_indicators": {"rsi_14": 25.0}},
            sentiment={"composite_score": 0.3},
            manipulation={"manipulation_score": 0.6, "evidence": {}},
            account={"positions": []},
        )

        assert decision.decision == "SKIP"
        assert "sentiment" in " ".join(decision.reasoning).lower()

    def test_make_decision_entry_buy_signal(self):
        """Test entry decision with good buy signal."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={
                "current_price": 175.0,
                "technical_indicators": {"rsi_14": 25.0, "ema_21": 180.0},
            },
            sentiment={"composite_score": -0.5},
            manipulation={
                "manipulation_score": 0.7,
                "evidence": {"no_news_catalyst": True},
            },
            account={"positions": []},
        )

        assert decision.decision in ["BUY", "HOLD"]

    def test_make_decision_exit_profit_target(self):
        """Test exit decision when profit target hit."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={"current_price": 184.0, "technical_indicators": {}},
            sentiment={},
            manipulation={},
            account={"positions": [{"symbol": "AAPL", "entry_price": 175.0}]},
        )

        assert decision.decision == "SELL"
        assert "profit target" in " ".join(decision.reasoning).lower()

    def test_make_decision_exit_stop_loss(self):
        """Test exit decision when stop loss triggered."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={"current_price": 172.0, "technical_indicators": {}},
            sentiment={},
            manipulation={},
            account={"positions": [{"symbol": "AAPL", "entry_price": 175.0}]},
        )

        assert decision.decision == "SELL"
        assert "stop loss" in " ".join(decision.reasoning).lower()

    def test_make_decision_exit_sentiment_normalized(self):
        """Test exit decision when sentiment normalizes."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={
                "current_price": 177.0,
                "technical_indicators": {"rsi_14": 45.0},
            },
            sentiment={"composite_score": 0.4},
            manipulation={},
            account={"positions": [{"symbol": "AAPL", "entry_price": 175.0}]},
        )

        assert decision.decision == "SELL"
        assert "normalized" in " ".join(decision.reasoning).lower()

    def test_make_decision_hold_position(self):
        """Test hold decision for existing position."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={
                "current_price": 176.0,
                "technical_indicators": {"rsi_14": 35.0},
            },
            sentiment={"composite_score": -0.2},
            manipulation={},
            account={"positions": [{"symbol": "AAPL", "entry_price": 175.0}]},
        )

        assert decision.decision == "HOLD"

    def test_generate_reasoning(self):
        """Test reasoning generation."""
        agent = TradingAgent()

        inputs = {
            "symbol": "AAPL",
            "market_data": {"technical_indicators": {"rsi_14": 28.5}},
            "sentiment": {"composite_score": -0.4},
            "manipulation": {
                "manipulation_score": 0.65,
                "evidence": {"sentiment_spike": True, "no_news_catalyst": True},
            },
        }

        reasoning = agent.generate_reasoning(inputs, "BUY")

        assert len(reasoning) > 0
        assert "AAPL" in reasoning[0]
        assert any("65%" in r or "0.65" in r for r in reasoning)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        agent = TradingAgent()

        shares = agent.calculate_position_size(
            account={"equity": 100000},
            current_price=100.0,
            stop_loss_price=98.75,
        )

        assert shares > 0
        assert shares * 100.0 <= 10000

    def test_calculate_position_size_zero_risk(self):
        """Test position size when stop equals current price."""
        agent = TradingAgent()

        shares = agent.calculate_position_size(
            account={"equity": 100000},
            current_price=100.0,
            stop_loss_price=100.0,
        )

        assert shares > 0

    def test_update_risk_state_daily_limit(self):
        """Test risk state update triggers daily halt."""
        agent = TradingAgent()

        agent.update_risk_state(daily_pnl_pct=-2.5, weekly_pnl_pct=-2.5)

        assert agent.risk_manager.is_halted is True
        assert "daily" in agent.risk_manager.halt_reason.lower()

    def test_update_risk_state_weekly_limit(self):
        """Test risk state update triggers weekly halt."""
        agent = TradingAgent()

        agent.update_risk_state(daily_pnl_pct=-1.0, weekly_pnl_pct=-5.5)

        assert agent.risk_manager.is_halted is True
        assert "weekly" in agent.risk_manager.halt_reason.lower()

    def test_reset_daily_risk(self):
        """Test daily risk reset."""
        agent = TradingAgent()
        agent.risk_manager.daily_loss_pct = 1.5

        agent.reset_daily_risk()

        assert agent.risk_manager.daily_loss_pct == 0.0

    def test_reset_weekly_risk(self):
        """Test weekly risk reset."""
        agent = TradingAgent()
        agent.risk_manager.daily_loss_pct = 1.5
        agent.risk_manager.weekly_loss_pct = 3.0

        agent.reset_weekly_risk()

        assert agent.risk_manager.daily_loss_pct == 0.0
        assert agent.risk_manager.weekly_loss_pct == 0.0

    def test_resume_trading_when_halted(self):
        """Test resuming trading after halt."""
        agent = TradingAgent()
        agent.risk_manager.is_halted = True
        agent.risk_manager.halt_reason = "Test"

        result = agent.resume_trading()

        assert result is True
        assert agent.risk_manager.is_halted is False
        assert agent.risk_manager.halt_reason is None

    def test_resume_trading_when_not_halted(self):
        """Test resuming when not halted returns False."""
        agent = TradingAgent()

        result = agent.resume_trading()

        assert result is False

    def test_make_decision_handles_decimal_values(self):
        """Test decision handles Decimal values correctly."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={
                "current_price": Decimal("175.50"),
                "technical_indicators": {
                    "rsi_14": Decimal("28.5"),
                    "ema_21": Decimal("180.00"),
                },
            },
            sentiment={"composite_score": -0.5},
            manipulation={"manipulation_score": 0.6, "evidence": {}},
            account={"positions": [], "equity": Decimal("100000.00")},
        )

        assert isinstance(decision, Decision)
        assert decision.symbol == "AAPL"

    def test_decision_includes_all_inputs(self):
        """Test decision includes all input data."""
        agent = TradingAgent()

        decision = agent.make_decision(
            symbol="AAPL",
            market_data={
                "current_price": 175.0,
                "technical_indicators": {"rsi_14": 30.0},
            },
            sentiment={"composite_score": -0.4},
            manipulation={"manipulation_score": 0.6, "evidence": {"test": True}},
            account={"positions": []},
        )

        assert "symbol" in decision.inputs
        assert "market_data" in decision.inputs
        assert "sentiment" in decision.inputs
        assert "manipulation" in decision.inputs

    def test_check_risk_limits_within_bounds(self):
        """Test risk limits check when within bounds."""
        agent = TradingAgent()
        agent.risk_manager.daily_loss_pct = 1.0
        agent.risk_manager.weekly_loss_pct = 2.0

        result = agent._check_risk_limits()

        assert result is False

    def test_check_risk_limits_exceeded(self):
        """Test risk limits check when exceeded."""
        agent = TradingAgent()
        agent.risk_manager.daily_loss_pct = 2.5

        result = agent._check_risk_limits()

        assert result is True
