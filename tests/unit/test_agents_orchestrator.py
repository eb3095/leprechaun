"""Unit tests for agent orchestrator."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.agents.decision_log import Decision
from src.agents.orchestrator import AgentOrchestrator
from src.agents.trading_agent import RiskManager, TradingStrategy


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self, sentiment=None, news=None, market=None, posts=None):
        self._sentiment = sentiment or []
        self._news = news or []
        self._market = market or {}
        self._posts = posts or []

    def get_sentiment(self, symbol):
        return self._sentiment

    def get_news(self, symbol):
        return self._news

    def get_market_data(self, symbol):
        return self._market

    def get_posts(self, symbol):
        return self._posts


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    def test_init_default(self):
        """Test initialization with defaults."""
        orchestrator = AgentOrchestrator()

        assert orchestrator.sentiment_agent is not None
        assert orchestrator.manipulation_agent is not None
        assert orchestrator.trading_agent is not None
        assert orchestrator.decision_logger is not None
        assert orchestrator.bayesian is not None

    def test_init_custom_strategy(self):
        """Test initialization with custom strategy."""
        strategy = TradingStrategy(profit_target_pct=3.0)
        orchestrator = AgentOrchestrator(trading_strategy=strategy)

        assert orchestrator.trading_agent.strategy.profit_target_pct == 3.0

    def test_init_custom_risk_manager(self):
        """Test initialization with custom risk manager."""
        risk = RiskManager(max_daily_loss_pct=3.0)
        orchestrator = AgentOrchestrator(risk_manager=risk)

        assert orchestrator.trading_agent.risk_manager.max_daily_loss_pct == 3.0

    def test_run_analysis_cycle_empty_symbols(self):
        """Test analysis cycle with no symbols."""
        orchestrator = AgentOrchestrator()

        decisions = orchestrator.run_analysis_cycle([])

        assert decisions == []

    def test_run_analysis_cycle_single_symbol(self):
        """Test analysis cycle with single symbol."""
        orchestrator = AgentOrchestrator()

        decisions = orchestrator.run_analysis_cycle(["AAPL"])

        assert len(decisions) == 1
        assert decisions[0].symbol == "AAPL"

    def test_run_analysis_cycle_multiple_symbols(self):
        """Test analysis cycle with multiple symbols."""
        orchestrator = AgentOrchestrator()

        decisions = orchestrator.run_analysis_cycle(["AAPL", "MSFT", "GOOGL"])

        assert len(decisions) == 3
        symbols = [d.symbol for d in decisions]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    def test_run_analysis_cycle_with_data_providers(self):
        """Test analysis cycle with data providers."""
        orchestrator = AgentOrchestrator()

        sentiment_provider = MockDataProvider(
            sentiment=[{"source": "reddit", "score": -0.5, "volume": 100}]
        )
        market_provider = MockDataProvider(
            market={"current_price": 175.0, "technical_indicators": {"rsi_14": 30.0}}
        )
        news_provider = MockDataProvider(news=[])

        decisions = orchestrator.run_analysis_cycle(
            ["AAPL"],
            market_data_provider=market_provider,
            sentiment_data_provider=sentiment_provider,
            news_data_provider=news_provider,
        )

        assert len(decisions) == 1

    def test_run_analysis_cycle_logs_decisions(self):
        """Test that analysis cycle logs decisions."""
        orchestrator = AgentOrchestrator()

        orchestrator.run_analysis_cycle(["AAPL", "MSFT"])

        stats = orchestrator.decision_logger.get_decision_statistics()
        assert stats["total_decisions"] == 2

    def test_run_analysis_cycle_handles_errors(self):
        """Test analysis cycle handles errors gracefully."""
        orchestrator = AgentOrchestrator()

        class ErrorProvider:
            def get_sentiment(self, symbol):
                raise Exception("Test error")

            def get_news(self, symbol):
                raise Exception("Test error")

            def get_market_data(self, symbol):
                raise Exception("Test error")

            def get_posts(self, symbol):
                raise Exception("Test error")

        decisions = orchestrator.run_analysis_cycle(
            ["AAPL"],
            sentiment_data_provider=ErrorProvider(),
        )

        assert len(decisions) == 1
        assert decisions[0].decision == "SKIP"
        assert "failed" in decisions[0].reasoning[0].lower()

    def test_run_monday_cycle(self):
        """Test Monday buy cycle."""
        orchestrator = AgentOrchestrator()

        results = orchestrator.run_monday_cycle(
            symbols=["AAPL", "MSFT"],
            account={"cash": 100000, "equity": 100000, "positions": []},
        )

        assert isinstance(results, list)

    def test_run_monday_cycle_sorts_by_confidence(self):
        """Test Monday cycle processes highest confidence first."""
        orchestrator = AgentOrchestrator()

        decisions = orchestrator.run_analysis_cycle(["AAPL", "MSFT", "GOOGL"])
        buy_decisions = [d for d in decisions if d.decision == "BUY"]

        if len(buy_decisions) >= 2:
            for i in range(len(buy_decisions) - 1):
                assert buy_decisions[i].confidence >= buy_decisions[i + 1].confidence

    def test_run_friday_cycle(self):
        """Test Friday sell cycle."""
        orchestrator = AgentOrchestrator()

        positions = [
            {"symbol": "AAPL", "entry_price": 175.0, "shares": 10},
            {"symbol": "MSFT", "entry_price": 380.0, "shares": 5},
        ]

        results = orchestrator.run_friday_cycle(positions)

        assert len(results) == 2
        assert all(r["symbol"] in ["AAPL", "MSFT"] for r in results)

    def test_run_friday_cycle_logs_decisions(self):
        """Test Friday cycle logs sell decisions."""
        orchestrator = AgentOrchestrator()

        positions = [{"symbol": "AAPL", "entry_price": 175.0, "shares": 10}]

        orchestrator.run_friday_cycle(positions)

        decisions = orchestrator.decision_logger.get_decisions(decision_type="SELL")
        assert len(decisions) >= 1

    def test_run_signal_check(self):
        """Test regular signal check for open positions."""
        orchestrator = AgentOrchestrator()

        market_provider = MockDataProvider(
            market={"current_price": 185.0, "technical_indicators": {"rsi_14": 45.0}}
        )
        sentiment_provider = MockDataProvider(
            sentiment=[{"source": "reddit", "score": 0.2, "volume": 50}]
        )

        positions = [{"symbol": "AAPL", "entry_price": 175.0, "shares": 10}]

        results = orchestrator.run_signal_check(
            positions,
            market_data_provider=market_provider,
            sentiment_data_provider=sentiment_provider,
        )

        assert len(results) == 1

    def test_get_analysis_summary(self):
        """Test getting comprehensive analysis summary."""
        orchestrator = AgentOrchestrator()

        summary = orchestrator.get_analysis_summary("AAPL")

        assert "symbol" in summary
        assert "timestamp" in summary
        assert "sentiment" in summary
        assert "manipulation" in summary
        assert "risk_assessment" in summary

    def test_get_analysis_summary_with_providers(self):
        """Test analysis summary with data providers."""
        orchestrator = AgentOrchestrator()

        sentiment_provider = MockDataProvider(
            sentiment=[{"source": "reddit", "score": -0.6, "volume": 200}]
        )
        market_provider = MockDataProvider(
            market={"current_price": 170.0, "technical_indicators": {"rsi_14": 25.0}}
        )

        summary = orchestrator.get_analysis_summary(
            "AAPL",
            market_data_provider=market_provider,
            sentiment_data_provider=sentiment_provider,
        )

        assert summary["sentiment"]["composite_score"] < 0
        assert "manipulation_score" in summary["manipulation"]

    def test_update_bayesian_priors(self):
        """Test updating Bayesian priors."""
        orchestrator = AgentOrchestrator()

        historical_data = [
            {"was_manipulation": True, "evidence": {"sentiment_spike": True}},
            {"was_manipulation": True, "evidence": {"sentiment_spike": True}},
            {"was_manipulation": False, "evidence": {"sentiment_spike": False}},
        ]

        result = orchestrator.update_bayesian_priors(historical_data)

        assert "prior_manipulation" in result

    def test_get_decision_statistics(self):
        """Test getting decision statistics."""
        orchestrator = AgentOrchestrator()

        orchestrator.run_analysis_cycle(["AAPL", "MSFT"])

        stats = orchestrator.get_decision_statistics()

        assert stats["total_decisions"] == 2

    def test_get_decision_statistics_with_time_range(self):
        """Test decision statistics with time range."""
        orchestrator = AgentOrchestrator()

        orchestrator.run_analysis_cycle(["AAPL"])

        now = datetime.utcnow()
        stats = orchestrator.get_decision_statistics(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
        )

        assert stats["total_decisions"] >= 1

    def test_export_decisions(self):
        """Test exporting decisions."""
        orchestrator = AgentOrchestrator()

        orchestrator.run_analysis_cycle(["AAPL"])

        now = datetime.utcnow()
        export = orchestrator.export_decisions(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
            format="json",
        )

        assert "AAPL" in export

    def test_export_decisions_csv(self):
        """Test exporting decisions as CSV."""
        orchestrator = AgentOrchestrator()

        orchestrator.run_analysis_cycle(["AAPL"])

        now = datetime.utcnow()
        export = orchestrator.export_decisions(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
            format="csv",
        )

        assert "timestamp,symbol" in export

    def test_halt_trading(self):
        """Test halting trading manually."""
        orchestrator = AgentOrchestrator()

        orchestrator.halt_trading("Manual halt for testing")

        is_halted, reason = orchestrator.is_halted()
        assert is_halted is True
        assert reason == "Manual halt for testing"

    def test_resume_trading(self):
        """Test resuming trading after halt."""
        orchestrator = AgentOrchestrator()

        orchestrator.halt_trading("Test halt")
        result = orchestrator.resume_trading()

        assert result is True
        is_halted, _ = orchestrator.is_halted()
        assert is_halted is False

    def test_resume_trading_when_not_halted(self):
        """Test resuming when not halted."""
        orchestrator = AgentOrchestrator()

        result = orchestrator.resume_trading()

        assert result is False

    def test_is_halted_default(self):
        """Test halted status default."""
        orchestrator = AgentOrchestrator()

        is_halted, reason = orchestrator.is_halted()

        assert is_halted is False
        assert reason is None

    def test_analysis_cycle_respects_halt(self):
        """Test that analysis respects halt state."""
        orchestrator = AgentOrchestrator()

        orchestrator.halt_trading("Testing")
        decisions = orchestrator.run_analysis_cycle(["AAPL"])

        assert len(decisions) == 1
        assert decisions[0].decision == "SKIP"
        assert "halted" in decisions[0].reasoning[0].lower()

    def test_monday_cycle_respects_halt(self):
        """Test that Monday cycle respects halt state."""
        orchestrator = AgentOrchestrator()

        orchestrator.halt_trading("Testing")
        results = orchestrator.run_monday_cycle(
            symbols=["AAPL"],
            account={"cash": 100000, "equity": 100000, "positions": []},
        )

        assert isinstance(results, list)

    def test_orchestrator_agents_share_bayesian(self):
        """Test that orchestrator and manipulation agent share Bayesian detector."""
        orchestrator = AgentOrchestrator()

        assert orchestrator.bayesian is orchestrator.manipulation_agent.bayesian

    def test_analysis_summary_includes_market_data(self):
        """Test analysis summary includes market data when provided."""
        orchestrator = AgentOrchestrator()

        market_provider = MockDataProvider(
            market={
                "current_price": 175.50,
                "technical_indicators": {"rsi_14": 32.0, "ema_21": 178.0},
            }
        )

        summary = orchestrator.get_analysis_summary(
            "AAPL", market_data_provider=market_provider
        )

        assert "market_data" in summary
        assert summary["market_data"]["current_price"] == 175.50

    def test_friday_cycle_empty_positions(self):
        """Test Friday cycle with no positions."""
        orchestrator = AgentOrchestrator()

        results = orchestrator.run_friday_cycle([])

        assert results == []

    def test_signal_check_empty_positions(self):
        """Test signal check with no positions."""
        orchestrator = AgentOrchestrator()

        results = orchestrator.run_signal_check([])

        assert results == []

    def test_signal_check_returns_hold_for_no_exit(self):
        """Test signal check returns hold when no exit signal."""
        orchestrator = AgentOrchestrator()

        market_provider = MockDataProvider(
            market={"current_price": 176.0, "technical_indicators": {"rsi_14": 35.0}}
        )
        sentiment_provider = MockDataProvider(
            sentiment=[{"source": "reddit", "score": -0.3, "volume": 50}]
        )

        positions = [{"symbol": "AAPL", "entry_price": 175.0, "shares": 10}]

        results = orchestrator.run_signal_check(
            positions,
            market_data_provider=market_provider,
            sentiment_data_provider=sentiment_provider,
        )

        assert len(results) == 1
        assert results[0]["decision"] == "HOLD" or results[0].get("executed") is False
