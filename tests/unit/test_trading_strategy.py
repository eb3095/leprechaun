"""Unit tests for trading strategy module."""

import pytest

from src.core.trading.strategy import (
    EntryConfidence,
    EntrySignal,
    ExitSignal,
    TradingStrategy,
)
from src.utils.config import TradingConfig


@pytest.fixture
def default_strategy():
    """Create strategy with default configuration."""
    return TradingStrategy()


@pytest.fixture
def custom_strategy():
    """Create strategy with custom configuration."""
    config = TradingConfig(
        profit_target_percent=3.0,
        stop_loss_percent=1.5,
        rsi_oversold=30,
        manipulation_threshold=0.6,
        sentiment_negative_threshold=-0.5,
    )
    return TradingStrategy(config)


@pytest.fixture
def strong_buy_data():
    """Stock data that should trigger a buy signal."""
    return {
        "symbol": "AAPL",
        "price": 150.0,
        "rsi": 25.0,
        "ema_21": 160.0,
        "sentiment_score": -0.6,
        "manipulation_score": 0.75,
        "has_news": False,
    }


@pytest.fixture
def weak_data():
    """Stock data that should not trigger a buy signal."""
    return {
        "symbol": "MSFT",
        "price": 300.0,
        "rsi": 55.0,
        "ema_21": 290.0,
        "sentiment_score": 0.2,
        "manipulation_score": 0.2,
        "has_news": False,
    }


class TestTradingStrategyInit:
    def test_default_config_values(self, default_strategy):
        assert default_strategy.profit_target == 0.025
        assert default_strategy.stop_loss == 0.0125
        assert default_strategy.rsi_oversold == 35
        assert default_strategy.manipulation_threshold == 0.5
        assert default_strategy.sentiment_threshold == -0.4

    def test_custom_config_values(self, custom_strategy):
        assert custom_strategy.profit_target == 0.03
        assert custom_strategy.stop_loss == 0.015
        assert custom_strategy.rsi_oversold == 30
        assert custom_strategy.manipulation_threshold == 0.6
        assert custom_strategy.sentiment_threshold == -0.5


class TestEvaluateEntry:
    def test_strong_buy_signal(self, default_strategy, strong_buy_data):
        signal = default_strategy.evaluate_entry(strong_buy_data)
        assert signal.should_enter is True
        assert signal.confidence in (EntryConfidence.HIGH, EntryConfidence.MEDIUM)
        assert len(signal.reasons) >= 3
        assert signal.score >= 0.60

    def test_weak_signal_no_entry(self, default_strategy, weak_data):
        signal = default_strategy.evaluate_entry(weak_data)
        assert signal.should_enter is False
        assert signal.score < 0.60

    def test_news_blocks_entry(self, default_strategy, strong_buy_data):
        strong_buy_data["has_news"] = True
        signal = default_strategy.evaluate_entry(strong_buy_data)
        assert signal.should_enter is False
        assert "news" in signal.reasons[0].lower()

    def test_missing_data_no_entry(self, default_strategy):
        incomplete_data = {"symbol": "TEST", "price": 100.0}
        signal = default_strategy.evaluate_entry(incomplete_data)
        assert signal.should_enter is False
        assert "insufficient" in signal.reasons[0].lower()

    def test_rsi_oversold_adds_score(self, default_strategy):
        data = {
            "symbol": "TEST",
            "price": 100.0,
            "rsi": 30.0,
            "sentiment_score": 0.0,
            "manipulation_score": 0.0,
            "has_news": False,
        }
        signal = default_strategy.evaluate_entry(data)
        assert any("rsi" in r.lower() for r in signal.reasons)

    def test_price_below_ema_adds_score(self, default_strategy):
        data = {
            "symbol": "TEST",
            "price": 95.0,
            "rsi": 50.0,
            "ema_21": 100.0,
            "sentiment_score": 0.0,
            "manipulation_score": 0.0,
            "has_news": False,
        }
        signal = default_strategy.evaluate_entry(data)
        assert any("ema" in r.lower() for r in signal.reasons)

    def test_negative_sentiment_adds_score(self, default_strategy):
        data = {
            "symbol": "TEST",
            "price": 100.0,
            "rsi": 50.0,
            "sentiment_score": -0.6,
            "manipulation_score": 0.0,
            "has_news": False,
        }
        signal = default_strategy.evaluate_entry(data)
        assert any("sentiment" in r.lower() for r in signal.reasons)

    def test_high_manipulation_adds_score(self, default_strategy):
        data = {
            "symbol": "TEST",
            "price": 100.0,
            "rsi": 50.0,
            "sentiment_score": 0.0,
            "manipulation_score": 0.7,
            "has_news": False,
        }
        signal = default_strategy.evaluate_entry(data)
        assert any("manipulation" in r.lower() for r in signal.reasons)

    def test_signal_to_dict(self, default_strategy, strong_buy_data):
        signal = default_strategy.evaluate_entry(strong_buy_data)
        d = signal.to_dict()
        assert "should_enter" in d
        assert "confidence" in d
        assert "reasons" in d
        assert "score" in d
        assert isinstance(d["confidence"], str)


class TestEvaluateExit:
    def test_profit_target_triggers_exit(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 103.0}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is True
        assert signal.exit_type == "TARGET"

    def test_stop_loss_triggers_exit(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 98.5}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is True
        assert signal.exit_type == "STOP_LOSS"

    def test_friday_close_triggers_exit(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 101.0, "is_friday_close": True}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is True
        assert signal.exit_type == "FRIDAY_CLOSE"

    def test_breaking_news_triggers_exit(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 101.0, "has_breaking_news": True}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is True
        assert signal.exit_type == "NEWS"

    def test_rsi_normalized_with_profit_triggers_exit(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 101.0, "rsi": 55.0}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is True
        assert signal.exit_type == "TARGET"

    def test_rsi_normalized_without_profit_holds(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 99.5, "rsi": 55.0}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is False

    def test_missing_price_data(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0}
        current = {}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is False
        assert "missing" in signal.reason.lower()

    def test_hold_position(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0, "shares": 10}
        current = {"price": 101.0}
        signal = default_strategy.evaluate_exit(position, current)
        assert signal.should_exit is False
        assert signal.exit_type == "NONE"

    def test_exit_signal_to_dict(self, default_strategy):
        position = {"symbol": "AAPL", "entry_price": 100.0}
        current = {"price": 103.0}
        signal = default_strategy.evaluate_exit(position, current)
        d = signal.to_dict()
        assert "should_exit" in d
        assert "reason" in d
        assert "exit_type" in d


class TestRankOpportunities:
    def test_ranks_by_score(self, default_strategy):
        candidates = [
            {
                "symbol": "LOW",
                "price": 100.0,
                "rsi": 50.0,
                "sentiment_score": -0.5,
                "manipulation_score": 0.6,
                "has_news": False,
            },
            {
                "symbol": "HIGH",
                "price": 100.0,
                "rsi": 25.0,
                "sentiment_score": -0.7,
                "manipulation_score": 0.8,
                "has_news": False,
            },
            {
                "symbol": "MED",
                "price": 100.0,
                "rsi": 30.0,
                "sentiment_score": -0.5,
                "manipulation_score": 0.65,
                "has_news": False,
            },
        ]
        ranked = default_strategy.rank_opportunities(candidates)
        assert ranked[0]["symbol"] == "HIGH"
        assert all("rank_score" in r for r in ranked)
        assert all("entry_signal" in r for r in ranked)

    def test_excludes_non_entries(self, default_strategy):
        candidates = [
            {
                "symbol": "GOOD",
                "price": 100.0,
                "rsi": 25.0,
                "sentiment_score": -0.7,
                "manipulation_score": 0.8,
                "has_news": False,
            },
            {
                "symbol": "BAD",
                "price": 100.0,
                "rsi": 60.0,
                "sentiment_score": 0.5,
                "manipulation_score": 0.1,
                "has_news": False,
            },
        ]
        ranked = default_strategy.rank_opportunities(candidates)
        assert len(ranked) == 1
        assert ranked[0]["symbol"] == "GOOD"

    def test_empty_candidates(self, default_strategy):
        ranked = default_strategy.rank_opportunities([])
        assert ranked == []

    def test_all_filtered_out(self, default_strategy):
        candidates = [
            {
                "symbol": "NEWS",
                "price": 100.0,
                "rsi": 25.0,
                "sentiment_score": -0.7,
                "manipulation_score": 0.8,
                "has_news": True,
            },
        ]
        ranked = default_strategy.rank_opportunities(candidates)
        assert ranked == []


class TestScoreToConfidence:
    def test_high_confidence(self, default_strategy):
        conf = default_strategy._score_to_confidence(0.90)
        assert conf == EntryConfidence.HIGH

    def test_medium_confidence(self, default_strategy):
        conf = default_strategy._score_to_confidence(0.75)
        assert conf == EntryConfidence.MEDIUM

    def test_low_confidence(self, default_strategy):
        conf = default_strategy._score_to_confidence(0.50)
        assert conf == EntryConfidence.LOW

    def test_boundary_high(self, default_strategy):
        conf = default_strategy._score_to_confidence(0.85)
        assert conf == EntryConfidence.HIGH

    def test_boundary_medium(self, default_strategy):
        conf = default_strategy._score_to_confidence(0.70)
        assert conf == EntryConfidence.MEDIUM
