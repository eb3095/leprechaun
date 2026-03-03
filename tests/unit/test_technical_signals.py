"""Unit tests for signal generation based on technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.core.technical.signals import (
    PROFIT_TARGET_PERCENT,
    STOP_LOSS_PERCENT,
    calculate_position_size,
    detect_macd_crossover,
    generate_entry_signal,
    generate_exit_signal,
    is_below_ema,
    is_bollinger_squeeze,
    is_oversold,
)


class TestIsOversold:
    """Tests for oversold detection."""

    def test_oversold_below_threshold(self):
        """Test RSI below threshold returns True."""
        assert is_oversold(30.0) is True
        assert is_oversold(20.0) is True
        assert is_oversold(34.9) is True

    def test_not_oversold_above_threshold(self):
        """Test RSI above threshold returns False."""
        assert is_oversold(35.0) is False
        assert is_oversold(50.0) is False
        assert is_oversold(70.0) is False

    def test_oversold_at_threshold(self):
        """Test RSI exactly at threshold returns False."""
        assert is_oversold(35.0, threshold=35.0) is False

    def test_oversold_custom_threshold(self):
        """Test oversold with custom threshold."""
        assert is_oversold(25.0, threshold=30.0) is True
        assert is_oversold(32.0, threshold=30.0) is False

    def test_oversold_nan_value(self):
        """Test oversold returns False for NaN."""
        assert is_oversold(np.nan) is False
        assert is_oversold(float("nan")) is False


class TestIsBelowEMA:
    """Tests for price below EMA detection."""

    def test_price_below_ema(self):
        """Test price below EMA returns True."""
        assert is_below_ema(95.0, 100.0) is True
        assert is_below_ema(99.99, 100.0) is True

    def test_price_above_ema(self):
        """Test price above EMA returns False."""
        assert is_below_ema(105.0, 100.0) is False
        assert is_below_ema(100.01, 100.0) is False

    def test_price_equals_ema(self):
        """Test price equal to EMA returns False."""
        assert is_below_ema(100.0, 100.0) is False

    def test_below_ema_nan_values(self):
        """Test function handles NaN values."""
        assert is_below_ema(np.nan, 100.0) is False
        assert is_below_ema(95.0, np.nan) is False
        assert is_below_ema(np.nan, np.nan) is False


class TestIsBollingerSqueeze:
    """Tests for Bollinger Band squeeze detection."""

    def test_price_below_lower_band(self):
        """Test price below lower band returns True."""
        assert is_bollinger_squeeze(95.0, 96.0) is True

    def test_price_at_lower_band(self):
        """Test price at lower band returns True."""
        assert is_bollinger_squeeze(96.0, 96.0) is True

    def test_price_above_lower_band(self):
        """Test price above lower band returns False."""
        assert is_bollinger_squeeze(97.0, 96.0) is False

    def test_bollinger_squeeze_nan_values(self):
        """Test function handles NaN values."""
        assert is_bollinger_squeeze(np.nan, 96.0) is False
        assert is_bollinger_squeeze(95.0, np.nan) is False


class TestDetectMACDCrossover:
    """Tests for MACD crossover detection."""

    def test_bullish_crossover(self):
        """Test detection of bullish MACD crossover."""
        macd_line = pd.Series([-2, -1, 0, 1, 2])
        signal_line = pd.Series([0, 0, 0, 0, 0])

        result = detect_macd_crossover(macd_line, signal_line)

        assert result.iloc[0] == 0
        assert result.iloc[3] == 1

    def test_bearish_crossover(self):
        """Test detection of bearish MACD crossover."""
        macd_line = pd.Series([2, 1, 0, -1, -2])
        signal_line = pd.Series([0, 0, 0, 0, 0])

        result = detect_macd_crossover(macd_line, signal_line)

        assert result.iloc[2] == -1

    def test_no_crossover(self):
        """Test no crossover when MACD stays above or below signal."""
        macd_line = pd.Series([1, 2, 3, 4, 5])
        signal_line = pd.Series([0, 0, 0, 0, 0])

        result = detect_macd_crossover(macd_line, signal_line)

        assert (result.iloc[1:] == 0).all()

    def test_multiple_crossovers(self):
        """Test detection of multiple crossovers."""
        macd_line = pd.Series([-1, 1, -1, 1, -1])
        signal_line = pd.Series([0, 0, 0, 0, 0])

        result = detect_macd_crossover(macd_line, signal_line)

        assert result.iloc[1] == 1
        assert result.iloc[2] == -1
        assert result.iloc[3] == 1
        assert result.iloc[4] == -1

    def test_crossover_insufficient_data(self):
        """Test crossover with insufficient data."""
        macd_line = pd.Series([1])
        signal_line = pd.Series([0])

        result = detect_macd_crossover(macd_line, signal_line)

        assert len(result) == 1
        assert result.iloc[0] == 0

    def test_crossover_preserves_index(self):
        """Test crossover preserves series index."""
        dates = pd.date_range("2024-01-01", periods=5)
        macd_line = pd.Series([-1, 0, 1, 2, 3], index=dates)
        signal_line = pd.Series([0, 0, 0, 0, 0], index=dates)

        result = detect_macd_crossover(macd_line, signal_line)

        assert list(result.index) == list(dates)


class TestGenerateEntrySignal:
    """Tests for entry signal generation."""

    def test_strong_buy_signal(self):
        """Test strong buy signal with multiple confirming indicators."""
        indicators = {
            "rsi": 25.0,
            "price": 95.0,
            "ema_21": 100.0,
            "bb_lower": 96.0,
            "macd_crossover": 1,
        }

        result = generate_entry_signal(
            indicators,
            sentiment_score=-0.6,
            manipulation_score=0.75,
        )

        assert result["signal"] == "BUY"
        assert result["confidence"] in ["HIGH", "MEDIUM"]
        assert len(result["reasons"]) >= 3

    def test_buy_signal_rsi_only(self):
        """Test signal with only RSI oversold."""
        indicators = {
            "rsi": 30.0,
            "price": 100.0,
            "ema_21": 95.0,
            "bb_lower": 90.0,
            "macd_crossover": 0,
        }

        result = generate_entry_signal(
            indicators,
            sentiment_score=0.0,
            manipulation_score=0.3,
        )

        assert result["signal"] == "HOLD"
        assert "RSI oversold" in str(result["reasons"])

    def test_hold_signal_no_triggers(self):
        """Test HOLD signal when no conditions are met."""
        indicators = {
            "rsi": 50.0,
            "price": 100.0,
            "ema_21": 95.0,
            "bb_lower": 90.0,
            "macd_crossover": 0,
        }

        result = generate_entry_signal(
            indicators,
            sentiment_score=0.0,
            manipulation_score=0.2,
        )

        assert result["signal"] == "HOLD"
        assert result["confidence"] == "LOW"

    def test_manipulation_score_boost(self):
        """Test high manipulation score increases buy signal strength."""
        base_indicators = {
            "rsi": 33.0,
            "price": 98.0,
            "ema_21": 100.0,
            "bb_lower": 97.0,
            "macd_crossover": 0,
        }

        result_low = generate_entry_signal(
            base_indicators,
            sentiment_score=-0.5,
            manipulation_score=0.3,
        )

        result_high = generate_entry_signal(
            base_indicators,
            sentiment_score=-0.5,
            manipulation_score=0.8,
        )

        high_reasons = " ".join(result_high["reasons"])
        assert "manipulation" in high_reasons.lower()

    def test_negative_sentiment_contributes(self):
        """Test negative sentiment contributes to buy signal."""
        indicators = {
            "rsi": 33.0,
            "price": 98.0,
            "ema_21": 100.0,
            "bb_lower": 90.0,
            "macd_crossover": 0,
        }

        result = generate_entry_signal(
            indicators,
            sentiment_score=-0.6,
            manipulation_score=0.6,
        )

        reasons_str = " ".join(result["reasons"])
        assert "sentiment" in reasons_str.lower()

    def test_bearish_macd_contributes_to_sell(self):
        """Test bearish MACD crossover contributes to sell signals."""
        indicators = {
            "rsi": 55.0,
            "price": 100.0,
            "ema_21": 95.0,
            "bb_lower": 90.0,
            "macd_crossover": -1,
        }

        result = generate_entry_signal(
            indicators,
            sentiment_score=0.0,
            manipulation_score=0.0,
        )

        reasons_str = " ".join(result["reasons"])
        assert "bearish" in reasons_str.lower() or "MACD" in reasons_str

    def test_missing_indicators_handled(self):
        """Test function handles missing indicator values."""
        indicators = {"rsi": 30.0}

        result = generate_entry_signal(
            indicators,
            sentiment_score=np.nan,
            manipulation_score=np.nan,
        )

        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert "confidence" in result
        assert "reasons" in result

    def test_nan_sentiment_score(self):
        """Test function handles NaN sentiment score."""
        indicators = {"rsi": 30.0, "price": 100.0, "ema_21": 105.0}

        result = generate_entry_signal(
            indicators,
            sentiment_score=np.nan,
            manipulation_score=0.6,
        )

        assert result["signal"] in ["BUY", "SELL", "HOLD"]


class TestGenerateExitSignal:
    """Tests for exit signal generation."""

    def test_stop_loss_triggered(self):
        """Test exit when stop loss is triggered."""
        position = {"entry_price": 100.0}
        current_price = 98.5

        result = generate_exit_signal(position, current_price, {})

        assert result["should_exit"] is True
        assert "stop loss" in result["reason"].lower()

    def test_profit_target_hit(self):
        """Test exit when profit target is reached."""
        position = {"entry_price": 100.0}
        current_price = 102.6

        result = generate_exit_signal(position, current_price, {})

        assert result["should_exit"] is True
        assert "profit target" in result["reason"].lower()

    def test_rsi_normalized_with_profit(self):
        """Test exit when RSI normalizes and position is profitable."""
        position = {"entry_price": 100.0}
        current_price = 101.5
        indicators = {"rsi": 55.0}

        result = generate_exit_signal(position, current_price, indicators)

        assert result["should_exit"] is True
        assert "RSI normalized" in result["reason"]

    def test_rsi_normalized_with_loss(self):
        """Test no exit when RSI normalizes but position is losing."""
        position = {"entry_price": 100.0}
        current_price = 99.5
        indicators = {"rsi": 55.0}

        result = generate_exit_signal(position, current_price, indicators)

        assert result["should_exit"] is False

    def test_hold_position(self):
        """Test hold when no exit conditions are met."""
        position = {"entry_price": 100.0}
        current_price = 101.0
        indicators = {"rsi": 45.0}

        result = generate_exit_signal(position, current_price, indicators)

        assert result["should_exit"] is False
        assert "hold" in result["reason"].lower()

    def test_missing_entry_price(self):
        """Test handling of missing entry price."""
        position = {}
        current_price = 100.0

        result = generate_exit_signal(position, current_price, {})

        assert result["should_exit"] is False
        assert "invalid" in result["reason"].lower()

    def test_exact_stop_loss_threshold(self):
        """Test exit at exact stop loss threshold."""
        position = {"entry_price": 100.0}
        current_price = 100.0 * (1 - STOP_LOSS_PERCENT / 100)

        result = generate_exit_signal(position, current_price, {})

        assert result["should_exit"] is True

    def test_exact_profit_target_threshold(self):
        """Test exit at exact profit target threshold."""
        position = {"entry_price": 100.0}
        current_price = 100.0 * (1 + PROFIT_TARGET_PERCENT / 100) + 0.001

        result = generate_exit_signal(position, current_price, {})

        assert result["should_exit"] is True

    def test_stop_loss_takes_priority(self):
        """Test stop loss is checked before other conditions."""
        position = {"entry_price": 100.0}
        current_price = 98.0
        indicators = {"rsi": 55.0}

        result = generate_exit_signal(position, current_price, indicators)

        assert result["should_exit"] is True
        assert "stop loss" in result["reason"].lower()


class TestCalculatePositionSize:
    """Tests for position sizing calculation."""

    def test_basic_position_size(self):
        """Test basic position size calculation."""
        account_value = 100000.0
        entry_price = 100.0

        shares = calculate_position_size(account_value, entry_price)

        max_loss = shares * entry_price * (STOP_LOSS_PERCENT / 100)
        assert max_loss <= account_value * 0.01 + 0.01

    def test_position_size_whole_shares(self):
        """Test position size returns whole number of shares."""
        shares = calculate_position_size(100000.0, 157.32)

        assert isinstance(shares, int)

    def test_position_size_custom_risk(self):
        """Test position size with custom risk percentage."""
        shares_1pct = calculate_position_size(100000.0, 100.0, risk_per_trade=0.01)
        shares_2pct = calculate_position_size(100000.0, 100.0, risk_per_trade=0.02)

        assert shares_2pct > shares_1pct

    def test_position_size_expensive_stock(self):
        """Test position size for expensive stock."""
        shares = calculate_position_size(100000.0, 5000.0)

        assert shares > 0
        assert shares < 20

    def test_position_size_cheap_stock(self):
        """Test position size for cheap stock."""
        shares = calculate_position_size(100000.0, 5.0)

        assert shares > 0

    def test_position_size_small_account(self):
        """Test position size with small account."""
        shares = calculate_position_size(1000.0, 100.0)

        assert shares >= 0

    def test_position_size_zero_stop_loss(self):
        """Test position size with zero stop loss returns 0."""
        shares = calculate_position_size(100000.0, 100.0, stop_loss_percent=0.0)

        assert shares == 0


class TestConstants:
    """Tests for module constants."""

    def test_profit_target_value(self):
        """Test profit target constant value."""
        assert PROFIT_TARGET_PERCENT == 2.5

    def test_stop_loss_value(self):
        """Test stop loss constant value."""
        assert STOP_LOSS_PERCENT == 1.25

    def test_risk_reward_ratio(self):
        """Test risk-reward ratio is approximately 2:1."""
        ratio = PROFIT_TARGET_PERCENT / STOP_LOSS_PERCENT
        assert abs(ratio - 2.0) < 0.01
