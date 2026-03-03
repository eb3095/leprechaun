"""Unit tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from src.core.technical.indicators import (
    calculate_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_volume_sma,
)


class TestCalculateRSI:
    """Tests for RSI calculation."""

    def test_rsi_basic_calculation(self):
        """Test RSI calculation with trending data."""
        prices = pd.Series([44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0,
                           43.5, 43.75, 44.0, 44.5, 44.75, 45.0, 45.5, 45.0])
        rsi = calculate_rsi(prices, period=14)

        assert not rsi.isna().all()
        assert 0 <= rsi.iloc[-1] <= 100

    def test_rsi_all_gains_returns_high_value(self):
        """Test RSI approaches 100 when all price changes are gains."""
        prices = pd.Series(range(20, 50))
        rsi = calculate_rsi(prices, period=14)

        assert rsi.iloc[-1] > 90

    def test_rsi_all_losses_returns_low_value(self):
        """Test RSI approaches 0 when all price changes are losses."""
        prices = pd.Series(range(50, 20, -1))
        rsi = calculate_rsi(prices, period=14)

        assert rsi.iloc[-1] < 10

    def test_rsi_insufficient_data(self):
        """Test RSI returns NaN with insufficient data."""
        prices = pd.Series([100, 101, 102])
        rsi = calculate_rsi(prices, period=14)

        assert rsi.isna().all()

    def test_rsi_custom_period(self):
        """Test RSI with custom period."""
        prices = pd.Series([100 + i * 0.5 for i in range(25)])
        rsi_7 = calculate_rsi(prices, period=7)
        rsi_21 = calculate_rsi(prices, period=21)

        assert not rsi_7.iloc[10:].isna().all()
        assert not rsi_21.iloc[-1:].isna().all()

    def test_rsi_preserves_index(self):
        """Test RSI preserves the original series index."""
        dates = pd.date_range("2024-01-01", periods=20)
        prices = pd.Series(range(100, 120), index=dates)
        rsi = calculate_rsi(prices, period=14)

        assert list(rsi.index) == list(prices.index)


class TestCalculateEMA:
    """Tests for EMA calculation."""

    def test_ema_basic_calculation(self):
        """Test EMA calculation with sample data."""
        prices = pd.Series([22.27, 22.19, 22.08, 22.17, 22.18, 22.13,
                           22.23, 22.43, 22.24, 22.29, 22.15])
        ema = calculate_ema(prices, period=9)

        assert not ema.isna().all()
        assert ema.iloc[-1] < prices.iloc[-1] + 1
        assert ema.iloc[-1] > prices.iloc[-1] - 1

    def test_ema_weights_recent_prices_more(self):
        """Test EMA gives more weight to recent prices than SMA."""
        prices = pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 15])
        ema = calculate_ema(prices, period=9)
        sma = prices.rolling(window=9).mean()

        assert ema.iloc[-1] > sma.iloc[-1]

    def test_ema_insufficient_data(self):
        """Test EMA returns NaN with insufficient data."""
        prices = pd.Series([100, 101, 102])
        ema = calculate_ema(prices, period=9)

        assert ema.isna().all()

    def test_ema_different_periods(self):
        """Test EMA with different periods (9, 21, 50)."""
        prices = pd.Series(range(100, 160))

        ema_9 = calculate_ema(prices, period=9)
        ema_21 = calculate_ema(prices, period=21)
        ema_50 = calculate_ema(prices, period=50)

        assert not ema_9.iloc[10:].isna().all()
        assert not ema_21.iloc[22:].isna().all()
        assert not ema_50.iloc[51:].isna().all()

        assert ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]

    def test_ema_constant_prices(self):
        """Test EMA equals price when prices are constant."""
        prices = pd.Series([100.0] * 20)
        ema = calculate_ema(prices, period=9)

        np.testing.assert_almost_equal(ema.iloc[-1], 100.0, decimal=5)


class TestCalculateMACD:
    """Tests for MACD calculation."""

    def test_macd_basic_calculation(self):
        """Test MACD calculation returns three series."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        macd_line, signal_line, histogram = calculate_macd(prices)

        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_macd_histogram_equals_difference(self):
        """Test histogram equals MACD line minus signal line."""
        prices = pd.Series([100 + np.sin(i / 5) * 5 for i in range(50)])
        macd_line, signal_line, histogram = calculate_macd(prices)

        valid_idx = ~(macd_line.isna() | signal_line.isna())
        expected = macd_line[valid_idx] - signal_line[valid_idx]
        np.testing.assert_array_almost_equal(
            histogram[valid_idx].values,
            expected.values,
            decimal=10,
        )

    def test_macd_insufficient_data(self):
        """Test MACD returns NaN with insufficient data."""
        prices = pd.Series([100, 101, 102])
        macd_line, signal_line, histogram = calculate_macd(prices)

        assert macd_line.isna().all()
        assert signal_line.isna().all()
        assert histogram.isna().all()

    def test_macd_uptrend_positive(self):
        """Test MACD line is positive in strong uptrend."""
        prices = pd.Series([100 + i * 2 for i in range(50)])
        macd_line, _, _ = calculate_macd(prices)

        assert macd_line.iloc[-1] > 0

    def test_macd_downtrend_negative(self):
        """Test MACD line is negative in strong downtrend."""
        prices = pd.Series([200 - i * 2 for i in range(50)])
        macd_line, _, _ = calculate_macd(prices)

        assert macd_line.iloc[-1] < 0

    def test_macd_custom_periods(self):
        """Test MACD with custom periods."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        macd_line, signal_line, histogram = calculate_macd(
            prices, fast=8, slow=17, signal=5
        )

        assert not macd_line.iloc[20:].isna().all()


class TestCalculateBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_basic_calculation(self):
        """Test Bollinger Bands returns three series."""
        prices = pd.Series([100 + np.random.randn() * 2 for _ in range(30)])
        upper, middle, lower = calculate_bollinger_bands(prices)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_bollinger_band_ordering(self):
        """Test upper > middle > lower for all valid values."""
        np.random.seed(42)
        prices = pd.Series([100 + np.random.randn() * 2 for _ in range(30)])
        upper, middle, lower = calculate_bollinger_bands(prices)

        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bollinger_middle_equals_sma(self):
        """Test middle band equals 20-period SMA."""
        prices = pd.Series(range(100, 130))
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        sma = prices.rolling(window=20).mean()

        valid_idx = ~middle.isna()
        np.testing.assert_array_almost_equal(
            middle[valid_idx].values,
            sma[valid_idx].values,
            decimal=10,
        )

    def test_bollinger_insufficient_data(self):
        """Test Bollinger Bands returns NaN with insufficient data."""
        prices = pd.Series([100, 101, 102])
        upper, middle, lower = calculate_bollinger_bands(prices)

        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_bollinger_std_dev_affects_width(self):
        """Test different std_dev values affect band width."""
        prices = pd.Series([100 + i * 0.5 for i in range(30)])

        upper_2, _, lower_2 = calculate_bollinger_bands(prices, std_dev=2.0)
        upper_1, _, lower_1 = calculate_bollinger_bands(prices, std_dev=1.0)

        width_2 = upper_2.iloc[-1] - lower_2.iloc[-1]
        width_1 = upper_1.iloc[-1] - lower_1.iloc[-1]

        assert width_2 > width_1


class TestCalculateATR:
    """Tests for ATR calculation."""

    def test_atr_basic_calculation(self):
        """Test ATR calculation with sample OHLC data."""
        high = pd.Series([102, 104, 103, 105, 106, 104, 107, 108, 106, 109,
                         110, 108, 111, 112, 110, 113])
        low = pd.Series([98, 100, 99, 101, 102, 100, 103, 104, 102, 105,
                        106, 104, 107, 108, 106, 109])
        close = pd.Series([100, 102, 101, 103, 104, 102, 105, 106, 104, 107,
                          108, 106, 109, 110, 108, 111])

        atr = calculate_atr(high, low, close, period=14)

        assert not atr.isna().all()
        assert atr.iloc[-1] > 0

    def test_atr_measures_volatility(self):
        """Test ATR is higher for more volatile data."""
        high_volatile = pd.Series([100, 110, 100, 110, 100, 110, 100, 110,
                                   100, 110, 100, 110, 100, 110, 100, 110])
        low_volatile = pd.Series([90, 100, 90, 100, 90, 100, 90, 100,
                                  90, 100, 90, 100, 90, 100, 90, 100])
        close_volatile = pd.Series([95, 105, 95, 105, 95, 105, 95, 105,
                                    95, 105, 95, 105, 95, 105, 95, 105])

        high_stable = pd.Series([101, 102, 101, 102, 101, 102, 101, 102,
                                 101, 102, 101, 102, 101, 102, 101, 102])
        low_stable = pd.Series([99, 100, 99, 100, 99, 100, 99, 100,
                                99, 100, 99, 100, 99, 100, 99, 100])
        close_stable = pd.Series([100, 101, 100, 101, 100, 101, 100, 101,
                                  100, 101, 100, 101, 100, 101, 100, 101])

        atr_volatile = calculate_atr(high_volatile, low_volatile, close_volatile)
        atr_stable = calculate_atr(high_stable, low_stable, close_stable)

        assert atr_volatile.iloc[-1] > atr_stable.iloc[-1]

    def test_atr_insufficient_data(self):
        """Test ATR handles insufficient data."""
        high = pd.Series([102])
        low = pd.Series([98])
        close = pd.Series([100])

        atr = calculate_atr(high, low, close)

        assert atr.isna().all()


class TestCalculateVolumeSMA:
    """Tests for Volume SMA calculation."""

    def test_volume_sma_basic_calculation(self):
        """Test Volume SMA calculation."""
        volume = pd.Series([1000000 + i * 10000 for i in range(25)])
        sma = calculate_volume_sma(volume, period=20)

        assert not sma.isna().all()
        assert sma.iloc[-1] > 0

    def test_volume_sma_equals_rolling_mean(self):
        """Test Volume SMA equals pandas rolling mean."""
        volume = pd.Series([1000000, 1100000, 1050000, 1200000, 1150000,
                           1300000, 1250000, 1400000, 1350000, 1500000,
                           1450000, 1600000, 1550000, 1700000, 1650000,
                           1800000, 1750000, 1900000, 1850000, 2000000])
        sma = calculate_volume_sma(volume, period=20)
        expected = volume.rolling(window=20).mean()

        np.testing.assert_array_almost_equal(
            sma.values,
            expected.values,
            decimal=5,
        )

    def test_volume_sma_insufficient_data(self):
        """Test Volume SMA returns NaN with insufficient data."""
        volume = pd.Series([1000000, 1100000, 1050000])
        sma = calculate_volume_sma(volume, period=20)

        assert sma.isna().all()


class TestCalculateAllIndicators:
    """Tests for the all-in-one indicator calculator."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV DataFrame."""
        np.random.seed(42)
        n = 60
        base = 100
        prices = base + np.cumsum(np.random.randn(n) * 0.5)

        return pd.DataFrame({
            "open": prices - np.random.rand(n) * 0.5,
            "high": prices + np.random.rand(n) * 1.0,
            "low": prices - np.random.rand(n) * 1.0,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
        })

    def test_all_indicators_returns_all_columns(self, sample_ohlcv):
        """Test all expected indicator columns are added."""
        result = calculate_all_indicators(sample_ohlcv)

        expected_columns = [
            "rsi_14", "ema_9", "ema_21", "ema_50",
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14", "volume_sma_20",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_indicators_preserves_original_data(self, sample_ohlcv):
        """Test original OHLCV columns are preserved."""
        result = calculate_all_indicators(sample_ohlcv)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(
                result[col],
                sample_ohlcv[col],
                check_names=True,
            )

    def test_all_indicators_with_uppercase_columns(self):
        """Test function handles uppercase column names."""
        np.random.seed(42)
        df = pd.DataFrame({
            "Open": [100, 101, 102] * 20,
            "High": [102, 103, 104] * 20,
            "Low": [98, 99, 100] * 20,
            "Close": [101, 102, 103] * 20,
            "Volume": [1000000] * 60,
        })

        result = calculate_all_indicators(df)

        assert "rsi_14" in result.columns

    def test_all_indicators_valid_values_at_end(self, sample_ohlcv):
        """Test indicators have valid values for last row (after warmup)."""
        result = calculate_all_indicators(sample_ohlcv)

        last_row = result.iloc[-1]
        assert not pd.isna(last_row["rsi_14"])
        assert not pd.isna(last_row["ema_21"])
        assert not pd.isna(last_row["bb_middle"])
        assert not pd.isna(last_row["volume_sma_20"])
