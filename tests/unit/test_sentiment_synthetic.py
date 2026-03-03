"""Unit tests for synthetic sentiment generator module."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.sentiment.synthetic import (
    BacktestSentimentProvider,
    SyntheticSentimentGenerator,
)


class TestSyntheticSentimentGenerator:
    """Tests for SyntheticSentimentGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator with fixed seed for reproducibility."""
        return SyntheticSentimentGenerator(seed=42)

    @pytest.fixture
    def price_df(self):
        """Create sample price DataFrame."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        np.random.seed(42)

        base_price = 100
        returns = np.random.normal(0, 0.02, size=30)
        prices = base_price * (1 + returns).cumprod()

        return pd.DataFrame({
            "date": dates.date,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, size=30),
        })

    def test_init_default_values(self):
        """Test initialization with default values."""
        gen = SyntheticSentimentGenerator()

        assert gen.noise_level == SyntheticSentimentGenerator.DEFAULT_NOISE_LEVEL
        assert gen.volume_weight == SyntheticSentimentGenerator.DEFAULT_VOLUME_WEIGHT

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        gen = SyntheticSentimentGenerator(
            seed=123,
            noise_level=0.25,
            volume_weight=0.5,
        )

        assert gen.noise_level == 0.25
        assert gen.volume_weight == 0.5

    def test_init_clamps_noise_level(self):
        """Test noise level is clamped to valid range."""
        gen_high = SyntheticSentimentGenerator(noise_level=2.0)
        gen_low = SyntheticSentimentGenerator(noise_level=-0.5)

        assert gen_high.noise_level == 1.0
        assert gen_low.noise_level == 0.0

    def test_generate_for_period_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        result = generator.generate_for_period(empty_df, "AAPL")

        assert result.empty

    def test_generate_for_period_returns_correct_columns(self, generator, price_df):
        """Test generated DataFrame has correct columns."""
        result = generator.generate_for_period(price_df, "AAPL")

        expected_columns = [
            "date",
            "sentiment_score",
            "sentiment_volume",
            "manipulation_score",
            "is_synthetic",
        ]
        assert list(result.columns) == expected_columns

    def test_generate_for_period_correct_length(self, generator, price_df):
        """Test generated DataFrame has same length as input."""
        result = generator.generate_for_period(price_df, "AAPL")

        assert len(result) == len(price_df)

    def test_generate_for_period_all_synthetic_flag(self, generator, price_df):
        """Test all rows have is_synthetic=True."""
        result = generator.generate_for_period(price_df, "AAPL")

        assert result["is_synthetic"].all()

    def test_generate_for_period_sentiment_in_range(self, generator, price_df):
        """Test sentiment scores are in valid range."""
        result = generator.generate_for_period(price_df, "AAPL")

        assert (result["sentiment_score"] >= -1).all()
        assert (result["sentiment_score"] <= 1).all()

    def test_generate_for_period_manipulation_in_range(self, generator, price_df):
        """Test manipulation scores are in valid range."""
        result = generator.generate_for_period(price_df, "AAPL")

        assert (result["manipulation_score"] >= 0).all()
        assert (result["manipulation_score"] <= 1).all()

    def test_generate_for_period_volume_positive(self, generator, price_df):
        """Test synthetic volume is always positive."""
        result = generator.generate_for_period(price_df, "AAPL")

        assert (result["sentiment_volume"] > 0).all()

    def test_generate_for_period_reproducible(self, price_df):
        """Test results are reproducible with same seed."""
        gen1 = SyntheticSentimentGenerator(seed=42)
        gen2 = SyntheticSentimentGenerator(seed=42)

        result1 = gen1.generate_for_period(price_df, "AAPL")
        result2 = gen2.generate_for_period(price_df, "AAPL")

        pd.testing.assert_frame_equal(result1, result2)

    def test_generate_for_period_different_with_different_seed(self, price_df):
        """Test results differ with different seeds."""
        gen1 = SyntheticSentimentGenerator(seed=42)
        gen2 = SyntheticSentimentGenerator(seed=99)

        result1 = gen1.generate_for_period(price_df, "AAPL")
        result2 = gen2.generate_for_period(price_df, "AAPL")

        assert not result1["sentiment_score"].equals(result2["sentiment_score"])

    def test_calculate_price_sentiment_negative_returns(self, generator):
        """Test negative returns produce negative sentiment."""
        returns = pd.Series([-0.05, -0.03, -0.02, -0.01, 0.0])

        sentiment = generator._calculate_price_sentiment(returns)

        assert sentiment.iloc[-1] > 0

    def test_calculate_price_sentiment_positive_returns(self, generator):
        """Test positive returns produce positive sentiment."""
        returns = pd.Series([0.05, 0.03, 0.02, 0.01, 0.0])

        sentiment = generator._calculate_price_sentiment(returns)

        assert sentiment.iloc[-1] < 0

    def test_calculate_volume_anomaly(self, generator):
        """Test volume anomaly detection."""
        normal_vol = [1000000] * 19 + [5000000]
        volume = pd.Series(normal_vol)

        anomaly = generator._calculate_volume_anomaly(volume)

        assert anomaly.iloc[-1] > anomaly.iloc[0]

    def test_calculate_manipulation_proxy_price_drop_low_volume(self, generator):
        """Test manipulation detection on price drop with low volume."""
        returns = pd.Series([0.0] * 19 + [-0.05])
        volume = pd.Series([1000000] * 19 + [500000])

        manip = generator._calculate_manipulation_proxy(returns, volume)

        assert manip.iloc[-1] > 0

    def test_add_noise_zero_noise_level(self, generator):
        """Test no noise is added when noise_level=0."""
        generator.noise_level = 0
        series = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])

        result = generator._add_noise(series)

        pd.testing.assert_series_equal(result, series)

    def test_set_noise_level(self, generator):
        """Test set_noise_level method."""
        generator.set_noise_level(0.3)
        assert generator.noise_level == 0.3

        generator.set_noise_level(1.5)
        assert generator.noise_level == 1.0

    def test_set_seed(self, generator, price_df):
        """Test set_seed method resets generator."""
        result1 = generator.generate_for_period(price_df, "AAPL")

        generator.set_seed(42)
        result2 = generator.generate_for_period(price_df, "AAPL")

        pd.testing.assert_frame_equal(result1, result2)


class TestBacktestSentimentProvider:
    """Tests for BacktestSentimentProvider class."""

    @pytest.fixture
    def mock_archive(self):
        """Create mock SentimentArchive."""
        archive = MagicMock()
        archive.session = MagicMock()
        return archive

    @pytest.fixture
    def generator(self):
        """Create generator with fixed seed."""
        return SyntheticSentimentGenerator(seed=42)

    @pytest.fixture
    def provider(self, mock_archive, generator):
        """Create BacktestSentimentProvider."""
        return BacktestSentimentProvider(mock_archive, generator)

    @pytest.fixture
    def price_df(self):
        """Create sample price DataFrame."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        return pd.DataFrame({
            "date": dates.date,
            "close": [100 + i * 0.5 for i in range(30)],
            "volume": [1000000] * 30,
            "open": [100 + i * 0.5 for i in range(30)],
            "high": [100 + i * 0.5 + 1 for i in range(30)],
            "low": [100 + i * 0.5 - 1 for i in range(30)],
        })

    def test_init_default_coverage(self, mock_archive, generator):
        """Test initialization with default min_coverage."""
        provider = BacktestSentimentProvider(mock_archive, generator)

        assert provider.min_coverage == BacktestSentimentProvider.DEFAULT_MIN_COVERAGE

    def test_init_custom_coverage(self, mock_archive, generator):
        """Test initialization with custom min_coverage."""
        provider = BacktestSentimentProvider(mock_archive, generator, min_coverage=0.5)

        assert provider.min_coverage == 0.5

    def test_get_sentiment_force_synthetic(self, provider, price_df):
        """Test forcing synthetic sentiment generation."""
        result = provider.get_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 30),
            price_df=price_df,
            force_synthetic=True,
        )

        assert not result.empty
        assert result["is_synthetic"].all()

    def test_get_sentiment_uses_archive_when_available(self, provider, mock_archive):
        """Test using archived data when coverage is sufficient."""
        archived_data = [
            {"date": date(2024, 1, i), "sentiment_score": 0.5, "manipulation_score": 0.3}
            for i in range(1, 31)
        ]
        mock_archive.get_daily_sentiment.return_value = archived_data

        result = provider.get_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 30),
        )

        assert len(result) == 30
        mock_archive.get_daily_sentiment.assert_called_once()

    def test_get_sentiment_falls_back_to_synthetic(self, provider, mock_archive, price_df):
        """Test falling back to synthetic when no archive data."""
        mock_archive.get_daily_sentiment.return_value = []

        result = provider.get_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 30),
            price_df=price_df,
        )

        assert not result.empty
        assert result["is_synthetic"].all()

    def test_get_sentiment_blends_data(self, provider, mock_archive, price_df):
        """Test blending archived and synthetic data for partial coverage."""
        archived_data = [
            {
                "date": date(2024, 1, i),
                "sentiment_score": 0.5,
                "manipulation_score": 0.3,
                "is_synthetic": False,
            }
            for i in range(1, 11)
        ]
        mock_archive.get_daily_sentiment.return_value = archived_data

        result = provider.get_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 30),
            price_df=price_df,
        )

        assert len(result) == 30
        archived_rows = result[result["is_synthetic"] == False]
        synthetic_rows = result[result["is_synthetic"] == True]
        assert len(archived_rows) == 10
        assert len(synthetic_rows) == 20

    def test_get_sentiment_no_data_no_price(self, provider, mock_archive):
        """Test returns empty when no archive and no price data."""
        mock_archive.session = None

        result = provider.get_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 30),
            price_df=None,
        )

        assert result.empty

    def test_get_sentiment_at_date_returns_dict(self, provider, mock_archive, price_df):
        """Test get_sentiment_at_date returns dictionary."""
        mock_archive.get_daily_sentiment.return_value = []

        result = provider.get_sentiment_at_date(
            "AAPL",
            date(2024, 1, 15),
            price_history=price_df,
        )

        assert isinstance(result, dict)
        assert "date" in result
        assert "sentiment_score" in result
        assert "manipulation_score" in result
        assert "is_synthetic" in result

    def test_get_sentiment_at_date_target_date(self, provider, mock_archive, price_df):
        """Test get_sentiment_at_date returns data for target date."""
        mock_archive.get_daily_sentiment.return_value = []

        target = date(2024, 1, 15)
        result = provider.get_sentiment_at_date(
            "AAPL",
            target,
            price_history=price_df,
        )

        assert result["date"] == target

    def test_get_coverage_info_with_session(self, provider, mock_archive):
        """Test coverage info retrieval."""
        mock_archive.get_coverage_report.return_value = {
            "total_days": 30,
            "symbol_coverage": {"AAPL": {"coverage_percent": 50.0}},
            "overall_coverage": 50.0,
        }

        result = provider.get_coverage_info(
            ["AAPL"],
            date(2024, 1, 1),
            date(2024, 1, 30),
        )

        assert result["overall_coverage"] == 50.0
        mock_archive.get_coverage_report.assert_called_once()

    def test_get_coverage_info_no_session(self, generator):
        """Test coverage info without database session."""
        mock_archive = MagicMock()
        mock_archive.session = None
        provider = BacktestSentimentProvider(mock_archive, generator)

        result = provider.get_coverage_info(
            ["AAPL", "MSFT"],
            date(2024, 1, 1),
            date(2024, 1, 30),
        )

        assert result["overall_coverage"] == 0.0
        assert "AAPL" in result["symbol_coverage"]
        assert "MSFT" in result["symbol_coverage"]
