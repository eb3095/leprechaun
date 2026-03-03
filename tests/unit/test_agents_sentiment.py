"""Unit tests for sentiment agent."""

from datetime import datetime, timedelta

import pytest

from src.agents.sentiment_agent import SentimentAgent, SentimentResult, SourceWeight


class TestSourceWeight:
    """Tests for SourceWeight dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = SourceWeight()

        assert weights.reddit == 0.4
        assert weights.stocktwits == 0.3
        assert weights.news == 0.3

    def test_custom_weights(self):
        """Test creating with custom weights."""
        weights = SourceWeight(reddit=0.5, stocktwits=0.25, news=0.25)

        assert weights.reddit == 0.5
        assert weights.stocktwits == 0.25


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_create_result(self):
        """Test creating a sentiment result."""
        result = SentimentResult(
            composite_score=-0.35,
            volume=150,
            velocity=-0.5,
            sources={"reddit": {"avg_score": -0.4, "volume": 100}},
            anomalies=[],
        )

        assert result.composite_score == -0.35
        assert result.volume == 150
        assert result.velocity == -0.5

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = SentimentResult(
            composite_score=-0.35,
            volume=150,
            velocity=-0.5,
            sources={},
            anomalies=[],
            timestamp=datetime(2026, 1, 15, 12, 0),
        )

        data = result.to_dict()

        assert data["composite_score"] == -0.35
        assert data["timestamp"] == "2026-01-15T12:00:00"


class TestSentimentAgent:
    """Tests for SentimentAgent."""

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        agent = SentimentAgent()

        assert agent.source_weights.reddit == 0.4
        assert agent.source_weights.stocktwits == 0.3

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        agent = SentimentAgent(source_weights={"reddit": 0.5, "news": 0.3})

        assert agent.source_weights.reddit == 0.5
        assert agent.source_weights.news == 0.3

    def test_aggregate_sentiment_empty(self):
        """Test aggregation with no data."""
        agent = SentimentAgent()

        result = agent.aggregate_sentiment("AAPL", [])

        assert result.composite_score == 0.0
        assert result.volume == 0
        assert result.velocity == 0.0

    def test_aggregate_sentiment_single_source(self):
        """Test aggregation with single source."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -0.5, "volume": 100},
            {"source": "reddit", "score": -0.3, "volume": 50},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert -0.5 <= result.composite_score <= -0.3
        assert result.volume == 150
        assert "reddit" in result.sources

    def test_aggregate_sentiment_multiple_sources(self):
        """Test aggregation with multiple sources."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -0.5, "volume": 100},
            {"source": "stocktwits", "score": -0.3, "volume": 80},
            {"source": "news", "score": -0.4, "volume": 50},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert -0.6 <= result.composite_score <= -0.2
        assert result.volume == 230
        assert len(result.sources) == 3

    def test_aggregate_sentiment_weighted(self):
        """Test that weighting affects composite score."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -0.8, "volume": 100},
            {"source": "news", "score": 0.2, "volume": 100},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert result.composite_score < 0

    def test_aggregate_sentiment_score_bounded(self):
        """Test composite score is bounded -1 to 1."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -1.5, "volume": 100},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert result.composite_score >= -1.0
        assert result.composite_score <= 1.0

    def test_detect_anomalies_insufficient_data(self):
        """Test anomaly detection with insufficient data."""
        agent = SentimentAgent()
        data = [
            {"score": -0.5, "volume": 100},
        ]

        anomalies = agent.detect_anomalies(data)

        assert anomalies == []

    def test_detect_anomalies_score_spike(self):
        """Test detection of score anomaly."""
        agent = SentimentAgent()
        data = [
            {"score": 0.1, "volume": 50},
            {"score": 0.11, "volume": 55},
            {"score": 0.09, "volume": 48},
            {"score": 0.1, "volume": 52},
            {"score": 0.1, "volume": 51},
            {"score": 0.1, "volume": 53},
            {"score": 0.1, "volume": 50},
            {"score": -0.9, "volume": 100},
        ]

        anomalies = agent.detect_anomalies(data)

        score_anomalies = [a for a in anomalies if a["type"] == "score_anomaly"]
        assert len(score_anomalies) > 0
        assert score_anomalies[0]["details"]["direction"] == "negative"

    def test_detect_anomalies_volume_spike(self):
        """Test detection of volume anomaly."""
        agent = SentimentAgent()
        data = [
            {"score": 0.1, "volume": 50},
            {"score": 0.1, "volume": 52},
            {"score": 0.1, "volume": 48},
            {"score": 0.1, "volume": 51},
            {"score": 0.1, "volume": 50},
            {"score": 0.1, "volume": 49},
            {"score": 0.1, "volume": 51},
            {"score": 0.1, "volume": 500},
        ]

        anomalies = agent.detect_anomalies(data)

        volume_anomalies = [a for a in anomalies if a["type"] == "volume_anomaly"]
        assert len(volume_anomalies) > 0

    def test_calculate_velocity_insufficient_data(self):
        """Test velocity calculation with insufficient data."""
        agent = SentimentAgent()
        data = [
            {"score": -0.5, "timestamp": datetime.utcnow().isoformat()},
        ]

        velocity = agent.calculate_velocity(data)

        assert velocity == 0.0

    def test_calculate_velocity_declining(self):
        """Test velocity calculation for declining sentiment."""
        agent = SentimentAgent()
        now = datetime.utcnow()

        data = []
        for i in range(10):
            data.append(
                {
                    "score": 0.5 - (i * 0.1),
                    "timestamp": (now - timedelta(hours=5 - i * 0.5)).isoformat(),
                }
            )

        velocity = agent.calculate_velocity(data)

        assert velocity < 0

    def test_calculate_velocity_improving(self):
        """Test velocity calculation for improving sentiment."""
        agent = SentimentAgent()
        now = datetime.utcnow()

        data = []
        for i in range(10):
            data.append(
                {
                    "score": -0.5 + (i * 0.1),
                    "timestamp": (now - timedelta(hours=5 - i * 0.5)).isoformat(),
                }
            )

        velocity = agent.calculate_velocity(data)

        assert velocity > 0

    def test_calculate_velocity_bounded(self):
        """Test velocity is bounded -1 to 1."""
        agent = SentimentAgent()
        now = datetime.utcnow()

        data = []
        for i in range(10):
            data.append(
                {
                    "score": 0.9 - (i * 0.2),
                    "timestamp": (now - timedelta(hours=5 - i * 0.5)).isoformat(),
                }
            )

        velocity = agent.calculate_velocity(data)

        assert velocity >= -1.0
        assert velocity <= 1.0

    def test_calculate_velocity_window(self):
        """Test velocity respects time window."""
        agent = SentimentAgent()
        now = datetime.utcnow()

        data = [
            {"score": 0.5, "timestamp": (now - timedelta(hours=24)).isoformat()},
            {"score": -0.5, "timestamp": (now - timedelta(hours=23)).isoformat()},
            {"score": 0.1, "timestamp": (now - timedelta(hours=2)).isoformat()},
            {"score": 0.1, "timestamp": (now - timedelta(hours=1)).isoformat()},
            {"score": 0.1, "timestamp": now.isoformat()},
        ]

        velocity = agent.calculate_velocity(data, window=6)

        assert abs(velocity) < 0.3

    def test_get_source_breakdown(self):
        """Test getting detailed source breakdown."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -0.5, "volume": 100},
            {"source": "reddit", "score": -0.3, "volume": 50},
            {"source": "news", "score": -0.4, "volume": 30},
        ]

        breakdown = agent.get_source_breakdown(data)

        assert "reddit" in breakdown
        assert "news" in breakdown
        assert breakdown["reddit"]["record_count"] == 2
        assert breakdown["reddit"]["total_volume"] == 150
        assert "mean_score" in breakdown["reddit"]
        assert "std_dev" in breakdown["reddit"]

    def test_is_sentiment_spike_true(self):
        """Test spike detection when score is below threshold."""
        agent = SentimentAgent()
        result = SentimentResult(
            composite_score=-0.5,
            volume=100,
            velocity=-0.3,
            sources={},
            anomalies=[],
        )

        assert agent.is_sentiment_spike(result) is True

    def test_is_sentiment_spike_false(self):
        """Test spike detection when score is above threshold."""
        agent = SentimentAgent()
        result = SentimentResult(
            composite_score=-0.2,
            volume=100,
            velocity=-0.1,
            sources={},
            anomalies=[],
        )

        assert agent.is_sentiment_spike(result) is False

    def test_is_sentiment_spike_custom_threshold(self):
        """Test spike detection with custom threshold."""
        agent = SentimentAgent()
        result = SentimentResult(
            composite_score=-0.3,
            volume=100,
            velocity=-0.1,
            sources={},
            anomalies=[],
        )

        assert agent.is_sentiment_spike(result, threshold=-0.2) is True
        assert agent.is_sentiment_spike(result, threshold=-0.5) is False

    def test_compare_sentiment(self):
        """Test comparing two sentiment results."""
        agent = SentimentAgent()

        previous = SentimentResult(
            composite_score=-0.5,
            volume=100,
            velocity=-0.3,
            sources={},
            anomalies=[],
        )

        current = SentimentResult(
            composite_score=-0.3,
            volume=150,
            velocity=-0.1,
            sources={},
            anomalies=[],
        )

        comparison = agent.compare_sentiment(current, previous)

        assert abs(comparison["score_change"] - 0.2) < 0.01
        assert abs(comparison["volume_change_pct"] - 50.0) < 0.01
        assert abs(comparison["velocity_change"] - 0.2) < 0.01
        assert comparison["is_improving"] is True
        assert comparison["is_accelerating"] is True
        assert comparison["volume_increasing"] is True

    def test_get_weights(self):
        """Test getting weights as dictionary."""
        agent = SentimentAgent()

        weights = agent.get_weights()

        assert weights["reddit"] == 0.4
        assert weights["stocktwits"] == 0.3
        assert weights["news"] == 0.3

    def test_set_weights(self):
        """Test setting weights from dictionary."""
        agent = SentimentAgent()

        agent.set_weights({"reddit": 0.5, "news": 0.25})

        assert agent.source_weights.reddit == 0.5
        assert agent.source_weights.news == 0.25

    def test_set_weights_clamped(self):
        """Test weights are clamped to 0-1."""
        agent = SentimentAgent()

        agent.set_weights({"reddit": 1.5, "news": -0.5})

        assert agent.source_weights.reddit == 1.0
        assert agent.source_weights.news == 0.0

    def test_aggregate_handles_missing_volume(self):
        """Test aggregation handles missing volume gracefully."""
        agent = SentimentAgent()
        data = [
            {"source": "reddit", "score": -0.5},
            {"source": "reddit", "score": -0.3},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert result.composite_score != 0.0
        assert result.volume == 2

    def test_aggregate_handles_unknown_source(self):
        """Test aggregation handles unknown sources."""
        agent = SentimentAgent()
        data = [
            {"source": "unknown_source", "score": -0.5, "volume": 100},
        ]

        result = agent.aggregate_sentiment("AAPL", data)

        assert "unknown_source" in result.sources
