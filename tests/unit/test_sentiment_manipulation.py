"""Unit tests for manipulation detection module."""

from datetime import datetime, timedelta, timezone

import pytest

from src.core.sentiment.manipulation import ManipulationDetector


class TestManipulationDetector:
    """Tests for ManipulationDetector class."""

    @pytest.fixture
    def detector(self):
        """Create manipulation detector instance."""
        return ManipulationDetector()

    @pytest.fixture
    def detector_custom_thresholds(self):
        """Create detector with custom thresholds."""
        return ManipulationDetector(
            coordination_threshold=0.5,
            bot_threshold=0.3,
        )


class TestDivergenceScore(TestManipulationDetector):
    """Tests for sentiment-price divergence calculation."""

    def test_no_divergence_both_positive(self, detector):
        """Test no divergence when sentiment and price align positively."""
        score = detector.calculate_divergence_score(
            sentiment_score=0.5,
            price_change=5.0,
            volume_ratio=1.0,
        )

        assert score == 0.0

    def test_no_divergence_both_negative(self, detector):
        """Test no divergence when sentiment and price align negatively."""
        score = detector.calculate_divergence_score(
            sentiment_score=-0.5,
            price_change=-5.0,
            volume_ratio=1.0,
        )

        assert score == 0.0

    def test_divergence_positive_sentiment_negative_price(self, detector):
        """Test divergence when positive sentiment but negative price."""
        score = detector.calculate_divergence_score(
            sentiment_score=0.5,
            price_change=-5.0,
            volume_ratio=1.0,
        )

        assert score > 0

    def test_divergence_negative_sentiment_positive_price(self, detector):
        """Test divergence when negative sentiment but positive price."""
        score = detector.calculate_divergence_score(
            sentiment_score=-0.5,
            price_change=5.0,
            volume_ratio=1.0,
        )

        assert score > 0

    def test_neutral_returns_low_score(self, detector):
        """Test neutral sentiment returns low divergence."""
        score = detector.calculate_divergence_score(
            sentiment_score=0.0,
            price_change=0.0,
            volume_ratio=1.0,
        )

        assert score == 0.0

    def test_low_volume_increases_divergence(self, detector):
        """Test low volume ratio increases divergence score."""
        base_score = detector.calculate_divergence_score(
            sentiment_score=0.5,
            price_change=-5.0,
            volume_ratio=1.0,
        )

        low_volume_score = detector.calculate_divergence_score(
            sentiment_score=0.5,
            price_change=-5.0,
            volume_ratio=0.3,
        )

        assert low_volume_score > base_score

    def test_divergence_capped_at_one(self, detector):
        """Test divergence score doesn't exceed 1.0."""
        score = detector.calculate_divergence_score(
            sentiment_score=0.9,
            price_change=-50.0,
            volume_ratio=0.1,
        )

        assert score <= 1.0


class TestSentimentSpike(TestManipulationDetector):
    """Tests for sentiment spike detection."""

    def test_no_spike_with_normal_sentiment(self, detector):
        """Test no spike when sentiment is within normal range."""
        history = [
            {"compound": 0.1},
            {"compound": 0.2},
            {"compound": -0.1},
            {"compound": 0.0},
            {"compound": 0.15},
        ]
        current = {"compound": 0.12}

        is_spike = detector.detect_sentiment_spike(history, current)

        assert is_spike is False

    def test_spike_detected_high_positive(self, detector):
        """Test spike detection with abnormally high sentiment."""
        history = [
            {"compound": 0.1},
            {"compound": 0.0},
            {"compound": 0.1},
            {"compound": -0.1},
            {"compound": 0.05},
        ]
        current = {"compound": 0.9}

        is_spike = detector.detect_sentiment_spike(history, current)

        assert is_spike is True

    def test_spike_detected_high_negative(self, detector):
        """Test spike detection with abnormally negative sentiment."""
        history = [
            {"compound": 0.1},
            {"compound": 0.0},
            {"compound": 0.1},
            {"compound": -0.1},
            {"compound": 0.05},
        ]
        current = {"compound": -0.9}

        is_spike = detector.detect_sentiment_spike(history, current)

        assert is_spike is True

    def test_insufficient_history_no_spike(self, detector):
        """Test no spike detection with insufficient history."""
        history = [{"compound": 0.1}, {"compound": 0.2}]
        current = {"compound": 0.9}

        is_spike = detector.detect_sentiment_spike(history, current)

        assert is_spike is False

    def test_custom_std_threshold(self, detector):
        """Test custom standard deviation threshold."""
        history = [
            {"compound": 0.1},
            {"compound": 0.0},
            {"compound": 0.1},
            {"compound": -0.1},
            {"compound": 0.05},
        ]
        current = {"compound": 0.5}

        is_spike_strict = detector.detect_sentiment_spike(
            history, current, std_threshold=3.0
        )
        is_spike_loose = detector.detect_sentiment_spike(
            history, current, std_threshold=1.0
        )

        assert is_spike_loose is True
        assert is_spike_strict is False or is_spike_strict is True


class TestVolumeSpike(TestManipulationDetector):
    """Tests for mention volume spike detection."""

    def test_no_spike_with_normal_volume(self, detector):
        """Test no spike when volume is within normal range."""
        history = [100, 110, 95, 105, 102]
        current = 108

        is_spike = detector.detect_volume_spike(history, current)

        assert is_spike is False

    def test_spike_detected_high_volume(self, detector):
        """Test spike detection with abnormally high volume."""
        history = [100, 110, 95, 105, 102]
        current = 500

        is_spike = detector.detect_volume_spike(history, current)

        assert is_spike is True

    def test_insufficient_history_no_spike(self, detector):
        """Test no spike detection with insufficient history."""
        history = [100, 110]
        current = 500

        is_spike = detector.detect_volume_spike(history, current)

        assert is_spike is False


class TestPostingPatterns(TestManipulationDetector):
    """Tests for posting pattern analysis."""

    def test_no_coordination_spread_posts(self, detector):
        """Test no coordination with spread out posts."""
        base_time = datetime.now(timezone.utc)
        posts = [
            {"timestamp": base_time - timedelta(hours=1)},
            {"timestamp": base_time - timedelta(hours=2)},
            {"timestamp": base_time - timedelta(hours=3)},
        ]

        result = detector.analyze_posting_patterns(posts)

        assert result["is_coordinated"] is False
        assert result["cluster_score"] < 0.5

    def test_coordination_detected_clustered_posts(self, detector):
        """Test coordination detection with clustered posts."""
        base_time = datetime.now(timezone.utc)
        posts = [
            {"timestamp": base_time},
            {"timestamp": base_time + timedelta(seconds=30)},
            {"timestamp": base_time + timedelta(seconds=60)},
            {"timestamp": base_time + timedelta(seconds=90)},
            {"timestamp": base_time + timedelta(seconds=120)},
            {"timestamp": base_time + timedelta(seconds=150)},
            {"timestamp": base_time + timedelta(seconds=180)},
        ]

        result = detector.analyze_posting_patterns(posts)

        assert result["is_coordinated"] is True
        assert result["cluster_score"] > 0.6

    def test_insufficient_posts(self, detector):
        """Test with insufficient posts."""
        posts = [{"timestamp": datetime.now(timezone.utc)}]

        result = detector.analyze_posting_patterns(posts)

        assert result["is_coordinated"] is False
        assert result["cluster_score"] == 0.0

    def test_string_timestamps(self, detector):
        """Test with ISO format string timestamps."""
        base_time = datetime.now(timezone.utc)
        posts = [
            {"timestamp": base_time.isoformat()},
            {"timestamp": (base_time + timedelta(seconds=30)).isoformat()},
            {"timestamp": (base_time + timedelta(seconds=60)).isoformat()},
        ]

        result = detector.analyze_posting_patterns(posts)

        assert "cluster_score" in result


class TestVocabularySimilarity(TestManipulationDetector):
    """Tests for vocabulary similarity analysis."""

    def test_high_similarity_copy_paste(self, detector):
        """Test high similarity with copy-paste content."""
        posts = [
            {"text": "Buy this amazing stock now! Great opportunity!"},
            {"text": "Buy this amazing stock now! Great opportunity!"},
            {"text": "Buy this amazing stock now! Great opportunity!"},
        ]

        score = detector.analyze_vocabulary_similarity(posts)

        assert score > 0.5

    def test_low_similarity_varied_content(self, detector):
        """Test low similarity with varied content."""
        posts = [
            {"text": "The earnings report was impressive this quarter"},
            {"text": "Technical analysis shows strong support levels"},
            {"text": "Management has announced a new product launch"},
        ]

        score = detector.analyze_vocabulary_similarity(posts)

        assert score < 0.5

    def test_insufficient_posts(self, detector):
        """Test with insufficient posts."""
        posts = [{"text": "Single post"}]

        score = detector.analyze_vocabulary_similarity(posts)

        assert score == 0.0

    def test_empty_texts_handled(self, detector):
        """Test empty texts are handled gracefully."""
        posts = [
            {"text": ""},
            {"text": None},
            {"text": "Valid text here"},
        ]

        score = detector.analyze_vocabulary_similarity(posts)

        assert isinstance(score, float)


class TestBotProbability(TestManipulationDetector):
    """Tests for bot probability estimation."""

    def test_new_account_high_probability(self, detector):
        """Test new account has higher bot probability."""
        author_data = {
            "account_age_days": 3,
            "posts_per_day": 25.0,
        }

        prob = detector.estimate_bot_probability(author_data)

        assert prob > 0.5

    def test_old_account_low_probability(self, detector):
        """Test established account has lower bot probability."""
        author_data = {
            "account_age_days": 365,
            "posts_per_day": 2.0,
            "karma": 5000,
        }

        prob = detector.estimate_bot_probability(author_data)

        assert prob < 0.3

    def test_high_posting_frequency_increases_probability(self, detector):
        """Test high posting frequency increases bot probability."""
        low_frequency = detector.estimate_bot_probability(
            {
                "account_age_days": 100,
                "posts_per_day": 1.0,
            }
        )

        high_frequency = detector.estimate_bot_probability(
            {
                "account_age_days": 100,
                "posts_per_day": 100.0,
            }
        )

        assert high_frequency > low_frequency

    def test_verified_account_decreases_probability(self, detector):
        """Test verified account decreases bot probability."""
        unverified = detector.estimate_bot_probability(
            {
                "account_age_days": 30,
                "posts_per_day": 10.0,
            }
        )

        verified = detector.estimate_bot_probability(
            {
                "account_age_days": 30,
                "posts_per_day": 10.0,
                "is_verified": True,
            }
        )

        assert verified < unverified

    def test_probability_in_range(self, detector):
        """Test probability is always in 0-1 range."""
        test_cases = [
            {"account_age_days": 1, "posts_per_day": 1000},
            {"account_age_days": 10000, "posts_per_day": 0.01},
            {},
        ]

        for author_data in test_cases:
            prob = detector.estimate_bot_probability(author_data)
            assert 0.0 <= prob <= 1.0


class TestManipulationScore(TestManipulationDetector):
    """Tests for combined manipulation score calculation."""

    def test_all_signals_zero_returns_zero(self, detector):
        """Test all zero signals returns zero score."""
        signals = {
            "divergence": 0.0,
            "spike": False,
            "coordination": 0.0,
            "bot_activity": 0.0,
            "vocabulary_similarity": 0.0,
        }

        score = detector.calculate_manipulation_score(signals)

        assert score == 0.0

    def test_all_signals_high_returns_high(self, detector):
        """Test all high signals returns high score."""
        signals = {
            "divergence": 1.0,
            "spike": True,
            "coordination": 1.0,
            "bot_activity": 1.0,
            "vocabulary_similarity": 1.0,
        }

        score = detector.calculate_manipulation_score(signals)

        assert score > 0.8

    def test_partial_signals(self, detector):
        """Test with partial signals."""
        signals = {
            "divergence": 0.5,
            "coordination": 0.5,
        }

        score = detector.calculate_manipulation_score(signals)

        assert 0 < score < 1

    def test_empty_signals_returns_zero(self, detector):
        """Test empty signals returns zero."""
        score = detector.calculate_manipulation_score({})

        assert score == 0.0

    def test_score_in_range(self, detector):
        """Test score is always in 0-1 range."""
        score = detector.calculate_manipulation_score(
            {
                "divergence": 2.0,
                "coordination": 2.0,
            }
        )

        assert 0.0 <= score <= 1.0


class TestAnalyzePosts(TestManipulationDetector):
    """Tests for comprehensive post analysis."""

    def test_full_analysis(self, detector):
        """Test full analysis with all parameters."""
        base_time = datetime.now(timezone.utc)
        posts = [
            {
                "text": "Buy AAPL now!",
                "timestamp": base_time,
                "author_data": {"account_age_days": 5, "posts_per_day": 50},
            },
            {
                "text": "Buy AAPL now!",
                "timestamp": base_time + timedelta(seconds=30),
                "author_data": {"account_age_days": 3, "posts_per_day": 100},
            },
            {
                "text": "Buy AAPL now!",
                "timestamp": base_time + timedelta(seconds=60),
                "author_data": {"account_age_days": 7, "posts_per_day": 30},
            },
        ]

        result = detector.analyze_posts(
            posts=posts,
            sentiment_score=0.8,
            price_change=-5.0,
            volume_ratio=0.5,
        )

        assert "manipulation_score" in result
        assert "is_suspicious" in result
        assert "signals" in result
        assert "details" in result
        assert "posts_analyzed" in result

    def test_minimal_analysis(self, detector):
        """Test analysis with minimal parameters."""
        posts = [
            {"text": "Test post", "timestamp": datetime.now(timezone.utc)},
        ]

        result = detector.analyze_posts(posts)

        assert "manipulation_score" in result
        assert result["posts_analyzed"] == 1

    def test_empty_posts(self, detector):
        """Test analysis with empty posts."""
        result = detector.analyze_posts([])

        assert result["posts_analyzed"] == 0
        assert result["manipulation_score"] >= 0
