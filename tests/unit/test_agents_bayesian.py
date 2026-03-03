"""Unit tests for Bayesian manipulation detector."""

import math

import pytest

from src.agents.bayesian import BayesianManipulationDetector, EvidenceWeights


class TestEvidenceWeights:
    """Tests for EvidenceWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = EvidenceWeights()

        assert weights.sentiment_spike == 3.0
        assert weights.no_news_catalyst == 2.5
        assert weights.coordination_detected == 4.0
        assert weights.high_bot_activity == 3.5
        assert weights.volume_sentiment_divergence == 2.0

    def test_custom_weights(self):
        """Test creating with custom weights."""
        weights = EvidenceWeights(
            sentiment_spike=5.0,
            coordination_detected=6.0,
        )

        assert weights.sentiment_spike == 5.0
        assert weights.coordination_detected == 6.0
        assert weights.no_news_catalyst == 2.5


class TestBayesianManipulationDetector:
    """Tests for BayesianManipulationDetector."""

    def test_init_default_prior(self):
        """Test initialization with default prior."""
        detector = BayesianManipulationDetector()

        assert detector.prior_manipulation == 0.1

    def test_init_custom_prior(self):
        """Test initialization with custom prior."""
        detector = BayesianManipulationDetector(prior_manipulation=0.2)

        assert detector.prior_manipulation == 0.2

    def test_init_prior_clamped_max(self):
        """Test prior is clamped to max value."""
        detector = BayesianManipulationDetector(prior_manipulation=0.9)

        assert detector.prior_manipulation == 0.5

    def test_init_prior_clamped_min(self):
        """Test prior is clamped to min value."""
        detector = BayesianManipulationDetector(prior_manipulation=0.001)

        assert detector.prior_manipulation == 0.01

    def test_init_custom_evidence_weights(self):
        """Test initialization with custom evidence weights."""
        detector = BayesianManipulationDetector(
            evidence_weights={"sentiment_spike": 5.0, "high_bot_activity": 2.0}
        )

        assert detector.evidence_weights.sentiment_spike == 5.0
        assert detector.evidence_weights.high_bot_activity == 2.0
        assert detector.evidence_weights.no_news_catalyst == 2.5

    def test_calculate_posterior_empty_evidence(self):
        """Test posterior with no evidence returns prior."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_posterior({})

        assert posterior == detector.prior_manipulation

    def test_calculate_posterior_single_positive_evidence(self):
        """Test posterior increases with positive evidence."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_posterior({"sentiment_spike": True})

        assert posterior > detector.prior_manipulation

    def test_calculate_posterior_single_negative_evidence(self):
        """Test posterior decreases with negative evidence."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_posterior({"sentiment_spike": False})

        assert posterior < detector.prior_manipulation

    def test_calculate_posterior_multiple_positive(self):
        """Test posterior with multiple positive evidence."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_posterior(
            {
                "sentiment_spike": True,
                "no_news_catalyst": True,
                "coordination_detected": True,
            }
        )

        assert posterior > 0.7

    def test_calculate_posterior_mixed_evidence(self):
        """Test posterior with mixed evidence."""
        detector = BayesianManipulationDetector()

        posterior_all_positive = detector.calculate_posterior(
            {
                "sentiment_spike": True,
                "no_news_catalyst": True,
            }
        )

        posterior_mixed = detector.calculate_posterior(
            {
                "sentiment_spike": True,
                "no_news_catalyst": False,
            }
        )

        assert posterior_mixed < posterior_all_positive

    def test_calculate_posterior_bounded(self):
        """Test posterior is bounded between 0.001 and 0.999."""
        detector = BayesianManipulationDetector()

        all_positive = {
            "sentiment_spike": True,
            "no_news_catalyst": True,
            "coordination_detected": True,
            "high_bot_activity": True,
            "volume_sentiment_divergence": True,
        }
        high_posterior = detector.calculate_posterior(all_positive)

        assert high_posterior <= 0.999

        all_negative = {k: False for k in all_positive}
        low_posterior = detector.calculate_posterior(all_negative)

        assert low_posterior >= 0.001

    def test_calculate_log_posterior(self):
        """Test log-odds calculation matches regular calculation."""
        detector = BayesianManipulationDetector()
        evidence = {"sentiment_spike": True, "no_news_catalyst": True}

        regular = detector.calculate_posterior(evidence)
        log_based = detector.calculate_log_posterior(evidence)

        assert abs(regular - log_based) < 0.01

    def test_calculate_log_posterior_empty(self):
        """Test log posterior with empty evidence."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_log_posterior({})

        assert posterior == detector.prior_manipulation

    def test_update_priors_empty_data(self):
        """Test updating priors with empty data."""
        detector = BayesianManipulationDetector()
        original_prior = detector.prior_manipulation

        result = detector.update_priors([])

        assert result["prior_manipulation"] == original_prior

    def test_update_priors_with_data(self):
        """Test updating priors with historical data."""
        detector = BayesianManipulationDetector()

        historical_data = [
            {"was_manipulation": True, "evidence": {"sentiment_spike": True}},
            {"was_manipulation": True, "evidence": {"sentiment_spike": True}},
            {"was_manipulation": False, "evidence": {"sentiment_spike": False}},
            {"was_manipulation": False, "evidence": {"sentiment_spike": False}},
            {"was_manipulation": False, "evidence": {"sentiment_spike": False}},
        ]

        result = detector.update_priors(historical_data)

        assert "prior_manipulation" in result
        assert 0.3 <= result["prior_manipulation"] <= 0.5

    def test_get_confidence_interval_low_sample(self):
        """Test confidence interval with low sample size."""
        detector = BayesianManipulationDetector()

        lower, upper = detector.get_confidence_interval(0.5, sample_size=0)

        assert lower == 0.0
        assert upper == 1.0

    def test_get_confidence_interval_high_sample(self):
        """Test confidence interval narrows with larger sample."""
        detector = BayesianManipulationDetector()

        lower_small, upper_small = detector.get_confidence_interval(0.5, sample_size=10)
        lower_large, upper_large = detector.get_confidence_interval(
            0.5, sample_size=1000
        )

        small_width = upper_small - lower_small
        large_width = upper_large - lower_large

        assert large_width < small_width

    def test_get_confidence_interval_bounded(self):
        """Test confidence interval is bounded 0-1."""
        detector = BayesianManipulationDetector()

        lower, upper = detector.get_confidence_interval(0.9, sample_size=10)

        assert lower >= 0.0
        assert upper <= 1.0

    def test_get_evidence_contribution(self):
        """Test evidence contribution calculation."""
        detector = BayesianManipulationDetector()
        evidence = {
            "sentiment_spike": True,
            "no_news_catalyst": False,
        }

        contributions = detector.get_evidence_contribution(evidence)

        assert "sentiment_spike" in contributions
        assert "no_news_catalyst" in contributions
        assert contributions["sentiment_spike"] > 0
        assert contributions["no_news_catalyst"] < 0

    def test_explain_posterior(self):
        """Test posterior explanation generation."""
        detector = BayesianManipulationDetector()
        evidence = {
            "sentiment_spike": True,
            "no_news_catalyst": True,
            "coordination_detected": True,
        }

        explanation = detector.explain_posterior(evidence)

        assert "posterior" in explanation
        assert "prior" in explanation
        assert "contributions" in explanation
        assert "interpretation" in explanation
        assert "top_factors" in explanation
        assert explanation["posterior"] > 0.5
        assert "High probability" in explanation["interpretation"]

    def test_explain_posterior_low_probability(self):
        """Test explanation for low manipulation probability."""
        detector = BayesianManipulationDetector()
        evidence = {
            "sentiment_spike": False,
            "no_news_catalyst": False,
        }

        explanation = detector.explain_posterior(evidence)

        assert "unlikely" in explanation["interpretation"].lower()

    def test_get_weights(self):
        """Test getting weights as dictionary."""
        detector = BayesianManipulationDetector()

        weights = detector.get_weights()

        assert isinstance(weights, dict)
        assert "sentiment_spike" in weights
        assert weights["sentiment_spike"] == 3.0

    def test_set_weights(self):
        """Test setting weights from dictionary."""
        detector = BayesianManipulationDetector()

        detector.set_weights({"sentiment_spike": 5.0, "high_bot_activity": 2.0})

        assert detector.evidence_weights.sentiment_spike == 5.0
        assert detector.evidence_weights.high_bot_activity == 2.0

    def test_set_weights_clamps_minimum(self):
        """Test weights are clamped to minimum 0.1."""
        detector = BayesianManipulationDetector()

        detector.set_weights({"sentiment_spike": -1.0})

        assert detector.evidence_weights.sentiment_spike == 0.1

    def test_unknown_evidence_type_ignored(self):
        """Test unknown evidence types don't break calculation."""
        detector = BayesianManipulationDetector()

        posterior = detector.calculate_posterior(
            {
                "sentiment_spike": True,
                "unknown_evidence": True,
            }
        )

        expected = detector.calculate_posterior({"sentiment_spike": True})
        assert abs(posterior - expected) < 0.01
