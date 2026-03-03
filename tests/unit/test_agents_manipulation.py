"""Unit tests for manipulation detection agent."""

from datetime import datetime, timedelta

import pytest

from src.agents.bayesian import BayesianManipulationDetector
from src.agents.manipulation_agent import ManipulationAgent


class TestManipulationAgent:
    """Tests for ManipulationAgent."""

    def test_init_default_bayesian(self):
        """Test initialization creates default Bayesian detector."""
        agent = ManipulationAgent()

        assert agent.bayesian is not None
        assert isinstance(agent.bayesian, BayesianManipulationDetector)

    def test_init_custom_bayesian(self):
        """Test initialization with custom Bayesian detector."""
        bayesian = BayesianManipulationDetector(prior_manipulation=0.2)
        agent = ManipulationAgent(bayesian=bayesian)

        assert agent.bayesian.prior_manipulation == 0.2

    def test_analyze_basic(self):
        """Test basic analysis with minimal data."""
        agent = ManipulationAgent()

        result = agent.analyze(
            "AAPL",
            {
                "sentiment": {"composite_score": -0.5},
                "price_history": [],
                "news": [],
                "technical_indicators": {},
                "posts": [],
            },
        )

        assert "manipulation_score" in result
        assert "bayesian_probability" in result
        assert "evidence" in result
        assert "confidence" in result
        assert "recommendation" in result
        assert result["symbol"] == "AAPL"

    def test_analyze_high_manipulation(self):
        """Test analysis with high manipulation signals."""
        agent = ManipulationAgent()

        data = {
            "sentiment": {"composite_score": -0.6},
            "price_history": [],
            "news": [],
            "technical_indicators": {},
            "posts": [],
        }

        result = agent.analyze("AAPL", data)

        assert result["evidence"]["sentiment_spike"] is True
        assert result["evidence"]["no_news_catalyst"] is True

    def test_analyze_with_news_catalyst(self):
        """Test analysis when news catalyst exists."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        data = {
            "sentiment": {"composite_score": -0.6},
            "sentiment_spike_time": now,
            "price_history": [],
            "news": [
                {
                    "title": "Company announces major lawsuit",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "sentiment": -0.5,
                }
            ],
            "technical_indicators": {},
            "posts": [],
        }

        result = agent.analyze("AAPL", data)

        assert result["evidence"]["no_news_catalyst"] is False

    def test_analyze_recommendation_high_score(self):
        """Test recommendation for high manipulation score."""
        bayesian = BayesianManipulationDetector(prior_manipulation=0.4)
        agent = ManipulationAgent(bayesian=bayesian)

        now = datetime.utcnow()
        posts = []
        for i in range(10):
            posts.append(
                {
                    "text": "buy this stock to the moon diamond hands",
                    "timestamp": (now + timedelta(seconds=i * 30)).isoformat(),
                    "author": f"user{i}1234567",
                    "account_age_days": 10,
                }
            )

        data = {
            "sentiment": {"composite_score": -0.7, "volume": 500},
            "price_history": [{"close": 100}, {"close": 99}],
            "news": [],
            "technical_indicators": {
                "volume_sma_20": 1000000,
                "current_volume": 100000,
            },
            "posts": posts,
        }

        result = agent.analyze("AAPL", data)

        assert (
            result["manipulation_score"] > 0.5
            or "CONTRARIAN" in result["recommendation"]
            or "MONITOR" in result["recommendation"]
        )

    def test_detect_coordination_insufficient_posts(self):
        """Test coordination detection with insufficient posts."""
        agent = ManipulationAgent()

        score = agent.detect_coordination(
            [
                {"text": "test", "timestamp": datetime.utcnow().isoformat()},
            ]
        )

        assert score == 0.0

    def test_detect_coordination_clustered_timing(self):
        """Test coordination detection with clustered timing."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        posts = []
        for i in range(10):
            posts.append(
                {
                    "text": "buy this stock now",
                    "timestamp": (now + timedelta(seconds=i * 30)).isoformat(),
                    "author": f"user{i}",
                }
            )

        score = agent.detect_coordination(posts)

        assert score > 0.3

    def test_detect_coordination_similar_vocabulary(self):
        """Test coordination detection with similar vocabulary."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        posts = []
        phrases = ["to the moon", "diamond hands", "buy the dip"]
        for i in range(10):
            posts.append(
                {
                    "text": f"This stock is going {phrases[i % 3]}! Buy now!",
                    "timestamp": (now + timedelta(minutes=i * 10)).isoformat(),
                    "author": f"user{i}",
                }
            )

        score = agent.detect_coordination(posts)

        assert score > 0.2

    def test_estimate_bot_activity_insufficient_posts(self):
        """Test bot estimation with insufficient posts."""
        agent = ManipulationAgent()

        fraction = agent.estimate_bot_activity(
            [
                {"text": "test"},
            ]
        )

        assert fraction == 0.0

    def test_estimate_bot_activity_new_accounts(self):
        """Test bot estimation with new accounts."""
        agent = ManipulationAgent()

        posts = []
        for i in range(10):
            posts.append(
                {
                    "text": "Great stock!",
                    "author": f"newuser{i}",
                    "account_age_days": 10,
                }
            )

        fraction = agent.estimate_bot_activity(posts)

        assert fraction > 0.3

    def test_estimate_bot_activity_generic_usernames(self):
        """Test bot estimation with generic usernames."""
        agent = ManipulationAgent()

        posts = []
        generic_names = ["john1234567", "user123456", "test-user-abc123"]
        for i, name in enumerate(generic_names * 3):
            posts.append(
                {
                    "text": "Buy this stock!",
                    "author": name,
                    "account_age_days": 365,
                }
            )

        fraction = agent.estimate_bot_activity(posts)

        assert fraction > 0.1

    def test_check_news_catalyst_no_news(self):
        """Test news catalyst check with no news."""
        agent = ManipulationAgent()

        has_catalyst = agent.check_news_catalyst("AAPL", [], datetime.utcnow())

        assert has_catalyst is False

    def test_check_news_catalyst_recent_negative_news(self):
        """Test news catalyst check with recent negative news."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        news = [
            {
                "title": "Company faces SEC investigation",
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "sentiment": -0.6,
            }
        ]

        has_catalyst = agent.check_news_catalyst("AAPL", news, now)

        assert has_catalyst is True

    def test_check_news_catalyst_old_news(self):
        """Test news catalyst check with old news."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        news = [
            {
                "title": "Company faces issues",
                "timestamp": (now - timedelta(days=5)).isoformat(),
                "sentiment": -0.6,
            }
        ]

        has_catalyst = agent.check_news_catalyst("AAPL", news, now)

        assert has_catalyst is False

    def test_check_news_catalyst_keyword_match(self):
        """Test news catalyst check with negative keyword."""
        agent = ManipulationAgent()

        now = datetime.utcnow()
        news = [
            {
                "title": "Major product recall announced",
                "timestamp": (now - timedelta(hours=6)).isoformat(),
                "sentiment": 0.0,
            }
        ]

        has_catalyst = agent.check_news_catalyst("AAPL", news, now)

        assert has_catalyst is True

    def test_get_risk_assessment_high(self):
        """Test risk assessment for high manipulation score."""
        agent = ManipulationAgent()

        analysis = {
            "manipulation_score": 0.85,
            "bayesian_probability": 0.8,
            "evidence": {
                "sentiment_spike": True,
                "no_news_catalyst": True,
                "coordination_detected": True,
                "high_bot_activity": True,
            },
        }

        assessment = agent.get_risk_assessment(analysis)

        assert assessment["risk_level"] == "HIGH"
        assert len(assessment["warnings"]) > 0

    def test_get_risk_assessment_medium(self):
        """Test risk assessment for medium manipulation score."""
        agent = ManipulationAgent()

        analysis = {
            "manipulation_score": 0.6,
            "bayesian_probability": 0.55,
            "evidence": {
                "sentiment_spike": True,
                "no_news_catalyst": False,
            },
        }

        assessment = agent.get_risk_assessment(analysis)

        assert assessment["risk_level"] == "MEDIUM"

    def test_get_risk_assessment_low(self):
        """Test risk assessment for low manipulation score."""
        agent = ManipulationAgent()

        analysis = {
            "manipulation_score": 0.3,
            "bayesian_probability": 0.25,
            "evidence": {
                "sentiment_spike": False,
                "no_news_catalyst": False,
            },
        }

        assessment = agent.get_risk_assessment(analysis)

        assert assessment["risk_level"] == "LOW"
        assert (
            len(assessment["warnings"]) == 0
            or "exists" in assessment["warnings"][0].lower()
        )

    def test_analyze_includes_explanation(self):
        """Test analysis includes Bayesian explanation."""
        agent = ManipulationAgent()

        result = agent.analyze(
            "AAPL",
            {
                "sentiment": {"composite_score": -0.5},
                "price_history": [],
                "news": [],
                "technical_indicators": {},
                "posts": [],
            },
        )

        assert "explanation" in result
        assert "posterior" in result["explanation"]
        assert "interpretation" in result["explanation"]

    def test_analyze_confidence_levels(self):
        """Test confidence level assignment."""
        agent = ManipulationAgent()

        low_data = {
            "sentiment": {"composite_score": 0.1},
            "price_history": [],
            "news": [
                {
                    "title": "Good news",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sentiment": 0.5,
                }
            ],
            "technical_indicators": {},
            "posts": [],
        }
        low_result = agent.analyze("AAPL", low_data)
        assert low_result["confidence"] == "LOW"

    def test_divergence_detection_high_sentiment_low_volume(self):
        """Test divergence when sentiment volume high but trading volume low."""
        agent = ManipulationAgent()

        data = {
            "sentiment": {"composite_score": -0.5, "volume": 500},
            "price_history": [{"close": 100}, {"close": 99}],
            "news": [],
            "technical_indicators": {
                "volume_sma_20": 1000000,
                "current_volume": 100000,
            },
            "posts": [],
        }

        result = agent.analyze("AAPL", data)

        assert result["evidence"]["volume_sentiment_divergence"] is True

    def test_analyze_with_sentiment_result_object(self):
        """Test analysis works with SentimentResult-like objects."""
        agent = ManipulationAgent()

        class MockSentimentResult:
            def __init__(self):
                self.composite_score = -0.5
                self.volume = 100

            def to_dict(self):
                return {"composite_score": -0.5, "volume": 100}

        data = {
            "sentiment": MockSentimentResult(),
            "price_history": [],
            "news": [],
            "technical_indicators": {},
            "posts": [],
        }

        result = agent.analyze("AAPL", data)

        assert result["evidence"]["sentiment_spike"] is True
