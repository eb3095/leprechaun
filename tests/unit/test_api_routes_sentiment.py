"""Unit tests for sentiment routes."""

import pytest
from unittest.mock import patch, MagicMock

from src.api.app import create_app


@pytest.fixture
def app():
    """Create test application."""
    with patch("src.api.app.create_db_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        test_app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
        })
        yield test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestSentimentBySymbolEndpoint:
    """Tests for /api/v1/sentiment/{symbol} endpoint."""

    def test_sentiment_requires_auth(self, client):
        """Sentiment endpoint should require authentication."""
        response = client.get("/api/v1/sentiment/AAPL")
        assert response.status_code == 401


class TestTrendingEndpoint:
    """Tests for /api/v1/sentiment/trending endpoint."""

    def test_trending_requires_auth(self, client):
        """Trending endpoint should require authentication."""
        response = client.get("/api/v1/sentiment/trending")
        assert response.status_code == 401


class TestManipulationScoreEndpoint:
    """Tests for /api/v1/manipulation/{symbol} endpoint."""

    def test_manipulation_requires_auth(self, client):
        """Manipulation endpoint should require authentication."""
        response = client.get("/api/v1/manipulation/AAPL")
        assert response.status_code == 401
