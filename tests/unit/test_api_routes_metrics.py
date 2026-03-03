"""Unit tests for metrics routes."""

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


class TestPortfolioEndpoint:
    """Tests for /api/v1/metrics/portfolio endpoint."""

    def test_portfolio_requires_auth(self, client):
        """Portfolio endpoint should require authentication."""
        response = client.get("/api/v1/metrics/portfolio")
        assert response.status_code == 401


class TestPerformanceEndpoint:
    """Tests for /api/v1/metrics/performance endpoint."""

    def test_performance_requires_auth(self, client):
        """Performance endpoint should require authentication."""
        response = client.get("/api/v1/metrics/performance")
        assert response.status_code == 401


class TestTradesEndpoint:
    """Tests for /api/v1/metrics/trades endpoint."""

    def test_trades_requires_auth(self, client):
        """Trades endpoint should require authentication."""
        response = client.get("/api/v1/metrics/trades")
        assert response.status_code == 401


class TestDailyPnlEndpoint:
    """Tests for /api/v1/metrics/daily endpoint."""

    def test_daily_requires_auth(self, client):
        """Daily endpoint should require authentication."""
        response = client.get("/api/v1/metrics/daily")
        assert response.status_code == 401


class TestWeeklyPnlEndpoint:
    """Tests for /api/v1/metrics/weekly endpoint."""

    def test_weekly_requires_auth(self, client):
        """Weekly endpoint should require authentication."""
        response = client.get("/api/v1/metrics/weekly")
        assert response.status_code == 401
