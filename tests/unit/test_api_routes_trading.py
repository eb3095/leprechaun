"""Unit tests for trading routes."""

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


@pytest.fixture
def auth_headers():
    """Create mock auth headers."""
    return {"Authorization": "Bearer mock-token"}


class TestTradingStatusEndpoint:
    """Tests for /api/v1/trading/status endpoint."""

    def test_status_requires_auth(self, client):
        """Status endpoint should require authentication."""
        response = client.get("/api/v1/trading/status")
        assert response.status_code == 401

    def test_status_without_token_returns_401(self, client):
        """Status endpoint without token should return 401."""
        response = client.get("/api/v1/trading/status")
        data = response.get_json()
        
        assert response.status_code == 401
        assert "error" in data
        assert data["error"] == "authorization_required"


class TestTradingStartEndpoint:
    """Tests for /api/v1/trading/start endpoint."""

    def test_start_requires_auth(self, client):
        """Start endpoint should require authentication."""
        response = client.post("/api/v1/trading/start")
        assert response.status_code == 401

    def test_start_uses_post_method(self, client):
        """Start endpoint should only accept POST."""
        response = client.get("/api/v1/trading/start")
        assert response.status_code in (401, 405)


class TestTradingStopEndpoint:
    """Tests for /api/v1/trading/stop endpoint."""

    def test_stop_requires_auth(self, client):
        """Stop endpoint should require authentication."""
        response = client.post("/api/v1/trading/stop")
        assert response.status_code == 401

    def test_stop_uses_post_method(self, client):
        """Stop endpoint should only accept POST."""
        response = client.get("/api/v1/trading/stop")
        assert response.status_code in (401, 405)


class TestPositionsEndpoint:
    """Tests for /api/v1/trading/positions endpoint."""

    def test_positions_requires_auth(self, client):
        """Positions endpoint should require authentication."""
        response = client.get("/api/v1/trading/positions")
        assert response.status_code == 401


class TestOrdersEndpoint:
    """Tests for /api/v1/trading/orders endpoint."""

    def test_orders_requires_auth(self, client):
        """Orders endpoint should require authentication."""
        response = client.get("/api/v1/trading/orders")
        assert response.status_code == 401


class TestHaltResumeEndpoint:
    """Tests for /api/v1/trading/halt/resume endpoint."""

    def test_resume_requires_auth(self, client):
        """Resume endpoint should require authentication."""
        response = client.post("/api/v1/trading/halt/resume")
        assert response.status_code == 401

    def test_resume_uses_post_method(self, client):
        """Resume endpoint should only accept POST."""
        response = client.get("/api/v1/trading/halt/resume")
        assert response.status_code in (401, 405)
