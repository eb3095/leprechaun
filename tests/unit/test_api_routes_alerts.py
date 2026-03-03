"""Unit tests for alerts routes."""

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


class TestAlertsListEndpoint:
    """Tests for /api/v1/alerts endpoint."""

    def test_alerts_requires_auth(self, client):
        """Alerts endpoint should require authentication."""
        response = client.get("/api/v1/alerts")
        assert response.status_code == 401


class TestTestAlertEndpoint:
    """Tests for /api/v1/alerts/test endpoint."""

    def test_test_alert_requires_auth(self, client):
        """Test alert endpoint should require authentication."""
        response = client.post("/api/v1/alerts/test")
        assert response.status_code == 401

    def test_test_alert_uses_post_method(self, client):
        """Test alert should only accept POST."""
        response = client.get("/api/v1/alerts/test")
        assert response.status_code == 405


class TestAlertSettingsEndpoint:
    """Tests for /api/v1/alerts/settings endpoint."""

    def test_get_settings_requires_auth(self, client):
        """Get settings endpoint should require authentication."""
        response = client.get("/api/v1/alerts/settings")
        assert response.status_code == 401

    def test_update_settings_requires_auth(self, client):
        """Update settings endpoint should require authentication."""
        response = client.put("/api/v1/alerts/settings")
        assert response.status_code == 401
