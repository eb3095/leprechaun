"""Unit tests for Flask API application factory."""

import pytest
from unittest.mock import patch, MagicMock

from src.api.app import create_app, add_token_to_blocklist


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    mock.exists.return_value = 0
    mock.setex.return_value = True
    return mock


@pytest.fixture
def app(mock_redis):
    """Create test application."""
    with patch("src.api.app.create_db_engine") as mock_engine, patch(
        "src.api.app._get_redis_client"
    ) as mock_get_redis:
        mock_engine.return_value = MagicMock()
        mock_get_redis.return_value = mock_redis
        test_app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test-secret-key",
                "JWT_SECRET_KEY": "test-jwt-secret",
                "RATELIMIT_STORAGE_URI": "redis://localhost:6379/0",
            }
        )
        yield test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_json(self, client):
        """Health endpoint should return correct JSON structure."""
        response = client.get("/health")
        data = response.get_json()

        assert data["status"] == "healthy"
        assert data["service"] == "leprechaun-api"


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_returns_json(self, client):
        """404 error should return JSON."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        data = response.get_json()
        assert data["error"] == "not_found"
        assert "message" in data


class TestBlocklist:
    """Tests for token blocklist functionality."""

    def test_add_token_to_blocklist(self, app, mock_redis):
        """Should add token JTI to blocklist via Redis."""
        test_jti = "test-jti-12345"

        with app.app_context():
            with patch("src.api.app._get_redis_client") as mock_get_redis:
                mock_get_redis.return_value = mock_redis
                add_token_to_blocklist(test_jti)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert "leprechaun:token:blocklist:" + test_jti == call_args[0][0]
        assert call_args[0][1] == 3600

    def test_add_token_to_blocklist_handles_no_redis(self, app):
        """Should handle missing Redis gracefully."""
        test_jti = "test-jti-12345"

        with app.app_context():
            with patch("src.api.app._get_redis_client") as mock_get_redis:
                mock_get_redis.return_value = None
                add_token_to_blocklist(test_jti)


class TestAppConfiguration:
    """Tests for application configuration."""

    def test_app_has_jwt_config(self, app):
        """Application should have JWT configuration."""
        assert "JWT_SECRET_KEY" in app.config
        assert app.config["JWT_SECRET_KEY"] == "test-jwt-secret"

    def test_app_has_testing_flag(self, app):
        """Application should have testing flag set."""
        assert app.config["TESTING"] is True

    def test_blueprints_registered(self, app):
        """All blueprints should be registered."""
        blueprint_names = [bp.name for bp in app.blueprints.values()]

        assert "auth" in blueprint_names
        assert "trading" in blueprint_names
        assert "metrics" in blueprint_names
        assert "sentiment" in blueprint_names
        assert "alerts" in blueprint_names
