"""Unit tests for authentication routes."""

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


class TestLoginEndpoint:
    """Tests for /api/v1/auth/login endpoint."""

    def test_login_requires_body(self, client):
        """Login should require request body."""
        response = client.post(
            "/api/v1/auth/login",
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_login_requires_username(self, client):
        """Login should require username."""
        response = client.post(
            "/api/v1/auth/login",
            json={"password": "test"}
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "username" in data.get("message", "").lower() or "required" in data.get("message", "").lower()

    def test_login_requires_password(self, client):
        """Login should require password."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "test"}
        )
        assert response.status_code == 400

    def test_login_rejects_invalid_credentials(self, client):
        """Login should reject invalid credentials."""
        with patch("src.api.routes.auth.get_db_session") as mock_session:
            mock_session.return_value = MagicMock()
            mock_session.return_value.query.return_value.filter_by.return_value.first.return_value = None
            
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "invalid", "password": "invalid"}
            )
            
            assert response.status_code == 401

    def test_login_uses_post_method(self, client):
        """Login should only accept POST."""
        response = client.get("/api/v1/auth/login")
        assert response.status_code == 405


class TestRefreshEndpoint:
    """Tests for /api/v1/auth/refresh endpoint."""

    def test_refresh_requires_token(self, client):
        """Refresh should require valid refresh token."""
        response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_uses_post_method(self, client):
        """Refresh should only accept POST."""
        response = client.get("/api/v1/auth/refresh")
        assert response.status_code == 405


class TestLogoutEndpoint:
    """Tests for /api/v1/auth/logout endpoint."""

    def test_logout_requires_token(self, client):
        """Logout should require valid token."""
        response = client.post("/api/v1/auth/logout")
        assert response.status_code == 401

    def test_logout_uses_post_method(self, client):
        """Logout should only accept POST."""
        response = client.get("/api/v1/auth/logout")
        assert response.status_code == 405


class TestMeEndpoint:
    """Tests for /api/v1/auth/me endpoint."""

    def test_me_requires_token(self, client):
        """Me endpoint should require valid token."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_me_uses_get_method(self, client):
        """Me endpoint should only accept GET."""
        response = client.post("/api/v1/auth/me")
        assert response.status_code == 405
