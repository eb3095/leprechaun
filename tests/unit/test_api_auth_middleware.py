"""Unit tests for authentication middleware."""

import pytest
from unittest.mock import patch, MagicMock

from src.api.middleware.auth import (
    _has_required_role,
    MSG_AUTH_REQUIRED,
)
from src.data.models import UserRole


class TestRoleHierarchy:
    """Tests for role hierarchy checking."""

    def test_admin_has_admin_access(self):
        """Admin should have admin access."""
        result = _has_required_role(UserRole.ADMIN, UserRole.ADMIN)
        assert result is True

    def test_admin_has_viewer_access(self):
        """Admin should have viewer access."""
        result = _has_required_role(UserRole.ADMIN, UserRole.VIEWER)
        assert result is True

    def test_viewer_has_viewer_access(self):
        """Viewer should have viewer access."""
        result = _has_required_role(UserRole.VIEWER, UserRole.VIEWER)
        assert result is True

    def test_viewer_lacks_admin_access(self):
        """Viewer should not have admin access."""
        result = _has_required_role(UserRole.VIEWER, UserRole.ADMIN)
        assert result is False


class TestAuthConstants:
    """Tests for authentication constants."""

    def test_auth_required_message_defined(self):
        """Auth required message should be defined."""
        assert MSG_AUTH_REQUIRED == "Valid authentication token required"
