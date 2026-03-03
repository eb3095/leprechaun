"""JWT authentication middleware for Leprechaun API."""

from functools import wraps
from typing import Callable, TypeVar

from flask import jsonify
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request

from src.data.models import User, UserRole


F = TypeVar("F", bound=Callable)

MSG_AUTH_REQUIRED = "Valid authentication token required"


def get_current_user() -> User | None:
    """Get the current authenticated user from the JWT token.

    Returns:
        User object if authenticated and found, None otherwise.
    """
    from src.api.app import get_db_session

    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
        session = get_db_session()
        if session is None:
            return None
        return session.query(User).filter_by(id=user_id, is_active=True).first()
    except Exception:
        return None


def get_current_user_role() -> UserRole | None:
    """Get the role of the current authenticated user.

    Returns:
        UserRole enum value if authenticated, None otherwise.
    """
    try:
        verify_jwt_in_request()
        claims = get_jwt()
        role_str = claims.get("role")
        if role_str:
            return UserRole(role_str)
        return None
    except Exception:
        return None


def jwt_required_with_role(required_role: UserRole):
    """Decorator that requires JWT authentication with a specific role.

    Args:
        required_role: The minimum required role for access.

    Returns:
        Decorator function.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
            except Exception:
                return jsonify({
                    "error": "authorization_required",
                    "message": MSG_AUTH_REQUIRED,
                }), 401

            claims = get_jwt()
            role_str = claims.get("role")

            if not role_str:
                return jsonify({
                    "error": "invalid_token",
                    "message": "Token missing role claim",
                }), 401

            try:
                user_role = UserRole(role_str)
            except ValueError:
                return jsonify({
                    "error": "invalid_token",
                    "message": "Invalid role in token",
                }), 401

            if not _has_required_role(user_role, required_role):
                return jsonify({
                    "error": "forbidden",
                    "message": f"Requires {required_role.value} role",
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def admin_required(fn: F) -> F:
    """Decorator that requires admin role.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function that enforces admin authentication.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            verify_jwt_in_request()
        except Exception:
            return jsonify({
                "error": "authorization_required",
                "message": MSG_AUTH_REQUIRED,
            }), 401

        claims = get_jwt()
        role_str = claims.get("role")

        if role_str != UserRole.ADMIN.value:
            return jsonify({
                "error": "forbidden",
                "message": "Admin access required",
            }), 403

        return fn(*args, **kwargs)
    return wrapper


def viewer_or_admin(fn: F) -> F:
    """Decorator that requires viewer or admin role.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function that enforces viewer-level authentication.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            verify_jwt_in_request()
        except Exception:
            return jsonify({
                "error": "authorization_required",
                "message": MSG_AUTH_REQUIRED,
            }), 401

        claims = get_jwt()
        role_str = claims.get("role")

        if role_str not in (UserRole.ADMIN.value, UserRole.VIEWER.value):
            return jsonify({
                "error": "forbidden",
                "message": "Viewer or admin access required",
            }), 403

        return fn(*args, **kwargs)
    return wrapper


def _has_required_role(user_role: UserRole, required_role: UserRole) -> bool:
    """Check if user role meets the required role level.

    Role hierarchy: ADMIN > VIEWER

    Args:
        user_role: The user's current role.
        required_role: The minimum required role.

    Returns:
        True if user has sufficient permissions.
    """
    role_hierarchy = {
        UserRole.ADMIN: 2,
        UserRole.VIEWER: 1,
    }

    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)

    return user_level >= required_level


def refresh_access_token():
    """Generate a new access token using the refresh token.

    Must be called within a request context with a valid refresh token.

    Returns:
        Tuple of (new_access_token, user_identity) or (None, None) on failure.
    """
    from flask_jwt_extended import create_access_token

    try:
        verify_jwt_in_request(refresh=True)
        user_id = get_jwt_identity()
        claims = get_jwt()

        additional_claims = {
            "role": claims.get("role"),
            "username": claims.get("username"),
        }

        new_token = create_access_token(
            identity=user_id,
            additional_claims=additional_claims,
        )

        return new_token, user_id
    except Exception:
        return None, None
