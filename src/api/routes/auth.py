"""Authentication routes for Leprechaun API."""

from datetime import datetime, timezone

import bcrypt
from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)

from src.api.app import add_token_to_blocklist, get_db_session
from src.api.middleware.auth import get_current_user, viewer_or_admin
from src.api.middleware.rate_limit import default_rate_limit
from src.data.models import User


auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")


@auth_bp.route("/login", methods=["POST"])
@default_rate_limit
def login():
    """Authenticate user and return JWT tokens.

    ---
    tags:
      - Authentication
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - username
              - password
            properties:
              username:
                type: string
                description: User's username
              password:
                type: string
                format: password
                description: User's password
    responses:
      200:
        description: Successfully authenticated
        content:
          application/json:
            schema:
              type: object
              properties:
                access_token:
                  type: string
                refresh_token:
                  type: string
                user:
                  type: object
      400:
        description: Missing required fields
      401:
        description: Invalid credentials
    """
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "bad_request",
            "message": "Request body required",
        }), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({
            "error": "bad_request",
            "message": "Username and password required",
        }), 400

    session = get_db_session()
    if session is None:
        return jsonify({
            "error": "service_unavailable",
            "message": "Database connection unavailable",
        }), 503

    user = session.query(User).filter_by(username=username, is_active=True).first()

    if not user:
        return jsonify({
            "error": "unauthorized",
            "message": "Invalid credentials",
        }), 401

    if not bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
        return jsonify({
            "error": "unauthorized",
            "message": "Invalid credentials",
        }), 401

    user.last_login = datetime.now(timezone.utc)
    session.commit()

    additional_claims = {
        "role": user.role.value,
        "username": user.username,
    }

    access_token = create_access_token(
        identity=user.id,
        additional_claims=additional_claims,
    )
    refresh_token = create_refresh_token(
        identity=user.id,
        additional_claims=additional_claims,
    )

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user.to_dict(),
    })


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Refresh the access token using a valid refresh token.

    ---
    tags:
      - Authentication
    security:
      - bearerAuth: []
    responses:
      200:
        description: New access token generated
        content:
          application/json:
            schema:
              type: object
              properties:
                access_token:
                  type: string
      401:
        description: Invalid or expired refresh token
    """
    user_id = get_jwt_identity()

    session = get_db_session()
    if session is None:
        return jsonify({
            "error": "service_unavailable",
            "message": "Database connection unavailable",
        }), 503

    user = session.query(User).filter_by(id=user_id, is_active=True).first()

    if not user:
        return jsonify({
            "error": "unauthorized",
            "message": "User not found or inactive",
        }), 401

    additional_claims = {
        "role": user.role.value,
        "username": user.username,
    }

    new_access_token = create_access_token(
        identity=user_id,
        additional_claims=additional_claims,
    )

    return jsonify({
        "access_token": new_access_token,
    })


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    """Invalidate the current access token.

    ---
    tags:
      - Authentication
    security:
      - bearerAuth: []
    responses:
      200:
        description: Successfully logged out
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
      401:
        description: Invalid or missing token
    """
    jti = get_jwt()["jti"]
    add_token_to_blocklist(jti)

    return jsonify({
        "message": "Successfully logged out",
    })


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
@viewer_or_admin
def get_me():
    """Get current authenticated user information.

    ---
    tags:
      - Authentication
    security:
      - bearerAuth: []
    responses:
      200:
        description: Current user information
        content:
          application/json:
            schema:
              type: object
              properties:
                user:
                  type: object
      401:
        description: Invalid or missing token
      404:
        description: User not found
    """
    user = get_current_user()

    if not user:
        return jsonify({
            "error": "not_found",
            "message": "User not found",
        }), 404

    return jsonify({
        "user": user.to_dict(),
    })
