"""Flask application factory for Leprechaun trading bot API."""

import os
from datetime import timedelta
from typing import Any, Optional

import redis
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy.orm import scoped_session

from src.api.extensions import jwt, limiter, metrics
from src.data.models import create_db_engine, get_session_factory
from src.utils.config import get_config, ConfigurationError


TOKEN_BLOCKLIST_PREFIX = "leprechaun:token:blocklist:"
TOKEN_EXPIRY_SECONDS = 3600


def _get_redis_client(app: Flask) -> Optional[redis.Redis]:
    """Get Redis client from app config or create one."""
    redis_url = app.config.get("RATELIMIT_STORAGE_URI")
    if not redis_url:
        return None
    try:
        return redis.from_url(redis_url, decode_responses=True)
    except Exception:
        return None


def create_app(config_override: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_override: Optional dictionary to override configuration values.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    _configure_app(app, config_override)
    _init_extensions(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _setup_db_session(app)
    _setup_jwt_callbacks(app)

    return app


def _configure_app(app: Flask, config_override: dict[str, Any] | None) -> None:
    """Configure Flask application settings."""
    try:
        config = get_config()
    except ConfigurationError:
        config = None

    app.config["ENV"] = os.getenv("ENV", "development")
    app.config["DEBUG"] = os.getenv("DEBUG", "false").lower() == "true"

    is_production = app.config["ENV"] == "production"

    secret_key = os.getenv("SECRET_KEY")
    jwt_secret_key = os.getenv("JWT_SECRET_KEY")

    if is_production:
        if not secret_key:
            raise ValueError(
                "SECRET_KEY environment variable is required in production"
            )
        if not jwt_secret_key:
            raise ValueError(
                "JWT_SECRET_KEY environment variable is required in production"
            )
        app.config["SECRET_KEY"] = secret_key
        app.config["JWT_SECRET_KEY"] = jwt_secret_key
    else:
        app.config["SECRET_KEY"] = secret_key or "dev-secret-key-change-in-prod"
        app.config["JWT_SECRET_KEY"] = jwt_secret_key or app.config["SECRET_KEY"]

    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
    app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=7)
    app.config["JWT_TOKEN_LOCATION"] = ["headers"]
    app.config["JWT_HEADER_NAME"] = "Authorization"
    app.config["JWT_HEADER_TYPE"] = "Bearer"

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        app.config["RATELIMIT_STORAGE_URI"] = redis_url
    elif config:
        app.config["RATELIMIT_STORAGE_URI"] = config.redis.url

    app.config["RATELIMIT_STRATEGY"] = "fixed-window"
    app.config["RATELIMIT_DEFAULT"] = "100/minute"

    cors_origins_env = os.getenv("CORS_ORIGINS")
    if is_production:
        if not cors_origins_env:
            raise ValueError(
                "CORS_ORIGINS environment variable is required in production"
            )
        cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
    else:
        if cors_origins_env:
            cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
        else:
            cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.config["CORS_ORIGINS"] = cors_origins

    if config_override:
        app.config.update(config_override)


def _init_extensions(app: Flask) -> None:
    """Initialize Flask extensions."""
    CORS(
        app,
        origins=app.config.get("CORS_ORIGINS", ["*"]),
        supports_credentials=True,
    )

    jwt.init_app(app)
    limiter.init_app(app)
    metrics.init_app(app)


def _register_blueprints(app: Flask) -> None:
    """Register API and web blueprints."""
    from src.api.routes.auth import auth_bp
    from src.api.routes.trading import trading_bp
    from src.api.routes.metrics import metrics_bp
    from src.api.routes.sentiment import sentiment_bp
    from src.api.routes.alerts import alerts_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(sentiment_bp)
    app.register_blueprint(alerts_bp)

    # Register web UI blueprint
    try:
        from web.views import web_bp

        app.register_blueprint(web_bp)
    except ImportError:
        pass

    @app.route("/health", methods=["GET"])
    @limiter.exempt
    def health_check():
        """Health check endpoint for Kubernetes probes."""
        return jsonify(
            {
                "status": "healthy",
                "service": "leprechaun-api",
            }
        )


def _register_error_handlers(app: Flask) -> None:
    """Register error handlers for common exceptions."""

    @app.errorhandler(400)
    def bad_request(error):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": (
                        str(error.description)
                        if hasattr(error, "description")
                        else "Bad request"
                    ),
                }
            ),
            400,
        )

    @app.errorhandler(401)
    def unauthorized(error):
        return (
            jsonify(
                {
                    "error": "unauthorized",
                    "message": "Authentication required",
                }
            ),
            401,
        )

    @app.errorhandler(403)
    def forbidden(error):
        return (
            jsonify(
                {
                    "error": "forbidden",
                    "message": "Insufficient permissions",
                }
            ),
            403,
        )

    @app.errorhandler(404)
    def not_found(error):
        return (
            jsonify(
                {
                    "error": "not_found",
                    "message": "Resource not found",
                }
            ),
            404,
        )

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return (
            jsonify(
                {
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                }
            ),
            429,
        )

    @app.errorhandler(500)
    def internal_error(error):
        return (
            jsonify(
                {
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )


def _setup_db_session(app: Flask) -> None:
    """Setup database session management."""
    engine = None
    session_factory = None

    try:
        engine = create_db_engine()
        session_factory = get_session_factory(engine)
        app.db_session = scoped_session(session_factory)
    except Exception:
        app.db_session = None

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        if app.db_session is not None:
            app.db_session.remove()


def _setup_jwt_callbacks(app: Flask) -> None:
    """Setup JWT-related callbacks."""

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload["jti"]
        redis_client = _get_redis_client(app)
        if redis_client is None:
            return False
        try:
            return redis_client.exists(f"{TOKEN_BLOCKLIST_PREFIX}{jti}") > 0
        except Exception:
            return False

    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return (
            jsonify(
                {
                    "error": "token_revoked",
                    "message": "Token has been revoked",
                }
            ),
            401,
        )

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return (
            jsonify(
                {
                    "error": "token_expired",
                    "message": "Token has expired",
                }
            ),
            401,
        )

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return (
            jsonify(
                {
                    "error": "invalid_token",
                    "message": "Invalid token",
                }
            ),
            401,
        )

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return (
            jsonify(
                {
                    "error": "authorization_required",
                    "message": "Authorization token is required",
                }
            ),
            401,
        )


def add_token_to_blocklist(jti: str) -> None:
    """Add a token JTI to the blocklist for logout.

    Stores the token JTI in Redis with TTL matching token expiry.
    """
    from flask import current_app

    redis_client = _get_redis_client(current_app)
    if redis_client is None:
        return

    try:
        redis_client.setex(
            f"{TOKEN_BLOCKLIST_PREFIX}{jti}",
            TOKEN_EXPIRY_SECONDS,
            "1",
        )
    except Exception:
        pass


def get_db_session():
    """Get the current database session from the application context."""
    from flask import current_app

    return current_app.db_session
