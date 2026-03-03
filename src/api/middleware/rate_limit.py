"""Rate limiting configuration for Leprechaun API."""

from src.api.extensions import limiter


DEFAULT_LIMIT = "100/minute"
TRADING_LIMIT = "10/minute"
ADMIN_LIMIT = "30/minute"


trading_rate_limit = limiter.limit(
    TRADING_LIMIT,
    error_message="Trading endpoint rate limit exceeded. Maximum 10 requests per minute.",
)

admin_rate_limit = limiter.limit(
    ADMIN_LIMIT,
    error_message="Admin endpoint rate limit exceeded. Maximum 30 requests per minute.",
)

default_rate_limit = limiter.limit(
    DEFAULT_LIMIT,
    error_message="Rate limit exceeded. Maximum 100 requests per minute.",
)

no_limit = limiter.exempt


def get_rate_limit_status():
    """Get current rate limit status for the client.

    Returns:
        Dictionary with rate limit information.
    """
    return {
        "limits": {
            "default": DEFAULT_LIMIT,
            "trading": TRADING_LIMIT,
            "admin": ADMIN_LIMIT,
        },
        "storage": "redis",
    }
