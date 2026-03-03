"""Utility functions for Leprechaun API routes."""

from flask import request

MAX_LIMIT = 100
DEFAULT_LIMIT = 50
DEFAULT_OFFSET = 0


def get_pagination_params() -> tuple[int, int]:
    """Extract and validate pagination parameters from request args.

    Returns:
        Tuple of (limit, offset) with validated values.
        - limit is clamped between 1 and MAX_LIMIT
        - offset is clamped to minimum of 0
    """
    limit = request.args.get("limit", DEFAULT_LIMIT, type=int)
    offset = request.args.get("offset", DEFAULT_OFFSET, type=int)

    limit = DEFAULT_LIMIT if limit is None else max(1, min(limit, MAX_LIMIT))
    offset = DEFAULT_OFFSET if offset is None else max(0, offset)

    return limit, offset
