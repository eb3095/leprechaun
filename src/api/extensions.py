"""Shared Flask extensions for Leprechaun API.

This module centralizes Flask extension instances to avoid circular imports.
"""

from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_flask_exporter import PrometheusMetrics


jwt = JWTManager()
limiter = Limiter(key_func=get_remote_address)
metrics = PrometheusMetrics.for_app_factory()
