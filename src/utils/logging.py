"""Structured logging setup for Leprechaun trading bot."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set a request ID in the current context.

    Args:
        request_id: Optional request ID. If None, generates a new UUID.

    Returns:
        The request ID that was set.
    """
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


class JsonFormatter(logging.Formatter):
    """JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            log_obj["request_id"] = request_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for development environments."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_id = get_request_id()
        rid_str = f"[{request_id}] " if request_id else ""

        base = f"{color}{timestamp} {record.levelname:8}{self.RESET} {rid_str}{record.module}: {record.getMessage()}"

        if hasattr(record, "extra_fields"):
            extras = " ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            base = f"{base} | {extras}"

        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


class TradeLogger(logging.LoggerAdapter):
    """Logger adapter with trade-specific context fields."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra", {})
        extra_fields = {**self.extra, **extra.get("extra_fields", {})}

        if extra_fields:
            extra["extra_fields"] = extra_fields

        kwargs["extra"] = extra
        return msg, kwargs


def create_trade_logger(
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    position_id: Optional[int] = None,
) -> TradeLogger:
    """Create a logger with trade-specific context.

    Args:
        symbol: Stock symbol being traded.
        order_id: Alpaca order ID.
        position_id: Internal position ID.

    Returns:
        TradeLogger with the specified context attached.
    """
    logger = logging.getLogger("leprechaun.trading")
    extra = {}
    if symbol:
        extra["symbol"] = symbol
    if order_id:
        extra["order_id"] = order_id
    if position_id:
        extra["position_id"] = position_id
    return TradeLogger(logger, extra)


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: 'json' for production, 'human' for development.
        log_file: Optional file path to write logs.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = HumanFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name, typically the module name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"leprechaun.{name}")


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """Log a message with additional context fields.

    Args:
        logger: Logger instance to use.
        level: Logging level (e.g., logging.INFO).
        message: Log message.
        **context: Additional key-value pairs to include in the log.
    """
    logger.log(level, message, extra={"extra_fields": context})
