"""Unit tests for logging module."""

import json
import logging
from io import StringIO

import pytest

from src.utils.logging import (
    JsonFormatter,
    HumanFormatter,
    TradeLogger,
    get_request_id,
    set_request_id,
    get_logger,
    setup_logging,
    create_trade_logger,
    log_with_context,
    request_id_var,
)


@pytest.fixture(autouse=True)
def reset_request_id():
    """Reset request ID before each test."""
    request_id_var.set(None)
    yield
    request_id_var.set(None)


class TestRequestIdContext:
    def test_set_and_get_request_id(self):
        rid = set_request_id("test-123")
        assert rid == "test-123"
        assert get_request_id() == "test-123"

    def test_auto_generate_request_id(self):
        rid = set_request_id()
        assert rid is not None
        assert len(rid) == 8
        assert get_request_id() == rid

    def test_request_id_none_by_default(self):
        assert get_request_id() is None


class TestJsonFormatter:
    def test_basic_format(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_includes_request_id(self):
        set_request_id("req-456")
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["request_id"] == "req-456"

    def test_includes_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"symbol": "AAPL", "price": 150.0}
        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["symbol"] == "AAPL"
        assert parsed["price"] == 150.0


class TestHumanFormatter:
    def test_basic_format(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        assert "INFO" in result
        assert "Test message" in result

    def test_includes_color_codes(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        assert "\033[31m" in result  # Red color for ERROR

    def test_includes_request_id(self):
        set_request_id("human-123")
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        assert "[human-123]" in result


class TestTradeLogger:
    def test_adds_symbol_to_extra(self):
        base_logger = logging.getLogger("test_trade")
        trade_logger = TradeLogger(base_logger, {"symbol": "TSLA"})

        msg, kwargs = trade_logger.process("Buy order", {})
        assert kwargs["extra"]["extra_fields"]["symbol"] == "TSLA"

    def test_merges_extra_fields(self):
        base_logger = logging.getLogger("test_trade")
        trade_logger = TradeLogger(base_logger, {"symbol": "TSLA"})

        msg, kwargs = trade_logger.process(
            "Buy order", {"extra": {"extra_fields": {"price": 200.0}}}
        )
        assert kwargs["extra"]["extra_fields"]["symbol"] == "TSLA"
        assert kwargs["extra"]["extra_fields"]["price"] == 200.0


class TestCreateTradeLogger:
    def test_creates_logger_with_context(self):
        logger = create_trade_logger(symbol="AAPL", order_id="ord-123", position_id=1)

        assert isinstance(logger, TradeLogger)
        assert logger.extra["symbol"] == "AAPL"
        assert logger.extra["order_id"] == "ord-123"
        assert logger.extra["position_id"] == 1

    def test_creates_logger_with_partial_context(self):
        logger = create_trade_logger(symbol="GOOG")

        assert logger.extra["symbol"] == "GOOG"
        assert "order_id" not in logger.extra


class TestSetupLogging:
    def test_configures_json_format(self):
        setup_logging(level="DEBUG", log_format="json")
        root_logger = logging.getLogger()

        assert root_logger.level == logging.DEBUG
        assert len(root_logger.handlers) > 0

    def test_configures_human_format(self):
        setup_logging(level="INFO", log_format="human")
        root_logger = logging.getLogger()

        assert root_logger.level == logging.INFO


class TestGetLogger:
    def test_returns_namespaced_logger(self):
        logger = get_logger("trading")
        assert logger.name == "leprechaun.trading"

    def test_different_modules_different_loggers(self):
        logger1 = get_logger("trading")
        logger2 = get_logger("sentiment")
        assert logger1 is not logger2


class TestLogWithContext:
    def test_logs_with_extra_context(self):
        logger = logging.getLogger("test_context")
        captured_record = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_record.append(record)
        
        handler = CapturingHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        log_with_context(logger, logging.INFO, "Test message", symbol="AAPL", price=100)
        
        assert len(captured_record) == 1
        assert hasattr(captured_record[0], "extra_fields")
        assert captured_record[0].extra_fields["symbol"] == "AAPL"
        assert captured_record[0].extra_fields["price"] == 100
