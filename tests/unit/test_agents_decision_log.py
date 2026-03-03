"""Unit tests for decision logging."""

import json
from datetime import datetime, timedelta

import pytest

from src.agents.decision_log import Decision, DecisionLogger


class TestDecision:
    """Tests for Decision dataclass."""

    def test_create_decision(self):
        """Test creating a decision."""
        decision = Decision(
            timestamp=datetime(2026, 1, 15, 10, 30),
            symbol="AAPL",
            decision="BUY",
            confidence=0.85,
            inputs={"rsi": 28.5, "sentiment": -0.3},
            reasoning=["RSI oversold", "Negative sentiment"],
            executed=False,
        )

        assert decision.symbol == "AAPL"
        assert decision.decision == "BUY"
        assert decision.confidence == 0.85
        assert decision.executed is False

    def test_decision_to_dict(self):
        """Test converting decision to dictionary."""
        decision = Decision(
            timestamp=datetime(2026, 1, 15, 10, 30),
            symbol="AAPL",
            decision="BUY",
            confidence=0.85,
            inputs={"rsi": 28.5},
            reasoning=["RSI oversold"],
            executed=True,
            execution_details={"order_id": "12345"},
        )

        result = decision.to_dict()

        assert result["symbol"] == "AAPL"
        assert result["decision"] == "BUY"
        assert result["timestamp"] == "2026-01-15T10:30:00"
        assert result["execution_details"]["order_id"] == "12345"

    def test_decision_from_dict(self):
        """Test creating decision from dictionary."""
        data = {
            "timestamp": "2026-01-15T10:30:00",
            "symbol": "AAPL",
            "decision": "SELL",
            "confidence": 0.9,
            "inputs": {"pnl": 2.5},
            "reasoning": ["Profit target reached"],
            "executed": True,
            "execution_details": {"price": 185.50},
        }

        decision = Decision.from_dict(data)

        assert decision.symbol == "AAPL"
        assert decision.decision == "SELL"
        assert decision.timestamp == datetime(2026, 1, 15, 10, 30)

    def test_decision_default_execution_details(self):
        """Test execution_details defaults to empty dict."""
        decision = Decision(
            timestamp=datetime.utcnow(),
            symbol="MSFT",
            decision="HOLD",
            confidence=0.5,
            inputs={},
            reasoning=[],
            executed=False,
        )

        assert decision.execution_details == {}


class TestDecisionLogger:
    """Tests for DecisionLogger."""

    def test_init_without_session(self):
        """Test initialization without database session."""
        logger = DecisionLogger()

        assert logger._session is None
        assert logger._memory_log == []

    def test_log_decision_memory(self):
        """Test logging decision to memory."""
        logger = DecisionLogger()
        decision = Decision(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            decision="BUY",
            confidence=0.8,
            inputs={},
            reasoning=["Test"],
            executed=False,
        )

        decision_id = logger.log_decision(decision)

        assert decision_id == 1
        assert decision.decision_id == 1
        assert len(logger._memory_log) == 1

    def test_log_multiple_decisions(self):
        """Test logging multiple decisions."""
        logger = DecisionLogger()

        for i in range(5):
            decision = Decision(
                timestamp=datetime.utcnow(),
                symbol=f"SYM{i}",
                decision="HOLD",
                confidence=0.5,
                inputs={},
                reasoning=[],
                executed=False,
            )
            decision_id = logger.log_decision(decision)
            assert decision_id == i + 1

        assert len(logger._memory_log) == 5

    def test_get_decisions_all(self):
        """Test getting all decisions."""
        logger = DecisionLogger()

        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    decision="BUY",
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions()

        assert len(decisions) == 3

    def test_get_decisions_filter_symbol(self):
        """Test filtering decisions by symbol."""
        logger = DecisionLogger()

        for symbol in ["AAPL", "AAPL", "MSFT"]:
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    decision="BUY",
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions(symbol="AAPL")

        assert len(decisions) == 2
        assert all(d.symbol == "AAPL" for d in decisions)

    def test_get_decisions_filter_time_range(self):
        """Test filtering decisions by time range."""
        logger = DecisionLogger()

        base_time = datetime(2026, 1, 15, 12, 0)
        for i in range(5):
            logger.log_decision(
                Decision(
                    timestamp=base_time + timedelta(hours=i),
                    symbol="AAPL",
                    decision="HOLD",
                    confidence=0.5,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions(
            start=base_time + timedelta(hours=1),
            end=base_time + timedelta(hours=3),
        )

        assert len(decisions) == 3

    def test_get_decisions_filter_decision_type(self):
        """Test filtering decisions by type."""
        logger = DecisionLogger()

        for decision_type in ["BUY", "BUY", "SELL", "HOLD"]:
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol="AAPL",
                    decision=decision_type,
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions(decision_type="BUY")

        assert len(decisions) == 2
        assert all(d.decision == "BUY" for d in decisions)

    def test_get_decisions_executed_only(self):
        """Test filtering to executed decisions only."""
        logger = DecisionLogger()

        for executed in [True, True, False, False, False]:
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol="AAPL",
                    decision="BUY",
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=executed,
                )
            )

        decisions = logger.get_decisions(executed_only=True)

        assert len(decisions) == 2
        assert all(d.executed for d in decisions)

    def test_get_decisions_limit(self):
        """Test limiting number of returned decisions."""
        logger = DecisionLogger()

        for i in range(10):
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol="AAPL",
                    decision="HOLD",
                    confidence=0.5,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions(limit=5)

        assert len(decisions) == 5

    def test_get_decisions_sorted_by_time(self):
        """Test decisions are sorted by time descending."""
        logger = DecisionLogger()

        base_time = datetime(2026, 1, 15, 12, 0)
        for i in range(5):
            logger.log_decision(
                Decision(
                    timestamp=base_time + timedelta(hours=i),
                    symbol="AAPL",
                    decision="HOLD",
                    confidence=0.5,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        decisions = logger.get_decisions()

        for i in range(len(decisions) - 1):
            assert decisions[i].timestamp >= decisions[i + 1].timestamp

    def test_get_decision_by_id(self):
        """Test getting specific decision by ID."""
        logger = DecisionLogger()

        decision = Decision(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            decision="BUY",
            confidence=0.85,
            inputs={"test": True},
            reasoning=["Test reason"],
            executed=False,
        )
        decision_id = logger.log_decision(decision)

        retrieved = logger.get_decision_by_id(decision_id)

        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.confidence == 0.85

    def test_get_decision_by_id_not_found(self):
        """Test getting non-existent decision returns None."""
        logger = DecisionLogger()

        retrieved = logger.get_decision_by_id(999)

        assert retrieved is None

    def test_export_decisions_json(self):
        """Test exporting decisions as JSON."""
        logger = DecisionLogger()

        base_time = datetime(2026, 1, 15, 12, 0)
        for i in range(3):
            logger.log_decision(
                Decision(
                    timestamp=base_time + timedelta(hours=i),
                    symbol="AAPL",
                    decision="BUY",
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        export = logger.export_decisions(
            start=base_time,
            end=base_time + timedelta(hours=5),
            format="json",
        )

        data = json.loads(export)
        assert len(data) == 3
        assert all(d["symbol"] == "AAPL" for d in data)

    def test_export_decisions_csv(self):
        """Test exporting decisions as CSV."""
        logger = DecisionLogger()

        base_time = datetime(2026, 1, 15, 12, 0)
        for i in range(3):
            logger.log_decision(
                Decision(
                    timestamp=base_time + timedelta(hours=i),
                    symbol="AAPL",
                    decision="BUY",
                    confidence=0.8,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        export = logger.export_decisions(
            start=base_time,
            end=base_time + timedelta(hours=5),
            format="csv",
        )

        lines = export.strip().split("\n")
        assert len(lines) == 4
        assert "timestamp,symbol,decision,confidence,executed" in lines[0]

    def test_export_decisions_empty(self):
        """Test exporting when no decisions exist."""
        logger = DecisionLogger()

        export = logger.export_decisions(
            start=datetime.utcnow(),
            end=datetime.utcnow() + timedelta(hours=1),
            format="csv",
        )

        assert "timestamp,symbol,decision,confidence,executed" in export

    def test_update_execution(self):
        """Test updating execution status."""
        logger = DecisionLogger()

        decision = Decision(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            decision="BUY",
            confidence=0.8,
            inputs={},
            reasoning=[],
            executed=False,
        )
        decision_id = logger.log_decision(decision)

        success = logger.update_execution(
            decision_id,
            executed=True,
            execution_details={"order_id": "ABC123", "price": 185.50},
        )

        assert success is True
        retrieved = logger.get_decision_by_id(decision_id)
        assert retrieved.executed is True
        assert retrieved.execution_details["order_id"] == "ABC123"

    def test_update_execution_not_found(self):
        """Test updating non-existent decision."""
        logger = DecisionLogger()

        success = logger.update_execution(999, executed=True, execution_details={})

        assert success is False

    def test_get_decision_statistics(self):
        """Test getting decision statistics."""
        logger = DecisionLogger()

        decisions_data = [
            ("BUY", 0.8, True),
            ("BUY", 0.7, True),
            ("SELL", 0.9, True),
            ("HOLD", 0.5, False),
            ("SKIP", 0.3, False),
        ]

        for decision_type, confidence, executed in decisions_data:
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol="AAPL",
                    decision=decision_type,
                    confidence=confidence,
                    inputs={},
                    reasoning=[],
                    executed=executed,
                )
            )

        stats = logger.get_decision_statistics()

        assert stats["total_decisions"] == 5
        assert stats["by_type"]["BUY"] == 2
        assert stats["by_type"]["SELL"] == 1
        assert stats["executed_count"] == 3
        assert 0.6 <= stats["avg_confidence"] <= 0.7

    def test_get_decision_statistics_empty(self):
        """Test statistics with no decisions."""
        logger = DecisionLogger()

        stats = logger.get_decision_statistics()

        assert stats["total_decisions"] == 0
        assert stats["by_type"] == {}
        assert stats["executed_count"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_clear_memory_log(self):
        """Test clearing memory log."""
        logger = DecisionLogger()

        for i in range(5):
            logger.log_decision(
                Decision(
                    timestamp=datetime.utcnow(),
                    symbol="AAPL",
                    decision="HOLD",
                    confidence=0.5,
                    inputs={},
                    reasoning=[],
                    executed=False,
                )
            )

        assert len(logger._memory_log) == 5

        logger.clear_memory_log()

        assert len(logger._memory_log) == 0
