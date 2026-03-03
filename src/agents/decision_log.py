"""Decision audit trail for agent decisions.

Logs all trading decisions with full context for analysis and compliance.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session


@dataclass
class Decision:
    """Represents a trading decision with full context."""

    timestamp: datetime
    symbol: str
    decision: str
    confidence: float
    inputs: dict[str, Any]
    reasoning: list[str]
    executed: bool
    execution_details: dict[str, Any] = field(default_factory=dict)
    decision_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        """Create Decision from dictionary."""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class DecisionLogger:
    """Logs and retrieves trading decisions for audit trail."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize logger with optional database session.

        Args:
            db_session: SQLAlchemy session. If None, operates in memory only.
        """
        self._session = db_session
        self._memory_log: list[Decision] = []

    def _get_model(self):
        """Lazy import to avoid circular dependencies."""
        from src.data.models import DecisionLog as DecisionLogModel, DecisionType

        return DecisionLogModel, DecisionType

    def log_decision(self, decision: Decision) -> int:
        """Log decision to database and return decision ID.

        Args:
            decision: Decision object to log.

        Returns:
            ID of the logged decision.
        """
        if self._session is not None:
            DecisionLogModel, DecisionType = self._get_model()

            decision_type = DecisionType[decision.decision.upper()]

            log_entry = DecisionLogModel(
                timestamp=decision.timestamp,
                symbol=decision.symbol,
                decision=decision_type,
                confidence=decision.confidence,
                inputs=decision.inputs,
                reasoning=decision.reasoning,
                executed=decision.executed,
                execution_details=decision.execution_details,
            )

            self._session.add(log_entry)
            self._session.flush()
            decision.decision_id = log_entry.id
            return log_entry.id

        self._memory_log.append(decision)
        decision_id = len(self._memory_log)
        decision.decision_id = decision_id
        return decision_id

    def get_decisions(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        decision_type: Optional[str] = None,
        executed_only: bool = False,
        limit: int = 100,
    ) -> list[Decision]:
        """Retrieve logged decisions with filters.

        Args:
            symbol: Filter by stock symbol.
            start: Start of time range (inclusive).
            end: End of time range (inclusive).
            decision_type: Filter by decision type (BUY, SELL, HOLD, SKIP).
            executed_only: If True, only return executed decisions.
            limit: Maximum number of decisions to return.

        Returns:
            List of Decision objects matching filters.
        """
        if self._session is not None:
            DecisionLogModel, DecisionType = self._get_model()

            query = self._session.query(DecisionLogModel)

            if symbol:
                query = query.filter(DecisionLogModel.symbol == symbol)
            if start:
                query = query.filter(DecisionLogModel.timestamp >= start)
            if end:
                query = query.filter(DecisionLogModel.timestamp <= end)
            if decision_type:
                query = query.filter(
                    DecisionLogModel.decision == DecisionType[decision_type.upper()]
                )
            if executed_only:
                query = query.filter(DecisionLogModel.executed == True)

            query = query.order_by(DecisionLogModel.timestamp.desc()).limit(limit)

            results = []
            for log in query.all():
                results.append(
                    Decision(
                        timestamp=log.timestamp,
                        symbol=log.symbol,
                        decision=log.decision.value,
                        confidence=float(log.confidence) if log.confidence else 0.0,
                        inputs=log.inputs or {},
                        reasoning=log.reasoning or [],
                        executed=log.executed,
                        execution_details=log.execution_details or {},
                        decision_id=log.id,
                    )
                )
            return results

        results = self._memory_log.copy()

        if symbol:
            results = [d for d in results if d.symbol == symbol]
        if start:
            results = [d for d in results if d.timestamp >= start]
        if end:
            results = [d for d in results if d.timestamp <= end]
        if decision_type:
            results = [
                d for d in results if d.decision.upper() == decision_type.upper()
            ]
        if executed_only:
            results = [d for d in results if d.executed]

        results.sort(key=lambda d: d.timestamp, reverse=True)
        return results[:limit]

    def get_decision_by_id(self, decision_id: int) -> Optional[Decision]:
        """Get specific decision by ID.

        Args:
            decision_id: ID of the decision to retrieve.

        Returns:
            Decision object if found, None otherwise.
        """
        if self._session is not None:
            DecisionLogModel, _ = self._get_model()

            log = self._session.query(DecisionLogModel).get(decision_id)
            if log is None:
                return None

            return Decision(
                timestamp=log.timestamp,
                symbol=log.symbol,
                decision=log.decision.value,
                confidence=float(log.confidence) if log.confidence else 0.0,
                inputs=log.inputs or {},
                reasoning=log.reasoning or [],
                executed=log.executed,
                execution_details=log.execution_details or {},
                decision_id=log.id,
            )

        if 0 < decision_id <= len(self._memory_log):
            return self._memory_log[decision_id - 1]
        return None

    def export_decisions(
        self,
        start: datetime,
        end: datetime,
        format: str = "json",
    ) -> str:
        """Export decisions for analysis.

        Args:
            start: Start of time range.
            end: End of time range.
            format: Export format ("json" or "csv").

        Returns:
            String with exported data in requested format.
        """
        decisions = self.get_decisions(start=start, end=end, limit=10000)

        if format == "csv":
            return self._export_csv(decisions)
        return self._export_json(decisions)

    def _export_json(self, decisions: list[Decision]) -> str:
        """Export decisions as JSON."""
        return json.dumps(
            [d.to_dict() for d in decisions],
            indent=2,
            default=str,
        )

    def _export_csv(self, decisions: list[Decision]) -> str:
        """Export decisions as CSV."""
        if not decisions:
            return "timestamp,symbol,decision,confidence,executed\n"

        lines = ["timestamp,symbol,decision,confidence,executed"]
        for d in decisions:
            lines.append(
                f"{d.timestamp.isoformat()},{d.symbol},{d.decision},"
                f"{d.confidence:.4f},{d.executed}"
            )
        return "\n".join(lines)

    def update_execution(
        self, decision_id: int, executed: bool, execution_details: dict[str, Any]
    ) -> bool:
        """Update execution status of a decision.

        Used when a decision is executed after being logged.

        Args:
            decision_id: ID of the decision to update.
            executed: Whether the decision was executed.
            execution_details: Details about the execution (order ID, price, etc).

        Returns:
            True if update was successful, False otherwise.
        """
        if self._session is not None:
            DecisionLogModel, _ = self._get_model()

            log = self._session.query(DecisionLogModel).get(decision_id)
            if log is None:
                return False

            log.executed = executed
            log.execution_details = execution_details
            return True

        if 0 < decision_id <= len(self._memory_log):
            decision = self._memory_log[decision_id - 1]
            decision.executed = executed
            decision.execution_details = execution_details
            return True
        return False

    def get_decision_statistics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get statistics on decisions for a time period.

        Args:
            start: Start of time range.
            end: End of time range.

        Returns:
            Dict with statistics:
                - total_decisions: total count
                - by_type: count by decision type
                - executed_count: number executed
                - avg_confidence: average confidence
        """
        decisions = self.get_decisions(start=start, end=end, limit=10000)

        if not decisions:
            return {
                "total_decisions": 0,
                "by_type": {},
                "executed_count": 0,
                "avg_confidence": 0.0,
            }

        by_type: dict[str, int] = {}
        for d in decisions:
            by_type[d.decision] = by_type.get(d.decision, 0) + 1

        executed_count = sum(1 for d in decisions if d.executed)
        avg_confidence = sum(d.confidence for d in decisions) / len(decisions)

        return {
            "total_decisions": len(decisions),
            "by_type": by_type,
            "executed_count": executed_count,
            "avg_confidence": avg_confidence,
        }

    def clear_memory_log(self) -> None:
        """Clear in-memory log. Does not affect database."""
        self._memory_log.clear()
