"""Sentiment aggregation agent for Leprechaun trading bot.

Aggregates sentiment from multiple sources (Reddit, StockTwits, News)
with source-specific weighting and anomaly detection.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional


@dataclass
class SourceWeight:
    """Weights for different sentiment sources."""

    reddit: float = 0.4
    stocktwits: float = 0.3
    news: float = 0.3


@dataclass
class SentimentResult:
    """Result of sentiment aggregation."""

    composite_score: float
    volume: int
    velocity: float
    sources: dict[str, dict[str, Any]]
    anomalies: list[dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "composite_score": self.composite_score,
            "volume": self.volume,
            "velocity": self.velocity,
            "sources": self.sources,
            "anomalies": self.anomalies,
            "timestamp": self.timestamp.isoformat(),
        }


class SentimentAgent:
    """Aggregates and analyzes sentiment from multiple sources."""

    ANOMALY_THRESHOLD_SCORE = 2.0
    ANOMALY_THRESHOLD_VELOCITY = 3.0
    DEFAULT_LOOKBACK_HOURS = 6
    MIN_POSTS_FOR_VELOCITY = 5

    def __init__(
        self,
        source_weights: Optional[dict[str, float]] = None,
    ):
        """Initialize sentiment agent.

        Args:
            source_weights: Dict of source name to weight. If None, uses defaults.
        """
        self.source_weights = SourceWeight()

        if source_weights:
            for source, weight in source_weights.items():
                if hasattr(self.source_weights, source):
                    setattr(self.source_weights, source, max(0.0, min(1.0, weight)))

    def aggregate_sentiment(
        self,
        symbol: str,
        sentiment_data: list[dict],
    ) -> SentimentResult:
        """Aggregate sentiment from multiple sources.

        Args:
            symbol: Stock symbol being analyzed.
            sentiment_data: List of sentiment records with keys:
                - source: str ("reddit", "stocktwits", "news")
                - score: float (-1 to 1)
                - volume: int (number of posts/mentions)
                - timestamp: datetime or ISO string

        Returns:
            SentimentResult with aggregated metrics.
        """
        if not sentiment_data:
            return SentimentResult(
                composite_score=0.0,
                volume=0,
                velocity=0.0,
                sources={},
                anomalies=[],
            )

        by_source: dict[str, list[dict]] = {}
        for record in sentiment_data:
            source = record.get("source", "unknown").lower()
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(record)

        source_summaries: dict[str, dict[str, Any]] = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        total_volume = 0

        for source, records in by_source.items():
            weight = getattr(self.source_weights, source, 0.1)

            source_scores = [r.get("score", 0.0) for r in records]
            source_volumes = [r.get("volume", 1) for r in records]

            weighted_scores = [
                score * vol for score, vol in zip(source_scores, source_volumes)
            ]
            total_source_volume = sum(source_volumes)

            if total_source_volume > 0:
                avg_score = sum(weighted_scores) / total_source_volume
            else:
                avg_score = (
                    sum(source_scores) / len(source_scores) if source_scores else 0.0
                )

            source_summaries[source] = {
                "avg_score": avg_score,
                "volume": total_source_volume,
                "count": len(records),
                "weight": weight,
            }

            total_weighted_score += avg_score * weight * total_source_volume
            total_weight += weight * total_source_volume
            total_volume += total_source_volume

        composite_score = (
            total_weighted_score / total_weight if total_weight > 0 else 0.0
        )
        composite_score = max(-1.0, min(1.0, composite_score))

        velocity = self.calculate_velocity(sentiment_data)

        anomalies = self.detect_anomalies(sentiment_data)

        return SentimentResult(
            composite_score=composite_score,
            volume=total_volume,
            velocity=velocity,
            sources=source_summaries,
            anomalies=anomalies,
        )

    def detect_anomalies(self, sentiment_history: list[dict]) -> list[dict[str, Any]]:
        """Detect unusual sentiment patterns.

        Detects:
        - Score anomalies (deviations from mean)
        - Velocity anomalies (rapid changes)
        - Volume anomalies (unusual post counts)

        Args:
            sentiment_history: List of sentiment records.

        Returns:
            List of detected anomalies with type, severity, and details.
        """
        if len(sentiment_history) < 3:
            return []

        anomalies = []

        scores = [r.get("score", 0.0) for r in sentiment_history]
        if len(scores) >= 3:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance) if variance > 0 else 0.001

            latest_score = scores[-1]
            z_score = abs(latest_score - mean_score) / std_dev

            if z_score > self.ANOMALY_THRESHOLD_SCORE:
                anomalies.append(
                    {
                        "type": "score_anomaly",
                        "severity": min(1.0, z_score / 5.0),
                        "details": {
                            "latest_score": latest_score,
                            "mean_score": mean_score,
                            "z_score": z_score,
                            "direction": (
                                "negative" if latest_score < mean_score else "positive"
                            ),
                        },
                    }
                )

        velocity = self.calculate_velocity(sentiment_history)
        if abs(velocity) > self.ANOMALY_THRESHOLD_VELOCITY:
            anomalies.append(
                {
                    "type": "velocity_anomaly",
                    "severity": min(1.0, abs(velocity) / 10.0),
                    "details": {
                        "velocity": velocity,
                        "direction": "decreasing" if velocity < 0 else "increasing",
                    },
                }
            )

        volumes = [r.get("volume", 1) for r in sentiment_history]
        if len(volumes) >= 3:
            mean_volume = sum(volumes) / len(volumes)
            if mean_volume > 0:
                variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
                std_dev = math.sqrt(variance) if variance > 0 else 1.0

                latest_volume = volumes[-1]
                z_score_vol = (latest_volume - mean_volume) / std_dev

                if z_score_vol > self.ANOMALY_THRESHOLD_SCORE:
                    anomalies.append(
                        {
                            "type": "volume_anomaly",
                            "severity": min(1.0, z_score_vol / 5.0),
                            "details": {
                                "latest_volume": latest_volume,
                                "mean_volume": mean_volume,
                                "z_score": z_score_vol,
                            },
                        }
                    )

        return anomalies

    def calculate_velocity(
        self,
        sentiment_history: list[dict],
        window: int = 6,
    ) -> float:
        """Calculate rate of sentiment change over time window.

        Uses linear regression slope for robust velocity estimation.

        Args:
            sentiment_history: List of sentiment records with timestamp and score.
            window: Time window in hours (default 6).

        Returns:
            Velocity as rate of change per hour (-1 to 1 scale).
        """
        if len(sentiment_history) < self.MIN_POSTS_FOR_VELOCITY:
            return 0.0

        now = datetime.utcnow()
        window_start = now - timedelta(hours=window)

        recent = []
        for record in sentiment_history:
            ts = record.get("timestamp")
            if ts is None:
                continue

            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                except ValueError:
                    continue

            if ts >= window_start:
                recent.append((ts, record.get("score", 0.0)))

        if len(recent) < 2:
            return 0.0

        recent.sort(key=lambda x: x[0])

        base_time = recent[0][0]
        points = [
            ((ts - base_time).total_seconds() / 3600, score) for ts, score in recent
        ]

        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_x2 = sum(p[0] ** 2 for p in points)

        denominator = n * sum_x2 - sum_x**2
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        return max(-1.0, min(1.0, slope))

    def get_source_breakdown(
        self,
        sentiment_data: list[dict],
    ) -> dict[str, dict[str, Any]]:
        """Get detailed breakdown by source.

        Args:
            sentiment_data: List of sentiment records.

        Returns:
            Dict mapping source to detailed statistics.
        """
        by_source: dict[str, list[dict]] = {}
        for record in sentiment_data:
            source = record.get("source", "unknown").lower()
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(record)

        breakdown = {}
        for source, records in by_source.items():
            scores = [r.get("score", 0.0) for r in records]
            volumes = [r.get("volume", 1) for r in records]

            mean_score = sum(scores) / len(scores) if scores else 0.0
            total_volume = sum(volumes)

            variance = (
                sum((s - mean_score) ** 2 for s in scores) / len(scores)
                if scores
                else 0.0
            )
            std_dev = math.sqrt(variance) if variance > 0 else 0.0

            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0

            breakdown[source] = {
                "mean_score": mean_score,
                "std_dev": std_dev,
                "min_score": min_score,
                "max_score": max_score,
                "total_volume": total_volume,
                "record_count": len(records),
                "weight": getattr(self.source_weights, source, 0.1),
            }

        return breakdown

    def is_sentiment_spike(
        self,
        current: SentimentResult,
        threshold: float = -0.4,
    ) -> bool:
        """Check if current sentiment represents a negative spike.

        Used to identify potential manipulation or panic selling.

        Args:
            current: Current sentiment result.
            threshold: Score threshold for spike detection (default -0.4).

        Returns:
            True if sentiment is spiking negatively.
        """
        return current.composite_score < threshold

    def compare_sentiment(
        self,
        current: SentimentResult,
        previous: SentimentResult,
    ) -> dict[str, Any]:
        """Compare two sentiment results.

        Args:
            current: Current sentiment.
            previous: Previous sentiment.

        Returns:
            Dict with comparison metrics.
        """
        score_change = current.composite_score - previous.composite_score
        volume_change = (
            (current.volume - previous.volume) / previous.volume
            if previous.volume > 0
            else 0.0
        )
        velocity_change = current.velocity - previous.velocity

        return {
            "score_change": score_change,
            "volume_change_pct": volume_change * 100,
            "velocity_change": velocity_change,
            "is_improving": score_change > 0,
            "is_accelerating": velocity_change > 0,
            "volume_increasing": volume_change > 0,
        }

    def get_weights(self) -> dict[str, float]:
        """Get current source weights as dictionary."""
        return {
            "reddit": self.source_weights.reddit,
            "stocktwits": self.source_weights.stocktwits,
            "news": self.source_weights.news,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set source weights from dictionary."""
        for source, weight in weights.items():
            if hasattr(self.source_weights, source):
                setattr(self.source_weights, source, max(0.0, min(1.0, weight)))
