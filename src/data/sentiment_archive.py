"""Sentiment archival system for storing historical sentiment data.

Archives real-time sentiment snapshots for future backtesting. Without
historical sentiment archives, backtesting must rely on synthetic data.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.models import SentimentData, SentimentSource, Stock

logger = logging.getLogger(__name__)


class SentimentArchive:
    """Archives sentiment data for future backtesting.

    Stores aggregated sentiment snapshots to the SentimentData table,
    enabling accurate backtesting with real historical sentiment.
    """

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the sentiment archive.

        Args:
            db_session: SQLAlchemy session for database operations.
        """
        self.session = db_session

    def archive_snapshot(self, symbol: str, sentiment_data: dict[str, Any]) -> int:
        """Archive a sentiment snapshot for a symbol.

        Args:
            symbol: Stock ticker symbol.
            sentiment_data: Dictionary containing:
                - timestamp: datetime of the snapshot
                - composite_score: float (-1 to 1)
                - volume: int (number of mentions)
                - sources: dict with per-source data
                - manipulation_score: float (0 to 1)
                - velocity: float (optional)
                - bot_fraction: float (optional)
                - coordination_score: float (optional)

        Returns:
            Archive ID (primary key of inserted record).

        Raises:
            RuntimeError: If no database session is available.
            ValueError: If symbol not found in database.
        """
        if self.session is None:
            raise RuntimeError("Database session required for archive_snapshot")

        stock = self.session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if stock is None:
            raise ValueError(f"Stock {symbol} not found in database")

        timestamp = sentiment_data.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)

        composite_score = sentiment_data.get("composite_score", 0.0)
        volume = sentiment_data.get("volume", 0)
        sources = sentiment_data.get("sources", {})
        velocity = sentiment_data.get("velocity")
        bot_fraction = sentiment_data.get("bot_fraction")
        coordination_score = sentiment_data.get("coordination_score")

        record = SentimentData(
            stock_id=stock.id,
            timestamp=timestamp,
            source=SentimentSource.REDDIT,
            sentiment_score=Decimal(str(composite_score)),
            sentiment_volume=volume,
            velocity=Decimal(str(velocity)) if velocity is not None else None,
            bot_fraction=Decimal(str(bot_fraction)) if bot_fraction is not None else None,
            coordination_score=(
                Decimal(str(coordination_score)) if coordination_score is not None else None
            ),
            raw_data={
                "sources": sources,
                "manipulation_score": sentiment_data.get("manipulation_score"),
                "is_archived": True,
                "archive_version": 1,
            },
        )

        self.session.add(record)
        self.session.flush()

        logger.debug(
            "Archived sentiment for %s: score=%.3f, volume=%d",
            symbol,
            composite_score,
            volume,
        )
        return record.id

    def archive_source_snapshot(
        self,
        symbol: str,
        source: str,
        score: float,
        volume: int,
        timestamp: Optional[datetime] = None,
        raw_data: Optional[dict] = None,
    ) -> int:
        """Archive sentiment from a specific source.

        Args:
            symbol: Stock ticker symbol.
            source: Source name (reddit, stocktwits, news).
            score: Sentiment score (-1 to 1).
            volume: Number of mentions.
            timestamp: Timestamp (defaults to now).
            raw_data: Optional raw source data.

        Returns:
            Archive ID.

        Raises:
            RuntimeError: If no database session is available.
            ValueError: If symbol not found or invalid source.
        """
        if self.session is None:
            raise RuntimeError("Database session required for archive_source_snapshot")

        stock = self.session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if stock is None:
            raise ValueError(f"Stock {symbol} not found in database")

        source_lower = source.lower()
        source_enum_map = {
            "reddit": SentimentSource.REDDIT,
            "stocktwits": SentimentSource.STOCKTWITS,
            "news": SentimentSource.NEWS,
        }
        if source_lower not in source_enum_map:
            raise ValueError(f"Invalid source: {source}")

        ts = timestamp or datetime.now()

        record = SentimentData(
            stock_id=stock.id,
            timestamp=ts,
            source=source_enum_map[source_lower],
            sentiment_score=Decimal(str(score)),
            sentiment_volume=volume,
            raw_data=raw_data or {},
        )

        self.session.add(record)
        self.session.flush()
        return record.id

    def get_archived_sentiment(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Retrieve archived sentiment for backtesting.

        Args:
            symbol: Stock ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of sentiment dictionaries sorted by timestamp.

        Raises:
            RuntimeError: If no database session is available.
        """
        if self.session is None:
            raise RuntimeError("Database session required for get_archived_sentiment")

        stock = self.session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if stock is None:
            return []

        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end, datetime.max.time())

        records = (
            self.session.query(SentimentData)
            .filter(
                SentimentData.stock_id == stock.id,
                SentimentData.timestamp >= start_dt,
                SentimentData.timestamp <= end_dt,
            )
            .order_by(SentimentData.timestamp)
            .all()
        )

        results = []
        for r in records:
            raw = r.raw_data or {}
            results.append({
                "id": r.id,
                "timestamp": r.timestamp,
                "date": r.timestamp.date(),
                "source": r.source.value if r.source else None,
                "sentiment_score": float(r.sentiment_score) if r.sentiment_score else 0.0,
                "sentiment_volume": r.sentiment_volume or 0,
                "velocity": float(r.velocity) if r.velocity else None,
                "bot_fraction": float(r.bot_fraction) if r.bot_fraction else None,
                "coordination_score": float(r.coordination_score) if r.coordination_score else None,
                "manipulation_score": raw.get("manipulation_score"),
                "sources": raw.get("sources", {}),
                "is_synthetic": False,
            })

        return results

    def get_daily_sentiment(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Get daily aggregated sentiment for backtesting.

        Aggregates multiple snapshots per day into a single daily value.

        Args:
            symbol: Stock ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of daily sentiment dictionaries.
        """
        raw_data = self.get_archived_sentiment(symbol, start, end)
        if not raw_data:
            return []

        daily: dict[date, list[dict]] = {}
        for record in raw_data:
            d = record["date"]
            if d not in daily:
                daily[d] = []
            daily[d].append(record)

        results = []
        for d in sorted(daily.keys()):
            records = daily[d]
            scores = [r["sentiment_score"] for r in records if r["sentiment_score"] is not None]
            volumes = [r["sentiment_volume"] for r in records]
            manipulation_scores = [
                r["manipulation_score"]
                for r in records
                if r["manipulation_score"] is not None
            ]

            avg_score = sum(scores) / len(scores) if scores else 0.0
            total_volume = sum(volumes)
            avg_manipulation = (
                sum(manipulation_scores) / len(manipulation_scores)
                if manipulation_scores
                else None
            )

            results.append({
                "date": d,
                "sentiment_score": avg_score,
                "sentiment_volume": total_volume,
                "manipulation_score": avg_manipulation,
                "snapshot_count": len(records),
                "is_synthetic": False,
            })

        return results

    def has_historical_data(self, symbol: str, start: date, end: date) -> bool:
        """Check if we have archived data for the period.

        Args:
            symbol: Stock ticker symbol.
            start: Start date.
            end: End date.

        Returns:
            True if any archived data exists in the period.

        Raises:
            RuntimeError: If no database session is available.
        """
        if self.session is None:
            raise RuntimeError("Database session required for has_historical_data")

        stock = self.session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if stock is None:
            return False

        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end, datetime.max.time())

        count = (
            self.session.query(func.count(SentimentData.id))
            .filter(
                SentimentData.stock_id == stock.id,
                SentimentData.timestamp >= start_dt,
                SentimentData.timestamp <= end_dt,
            )
            .scalar()
        )

        return count > 0

    def get_coverage_report(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, Any]:
        """Report what percentage of the requested period has data.

        Args:
            symbols: List of stock symbols.
            start: Start date.
            end: End date.

        Returns:
            Dictionary with coverage statistics:
                - total_days: Number of trading days in period
                - symbol_coverage: Dict mapping symbol to coverage info
                - overall_coverage: Average coverage percentage
        """
        if self.session is None:
            raise RuntimeError("Database session required for get_coverage_report")

        total_days = (end - start).days + 1
        symbol_coverage = {}
        total_coverage = 0.0

        for symbol in symbols:
            stock = self.session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
            if stock is None:
                symbol_coverage[symbol] = {
                    "days_with_data": 0,
                    "coverage_percent": 0.0,
                    "earliest_date": None,
                    "latest_date": None,
                }
                continue

            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())

            date_counts = (
                self.session.query(
                    func.date(SentimentData.timestamp).label("date"),
                    func.count(SentimentData.id).label("count"),
                )
                .filter(
                    SentimentData.stock_id == stock.id,
                    SentimentData.timestamp >= start_dt,
                    SentimentData.timestamp <= end_dt,
                )
                .group_by(func.date(SentimentData.timestamp))
                .all()
            )

            days_with_data = len(date_counts)
            coverage_pct = (days_with_data / total_days * 100) if total_days > 0 else 0.0

            dates = [d[0] for d in date_counts]
            earliest = min(dates) if dates else None
            latest = max(dates) if dates else None

            symbol_coverage[symbol] = {
                "days_with_data": days_with_data,
                "coverage_percent": coverage_pct,
                "earliest_date": earliest,
                "latest_date": latest,
            }
            total_coverage += coverage_pct

        overall = total_coverage / len(symbols) if symbols else 0.0

        return {
            "total_days": total_days,
            "symbol_coverage": symbol_coverage,
            "overall_coverage": overall,
            "start_date": start,
            "end_date": end,
        }

    def delete_old_data(self, older_than: date) -> int:
        """Delete archived sentiment data older than a date.

        Args:
            older_than: Delete data before this date.

        Returns:
            Number of records deleted.

        Raises:
            RuntimeError: If no database session is available.
        """
        if self.session is None:
            raise RuntimeError("Database session required for delete_old_data")

        cutoff = datetime.combine(older_than, datetime.min.time())

        deleted = (
            self.session.query(SentimentData)
            .filter(SentimentData.timestamp < cutoff)
            .delete(synchronize_session=False)
        )

        self.session.flush()
        logger.info("Deleted %d sentiment records older than %s", deleted, older_than)
        return deleted


def create_archival_job(archive: SentimentArchive, sentiment_agent, universe) -> callable:
    """Create job function to archive sentiment for all symbols.

    Should be registered to run every hour during market hours.

    Args:
        archive: SentimentArchive instance.
        sentiment_agent: SentimentAgent for getting current sentiment.
        universe: StockUniverse or list of symbols.

    Returns:
        Job function that can be scheduled.
    """

    def archive_sentiment_job() -> dict[str, int]:
        """Archive current sentiment for all universe symbols."""
        archived_count = 0
        error_count = 0

        if hasattr(universe, "get_all_symbols"):
            symbols = universe.get_all_symbols()
        else:
            symbols = list(universe)

        for symbol in symbols:
            try:
                if hasattr(sentiment_agent, "get_sentiment"):
                    sentiment_data = sentiment_agent.get_sentiment(symbol)
                elif hasattr(sentiment_agent, "aggregate_sentiment"):
                    sentiment_data = sentiment_agent.aggregate_sentiment(symbol, [])
                    if hasattr(sentiment_data, "to_dict"):
                        sentiment_data = sentiment_data.to_dict()
                else:
                    continue

                if sentiment_data:
                    archive.archive_snapshot(symbol, sentiment_data)
                    archived_count += 1

            except Exception as e:
                logger.warning("Failed to archive sentiment for %s: %s", symbol, e)
                error_count += 1

        logger.info(
            "Sentiment archival complete: %d archived, %d errors",
            archived_count,
            error_count,
        )

        return {"archived": archived_count, "errors": error_count}

    archive_sentiment_job.__name__ = "archive_sentiment"
    return archive_sentiment_job
