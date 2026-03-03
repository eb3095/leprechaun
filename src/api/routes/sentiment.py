"""Sentiment routes for Leprechaun API."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from src.api.middleware.auth import viewer_or_admin
from src.api.middleware.rate_limit import default_rate_limit
from src.api.services import get_services

logger = logging.getLogger(__name__)

sentiment_bp = Blueprint("sentiment", __name__, url_prefix="/api/v1")


def _get_period_hours(period: str) -> int:
    """Convert period string to hours."""
    period_map = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    return period_map.get(period, 24)


def _get_sentiment_from_db(
    symbol: str, hours: int = 24
) -> tuple[list[dict], Optional[int]]:
    """Get sentiment data from database for a symbol."""
    try:
        from flask import current_app
        from src.data.models import SentimentData, Stock

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return [], None

        session = current_app.db_session()
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return [], None

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        records = (
            session.query(SentimentData)
            .filter(
                SentimentData.stock_id == stock.id,
                SentimentData.timestamp >= cutoff,
            )
            .order_by(SentimentData.timestamp.desc())
            .all()
        )

        sentiment_data = []
        for record in records:
            sentiment_data.append(
                {
                    "source": record.source.value,
                    "score": float(record.sentiment_score) if record.sentiment_score else 0.0,
                    "volume": record.sentiment_volume or 0,
                    "timestamp": record.timestamp.isoformat(),
                    "velocity": float(record.velocity) if record.velocity else 0.0,
                    "bot_fraction": float(record.bot_fraction) if record.bot_fraction else 0.0,
                    "coordination_score": float(record.coordination_score) if record.coordination_score else 0.0,
                }
            )

        return sentiment_data, stock.id
    except Exception as e:
        logger.warning("Could not get sentiment from database: %s", e)
        return [], None


def _get_historical_sentiment(stock_id: int, hours: int = 24) -> list[dict]:
    """Get historical sentiment aggregated by hour."""
    try:
        from flask import current_app
        from sqlalchemy import func
        from src.data.models import SentimentData

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return []

        session = current_app.db_session()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        results = (
            session.query(
                func.date_format(SentimentData.timestamp, "%Y-%m-%d %H:00:00").label("hour"),
                func.avg(SentimentData.sentiment_score).label("avg_score"),
                func.sum(SentimentData.sentiment_volume).label("total_volume"),
            )
            .filter(
                SentimentData.stock_id == stock_id,
                SentimentData.timestamp >= cutoff,
            )
            .group_by(func.date_format(SentimentData.timestamp, "%Y-%m-%d %H:00:00"))
            .order_by(func.date_format(SentimentData.timestamp, "%Y-%m-%d %H:00:00"))
            .all()
        )

        return [
            {
                "timestamp": row.hour,
                "score": float(row.avg_score) if row.avg_score else 0.0,
                "volume": int(row.total_volume) if row.total_volume else 0,
            }
            for row in results
        ]
    except Exception as e:
        logger.warning("Could not get historical sentiment: %s", e)
        return []


@sentiment_bp.route("/sentiment/<symbol>", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_sentiment(symbol: str):
    """Get sentiment data for a specific stock.

    ---
    tags:
      - Sentiment
    security:
      - bearerAuth: []
    parameters:
      - in: path
        name: symbol
        schema:
          type: string
        required: true
        description: Stock symbol (e.g., AAPL, TSLA)
      - in: query
        name: period
        schema:
          type: string
          enum: [1h, 4h, 1d, 1w]
          default: 1d
        description: Time period for aggregated sentiment
    responses:
      200:
        description: Sentiment data for the stock
        content:
          application/json:
            schema:
              type: object
              properties:
                symbol:
                  type: string
                period:
                  type: string
                sentiment_score:
                  type: number
                  description: Aggregated sentiment score (-1 to 1)
                sentiment_volume:
                  type: integer
                  description: Number of mentions/posts
                velocity:
                  type: number
                  description: Rate of change in mentions
                sources:
                  type: object
                  properties:
                    reddit:
                      type: object
                    stocktwits:
                      type: object
                    news:
                      type: object
                historical:
                  type: array
                  items:
                    type: object
      401:
        description: Authentication required
      404:
        description: Symbol not found
    """
    services = get_services()
    symbol = symbol.upper()
    period = request.args.get("period", "1d")
    hours = _get_period_hours(period)

    sentiment_data, stock_id = _get_sentiment_from_db(symbol, hours)

    if not sentiment_data:
        if not _stock_exists(symbol):
            return (
                jsonify(
                    {
                        "error": "not_found",
                        "message": f"Symbol {symbol} not found in universe",
                    }
                ),
                404,
            )

        return jsonify(
            {
                "symbol": symbol,
                "period": period,
                "sentiment_score": 0.0,
                "sentiment_volume": 0,
                "velocity": 0.0,
                "sources": {
                    "reddit": {"score": 0.0, "volume": 0},
                    "stocktwits": {"score": 0.0, "volume": 0},
                    "news": {"score": 0.0, "volume": 0},
                },
                "historical": [],
                "message": "No sentiment data available for this period",
            }
        )

    sentiment_result = services.sentiment_agent.aggregate_sentiment(symbol, sentiment_data)

    sources_response = {}
    for source, data in sentiment_result.sources.items():
        sources_response[source] = {
            "score": data.get("avg_score", 0.0),
            "volume": data.get("volume", 0),
            "count": data.get("count", 0),
        }

    for default_source in ["reddit", "stocktwits", "news"]:
        if default_source not in sources_response:
            sources_response[default_source] = {"score": 0.0, "volume": 0}

    historical = []
    if stock_id:
        historical = _get_historical_sentiment(stock_id, hours)

    return jsonify(
        {
            "symbol": symbol,
            "period": period,
            "sentiment_score": sentiment_result.composite_score,
            "sentiment_volume": sentiment_result.volume,
            "velocity": sentiment_result.velocity,
            "sources": sources_response,
            "historical": historical,
            "anomalies": sentiment_result.anomalies,
            "timestamp": sentiment_result.timestamp.isoformat(),
        }
    )


@sentiment_bp.route("/sentiment/trending", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_trending():
    """Get trending stocks by sentiment activity.

    ---
    tags:
      - Sentiment
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: limit
        schema:
          type: integer
          default: 20
        description: Maximum number of stocks to return
      - in: query
        name: sort_by
        schema:
          type: string
          enum: [volume, velocity, score]
          default: volume
        description: Sort criteria
      - in: query
        name: filter
        schema:
          type: string
          enum: [positive, negative, all]
          default: all
        description: Filter by sentiment direction
    responses:
      200:
        description: List of trending stocks
        content:
          application/json:
            schema:
              type: object
              properties:
                trending:
                  type: array
                  items:
                    type: object
                    properties:
                      symbol:
                        type: string
                      name:
                        type: string
                      sentiment_score:
                        type: number
                      sentiment_volume:
                        type: integer
                      velocity:
                        type: number
                      change_24h:
                        type: number
                timestamp:
                  type: string
                  format: date-time
      401:
        description: Authentication required
    """
    limit = request.args.get("limit", 20, type=int)
    limit = 20 if limit is None else min(max(1, limit), 100)
    sort_by = request.args.get("sort_by", "volume")
    sentiment_filter = request.args.get("filter", "all")

    trending = _get_trending_from_db(limit * 2, sort_by)

    if sentiment_filter == "positive":
        trending = [t for t in trending if t.get("sentiment_score", 0) > 0]
    elif sentiment_filter == "negative":
        trending = [t for t in trending if t.get("sentiment_score", 0) < 0]

    trending = trending[:limit]

    return jsonify(
        {
            "trending": trending,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sort_by": sort_by,
            "filter": sentiment_filter,
        }
    )


def _get_trending_from_db(limit: int, sort_by: str) -> list[dict]:
    """Get trending stocks from database based on sentiment activity."""
    try:
        from flask import current_app
        from sqlalchemy import func, desc
        from src.data.models import SentimentData, Stock

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return []

        session = current_app.db_session()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        sort_column = {
            "volume": func.sum(SentimentData.sentiment_volume),
            "velocity": func.max(SentimentData.velocity),
            "score": func.abs(func.avg(SentimentData.sentiment_score)),
        }.get(sort_by, func.sum(SentimentData.sentiment_volume))

        results = (
            session.query(
                Stock.symbol,
                Stock.name,
                func.avg(SentimentData.sentiment_score).label("avg_score"),
                func.sum(SentimentData.sentiment_volume).label("total_volume"),
                func.max(SentimentData.velocity).label("max_velocity"),
            )
            .join(SentimentData, Stock.id == SentimentData.stock_id)
            .filter(SentimentData.timestamp >= cutoff)
            .group_by(Stock.id, Stock.symbol, Stock.name)
            .having(func.sum(SentimentData.sentiment_volume) > 0)
            .order_by(desc(sort_column))
            .limit(limit)
            .all()
        )

        return [
            {
                "symbol": row.symbol,
                "name": row.name or row.symbol,
                "sentiment_score": float(row.avg_score) if row.avg_score else 0.0,
                "sentiment_volume": int(row.total_volume) if row.total_volume else 0,
                "velocity": float(row.max_velocity) if row.max_velocity else 0.0,
            }
            for row in results
        ]
    except Exception as e:
        logger.warning("Could not get trending from database: %s", e)
        return []


@sentiment_bp.route("/manipulation/<symbol>", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_manipulation_score(symbol: str):
    """Get manipulation detection score for a stock.

    ---
    tags:
      - Sentiment
    security:
      - bearerAuth: []
    parameters:
      - in: path
        name: symbol
        schema:
          type: string
        required: true
        description: Stock symbol (e.g., AAPL, TSLA)
    responses:
      200:
        description: Manipulation analysis for the stock
        content:
          application/json:
            schema:
              type: object
              properties:
                symbol:
                  type: string
                manipulation_score:
                  type: number
                  description: Probability of manipulation (0 to 1)
                bayesian_probability:
                  type: number
                divergence_score:
                  type: number
                  description: Sentiment vs price divergence
                has_news_catalyst:
                  type: boolean
                bot_activity:
                  type: number
                coordination_score:
                  type: number
                triggered_signals:
                  type: array
                  items:
                    type: string
                recommendation:
                  type: string
                  enum: [strong_buy, buy, neutral, avoid]
      401:
        description: Authentication required
      404:
        description: Symbol not found
    """
    services = get_services()
    symbol = symbol.upper()

    if not _stock_exists(symbol):
        return (
            jsonify(
                {
                    "error": "not_found",
                    "message": f"Symbol {symbol} not found in universe",
                }
            ),
            404,
        )

    db_result = _get_latest_manipulation_from_db(symbol)
    if db_result:
        return jsonify(db_result)

    sentiment_data, _ = _get_sentiment_from_db(symbol, hours=24)

    if sentiment_data:
        sentiment_result = services.sentiment_agent.aggregate_sentiment(
            symbol, sentiment_data
        )

        analysis_data = {
            "sentiment": sentiment_result.to_dict(),
            "price_history": [],
            "news": [],
            "technical_indicators": {},
            "posts": [],
        }

        result = services.manipulation_agent.analyze(symbol, analysis_data)

        evidence = result.get("evidence", {})
        triggered = [k for k, v in evidence.items() if v]

        return jsonify(
            {
                "symbol": symbol,
                "manipulation_score": result.get("manipulation_score", 0.0),
                "bayesian_probability": result.get("bayesian_probability", 0.0),
                "divergence_score": 0.0,
                "has_news_catalyst": not evidence.get("no_news_catalyst", True),
                "bot_activity": 0.0,
                "coordination_score": 0.0,
                "triggered_signals": triggered,
                "recommendation": _map_recommendation(result.get("recommendation", "")),
                "confidence": result.get("confidence", "LOW"),
                "explanation": result.get("explanation", {}),
                "timestamp": result.get("timestamp"),
            }
        )

    return jsonify(
        {
            "symbol": symbol,
            "manipulation_score": 0.0,
            "bayesian_probability": 0.0,
            "divergence_score": 0.0,
            "has_news_catalyst": True,
            "bot_activity": 0.0,
            "coordination_score": 0.0,
            "triggered_signals": [],
            "recommendation": "neutral",
            "message": "Insufficient data for manipulation analysis",
        }
    )


def _get_latest_manipulation_from_db(symbol: str) -> Optional[dict]:
    """Get the latest manipulation score from database."""
    try:
        from flask import current_app
        from src.data.models import ManipulationScore, Stock

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return None

        session = current_app.db_session()
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        record = (
            session.query(ManipulationScore)
            .filter(
                ManipulationScore.stock_id == stock.id,
                ManipulationScore.timestamp >= cutoff,
            )
            .order_by(ManipulationScore.timestamp.desc())
            .first()
        )

        if not record:
            return None

        triggered = []
        if record.triggered_signals:
            triggered = [k for k, v in record.triggered_signals.items() if v]

        return {
            "symbol": symbol,
            "manipulation_score": float(record.manipulation_score) if record.manipulation_score else 0.0,
            "bayesian_probability": float(record.bayesian_probability) if record.bayesian_probability else 0.0,
            "divergence_score": float(record.divergence_score) if record.divergence_score else 0.0,
            "has_news_catalyst": record.has_news_catalyst if record.has_news_catalyst is not None else True,
            "bot_activity": 0.0,
            "coordination_score": 0.0,
            "triggered_signals": triggered,
            "recommendation": _score_to_recommendation(float(record.manipulation_score) if record.manipulation_score else 0.0),
            "timestamp": record.timestamp.isoformat(),
        }
    except Exception as e:
        logger.warning("Could not get manipulation from database: %s", e)
        return None


def _stock_exists(symbol: str) -> bool:
    """Check if stock exists in universe."""
    try:
        from flask import current_app
        from src.data.models import Stock

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return True

        session = current_app.db_session()
        return session.query(Stock).filter(Stock.symbol == symbol).first() is not None
    except Exception:
        return True


def _map_recommendation(rec: str) -> str:
    """Map internal recommendation to API response format."""
    rec_lower = rec.lower()
    if "strong" in rec_lower and "buy" in rec_lower:
        return "strong_buy"
    elif "buy" in rec_lower:
        return "buy"
    elif "avoid" in rec_lower or "pass" in rec_lower:
        return "avoid"
    return "neutral"


def _score_to_recommendation(score: float) -> str:
    """Convert manipulation score to recommendation."""
    if score > 0.7:
        return "strong_buy"
    elif score > 0.5:
        return "buy"
    elif score < 0.3:
        return "avoid"
    return "neutral"
