"""Metrics routes for Leprechaun API."""

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from src.api.middleware.auth import viewer_or_admin
from src.api.middleware.rate_limit import default_rate_limit
from src.api.services import get_services
from src.api.utils import get_pagination_params

logger = logging.getLogger(__name__)

metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/v1/metrics")


def _to_float(value: Any) -> float:
    """Safely convert a value to float."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@metrics_bp.route("/portfolio", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_portfolio():
    """Get current portfolio metrics.

    ---
    tags:
      - Metrics
    security:
      - bearerAuth: []
    responses:
      200:
        description: Portfolio metrics
        content:
          application/json:
            schema:
              type: object
              properties:
                total_value:
                  type: number
                cash:
                  type: number
                positions_value:
                  type: number
                unrealized_pnl:
                  type: number
                unrealized_pnl_percent:
                  type: number
                daily_pnl:
                  type: number
                daily_pnl_percent:
                  type: number
      401:
        description: Authentication required
    """
    services = get_services()

    try:
        account = services.order_executor.get_account()
        positions = services.order_executor.get_positions()
    except Exception as e:
        logger.warning("Could not get account data: %s", e)
        account = {"equity": 0, "cash": 0, "portfolio_value": 0}
        positions = []

    total_value = _to_float(account.get("equity", 0))
    cash = _to_float(account.get("cash", 0))
    positions_value = sum(_to_float(p.get("market_value", 0)) for p in positions)
    unrealized_pnl = sum(_to_float(p.get("unrealized_pnl", 0)) for p in positions)
    unrealized_pnl_pct = (
        sum(_to_float(p.get("unrealized_pnl_pct", 0)) for p in positions) / len(positions)
        if positions
        else 0.0
    )

    daily_metrics = _get_latest_daily_metrics()
    daily_pnl = _to_float(daily_metrics.get("daily_pnl", 0)) if daily_metrics else 0.0
    daily_pnl_pct = _to_float(daily_metrics.get("daily_pnl_percent", 0)) if daily_metrics else 0.0

    return jsonify(
        {
            "total_value": total_value,
            "cash": cash,
            "positions_value": positions_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": unrealized_pnl_pct,
            "daily_pnl": daily_pnl,
            "daily_pnl_percent": daily_pnl_pct,
            "active_positions": len(positions),
            "is_simulated": services.order_executor.is_simulated,
        }
    )


@metrics_bp.route("/performance", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_performance():
    """Get historical performance metrics.

    ---
    tags:
      - Metrics
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: period
        schema:
          type: string
          enum: [1w, 1m, 3m, 6m, 1y, all]
          default: 1m
        description: Time period for performance data
    responses:
      200:
        description: Performance metrics
        content:
          application/json:
            schema:
              type: object
              properties:
                period:
                  type: string
                total_return:
                  type: number
                total_return_percent:
                  type: number
                sharpe_ratio:
                  type: number
                max_drawdown:
                  type: number
                win_rate:
                  type: number
                total_trades:
                  type: integer
                winning_trades:
                  type: integer
                losing_trades:
                  type: integer
                equity_curve:
                  type: array
                  items:
                    type: object
      401:
        description: Authentication required
    """
    period = request.args.get("period", "1m")

    period_days = {
        "1w": 7,
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365,
        "all": None,
    }
    days = period_days.get(period, 30)

    metrics_data = _get_performance_metrics(days)
    trade_stats = _get_trade_statistics(days)
    equity_curve = _get_equity_curve(days)

    return jsonify(
        {
            "period": period,
            "total_return": metrics_data.get("total_pnl", 0.0),
            "total_return_percent": metrics_data.get("total_pnl_percent", 0.0),
            "sharpe_ratio": metrics_data.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics_data.get("max_drawdown", 0.0),
            "win_rate": trade_stats.get("win_rate", 0.0),
            "total_trades": trade_stats.get("total", 0),
            "winning_trades": trade_stats.get("wins", 0),
            "losing_trades": trade_stats.get("losses", 0),
            "avg_win": trade_stats.get("avg_win", 0.0),
            "avg_loss": trade_stats.get("avg_loss", 0.0),
            "profit_factor": trade_stats.get("profit_factor", 0.0),
            "equity_curve": equity_curve,
        }
    )


@metrics_bp.route("/trades", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_trades():
    """Get trade history with details.

    ---
    tags:
      - Metrics
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: symbol
        schema:
          type: string
        description: Filter by stock symbol
      - in: query
        name: start_date
        schema:
          type: string
          format: date
        description: Start date for filtering
      - in: query
        name: end_date
        schema:
          type: string
          format: date
        description: End date for filtering
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
        description: Maximum number of trades to return
      - in: query
        name: offset
        schema:
          type: integer
          default: 0
        description: Number of trades to skip
    responses:
      200:
        description: Trade history
        content:
          application/json:
            schema:
              type: object
              properties:
                trades:
                  type: array
                  items:
                    type: object
                total:
                  type: integer
                limit:
                  type: integer
                offset:
                  type: integer
      401:
        description: Authentication required
    """
    symbol_filter = request.args.get("symbol")
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")
    limit, offset = get_pagination_params()

    start_date = None
    end_date = None
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str).date()
        except ValueError:
            pass
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str).date()
        except ValueError:
            pass

    trades, total = _get_trades_from_db(
        symbol=symbol_filter,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    return jsonify(
        {
            "trades": trades,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@metrics_bp.route("/daily", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_daily_pnl():
    """Get daily profit and loss data.

    ---
    tags:
      - Metrics
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: days
        schema:
          type: integer
          default: 30
        description: Number of days to return
    responses:
      200:
        description: Daily P&L data
        content:
          application/json:
            schema:
              type: object
              properties:
                daily_pnl:
                  type: array
                  items:
                    type: object
                    properties:
                      date:
                        type: string
                        format: date
                      pnl:
                        type: number
                      pnl_percent:
                        type: number
                      cumulative_pnl:
                        type: number
                total_pnl:
                  type: number
                average_daily_pnl:
                  type: number
      401:
        description: Authentication required
    """
    days = request.args.get("days", 30, type=int)
    days = 30 if days is None else min(max(1, days), 365)

    daily_data = _get_daily_pnl_from_db(days)

    total_pnl = sum(d.get("pnl", 0) for d in daily_data)
    avg_pnl = total_pnl / len(daily_data) if daily_data else 0.0

    cumulative = 0.0
    for entry in daily_data:
        cumulative += entry.get("pnl", 0)
        entry["cumulative_pnl"] = cumulative

    return jsonify(
        {
            "daily_pnl": daily_data,
            "total_pnl": total_pnl,
            "average_daily_pnl": avg_pnl,
            "days_returned": len(daily_data),
        }
    )


@metrics_bp.route("/weekly", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_weekly_pnl():
    """Get weekly profit and loss data.

    ---
    tags:
      - Metrics
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: weeks
        schema:
          type: integer
          default: 12
        description: Number of weeks to return
    responses:
      200:
        description: Weekly P&L data
        content:
          application/json:
            schema:
              type: object
              properties:
                weekly_pnl:
                  type: array
                  items:
                    type: object
                    properties:
                      week_start:
                        type: string
                        format: date
                      week_end:
                        type: string
                        format: date
                      pnl:
                        type: number
                      pnl_percent:
                        type: number
                      trades_count:
                        type: integer
                total_pnl:
                  type: number
                average_weekly_pnl:
                  type: number
      401:
        description: Authentication required
    """
    weeks = request.args.get("weeks", 12, type=int)
    weeks = 12 if weeks is None else min(max(1, weeks), 52)

    weekly_data = _get_weekly_pnl_from_db(weeks)

    total_pnl = sum(w.get("pnl", 0) for w in weekly_data)
    avg_pnl = total_pnl / len(weekly_data) if weekly_data else 0.0

    return jsonify(
        {
            "weekly_pnl": weekly_data,
            "total_pnl": total_pnl,
            "average_weekly_pnl": avg_pnl,
            "weeks_returned": len(weekly_data),
        }
    )


def _get_latest_daily_metrics() -> Optional[dict]:
    """Get the most recent portfolio metrics from database."""
    try:
        from flask import current_app
        from src.data.models import PortfolioMetric

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return None

        session = current_app.db_session()
        metric = (
            session.query(PortfolioMetric)
            .order_by(PortfolioMetric.date.desc())
            .first()
        )

        if metric:
            return metric.to_dict()
        return None
    except Exception as e:
        logger.warning("Could not get latest metrics: %s", e)
        return None


def _get_performance_metrics(days: Optional[int]) -> dict:
    """Get performance metrics from database."""
    try:
        from flask import current_app
        from src.data.models import PortfolioMetric

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return {}

        session = current_app.db_session()
        query = session.query(PortfolioMetric)

        if days:
            start_date = date.today() - timedelta(days=days)
            query = query.filter(PortfolioMetric.date >= start_date)

        metrics = query.order_by(PortfolioMetric.date.desc()).all()

        if not metrics:
            return {}

        latest = metrics[0]

        daily_returns = []
        for m in metrics:
            if m.daily_pnl_percent:
                daily_returns.append(_to_float(m.daily_pnl_percent))

        equity_values = [_to_float(m.total_value) for m in metrics if m.total_value]
        max_drawdown = 0.0
        if equity_values:
            peak = equity_values[0]
            for val in equity_values:
                if val > peak:
                    peak = val
                drawdown = (peak - val) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

        sharpe = 0.0
        if daily_returns and len(daily_returns) > 1:
            avg_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = variance ** 0.5
            if std_dev > 0:
                sharpe = (avg_return * 252 ** 0.5) / (std_dev * 252 ** 0.5)

        return {
            "total_pnl": _to_float(latest.total_pnl),
            "total_pnl_percent": _to_float(latest.total_pnl_percent),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown * 100,
        }
    except Exception as e:
        logger.warning("Could not get performance metrics: %s", e)
        return {}


def _get_trade_statistics(days: Optional[int]) -> dict:
    """Get trade statistics from closed positions."""
    try:
        from flask import current_app
        from src.data.models import Position, PositionStatus

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return {}

        session = current_app.db_session()
        query = session.query(Position).filter(Position.status == PositionStatus.CLOSED)

        if days:
            start_date = date.today() - timedelta(days=days)
            query = query.filter(Position.exit_date >= start_date)

        positions = query.all()

        if not positions:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        wins = [p for p in positions if p.realized_pnl and p.realized_pnl > 0]
        losses = [p for p in positions if p.realized_pnl and p.realized_pnl <= 0]

        win_rate = len(wins) / len(positions) * 100 if positions else 0.0

        avg_win = (
            sum(_to_float(p.realized_pnl) for p in wins) / len(wins) if wins else 0.0
        )
        avg_loss = (
            sum(_to_float(p.realized_pnl) for p in losses) / len(losses)
            if losses
            else 0.0
        )

        total_wins = sum(_to_float(p.realized_pnl) for p in wins)
        total_losses = abs(sum(_to_float(p.realized_pnl) for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "total": len(positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }
    except Exception as e:
        logger.warning("Could not get trade statistics: %s", e)
        return {}


def _get_equity_curve(days: Optional[int]) -> list[dict]:
    """Get equity curve data."""
    try:
        from flask import current_app
        from src.data.models import PortfolioMetric

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return []

        session = current_app.db_session()
        query = session.query(PortfolioMetric)

        if days:
            start_date = date.today() - timedelta(days=days)
            query = query.filter(PortfolioMetric.date >= start_date)

        metrics = query.order_by(PortfolioMetric.date.asc()).all()

        return [
            {
                "date": m.date.isoformat(),
                "value": _to_float(m.total_value),
                "daily_pnl": _to_float(m.daily_pnl),
            }
            for m in metrics
        ]
    except Exception as e:
        logger.warning("Could not get equity curve: %s", e)
        return []


def _get_trades_from_db(
    symbol: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    limit: int,
    offset: int,
) -> tuple[list[dict], int]:
    """Get trades from database with filtering."""
    try:
        from flask import current_app
        from src.data.models import Position, PositionStatus, Stock

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return [], 0

        session = current_app.db_session()
        query = session.query(Position).join(Stock)
        query = query.filter(Position.status == PositionStatus.CLOSED)

        if symbol:
            query = query.filter(Stock.symbol == symbol.upper())
        if start_date:
            query = query.filter(Position.exit_date >= start_date)
        if end_date:
            query = query.filter(Position.exit_date <= end_date)

        total = query.count()
        positions = (
            query.order_by(Position.exit_date.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        trades = []
        for pos in positions:
            trades.append(
                {
                    "id": pos.id,
                    "symbol": pos.stock.symbol,
                    "entry_date": pos.entry_date.isoformat(),
                    "entry_price": _to_float(pos.entry_price),
                    "exit_date": pos.exit_date.isoformat() if pos.exit_date else None,
                    "exit_price": _to_float(pos.exit_price),
                    "shares": _to_float(pos.shares),
                    "exit_reason": pos.exit_reason.value if pos.exit_reason else None,
                    "pnl": _to_float(pos.realized_pnl),
                    "pnl_percent": _to_float(pos.realized_pnl_percent),
                }
            )

        return trades, total
    except Exception as e:
        logger.warning("Could not get trades from database: %s", e)
        return [], 0


def _get_daily_pnl_from_db(days: int) -> list[dict]:
    """Get daily P&L data from database."""
    try:
        from flask import current_app
        from src.data.models import PortfolioMetric

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return []

        session = current_app.db_session()
        start_date = date.today() - timedelta(days=days)

        metrics = (
            session.query(PortfolioMetric)
            .filter(PortfolioMetric.date >= start_date)
            .order_by(PortfolioMetric.date.asc())
            .all()
        )

        return [
            {
                "date": m.date.isoformat(),
                "pnl": _to_float(m.daily_pnl),
                "pnl_percent": _to_float(m.daily_pnl_percent),
            }
            for m in metrics
        ]
    except Exception as e:
        logger.warning("Could not get daily P&L: %s", e)
        return []


def _get_weekly_pnl_from_db(weeks: int) -> list[dict]:
    """Get weekly P&L data from database."""
    try:
        from flask import current_app
        from sqlalchemy import func
        from src.data.models import PortfolioMetric

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return []

        session = current_app.db_session()
        start_date = date.today() - timedelta(weeks=weeks)

        results = (
            session.query(
                func.yearweek(PortfolioMetric.date, 1).label("yearweek"),
                func.min(PortfolioMetric.date).label("week_start"),
                func.max(PortfolioMetric.date).label("week_end"),
                func.sum(PortfolioMetric.daily_pnl).label("weekly_pnl"),
            )
            .filter(PortfolioMetric.date >= start_date)
            .group_by(func.yearweek(PortfolioMetric.date, 1))
            .order_by(func.yearweek(PortfolioMetric.date, 1).asc())
            .all()
        )

        trade_counts = _get_weekly_trade_counts(start_date)

        return [
            {
                "week_start": r.week_start.isoformat(),
                "week_end": r.week_end.isoformat(),
                "pnl": _to_float(r.weekly_pnl),
                "pnl_percent": 0.0,
                "trades_count": trade_counts.get(r.yearweek, 0),
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("Could not get weekly P&L: %s", e)
        return []


def _get_weekly_trade_counts(start_date: date) -> dict[int, int]:
    """Get trade counts grouped by week."""
    try:
        from flask import current_app
        from sqlalchemy import func
        from src.data.models import Position, PositionStatus

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return {}

        session = current_app.db_session()

        results = (
            session.query(
                func.yearweek(Position.exit_date, 1).label("yearweek"),
                func.count(Position.id).label("count"),
            )
            .filter(
                Position.status == PositionStatus.CLOSED,
                Position.exit_date >= start_date,
            )
            .group_by(func.yearweek(Position.exit_date, 1))
            .all()
        )

        return {r.yearweek: r.count for r in results}
    except Exception:
        return {}
