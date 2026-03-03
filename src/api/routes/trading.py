"""Trading routes for Leprechaun API."""

import logging
from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from src.api.middleware.auth import admin_required, viewer_or_admin
from src.api.middleware.rate_limit import trading_rate_limit, admin_rate_limit
from src.api.services import get_services
from src.api.utils import get_pagination_params

logger = logging.getLogger(__name__)

trading_bp = Blueprint("trading", __name__, url_prefix="/api/v1/trading")


def _get_halt_info() -> dict[str, Any]:
    """Get active halt information from database if available."""
    try:
        from flask import current_app
        from src.data.models import TradingHalt

        if hasattr(current_app, "db_session") and current_app.db_session:
            session = current_app.db_session()
            halt = (
                session.query(TradingHalt)
                .filter(TradingHalt.is_active == True)
                .order_by(TradingHalt.start_time.desc())
                .first()
            )
            if halt:
                return {
                    "id": halt.id,
                    "start_time": halt.start_time.isoformat(),
                    "reason": halt.reason,
                    "daily_loss": float(halt.daily_loss) if halt.daily_loss else None,
                    "weekly_loss": float(halt.weekly_loss) if halt.weekly_loss else None,
                    "sandbox_test_required": halt.sandbox_test_required,
                    "sandbox_test_passed": halt.sandbox_test_passed,
                }
    except Exception as e:
        logger.warning("Could not get halt info: %s", e)
    return None


@trading_bp.route("/status", methods=["GET"])
@jwt_required()
@viewer_or_admin
@trading_rate_limit
def get_status():
    """Get current trading bot status.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    responses:
      200:
        description: Trading bot status
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  enum: [running, stopped, halted]
                is_market_open:
                  type: boolean
                active_positions:
                  type: integer
                last_signal_check:
                  type: string
                  format: date-time
                halt_reason:
                  type: string
                  nullable: true
      401:
        description: Authentication required
    """
    services = get_services()
    status = services.get_trading_status()

    halt_info = _get_halt_info()
    if halt_info:
        status["halt_info"] = halt_info

    return jsonify(status)


@trading_bp.route("/start", methods=["POST"])
@jwt_required()
@admin_required
@admin_rate_limit
def start_trading():
    """Start the trading bot.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    responses:
      200:
        description: Trading bot started
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                status:
                  type: string
      400:
        description: Cannot start trading (e.g., already running, market closed)
      401:
        description: Authentication required
      403:
        description: Admin access required
    """
    services = get_services()

    current_status = services.get_trading_status()
    if current_status["status"] == "running":
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "Trading is already running",
                }
            ),
            400,
        )

    if current_status["status"] == "halted":
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": f"Trading is halted: {current_status.get('halt_reason', 'Unknown')}. Use /halt/resume to restart.",
                }
            ),
            400,
        )

    result = services.start_trading()

    if not result.get("success"):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": result.get("message", "Failed to start trading"),
                }
            ),
            400,
        )

    return jsonify(
        {
            "message": result.get("message", "Trading bot started"),
            "status": "running",
            "is_simulated": services.order_executor.is_simulated,
        }
    )


@trading_bp.route("/stop", methods=["POST"])
@jwt_required()
@admin_required
@admin_rate_limit
def stop_trading():
    """Stop the trading bot.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    responses:
      200:
        description: Trading bot stopped
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                status:
                  type: string
      400:
        description: Cannot stop trading (e.g., already stopped)
      401:
        description: Authentication required
      403:
        description: Admin access required
    """
    services = get_services()

    current_status = services.get_trading_status()
    if current_status["status"] == "stopped":
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "Trading is not running",
                }
            ),
            400,
        )

    result = services.stop_trading()

    if not result.get("success"):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": result.get("message", "Failed to stop trading"),
                }
            ),
            400,
        )

    return jsonify(
        {
            "message": result.get("message", "Trading bot stopped"),
            "status": "stopped",
        }
    )


@trading_bp.route("/positions", methods=["GET"])
@jwt_required()
@viewer_or_admin
@trading_rate_limit
def get_positions():
    """Get current open positions.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: status
        schema:
          type: string
          enum: [open, closed, all]
        description: Filter by position status
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
        description: Maximum number of positions to return
      - in: query
        name: offset
        schema:
          type: integer
          default: 0
        description: Number of positions to skip
    responses:
      200:
        description: List of positions
        content:
          application/json:
            schema:
              type: object
              properties:
                positions:
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
    services = get_services()
    status_filter = request.args.get("status", "open")
    limit, offset = get_pagination_params()

    positions = []
    total = 0

    try:
        if status_filter in ("open", "all"):
            live_positions = services.order_executor.get_positions()
            positions.extend(live_positions)
    except Exception as e:
        logger.warning("Could not get live positions: %s", e)

    try:
        from flask import current_app
        from src.data.models import Position, PositionStatus, Stock

        if hasattr(current_app, "db_session") and current_app.db_session:
            session = current_app.db_session()
            query = session.query(Position).join(Stock)

            if status_filter == "open":
                query = query.filter(Position.status == PositionStatus.OPEN)
            elif status_filter == "closed":
                query = query.filter(Position.status == PositionStatus.CLOSED)

            total = query.count()
            db_positions = query.order_by(Position.entry_date.desc()).offset(offset).limit(limit).all()

            for pos in db_positions:
                existing = next(
                    (p for p in positions if p.get("symbol") == pos.stock.symbol), None
                )
                if existing:
                    existing["db_id"] = pos.id
                    existing["entry_signal_id"] = pos.entry_signal_id
                else:
                    positions.append(
                        {
                            "db_id": pos.id,
                            "symbol": pos.stock.symbol,
                            "qty": float(pos.shares),
                            "avg_entry_price": float(pos.entry_price),
                            "entry_date": pos.entry_date.isoformat(),
                            "exit_date": pos.exit_date.isoformat() if pos.exit_date else None,
                            "exit_price": float(pos.exit_price) if pos.exit_price else None,
                            "exit_reason": pos.exit_reason.value if pos.exit_reason else None,
                            "realized_pnl": float(pos.realized_pnl) if pos.realized_pnl else None,
                            "status": pos.status.value,
                        }
                    )
    except Exception as e:
        logger.warning("Could not get database positions: %s", e)

    paginated = positions[offset : offset + limit] if status_filter == "all" else positions

    return jsonify(
        {
            "positions": paginated,
            "total": total or len(positions),
            "limit": limit,
            "offset": offset,
        }
    )


@trading_bp.route("/orders", methods=["GET"])
@jwt_required()
@viewer_or_admin
@trading_rate_limit
def get_orders():
    """Get recent orders.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: status
        schema:
          type: string
          enum: [pending, filled, cancelled, all]
        description: Filter by order status
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
        description: Maximum number of orders to return
      - in: query
        name: offset
        schema:
          type: integer
          default: 0
        description: Number of orders to skip
    responses:
      200:
        description: List of orders
        content:
          application/json:
            schema:
              type: object
              properties:
                orders:
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
    services = get_services()
    status_filter = request.args.get("status", "all")
    limit, offset = get_pagination_params()

    orders = []

    try:
        executor = services.order_executor
        if executor.is_simulated:
            sim_orders = list(executor._simulated.orders.values())
            for order in sim_orders:
                order_dict = order.to_dict()
                if status_filter == "all" or order_dict.get("status") == status_filter:
                    orders.append(order_dict)
        else:
            from alpaca.trading.enums import QueryOrderStatus

            status_map = {
                "pending": QueryOrderStatus.OPEN,
                "filled": QueryOrderStatus.CLOSED,
                "cancelled": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            alpaca_status = status_map.get(status_filter, QueryOrderStatus.ALL)
            alpaca_orders = executor.client.get_orders(status=alpaca_status, limit=limit)

            for order in alpaca_orders:
                orders.append(executor._alpaca_order_to_dict(order))
    except Exception as e:
        logger.warning("Could not get orders: %s", e)

    orders.sort(key=lambda o: o.get("created_at", ""), reverse=True)
    paginated = orders[offset : offset + limit]

    return jsonify(
        {
            "orders": paginated,
            "total": len(orders),
            "limit": limit,
            "offset": offset,
        }
    )


@trading_bp.route("/halt/resume", methods=["POST"])
@jwt_required()
@admin_required
@admin_rate_limit
def resume_from_halt():
    """Resume trading after a halt.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              skip_sandbox_test:
                type: boolean
                default: false
                description: Skip sandbox testing requirement (not recommended)
              reason:
                type: string
                description: Reason for resuming trading
    responses:
      200:
        description: Trading resumed
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                status:
                  type: string
                sandbox_test_passed:
                  type: boolean
      400:
        description: Cannot resume (e.g., not halted, sandbox test failed)
      401:
        description: Authentication required
      403:
        description: Admin access required
    """
    services = get_services()
    data = request.get_json() or {}
    skip_sandbox = data.get("skip_sandbox_test", False)
    resume_reason = data.get("reason", "")

    is_halted, _ = services.orchestrator.is_halted()

    if not is_halted:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "Trading is not halted",
                }
            ),
            400,
        )

    sandbox_test_passed = None
    user_identity = get_jwt_identity()

    if not skip_sandbox:
        sandbox_test_passed = _run_sandbox_test(services)
        if not sandbox_test_passed:
            return (
                jsonify(
                    {
                        "error": "bad_request",
                        "message": "Sandbox test failed. Use skip_sandbox_test=true to bypass (not recommended).",
                        "sandbox_test_passed": False,
                    }
                ),
                400,
            )

    try:
        services.orchestrator.resume_trading()

        _update_halt_record(
            resumed_by=user_identity,
            sandbox_test_passed=sandbox_test_passed,
        )

        if services.notification_service:
            services.notification_service.send_resume_alert()

        logger.info(
            "Trading resumed by %s. Reason: %s",
            user_identity,
            resume_reason or "Not provided",
        )

        return jsonify(
            {
                "message": "Trading resumed successfully",
                "status": "running",
                "sandbox_test_passed": sandbox_test_passed,
                "resumed_by": user_identity,
            }
        )
    except Exception as e:
        logger.error("Failed to resume trading: %s", e)
        return (
            jsonify(
                {
                    "error": "internal_error",
                    "message": f"Failed to resume trading: {str(e)}",
                }
            ),
            500,
        )


def _run_sandbox_test(services) -> bool:
    """Run a simple sandbox test to verify trading is working."""
    try:
        account = services.order_executor.get_account()
        if account.get("equity", 0) <= 0:
            return False

        _ = services.order_executor.get_positions()
        return True
    except Exception as e:
        logger.error("Sandbox test failed: %s", e)
        return False


def _update_halt_record(resumed_by: str, sandbox_test_passed: bool) -> None:
    """Update the halt record in database."""
    try:
        from flask import current_app
        from src.data.models import TradingHalt

        if hasattr(current_app, "db_session") and current_app.db_session:
            session = current_app.db_session()
            halt = (
                session.query(TradingHalt)
                .filter(TradingHalt.is_active == True)
                .order_by(TradingHalt.start_time.desc())
                .first()
            )
            if halt:
                halt.is_active = False
                halt.end_time = datetime.now(timezone.utc)
                halt.sandbox_test_passed = sandbox_test_passed
                halt.resumed_by = resumed_by
                session.commit()
    except Exception as e:
        logger.warning("Could not update halt record: %s", e)


@trading_bp.route("/account", methods=["GET"])
@jwt_required()
@viewer_or_admin
@trading_rate_limit
def get_account():
    """Get trading account information.

    ---
    tags:
      - Trading
    security:
      - bearerAuth: []
    responses:
      200:
        description: Account information
      401:
        description: Authentication required
    """
    services = get_services()

    try:
        account = services.order_executor.get_account()
        account["is_simulated"] = services.order_executor.is_simulated
        return jsonify(account)
    except Exception as e:
        logger.error("Could not get account info: %s", e)
        return (
            jsonify(
                {
                    "error": "service_unavailable",
                    "message": "Could not retrieve account information",
                }
            ),
            503,
        )
