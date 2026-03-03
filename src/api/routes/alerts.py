"""Alerts routes for Leprechaun API."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from src.api.middleware.auth import admin_required, viewer_or_admin
from src.api.middleware.rate_limit import default_rate_limit, admin_rate_limit
from src.api.services import get_services
from src.api.utils import get_pagination_params

logger = logging.getLogger(__name__)

TEST_ALERT_TITLE = "Test Alert"

alerts_bp = Blueprint("alerts", __name__, url_prefix="/api/v1/alerts")


@alerts_bp.route("", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_alerts():
    """Get alert history.

    ---
    tags:
      - Alerts
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: type
        schema:
          type: string
          enum: [trade, halt, resume, error, signal, daily_summary, all]
          default: all
        description: Filter by alert type
      - in: query
        name: channel
        schema:
          type: string
          enum: [discord, push, both, all]
          default: all
        description: Filter by delivery channel
      - in: query
        name: start_date
        schema:
          type: string
          format: date-time
        description: Start date for filtering
      - in: query
        name: end_date
        schema:
          type: string
          format: date-time
        description: End date for filtering
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
        description: Maximum number of alerts to return
      - in: query
        name: offset
        schema:
          type: integer
          default: 0
        description: Number of alerts to skip
    responses:
      200:
        description: List of alerts
        content:
          application/json:
            schema:
              type: object
              properties:
                alerts:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      timestamp:
                        type: string
                        format: date-time
                      alert_type:
                        type: string
                      channel:
                        type: string
                      title:
                        type: string
                      message:
                        type: string
                      sent_successfully:
                        type: boolean
                total:
                  type: integer
                limit:
                  type: integer
                offset:
                  type: integer
      401:
        description: Authentication required
    """
    alert_type = request.args.get("type", "all")
    channel = request.args.get("channel", "all")
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")
    limit, offset = get_pagination_params()

    start_date = None
    end_date = None
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        except ValueError:
            pass
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    alerts, total = _get_alerts_from_db(
        alert_type=alert_type if alert_type != "all" else None,
        channel=channel if channel != "all" else None,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    return jsonify(
        {
            "alerts": alerts,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@alerts_bp.route("/test", methods=["POST"])
@jwt_required()
@admin_required
@admin_rate_limit
def send_test_alert():
    """Send a test alert to verify notification channels.

    ---
    tags:
      - Alerts
    security:
      - bearerAuth: []
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              channel:
                type: string
                enum: [discord, push, both]
                default: both
                description: Channel to send test alert
              message:
                type: string
                description: Custom message for test alert
    responses:
      200:
        description: Test alert sent
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                channels_tested:
                  type: array
                  items:
                    type: string
                results:
                  type: object
                  properties:
                    discord:
                      type: object
                      properties:
                        success:
                          type: boolean
                        error:
                          type: string
                    push:
                      type: object
                      properties:
                        success:
                          type: boolean
                        error:
                          type: string
      400:
        description: Invalid channel specified
      401:
        description: Authentication required
      403:
        description: Admin access required
    """
    services = get_services()
    data = request.get_json() or {}
    channel = data.get("channel", "both")
    custom_message = data.get("message", "This is a test alert from Leprechaun.")

    valid_channels = ("discord", "push", "both")
    if channel not in valid_channels:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": f"Invalid channel. Must be one of: {', '.join(valid_channels)}",
                }
            ),
            400,
        )

    results = {"discord": None, "push": None}
    channels_tested = []

    notification_service = services.notification_service

    if channel in ("discord", "both"):
        channels_tested.append("discord")
        discord_result = _send_discord_test(notification_service, custom_message)
        results["discord"] = discord_result

    if channel in ("push", "both"):
        channels_tested.append("push")
        push_result = _send_push_test(notification_service, custom_message)
        results["push"] = push_result

    _log_test_alert(channels_tested, results)

    return jsonify(
        {
            "message": "Test alert sent",
            "channels_tested": channels_tested,
            "results": results,
        }
    )


def _send_discord_test(notification_service, message: str) -> dict[str, Any]:
    """Send a test alert via Discord."""
    if not notification_service or not notification_service.discord:
        return {"success": False, "error": "Discord not configured"}

    try:
        from src.utils.notifications import AlertMessage, AlertPriority
        from src.data.models import AlertType

        alert = AlertMessage(
            alert_type=AlertType.SIGNAL,
            title=TEST_ALERT_TITLE,
            message=message,
            priority=AlertPriority.NORMAL,
        )

        success = notification_service.discord.send(alert)
        return {"success": success, "error": None if success else "Send failed"}
    except Exception as e:
        logger.error("Discord test failed: %s", e)
        return {"success": False, "error": str(e)}


def _send_push_test(notification_service, message: str) -> dict[str, Any]:
    """Send a test alert via Firebase push notification."""
    if not notification_service or not notification_service.firebase:
        return {"success": False, "error": "Firebase not configured"}

    try:
        from src.utils.notifications import AlertMessage, AlertPriority
        from src.data.models import AlertType

        alert = AlertMessage(
            alert_type=AlertType.SIGNAL,
            title=TEST_ALERT_TITLE,
            message=message,
            priority=AlertPriority.NORMAL,
        )

        success = notification_service.firebase.send(alert, topic="test")
        return {"success": success, "error": None if success else "Send failed"}
    except Exception as e:
        logger.error("Push test failed: %s", e)
        return {"success": False, "error": str(e)}


def _log_test_alert(channels: list[str], results: dict) -> None:
    """Log test alert to database."""
    try:
        from flask import current_app
        from src.data.models import Alert, AlertType, AlertChannel

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return

        session = current_app.db_session()

        channel_map = {
            "discord": AlertChannel.DISCORD,
            "push": AlertChannel.PUSH,
        }

        for ch in channels:
            result = results.get(ch, {})
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                alert_type=AlertType.SIGNAL,
                channel=channel_map.get(ch, AlertChannel.DISCORD),
                title=TEST_ALERT_TITLE,
                message="Test alert sent via API",
                sent_successfully=result.get("success", False),
                error_message=result.get("error"),
            )
            session.add(alert)

        session.commit()
    except Exception as e:
        logger.warning("Could not log test alert: %s", e)


@alerts_bp.route("/settings", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_alert_settings():
    """Get current alert configuration settings.

    ---
    tags:
      - Alerts
    security:
      - bearerAuth: []
    responses:
      200:
        description: Alert settings
        content:
          application/json:
            schema:
              type: object
              properties:
                discord:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                    webhook_configured:
                      type: boolean
                push:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                    firebase_configured:
                      type: boolean
                alert_types:
                  type: object
                  description: Per-type notification preferences
      401:
        description: Authentication required
    """
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
    discord_enabled = os.getenv("DISCORD_ENABLED", "true").lower() == "true"

    firebase_creds = os.getenv("FIREBASE_CREDENTIALS_PATH")
    firebase_enabled = os.getenv("FIREBASE_ENABLED", "false").lower() == "true"

    stored_settings = _get_stored_alert_settings()

    default_alert_types = {
        "trade": {"discord": True, "push": True},
        "halt": {"discord": True, "push": True},
        "resume": {"discord": True, "push": True},
        "error": {"discord": True, "push": False},
        "signal": {"discord": False, "push": False},
        "daily_summary": {"discord": True, "push": False},
    }

    alert_types = stored_settings.get("alert_types", default_alert_types)

    return jsonify(
        {
            "discord": {
                "enabled": discord_enabled and bool(discord_webhook),
                "webhook_configured": bool(discord_webhook),
            },
            "push": {
                "enabled": firebase_enabled and bool(firebase_creds),
                "firebase_configured": bool(firebase_creds),
            },
            "alert_types": alert_types,
        }
    )


@alerts_bp.route("/settings", methods=["PUT"])
@jwt_required()
@admin_required
@admin_rate_limit
def update_alert_settings():
    """Update alert configuration settings.

    ---
    tags:
      - Alerts
    security:
      - bearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              discord:
                type: object
                properties:
                  enabled:
                    type: boolean
              push:
                type: object
                properties:
                  enabled:
                    type: boolean
              alert_types:
                type: object
                description: Per-type notification preferences
    responses:
      200:
        description: Settings updated successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                settings:
                  type: object
      400:
        description: Invalid settings
      401:
        description: Authentication required
      403:
        description: Admin access required
    """
    data = request.get_json()

    if not data:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "Request body required",
                }
            ),
            400,
        )

    _save_alert_settings(data)

    current_settings = {
        "discord": data.get("discord", {}),
        "push": data.get("push", {}),
        "alert_types": data.get("alert_types", {}),
    }

    return jsonify(
        {
            "message": "Alert settings updated",
            "settings": current_settings,
        }
    )


def _get_alerts_from_db(
    alert_type: Optional[str],
    channel: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    limit: int,
    offset: int,
) -> tuple[list[dict], int]:
    """Get alerts from database with filtering."""
    try:
        from flask import current_app
        from src.data.models import Alert, AlertType, AlertChannel

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return [], 0

        session = current_app.db_session()
        query = session.query(Alert)

        if alert_type:
            try:
                type_enum = AlertType(alert_type)
                query = query.filter(Alert.alert_type == type_enum)
            except ValueError:
                pass

        if channel:
            try:
                channel_enum = AlertChannel(channel.upper())
                query = query.filter(Alert.channel == channel_enum)
            except ValueError:
                pass

        if start_date:
            query = query.filter(Alert.timestamp >= start_date)
        if end_date:
            query = query.filter(Alert.timestamp <= end_date)

        total = query.count()
        alerts = (
            query.order_by(Alert.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return [
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "alert_type": a.alert_type.value,
                "channel": a.channel.value,
                "title": a.title,
                "message": a.message,
                "sent_successfully": a.sent_successfully,
                "error_message": a.error_message,
                "metadata": a.alert_metadata,
            }
            for a in alerts
        ], total
    except Exception as e:
        logger.warning("Could not get alerts from database: %s", e)
        return [], 0


def _get_stored_alert_settings() -> dict:
    """Get stored alert settings from Redis or database."""
    try:
        from flask import current_app
        import redis

        redis_url = current_app.config.get("RATELIMIT_STORAGE_URI")
        if redis_url:
            r = redis.from_url(redis_url, decode_responses=True)
            settings_json = r.get("leprechaun:alert_settings")
            if settings_json:
                import json

                return json.loads(settings_json)
    except Exception as e:
        logger.debug("Could not get stored settings from Redis: %s", e)

    return {}


def _save_alert_settings(settings: dict) -> None:
    """Save alert settings to Redis."""
    try:
        from flask import current_app
        import redis
        import json

        redis_url = current_app.config.get("RATELIMIT_STORAGE_URI")
        if redis_url:
            r = redis.from_url(redis_url, decode_responses=True)
            r.set("leprechaun:alert_settings", json.dumps(settings))
    except Exception as e:
        logger.warning("Could not save alert settings: %s", e)


@alerts_bp.route("/summary", methods=["GET"])
@jwt_required()
@viewer_or_admin
@default_rate_limit
def get_alerts_summary():
    """Get summary of recent alerts.

    ---
    tags:
      - Alerts
    security:
      - bearerAuth: []
    parameters:
      - in: query
        name: days
        schema:
          type: integer
          default: 7
        description: Number of days to summarize
    responses:
      200:
        description: Alerts summary
      401:
        description: Authentication required
    """
    days = request.args.get("days", 7, type=int)
    days = 7 if days is None else min(max(1, days), 30)

    summary = _get_alerts_summary(days)

    return jsonify(summary)


def _get_alerts_summary(days: int) -> dict:
    """Get summary statistics of alerts."""
    try:
        from flask import current_app
        from sqlalchemy import func
        from src.data.models import Alert, AlertType

        if not hasattr(current_app, "db_session") or not current_app.db_session:
            return {"total": 0, "by_type": {}, "success_rate": 0.0}

        session = current_app.db_session()
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        total = (
            session.query(func.count(Alert.id))
            .filter(Alert.timestamp >= start_date)
            .scalar()
        )

        by_type_results = (
            session.query(Alert.alert_type, func.count(Alert.id))
            .filter(Alert.timestamp >= start_date)
            .group_by(Alert.alert_type)
            .all()
        )
        by_type = {t.value: c for t, c in by_type_results}

        successful = (
            session.query(func.count(Alert.id))
            .filter(
                Alert.timestamp >= start_date,
                Alert.sent_successfully == True,
            )
            .scalar()
        )

        success_rate = (successful / total * 100) if total > 0 else 0.0

        return {
            "total": total,
            "by_type": by_type,
            "successful": successful,
            "failed": total - successful,
            "success_rate": success_rate,
            "period_days": days,
        }
    except Exception as e:
        logger.warning("Could not get alerts summary: %s", e)
        return {"total": 0, "by_type": {}, "success_rate": 0.0}
