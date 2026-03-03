"""Notification utilities for Leprechaun trading bot."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import requests

from src.data.models import AlertType
from src.utils.logging import get_logger
from src.utils.metrics import record_alert

logger = get_logger("notifications")


class AlertPriority(Enum):
    """Priority levels for alerts."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertMessage:
    """Structured alert message."""

    alert_type: AlertType
    title: str
    message: str
    priority: AlertPriority = AlertPriority.NORMAL
    symbol: Optional[str] = None
    data: Optional[dict[str, Any]] = None


class RateLimiter:
    """Simple rate limiter to prevent notification spam."""

    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list[float] = []

    def can_send(self) -> bool:
        """Check if a message can be sent within rate limits."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        return len(self.calls) < self.max_calls

    def record(self) -> None:
        """Record that a message was sent."""
        self.calls.append(time.time())


class DiscordNotifier:
    """Send notifications via Discord webhooks."""

    PRIORITY_COLORS = {
        AlertPriority.LOW: 0x808080,
        AlertPriority.NORMAL: 0x3498DB,
        AlertPriority.HIGH: 0xF1C40F,
        AlertPriority.CRITICAL: 0xE74C3C,
    }

    def __init__(
        self,
        webhook_url: str,
        enabled: bool = True,
        rate_limit: int = 30,
        rate_period: int = 60,
        max_retries: int = 3,
    ):
        self.webhook_url = webhook_url
        self.enabled = enabled
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rate_limit, rate_period)

    def send(self, alert: AlertMessage) -> bool:
        """Send an alert to Discord.

        Args:
            alert: The alert message to send.

        Returns:
            True if the message was sent successfully.
        """
        if not self.enabled:
            logger.debug("Discord notifications disabled, skipping")
            return False

        if not self.rate_limiter.can_send():
            logger.warning("Discord rate limit exceeded, message dropped")
            return False

        embed = self._build_embed(alert)
        payload = {"embeds": [embed]}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                self.rate_limiter.record()
                record_alert("discord", alert.alert_type.value)
                logger.info(
                    "Discord notification sent",
                    extra={
                        "extra_fields": {
                            "alert_type": alert.alert_type.value,
                            "title": alert.title,
                        }
                    },
                )
                return True

            except requests.exceptions.RequestException as e:
                wait_time = 2**attempt
                logger.warning(
                    "Discord webhook failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    type(e).__name__,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        logger.error("Discord notification failed after all retries")
        return False

    def _build_embed(self, alert: AlertMessage) -> dict[str, Any]:
        """Build a Discord embed from an alert message."""
        embed: dict[str, Any] = {
            "title": alert.title,
            "description": alert.message,
            "color": self.PRIORITY_COLORS[alert.priority],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"Leprechaun • {alert.alert_type.value}"},
        }

        fields = []
        if alert.symbol:
            fields.append({"name": "Symbol", "value": alert.symbol, "inline": True})

        if alert.data:
            for key, value in alert.data.items():
                if len(fields) >= 10:
                    break
                fields.append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value),
                        "inline": True,
                    }
                )

        if fields:
            embed["fields"] = fields

        return embed


class FirebaseNotifier:
    """Send push notifications via Firebase Cloud Messaging.

    This is a placeholder implementation. Full implementation requires
    firebase-admin SDK setup with service account credentials.
    """

    def __init__(self, credentials_path: Optional[str] = None, enabled: bool = False):
        self.enabled = enabled
        self.credentials_path = credentials_path
        self._initialized = False

    def _initialize(self) -> bool:
        """Initialize Firebase Admin SDK if not already done."""
        if self._initialized:
            return True

        if not self.enabled or not self.credentials_path:
            return False

        try:
            import firebase_admin
            from firebase_admin import credentials

            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_path)
                firebase_admin.initialize_app(cred)

            self._initialized = True
            logger.info("Firebase Admin SDK initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize Firebase: %s", type(e).__name__)
            return False

    def send(
        self,
        alert: AlertMessage,
        topic: Optional[str] = None,
        token: Optional[str] = None,
    ) -> bool:
        """Send a push notification via Firebase.

        Args:
            alert: The alert message to send.
            topic: Firebase topic to send to (e.g., 'all', 'trading').
            token: Specific device token to send to.

        Returns:
            True if the message was sent successfully.
        """
        if not self.enabled:
            logger.debug("Firebase notifications disabled, skipping")
            return False

        if not self._initialize():
            return False

        if not topic and not token:
            topic = "all"

        try:
            from firebase_admin import messaging

            notification = messaging.Notification(
                title=alert.title,
                body=alert.message,
            )

            data = {
                "alert_type": alert.alert_type.value,
                "priority": alert.priority.value,
            }
            if alert.symbol:
                data["symbol"] = alert.symbol
            if alert.data:
                for k, v in alert.data.items():
                    data[k] = str(v)

            if token:
                message = messaging.Message(
                    notification=notification,
                    data=data,
                    token=token,
                )
            else:
                message = messaging.Message(
                    notification=notification,
                    data=data,
                    topic=topic,
                )

            response = messaging.send(message)
            record_alert("push", alert.alert_type.value)
            logger.info(
                "Firebase notification sent",
                extra={
                    "extra_fields": {
                        "message_id": response,
                        "alert_type": alert.alert_type.value,
                    }
                },
            )
            return True

        except Exception as e:
            logger.error("Firebase notification failed: %s", type(e).__name__)
            return False


class NotificationService:
    """Unified notification service for all channels."""

    def __init__(
        self,
        discord: Optional[DiscordNotifier] = None,
        firebase: Optional[FirebaseNotifier] = None,
    ):
        self.discord = discord
        self.firebase = firebase

    def send_trade_alert(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        reason: Optional[str] = None,
    ) -> None:
        """Send a trade execution alert."""
        alert = AlertMessage(
            alert_type=AlertType.TRADE,
            title=f"{action.upper()} {symbol}",
            message=f"Executed {action} order for {shares:.2f} shares at ${price:.2f}",
            priority=AlertPriority.NORMAL,
            symbol=symbol,
            data={
                "shares": f"{shares:.2f}",
                "price": f"${price:.2f}",
                "reason": reason or "Signal",
            },
        )
        self._send_to_all(alert)

    def send_halt_alert(
        self,
        reason: str,
        daily_loss: Optional[float] = None,
        weekly_loss: Optional[float] = None,
    ) -> None:
        """Send a trading halt alert."""
        data: dict[str, Any] = {}
        if daily_loss is not None:
            data["daily_loss"] = f"{daily_loss:.2f}%"
        if weekly_loss is not None:
            data["weekly_loss"] = f"{weekly_loss:.2f}%"

        alert = AlertMessage(
            alert_type=AlertType.HALT,
            title="Trading Halted",
            message=reason,
            priority=AlertPriority.CRITICAL,
            data=data,
        )
        self._send_to_all(alert)

    def send_resume_alert(self) -> None:
        """Send a trading resumed alert."""
        alert = AlertMessage(
            alert_type=AlertType.RESUME,
            title="Trading Resumed",
            message="Trading has been resumed and is now active.",
            priority=AlertPriority.HIGH,
        )
        self._send_to_all(alert)

    def send_error_alert(
        self, error: str, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Send an error alert."""
        alert = AlertMessage(
            alert_type=AlertType.ERROR,
            title="System Error",
            message=error,
            priority=AlertPriority.HIGH,
            data=context,
        )
        self._send_to_all(alert)

    def send_signal_alert(
        self,
        symbol: str,
        signal_type: str,
        confidence: str,
        manipulation_score: Optional[float] = None,
    ) -> None:
        """Send a trading signal alert."""
        data: dict[str, Any] = {"signal": signal_type, "confidence": confidence}
        if manipulation_score is not None:
            data["manipulation_score"] = f"{manipulation_score:.2f}"

        alert = AlertMessage(
            alert_type=AlertType.SIGNAL,
            title=f"Signal: {signal_type.upper()} {symbol}",
            message=f"Generated {confidence} confidence {signal_type} signal",
            priority=(
                AlertPriority.NORMAL if confidence == "low" else AlertPriority.HIGH
            ),
            symbol=symbol,
            data=data,
        )
        self._send_to_all(alert)

    def send_daily_summary(
        self,
        pnl: float,
        pnl_percent: float,
        trades_count: int,
        win_count: int,
        portfolio_value: float,
    ) -> None:
        """Send a daily summary alert."""
        win_rate = (win_count / trades_count * 100) if trades_count > 0 else 0
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        alert = AlertMessage(
            alert_type=AlertType.DAILY_SUMMARY,
            title=f"Daily Summary {pnl_emoji}",
            message=f"P&L: ${pnl:+.2f} ({pnl_percent:+.2f}%)",
            priority=AlertPriority.NORMAL,
            data={
                "portfolio_value": f"${portfolio_value:,.2f}",
                "trades": str(trades_count),
                "win_rate": f"{win_rate:.1f}%",
            },
        )
        self._send_to_all(alert)

    def _send_to_all(self, alert: AlertMessage) -> None:
        """Send an alert to all configured channels."""
        if self.discord:
            self.discord.send(alert)
        if self.firebase:
            self.firebase.send(alert)
