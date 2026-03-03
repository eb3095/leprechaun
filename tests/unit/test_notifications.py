"""Unit tests for notifications module."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.notifications import (
    AlertMessage,
    AlertType,
    AlertPriority,
    RateLimiter,
    DiscordNotifier,
    FirebaseNotifier,
    NotificationService,
)


class TestAlertMessage:
    def test_create_basic_alert(self):
        alert = AlertMessage(
            alert_type=AlertType.TRADE,
            title="Test Trade",
            message="Test message",
        )
        assert alert.alert_type == AlertType.TRADE
        assert alert.title == "Test Trade"
        assert alert.priority == AlertPriority.NORMAL

    def test_create_alert_with_data(self):
        alert = AlertMessage(
            alert_type=AlertType.HALT,
            title="Trading Halted",
            message="Daily loss limit exceeded",
            priority=AlertPriority.CRITICAL,
            data={"daily_loss": "2.5%"},
        )
        assert alert.data["daily_loss"] == "2.5%"
        assert alert.priority == AlertPriority.CRITICAL


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_calls=3, period_seconds=60)
        assert limiter.can_send() is True
        limiter.record()
        assert limiter.can_send() is True
        limiter.record()
        assert limiter.can_send() is True
        limiter.record()
        assert limiter.can_send() is False

    def test_resets_after_period(self):
        limiter = RateLimiter(max_calls=1, period_seconds=1)
        limiter.record()
        assert limiter.can_send() is False
        time.sleep(1.1)
        assert limiter.can_send() is True


class TestDiscordNotifier:
    @pytest.fixture
    def notifier(self):
        return DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=True,
            rate_limit=10,
            rate_period=60,
            max_retries=2,
        )

    def test_disabled_notifier(self, notifier):
        notifier.enabled = False
        alert = AlertMessage(AlertType.TRADE, "Test", "Test message")
        assert notifier.send(alert) is False

    def test_rate_limited(self, notifier):
        notifier.rate_limiter.max_calls = 0
        alert = AlertMessage(AlertType.TRADE, "Test", "Test message")
        assert notifier.send(alert) is False

    @patch("src.utils.notifications.requests.post")
    def test_successful_send(self, mock_post, notifier):
        mock_post.return_value.raise_for_status = MagicMock()
        alert = AlertMessage(
            alert_type=AlertType.TRADE,
            title="Buy AAPL",
            message="Executed buy order",
            symbol="AAPL",
        )
        result = notifier.send(alert)
        assert result is True
        mock_post.assert_called_once()

    @patch("src.utils.notifications.requests.post")
    def test_retry_on_failure(self, mock_post, notifier):
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        alert = AlertMessage(AlertType.ERROR, "Error", "System error")

        with patch("time.sleep"):
            result = notifier.send(alert)

        assert result is False
        assert mock_post.call_count == 2

    def test_build_embed_with_fields(self, notifier):
        alert = AlertMessage(
            alert_type=AlertType.TRADE,
            title="Trade Alert",
            message="Buy executed",
            priority=AlertPriority.HIGH,
            symbol="TSLA",
            data={"shares": "10", "price": "$200.00"},
        )
        embed = notifier._build_embed(alert)

        assert embed["title"] == "Trade Alert"
        assert embed["description"] == "Buy executed"
        assert embed["color"] == DiscordNotifier.PRIORITY_COLORS[AlertPriority.HIGH]
        assert len(embed["fields"]) == 3

    def test_build_embed_priority_colors(self, notifier):
        for priority in AlertPriority:
            alert = AlertMessage(AlertType.TRADE, "Test", "Test", priority=priority)
            embed = notifier._build_embed(alert)
            assert embed["color"] == DiscordNotifier.PRIORITY_COLORS[priority]


class TestFirebaseNotifier:
    def test_disabled_notifier(self):
        notifier = FirebaseNotifier(enabled=False)
        alert = AlertMessage(AlertType.TRADE, "Test", "Test message")
        assert notifier.send(alert) is False

    def test_not_initialized_without_credentials(self):
        notifier = FirebaseNotifier(enabled=True, credentials_path=None)
        assert notifier._initialize() is False

    def test_already_initialized_returns_true(self):
        notifier = FirebaseNotifier(enabled=True, credentials_path="/path/to/creds.json")
        notifier._initialized = True
        assert notifier._initialize() is True


class TestNotificationService:
    @pytest.fixture
    def service(self):
        discord = MagicMock(spec=DiscordNotifier)
        discord.send.return_value = True
        firebase = MagicMock(spec=FirebaseNotifier)
        firebase.send.return_value = True
        return NotificationService(discord=discord, firebase=firebase)

    def test_send_trade_alert(self, service):
        service.send_trade_alert(
            symbol="AAPL",
            action="buy",
            shares=10.0,
            price=150.0,
            reason="Signal triggered",
        )
        service.discord.send.assert_called_once()
        service.firebase.send.assert_called_once()
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.TRADE
        assert call_args.symbol == "AAPL"

    def test_send_halt_alert(self, service):
        service.send_halt_alert(
            reason="Daily loss limit exceeded",
            daily_loss=2.5,
            weekly_loss=3.0,
        )
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.HALT
        assert call_args.priority == AlertPriority.CRITICAL

    def test_send_resume_alert(self, service):
        service.send_resume_alert()
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.RESUME
        assert call_args.priority == AlertPriority.HIGH

    def test_send_error_alert(self, service):
        service.send_error_alert("Database connection failed", {"retry_count": 3})
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.ERROR
        assert call_args.data["retry_count"] == 3

    def test_send_signal_alert(self, service):
        service.send_signal_alert(
            symbol="GME",
            signal_type="buy",
            confidence="high",
            manipulation_score=0.75,
        )
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.SIGNAL
        assert call_args.symbol == "GME"

    def test_send_daily_summary(self, service):
        service.send_daily_summary(
            pnl=500.0,
            pnl_percent=0.5,
            trades_count=5,
            win_count=3,
            portfolio_value=100500.0,
        )
        call_args = service.discord.send.call_args[0][0]
        assert call_args.alert_type == AlertType.DAILY_SUMMARY
        assert "500" in call_args.message

    def test_works_without_firebase(self):
        discord = MagicMock(spec=DiscordNotifier)
        service = NotificationService(discord=discord, firebase=None)
        service.send_trade_alert("AAPL", "buy", 10.0, 150.0)
        discord.send.assert_called_once()

    def test_works_without_discord(self):
        firebase = MagicMock(spec=FirebaseNotifier)
        service = NotificationService(discord=None, firebase=firebase)
        service.send_trade_alert("AAPL", "buy", 10.0, 150.0)
        firebase.send.assert_called_once()
