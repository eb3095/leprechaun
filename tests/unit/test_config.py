"""Unit tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.utils.config import (
    Config,
    ConfigurationError,
    DatabaseConfig,
    RedisConfig,
    AlpacaConfig,
    TradingConfig,
    load_config,
    validate_config,
)


@pytest.fixture
def mock_env():
    """Provide a complete mock environment."""
    return {
        "ENV": "development",
        "DEBUG": "false",
        "SECRET_KEY": "test-secret-key",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_NAME": "testdb",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "ALPACA_API_KEY": "test-alpaca-key",
        "ALPACA_API_SECRET": "test-alpaca-secret",
        "ALPACA_PAPER": "true",
        "REDDIT_CLIENT_ID": "test-reddit-id",
        "REDDIT_CLIENT_SECRET": "test-reddit-secret",
        "REDDIT_USER_AGENT": "TestAgent/1.0",
        "FINNHUB_API_KEY": "test-finnhub-key",
        "STOCKTWITS_API_KEY": "test-stocktwits-key",
        "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test",
        "DISCORD_ENABLED": "true",
    }


class TestDatabaseConfig:
    def test_url_generation(self):
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        assert config.url == "mysql+pymysql://testuser:testpass@localhost:3306/testdb"

    def test_url_encodes_special_characters(self):
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            name="testdb",
            user="user@domain",
            password="p@ss:word#123",
        )
        assert "user%40domain" in config.url
        assert "p%40ss%3Aword%23123" in config.url

    def test_immutable(self):
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        with pytest.raises(AttributeError):
            config.host = "newhost"

    def test_repr_hides_password(self):
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            name="testdb",
            user="testuser",
            password="supersecret",
        )
        repr_str = repr(config)
        assert "supersecret" not in repr_str
        assert "***" in repr_str
        assert "localhost" in repr_str


class TestRedisConfig:
    def test_url_without_password(self):
        config = RedisConfig(host="localhost", port=6379, db=0)
        assert config.url == "redis://localhost:6379/0"

    def test_url_with_password(self):
        config = RedisConfig(host="localhost", port=6379, db=0, password="secret")
        assert config.url == "redis://:secret@localhost:6379/0"

    def test_url_encodes_special_password(self):
        config = RedisConfig(host="localhost", port=6379, db=0, password="p@ss:word")
        assert "p%40ss%3Aword" in config.url

    def test_repr_hides_password(self):
        config = RedisConfig(host="localhost", port=6379, db=0, password="secret")
        repr_str = repr(config)
        assert "secret" not in repr_str
        assert "***" in repr_str


class TestAlpacaConfig:
    def test_paper_base_url(self):
        config = AlpacaConfig(api_key="key", api_secret="secret", paper=True)
        assert config.base_url == "https://paper-api.alpaca.markets"

    def test_live_base_url(self):
        config = AlpacaConfig(api_key="key", api_secret="secret", paper=False)
        assert config.base_url == "https://api.alpaca.markets"

    def test_repr_hides_credentials(self):
        config = AlpacaConfig(api_key="my-api-key", api_secret="my-secret", paper=True)
        repr_str = repr(config)
        assert "my-api-key" not in repr_str
        assert "my-secret" not in repr_str
        assert "***" in repr_str


class TestTradingConfig:
    def test_default_values(self):
        config = TradingConfig()
        assert config.profit_target_percent == 2.5
        assert config.stop_loss_percent == 1.25
        assert config.position_risk_percent == 1.0
        assert config.daily_loss_limit_percent == 2.0
        assert config.weekly_loss_limit_percent == 5.0
        assert config.rsi_oversold == 35
        assert config.rsi_normalized == 50
        assert config.manipulation_threshold == 0.5
        assert config.sentiment_negative_threshold == -0.4

    def test_risk_reward_ratio(self):
        config = TradingConfig()
        ratio = config.profit_target_percent / config.stop_loss_percent
        assert ratio == 2.0


class TestLoadConfig:
    @patch("src.utils.config.load_dotenv")
    def test_load_complete_config(self, mock_dotenv, mock_env):
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            assert config.env == "development"
            assert config.debug is False
            assert config.database.host == "localhost"
            assert config.alpaca.paper is True
            assert config.trading.profit_target_percent == 2.5

    @patch("src.utils.config.load_dotenv")
    def test_missing_required_field(self, mock_dotenv, mock_env):
        del mock_env["SECRET_KEY"]
        with patch.dict(os.environ, mock_env, clear=True):
            with pytest.raises(ConfigurationError) as exc:
                load_config()
            assert "SECRET_KEY" in str(exc.value)

    @patch("src.utils.config.load_dotenv")
    def test_custom_trading_params(self, mock_dotenv, mock_env):
        mock_env["PROFIT_TARGET_PERCENT"] = "3.0"
        mock_env["STOP_LOSS_PERCENT"] = "1.5"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            assert config.trading.profit_target_percent == 3.0
            assert config.trading.stop_loss_percent == 1.5

    @patch("src.utils.config.load_dotenv")
    def test_debug_mode(self, mock_dotenv, mock_env):
        mock_env["DEBUG"] = "true"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            assert config.debug is True

    @patch("src.utils.config.load_dotenv")
    def test_reddit_subreddits_parsing(self, mock_dotenv, mock_env):
        mock_env["REDDIT_SUBREDDITS"] = "stocks, investing, options"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            assert config.reddit.subreddits == ("stocks", "investing", "options")


class TestValidateConfig:
    @patch("src.utils.config.load_dotenv")
    def test_paper_in_production_warning(self, mock_dotenv, mock_env):
        mock_env["ENV"] = "production"
        mock_env["ALPACA_PAPER"] = "true"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            warnings = validate_config(config)
            assert any("paper trading" in w.lower() for w in warnings)

    @patch("src.utils.config.load_dotenv")
    def test_disabled_notifications_warning(self, mock_dotenv, mock_env):
        mock_env["DISCORD_ENABLED"] = "false"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            warnings = validate_config(config)
            assert any("discord" in w.lower() for w in warnings)

    @patch("src.utils.config.load_dotenv")
    def test_unfavorable_risk_reward_warning(self, mock_dotenv, mock_env):
        mock_env["PROFIT_TARGET_PERCENT"] = "1.0"
        mock_env["STOP_LOSS_PERCENT"] = "2.0"
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            warnings = validate_config(config)
            assert any("risk/reward" in w.lower() for w in warnings)
