"""Configuration management for Leprechaun trading bot."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import quote as url_encode

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str
    pool_size: int = 5
    pool_recycle: int = 3600

    @property
    def url(self) -> str:
        user = url_encode(self.user, safe="")
        password = url_encode(self.password, safe="")
        return f"mysql+pymysql://{user}:{password}@{self.host}:{self.port}/{self.name}"

    def __repr__(self) -> str:
        return f"DatabaseConfig(host={self.host!r}, port={self.port}, name={self.name!r}, user={self.user!r}, password='***')"


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection configuration."""

    host: str
    port: int
    db: int = 0
    password: Optional[str] = None

    @property
    def url(self) -> str:
        if self.password:
            password = url_encode(self.password, safe="")
            auth = f":{password}@"
        else:
            auth = ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    def __repr__(self) -> str:
        pwd = "'***'" if self.password else "None"
        return f"RedisConfig(host={self.host!r}, port={self.port}, db={self.db}, password={pwd})"


@dataclass(frozen=True)
class AlpacaConfig:
    """Alpaca trading API configuration."""

    api_key: str
    api_secret: str
    paper: bool = True

    @property
    def base_url(self) -> str:
        return "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

    def __repr__(self) -> str:
        return f"AlpacaConfig(api_key='***', api_secret='***', paper={self.paper})"


@dataclass(frozen=True)
class RedditConfig:
    """Reddit API configuration."""

    client_id: str
    client_secret: str
    user_agent: str
    subreddits: tuple[str, ...] = ("wallstreetbets", "stocks", "investing")

    def __repr__(self) -> str:
        return f"RedditConfig(client_id='***', client_secret='***', user_agent={self.user_agent!r}, subreddits={self.subreddits!r})"


@dataclass(frozen=True)
class FinnhubConfig:
    """Finnhub API configuration."""

    api_key: str

    def __repr__(self) -> str:
        return "FinnhubConfig(api_key='***')"


@dataclass(frozen=True)
class StockTwitsConfig:
    """StockTwits RapidAPI configuration."""

    api_key: str
    api_host: str = "stocktwits.p.rapidapi.com"

    def __repr__(self) -> str:
        return f"StockTwitsConfig(api_key='***', api_host={self.api_host!r})"


@dataclass(frozen=True)
class DiscordConfig:
    """Discord webhook configuration."""

    webhook_url: str
    enabled: bool = True

    def __repr__(self) -> str:
        return f"DiscordConfig(webhook_url='***', enabled={self.enabled})"


@dataclass(frozen=True)
class FirebaseConfig:
    """Firebase push notification configuration."""

    credentials_path: Optional[str] = None
    enabled: bool = False


@dataclass(frozen=True)
class TradingConfig:
    """Trading parameters and risk management settings."""

    profit_target_percent: float = 2.5
    stop_loss_percent: float = 1.25
    position_risk_percent: float = 1.0
    daily_loss_limit_percent: float = 2.0
    weekly_loss_limit_percent: float = 5.0
    rsi_oversold: int = 35
    rsi_normalized: int = 50
    manipulation_threshold: float = 0.5
    sentiment_negative_threshold: float = -0.4


@dataclass
class BacktestConfig:
    """Configuration for backtesting mode.

    Controls how sentiment data is sourced during backtests:
    - Archived sentiment: Real historical data (most accurate)
    - Synthetic sentiment: Generated from price/volume (approximation)
    """

    use_synthetic_sentiment: bool = True
    synthetic_noise_level: float = 0.15
    min_archive_coverage: float = 0.8
    technicals_only: bool = False


@dataclass
class Config:
    """Main application configuration."""

    env: str
    debug: bool
    secret_key: str

    database: DatabaseConfig
    redis: RedisConfig
    alpaca: AlpacaConfig
    reddit: RedditConfig
    finnhub: FinnhubConfig
    stocktwits: StockTwitsConfig
    discord: DiscordConfig
    firebase: FirebaseConfig
    trading: TradingConfig

    log_level: str = "INFO"
    log_format: str = "json"


def _get_required(key: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.getenv(key)
    if value is None:
        raise ConfigurationError(f"Missing required environment variable: {key}")
    return value


def _get_optional(key: str, default: str = "") -> str:
    """Get an optional environment variable with a default."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_int(key: str, default: int) -> int:
    """Get an integer environment variable."""
    return int(os.getenv(key, str(default)))


def _get_float(key: str, default: float) -> float:
    """Get a float environment variable."""
    return float(os.getenv(key, str(default)))


def _load_database_config() -> DatabaseConfig:
    return DatabaseConfig(
        host=_get_required("DB_HOST"),
        port=_get_int("DB_PORT", 3306),
        name=_get_required("DB_NAME"),
        user=_get_required("DB_USER"),
        password=_get_required("DB_PASSWORD"),
        pool_size=_get_int("DB_POOL_SIZE", 5),
        pool_recycle=_get_int("DB_POOL_RECYCLE", 3600),
    )


def _load_redis_config() -> RedisConfig:
    return RedisConfig(
        host=_get_required("REDIS_HOST"),
        port=_get_int("REDIS_PORT", 6379),
        db=_get_int("REDIS_DB", 0),
        password=_get_optional("REDIS_PASSWORD") or None,
    )


def _load_alpaca_config() -> AlpacaConfig:
    return AlpacaConfig(
        api_key=_get_required("ALPACA_API_KEY"),
        api_secret=_get_required("ALPACA_API_SECRET"),
        paper=_get_bool("ALPACA_PAPER", True),
    )


def _load_reddit_config() -> RedditConfig:
    subreddits_str = _get_optional("REDDIT_SUBREDDITS", "wallstreetbets,stocks,investing")
    subreddits = tuple(s.strip() for s in subreddits_str.split(",") if s.strip())
    return RedditConfig(
        client_id=_get_required("REDDIT_CLIENT_ID"),
        client_secret=_get_required("REDDIT_CLIENT_SECRET"),
        user_agent=_get_optional("REDDIT_USER_AGENT", "Leprechaun/1.0"),
        subreddits=subreddits,
    )


def _load_finnhub_config() -> FinnhubConfig:
    return FinnhubConfig(api_key=_get_required("FINNHUB_API_KEY"))


def _load_stocktwits_config() -> StockTwitsConfig:
    return StockTwitsConfig(
        api_key=_get_required("STOCKTWITS_API_KEY"),
        api_host=_get_optional("STOCKTWITS_API_HOST", "stocktwits.p.rapidapi.com"),
    )


def _load_discord_config() -> DiscordConfig:
    return DiscordConfig(
        webhook_url=_get_required("DISCORD_WEBHOOK_URL"),
        enabled=_get_bool("DISCORD_ENABLED", True),
    )


def _load_firebase_config() -> FirebaseConfig:
    creds_path = _get_optional("FIREBASE_CREDENTIALS_PATH")
    return FirebaseConfig(
        credentials_path=creds_path or None,
        enabled=_get_bool("FIREBASE_ENABLED", False),
    )


def _load_trading_config() -> TradingConfig:
    return TradingConfig(
        profit_target_percent=_get_float("PROFIT_TARGET_PERCENT", 2.5),
        stop_loss_percent=_get_float("STOP_LOSS_PERCENT", 1.25),
        position_risk_percent=_get_float("POSITION_RISK_PERCENT", 1.0),
        daily_loss_limit_percent=_get_float("DAILY_LOSS_LIMIT_PERCENT", 2.0),
        weekly_loss_limit_percent=_get_float("WEEKLY_LOSS_LIMIT_PERCENT", 5.0),
        rsi_oversold=_get_int("RSI_OVERSOLD", 35),
        rsi_normalized=_get_int("RSI_NORMALIZED", 50),
        manipulation_threshold=_get_float("MANIPULATION_THRESHOLD", 0.5),
        sentiment_negative_threshold=_get_float("SENTIMENT_NEGATIVE_THRESHOLD", -0.4),
    )


def load_config(env_file: Optional[str] = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_file: Path to .env file. If None, searches for .env in current
                  directory and parent directories.

    Returns:
        Fully populated Config object.

    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Config(
        env=_get_optional("ENV", "development"),
        debug=_get_bool("DEBUG", False),
        secret_key=_get_required("SECRET_KEY"),
        log_level=_get_optional("LOG_LEVEL", "INFO"),
        log_format=_get_optional("LOG_FORMAT", "json"),
        database=_load_database_config(),
        redis=_load_redis_config(),
        alpaca=_load_alpaca_config(),
        reddit=_load_reddit_config(),
        finnhub=_load_finnhub_config(),
        stocktwits=_load_stocktwits_config(),
        discord=_load_discord_config(),
        firebase=_load_firebase_config(),
        trading=_load_trading_config(),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the singleton configuration instance.

    Uses lru_cache to ensure the configuration is only loaded once.
    """
    return load_config()


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of warnings.

    Required fields are validated at load time and raise ConfigurationError.
    This function checks for potential issues that won't prevent startup.
    """
    warnings = []

    if config.alpaca.paper and config.env == "production":
        warnings.append("Using paper trading in production environment")

    if not config.discord.enabled:
        warnings.append("Discord notifications are disabled")

    if not config.firebase.enabled:
        warnings.append("Firebase push notifications are disabled")

    if config.trading.stop_loss_percent >= config.trading.profit_target_percent:
        warnings.append(
            f"Stop loss ({config.trading.stop_loss_percent}%) >= profit target "
            f"({config.trading.profit_target_percent}%): unfavorable risk/reward"
        )

    return warnings
