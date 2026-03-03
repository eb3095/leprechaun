"""Synthetic sentiment generator for backtesting.

Generates synthetic sentiment from price/volume data when historical
sentiment archives are unavailable. Uses price patterns and volume
anomalies to approximate what sentiment might have been.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.sentiment_archive import SentimentArchive

logger = logging.getLogger(__name__)


class SyntheticSentimentGenerator:
    """Generates synthetic sentiment from price/volume data.

    Strategy:
    - Sharp price drops with normal/low volume = potential FUD (negative sentiment)
    - Sharp price drops with high volume = legitimate selling (neutral sentiment)
    - Price drops with high volatility divergence = manipulation signal
    - Add random noise to simulate real sentiment variability
    """

    DEFAULT_NOISE_LEVEL = 0.15
    DEFAULT_VOLUME_WEIGHT = 0.3
    DEFAULT_RETURN_LOOKBACK = 5
    DEFAULT_VOLUME_LOOKBACK = 20
    DEFAULT_VOLATILITY_LOOKBACK = 20

    def __init__(
        self,
        seed: Optional[int] = None,
        noise_level: float = DEFAULT_NOISE_LEVEL,
        volume_weight: float = DEFAULT_VOLUME_WEIGHT,
    ):
        """Initialize the synthetic sentiment generator.

        Args:
            seed: Random seed for reproducibility.
            noise_level: Amplitude of random noise (0 to 1).
            volume_weight: Weight for volume in sentiment calculation.
        """
        self.rng = np.random.default_rng(seed)
        self.noise_level = max(0.0, min(1.0, noise_level))
        self.volume_weight = max(0.0, min(1.0, volume_weight))

    def generate_for_period(
        self, price_df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate synthetic sentiment for a price history DataFrame.

        Args:
            price_df: DataFrame with columns: date, open, high, low, close, volume.
            symbol: Stock symbol (for logging).

        Returns:
            DataFrame with columns:
                - date
                - sentiment_score (-1 to 1)
                - sentiment_volume (synthetic mention count)
                - manipulation_score (0 to 1)
                - is_synthetic (always True)
        """
        if price_df.empty:
            return pd.DataFrame(columns=[
                "date", "sentiment_score", "sentiment_volume",
                "manipulation_score", "is_synthetic",
            ])

        df = price_df.copy()

        if "date" not in df.columns:
            if df.index.name == "date":
                df = df.reset_index()
            else:
                raise ValueError("price_df must have a 'date' column")

        df = df.sort_values("date").reset_index(drop=True)

        returns = df["close"].pct_change()

        price_sentiment = self._calculate_price_sentiment(returns)

        volume_anomaly = self._calculate_volume_anomaly(df["volume"])

        manipulation = self._calculate_manipulation_proxy(returns, df["volume"])

        combined_sentiment = self._combine_signals(
            price_sentiment, volume_anomaly, manipulation
        )

        sentiment_with_noise = self._add_noise(combined_sentiment)

        synthetic_volume = self._generate_synthetic_volume(
            df["volume"], returns, manipulation
        )

        result = pd.DataFrame({
            "date": df["date"],
            "sentiment_score": sentiment_with_noise.clip(-1, 1),
            "sentiment_volume": synthetic_volume.astype(int),
            "manipulation_score": manipulation.clip(0, 1),
            "is_synthetic": True,
        })

        logger.debug(
            "Generated synthetic sentiment for %s: %d days, avg_score=%.3f",
            symbol,
            len(result),
            result["sentiment_score"].mean(),
        )

        return result

    def _calculate_price_sentiment(self, returns: pd.Series) -> pd.Series:
        """Convert price returns to sentiment score.

        Negative returns -> negative sentiment (with lag and smoothing).
        Uses exponential smoothing to simulate sentiment inertia.

        Args:
            returns: Daily return series.

        Returns:
            Sentiment series (-1 to 1).
        """
        smoothed = returns.ewm(span=3, adjust=False).mean()

        sentiment = -smoothed * 10

        sentiment = sentiment.clip(-1, 1)

        sentiment = sentiment.shift(1).fillna(0)

        return sentiment

    def _calculate_volume_anomaly(self, volume: pd.Series) -> pd.Series:
        """Detect volume anomalies that might indicate real news vs manipulation.

        High volume typically indicates legitimate market activity (news, earnings).
        Low/normal volume during price drops suggests potential manipulation.

        Args:
            volume: Volume series.

        Returns:
            Volume anomaly score (0 to 1, higher = more anomalous).
        """
        vol_sma = volume.rolling(window=self.DEFAULT_VOLUME_LOOKBACK, min_periods=1).mean()
        vol_std = volume.rolling(window=self.DEFAULT_VOLUME_LOOKBACK, min_periods=1).std()

        vol_std = vol_std.replace(0, 1)

        z_score = (volume - vol_sma) / vol_std

        anomaly = (1 / (1 + np.exp(-z_score))) * 2 - 1

        return anomaly.fillna(0)

    def _calculate_manipulation_proxy(
        self, returns: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Estimate manipulation probability from price/volume patterns.

        High manipulation score when:
        - Price drops sharply but volume is normal (no real selling pressure)
        - Volatility spikes without volume increase

        Args:
            returns: Daily return series.
            volume: Volume series.

        Returns:
            Manipulation score (0 to 1).
        """
        vol_sma = volume.rolling(window=self.DEFAULT_VOLUME_LOOKBACK, min_periods=1).mean()
        volume_ratio = volume / vol_sma.replace(0, 1)

        volatility = returns.rolling(window=self.DEFAULT_VOLATILITY_LOOKBACK, min_periods=1).std()
        vol_sma_price = volatility.rolling(window=self.DEFAULT_VOLATILITY_LOOKBACK, min_periods=1).mean()
        vol_ratio = volatility / vol_sma_price.replace(0, 0.01)

        negative_returns = returns < -0.01

        low_volume = volume_ratio < 1.0

        high_volatility = vol_ratio > 1.5

        manipulation_raw = pd.Series(0.0, index=returns.index)

        manipulation_raw = manipulation_raw.where(
            ~(negative_returns & low_volume),
            0.5 + 0.3 * (-returns * 10).clip(-1, 1),
        )

        manipulation_raw = manipulation_raw.where(
            ~(negative_returns & high_volatility & low_volume),
            manipulation_raw + 0.2,
        )

        manipulation = manipulation_raw.clip(0, 1)

        return manipulation.fillna(0)

    def _combine_signals(
        self,
        price_sentiment: pd.Series,
        volume_anomaly: pd.Series,
        manipulation: pd.Series,
    ) -> pd.Series:
        """Combine multiple signals into final sentiment score.

        Args:
            price_sentiment: Sentiment from price returns.
            volume_anomaly: Volume anomaly signal.
            manipulation: Manipulation probability.

        Returns:
            Combined sentiment score.
        """
        volume_adjustment = volume_anomaly * self.volume_weight

        combined = price_sentiment - manipulation * 0.3 + volume_adjustment * 0.2

        return combined

    def _add_noise(self, series: pd.Series) -> pd.Series:
        """Add realistic random noise to synthetic data.

        Args:
            series: Input series.

        Returns:
            Series with added noise.
        """
        if self.noise_level == 0:
            return series

        noise = self.rng.normal(0, self.noise_level, size=len(series))

        noisy = series + noise

        return noisy

    def _generate_synthetic_volume(
        self,
        price_volume: pd.Series,
        returns: pd.Series,
        manipulation: pd.Series,
    ) -> pd.Series:
        """Generate synthetic mention volume based on price action.

        Args:
            price_volume: Trading volume.
            returns: Price returns.
            manipulation: Manipulation score.

        Returns:
            Synthetic mention count series.
        """
        base_volume = 100

        vol_factor = (price_volume / price_volume.mean()).fillna(1)
        vol_factor = vol_factor.clip(0.5, 3)

        return_magnitude = returns.abs().fillna(0)
        return_factor = 1 + return_magnitude * 20

        manip_factor = 1 + manipulation * 2

        synthetic = base_volume * vol_factor * return_factor * manip_factor

        noise = self.rng.uniform(0.8, 1.2, size=len(synthetic))
        synthetic = synthetic * noise

        return synthetic.fillna(base_volume).clip(10, 10000)

    def set_noise_level(self, level: float) -> None:
        """Update noise level.

        Args:
            level: New noise level (0 to 1).
        """
        self.noise_level = max(0.0, min(1.0, level))

    def set_seed(self, seed: int) -> None:
        """Reset random generator with new seed.

        Args:
            seed: New random seed.
        """
        self.rng = np.random.default_rng(seed)


class BacktestSentimentProvider:
    """Provides sentiment data for backtesting.

    Uses real archived data when available, falls back to synthetic
    generation from price data when historical sentiment is unavailable.
    """

    DEFAULT_MIN_COVERAGE = 0.8

    def __init__(
        self,
        archive: SentimentArchive,
        generator: SyntheticSentimentGenerator,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ):
        """Initialize the backtest sentiment provider.

        Args:
            archive: SentimentArchive for real historical data.
            generator: SyntheticSentimentGenerator for synthetic data.
            min_coverage: Minimum coverage to use archived data (0 to 1).
        """
        self.archive = archive
        self.generator = generator
        self.min_coverage = min_coverage

    def get_sentiment(
        self,
        symbol: str,
        start: date,
        end: date,
        price_df: Optional[pd.DataFrame] = None,
        force_synthetic: bool = False,
    ) -> pd.DataFrame:
        """Get sentiment for backtesting period.

        1. Check archive for real data
        2. If coverage < threshold, generate synthetic
        3. Can blend real + synthetic for partial coverage

        Args:
            symbol: Stock ticker symbol.
            start: Start date.
            end: End date.
            price_df: Price DataFrame (required for synthetic generation).
            force_synthetic: If True, always use synthetic data.

        Returns:
            DataFrame with sentiment data and 'is_synthetic' flag.
        """
        if force_synthetic and price_df is not None:
            return self.generator.generate_for_period(price_df, symbol)

        if self.archive.session is not None and not force_synthetic:
            archived = self.archive.get_daily_sentiment(symbol, start, end)

            if archived:
                total_days = (end - start).days + 1
                coverage = len(archived) / total_days

                if coverage >= self.min_coverage:
                    logger.debug(
                        "Using archived sentiment for %s (%.1f%% coverage)",
                        symbol,
                        coverage * 100,
                    )
                    return pd.DataFrame(archived)

                if price_df is not None and coverage > 0:
                    return self._blend_sentiment(symbol, archived, price_df)

        if price_df is not None:
            logger.debug("Using synthetic sentiment for %s (no archive coverage)", symbol)
            return self.generator.generate_for_period(price_df, symbol)

        logger.warning(
            "No sentiment data available for %s: no archive and no price data",
            symbol,
        )
        return pd.DataFrame(columns=[
            "date", "sentiment_score", "sentiment_volume",
            "manipulation_score", "is_synthetic",
        ])

    def _blend_sentiment(
        self,
        symbol: str,
        archived: list[dict],
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Blend archived and synthetic sentiment for partial coverage.

        Args:
            symbol: Stock symbol.
            start: Start date.
            end: End date.
            archived: List of archived sentiment records.
            price_df: Price DataFrame for synthetic generation.

        Returns:
            Blended DataFrame.
        """
        synthetic = self.generator.generate_for_period(price_df, symbol)

        if synthetic.empty:
            return pd.DataFrame(archived)

        archived_df = pd.DataFrame(archived)
        if "date" not in archived_df.columns:
            archived_df["date"] = pd.to_datetime(archived_df["timestamp"]).dt.date

        archived_dates = set(archived_df["date"])

        synthetic_fill = synthetic[~synthetic["date"].isin(archived_dates)].copy()

        result = pd.concat([archived_df, synthetic_fill], ignore_index=True)
        result = result.sort_values("date").reset_index(drop=True)

        logger.debug(
            "Blended sentiment for %s: %d archived + %d synthetic",
            symbol,
            len(archived_df),
            len(synthetic_fill),
        )

        return result

    def get_sentiment_at_date(
        self,
        symbol: str,
        target_date: date,
        price_history: Optional[pd.DataFrame] = None,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Get sentiment for a specific date (for step-by-step backtesting).

        Args:
            symbol: Stock ticker symbol.
            target_date: The date to get sentiment for.
            price_history: Price history ending at or before target_date.
            lookback_days: Days of history to consider for synthetic.

        Returns:
            Dictionary with sentiment data for the date.
        """
        start = target_date - timedelta(days=lookback_days)

        sentiment_df = self.get_sentiment(
            symbol, start, target_date, price_df=price_history
        )

        if sentiment_df.empty:
            return {
                "date": target_date,
                "sentiment_score": 0.0,
                "sentiment_volume": 0,
                "manipulation_score": 0.0,
                "is_synthetic": True,
            }

        target_row = sentiment_df[sentiment_df["date"] == target_date]

        if not target_row.empty:
            return target_row.iloc[0].to_dict()

        latest = sentiment_df.iloc[-1]
        return {
            "date": target_date,
            "sentiment_score": float(latest.get("sentiment_score", 0.0)),
            "sentiment_volume": int(latest.get("sentiment_volume", 0)),
            "manipulation_score": float(latest.get("manipulation_score", 0.0)),
            "is_synthetic": True,
        }

    def get_coverage_info(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, Any]:
        """Get coverage information for symbols.

        Args:
            symbols: List of stock symbols.
            start: Start date.
            end: End date.

        Returns:
            Coverage report from archive.
        """
        if self.archive.session is None:
            return {
                "total_days": (end - start).days + 1,
                "symbol_coverage": {s: {"coverage_percent": 0.0} for s in symbols},
                "overall_coverage": 0.0,
            }

        return self.archive.get_coverage_report(symbols, start, end)
