"""Manipulation detection agent for Leprechaun trading bot.

Analyzes market data to detect potential manipulation using
Bayesian inference and pattern recognition.
"""

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Optional

from src.agents.bayesian import BayesianManipulationDetector


class ManipulationAgent:
    """Detects potential market manipulation in stock sentiment and price data."""

    COORDINATION_WINDOW_MINUTES = 30
    COORDINATION_THRESHOLD = 0.4
    BOT_THRESHOLD = 0.3
    NEWS_LOOKBACK_HOURS = 24

    def __init__(
        self,
        bayesian: Optional[BayesianManipulationDetector] = None,
    ):
        """Initialize manipulation agent.

        Args:
            bayesian: Bayesian detector instance. Creates default if None.
        """
        self.bayesian = bayesian or BayesianManipulationDetector()

    def analyze(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze stock for manipulation signals.

        Args:
            symbol: Stock symbol to analyze.
            data: Dict containing:
                - sentiment: SentimentResult or dict with sentiment data
                - price_history: List of price records
                - news: List of news articles
                - technical_indicators: Dict of technical indicators
                - posts: List of social media posts (optional)

        Returns:
            Dict with:
                - manipulation_score: float (0 to 1)
                - bayesian_probability: float (0 to 1)
                - evidence: dict of which signals triggered
                - confidence: str ("HIGH", "MEDIUM", "LOW")
                - recommendation: str
        """
        evidence = self._gather_evidence(symbol, data)

        bayesian_prob = self.bayesian.calculate_posterior(evidence)

        weighted_signals = [
            (evidence.get("sentiment_spike", False), 0.25),
            (evidence.get("no_news_catalyst", False), 0.20),
            (evidence.get("coordination_detected", False), 0.25),
            (evidence.get("high_bot_activity", False), 0.15),
            (evidence.get("volume_sentiment_divergence", False), 0.15),
        ]
        manipulation_score = sum(w for present, w in weighted_signals if present)

        combined_score = 0.6 * bayesian_prob + 0.4 * manipulation_score

        if combined_score > 0.7:
            confidence = "HIGH"
        elif combined_score > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        recommendation = self._generate_recommendation(
            combined_score, confidence, evidence
        )

        explanation = self.bayesian.explain_posterior(evidence)

        return {
            "manipulation_score": combined_score,
            "bayesian_probability": bayesian_prob,
            "evidence": evidence,
            "confidence": confidence,
            "recommendation": recommendation,
            "explanation": explanation,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _gather_evidence(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> dict[str, bool]:
        """Gather all evidence signals for Bayesian analysis."""
        evidence = {}

        sentiment = data.get("sentiment", {})
        if hasattr(sentiment, "to_dict"):
            sentiment = sentiment.to_dict()

        sentiment_score = sentiment.get("composite_score", 0.0)
        evidence["sentiment_spike"] = sentiment_score < -0.4

        news = data.get("news", [])
        sentiment_spike_time = data.get("sentiment_spike_time")
        if sentiment_spike_time is None and evidence["sentiment_spike"]:
            sentiment_spike_time = datetime.utcnow()
        evidence["no_news_catalyst"] = not self.check_news_catalyst(
            symbol, news, sentiment_spike_time
        )

        posts = data.get("posts", [])
        coordination_score = self.detect_coordination(posts)
        evidence["coordination_detected"] = (
            coordination_score > self.COORDINATION_THRESHOLD
        )

        bot_fraction = self.estimate_bot_activity(posts)
        evidence["high_bot_activity"] = bot_fraction > self.BOT_THRESHOLD

        evidence["volume_sentiment_divergence"] = self._check_divergence(data)

        return evidence

    def _check_divergence(self, data: dict[str, Any]) -> bool:
        """Check for divergence between sentiment volume and price movement."""
        sentiment = data.get("sentiment", {})
        if hasattr(sentiment, "to_dict"):
            sentiment = sentiment.to_dict()

        sentiment_volume = sentiment.get("volume", 0)
        technical = data.get("technical_indicators", {})

        volume_sma = technical.get("volume_sma_20", 0)
        current_volume = technical.get("current_volume", 0)

        if volume_sma > 0 and sentiment_volume > 100:
            if current_volume < volume_sma * 0.5:
                return True

        price_history = data.get("price_history", [])
        if len(price_history) >= 2:
            recent_prices = [p.get("close", 0) for p in price_history[-5:]]
            if recent_prices and recent_prices[0] > 0:
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                sentiment_score = sentiment.get("composite_score", 0.0)

                if sentiment_score < -0.3 and price_change > -0.02:
                    return True
                if sentiment_score > 0.3 and price_change < 0.02:
                    return True

        return False

    def detect_coordination(self, posts: list[dict]) -> float:
        """Detect coordinated posting patterns.

        Looks for:
        - Similar timing (posts clustered in time)
        - Similar vocabulary (shared phrases)
        - Similar structure

        Args:
            posts: List of post dicts with text, timestamp, author.

        Returns:
            Coordination score from 0 (natural) to 1 (highly coordinated).
        """
        if len(posts) < 5:
            return 0.0

        timing_score = self._analyze_timing_coordination(posts)

        vocabulary_score = self._analyze_vocabulary_coordination(posts)

        author_score = self._analyze_author_patterns(posts)

        combined = 0.4 * timing_score + 0.4 * vocabulary_score + 0.2 * author_score

        return min(1.0, combined)

    def _analyze_timing_coordination(self, posts: list[dict]) -> float:
        """Analyze temporal clustering of posts."""
        timestamps = []
        for post in posts:
            ts = post.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                except ValueError:
                    continue
            timestamps.append(ts)

        if len(timestamps) < 3:
            return 0.0

        timestamps.sort()

        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(delta)

        if not intervals:
            return 0.0

        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)

        if avg_interval > 0:
            cv = (variance**0.5) / avg_interval
        else:
            cv = 0

        if cv < 0.3:
            return 0.8
        elif cv < 0.5:
            return 0.5
        elif cv < 0.8:
            return 0.3
        return 0.0

    def _analyze_vocabulary_coordination(self, posts: list[dict]) -> float:
        """Analyze shared vocabulary patterns."""
        texts = [post.get("text", "") for post in posts if post.get("text")]

        if len(texts) < 3:
            return 0.0

        all_phrases: Counter = Counter()
        for text in texts:
            phrases = self._extract_phrases(text)
            all_phrases.update(phrases)

        shared_phrases = sum(1 for phrase, count in all_phrases.items() if count > 2)
        total_unique = len(all_phrases)

        if total_unique > 0:
            shared_ratio = shared_phrases / total_unique
        else:
            shared_ratio = 0.0

        if shared_ratio > 0.5:
            return 0.9
        elif shared_ratio > 0.3:
            return 0.6
        elif shared_ratio > 0.1:
            return 0.3
        return 0.0

    def _extract_phrases(self, text: str) -> list[str]:
        """Extract meaningful phrases from text."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()

        stock_terms = {
            "buy",
            "sell",
            "hold",
            "moon",
            "rocket",
            "squeeze",
            "short",
            "long",
            "calls",
            "puts",
            "diamond",
            "hands",
            "dump",
            "pump",
            "dip",
            "rip",
            "tendies",
            "ape",
        }

        phrases = []
        for i in range(len(words) - 1):
            if words[i] in stock_terms or words[i + 1] in stock_terms:
                phrases.append(f"{words[i]} {words[i + 1]}")

        for word in words:
            if word in stock_terms:
                phrases.append(word)

        return phrases

    def _analyze_author_patterns(self, posts: list[dict]) -> float:
        """Analyze author account patterns."""
        authors = [post.get("author", "") for post in posts if post.get("author")]

        if len(authors) < 3:
            return 0.0

        author_counts = Counter(authors)

        unique_authors = len(author_counts)
        total_posts = len(authors)

        if total_posts > 0:
            diversity = unique_authors / total_posts
        else:
            diversity = 1.0

        if diversity < 0.3:
            return 0.7
        elif diversity < 0.5:
            return 0.4
        return 0.0

    def estimate_bot_activity(self, posts: list[dict]) -> float:
        """Estimate fraction of bot-generated content.

        Looks for bot indicators:
        - Account age < 30 days
        - Generic usernames
        - High posting frequency
        - Template-like content

        Args:
            posts: List of post dicts.

        Returns:
            Estimated fraction of bot posts (0 to 1).
        """
        if len(posts) < 3:
            return 0.0

        bot_indicators = 0
        total_indicators = 0

        for post in posts:
            account_age = post.get("account_age_days", 365)
            if account_age < 30:
                bot_indicators += 1
            total_indicators += 1

            author = post.get("author", "")
            if self._is_generic_username(author):
                bot_indicators += 0.5
            total_indicators += 1

            text = post.get("text", "")
            if len(text) > 0:
                if self._is_template_content(text):
                    bot_indicators += 1
                total_indicators += 1

        if total_indicators > 0:
            return min(1.0, bot_indicators / total_indicators)
        return 0.0

    def _is_generic_username(self, username: str) -> bool:
        """Check if username follows generic bot patterns."""
        if not username:
            return True

        patterns = [
            r"^[a-z]+[0-9]{4,}$",
            r"^[A-Z][a-z]+[A-Z][a-z]+[0-9]+$",
            r"^user[0-9]+$",
            r"^[a-z]+-[a-z]+-[a-z0-9]+$",
        ]

        for pattern in patterns:
            if re.match(pattern, username):
                return True

        return False

    def _is_template_content(self, text: str) -> bool:
        """Check if text appears to be template-generated."""
        template_phrases = [
            "this is financial advice",
            "buy now before it's too late",
            "to the moon",
            "not financial advice",
            "this stock will",
            "guaranteed returns",
            "don't miss out",
        ]

        text_lower = text.lower()
        matches = sum(1 for phrase in template_phrases if phrase in text_lower)

        return matches >= 2

    def check_news_catalyst(
        self,
        symbol: str,
        news: list[dict],
        sentiment_spike_time: Optional[datetime] = None,
    ) -> bool:
        """Check if there's news that explains the sentiment.

        Args:
            symbol: Stock symbol.
            news: List of news articles with title, timestamp, sentiment.
            sentiment_spike_time: When the sentiment spike occurred.

        Returns:
            True if relevant news catalyst found, False otherwise.
        """
        if not news:
            return False

        if sentiment_spike_time is None:
            sentiment_spike_time = datetime.utcnow()

        lookback = sentiment_spike_time - timedelta(hours=self.NEWS_LOOKBACK_HOURS)

        for article in news:
            article_time = article.get("timestamp")
            if article_time is None:
                continue

            if isinstance(article_time, str):
                try:
                    article_time = datetime.fromisoformat(
                        article_time.replace("Z", "+00:00")
                    )
                    if article_time.tzinfo:
                        article_time = article_time.replace(tzinfo=None)
                except ValueError:
                    continue

            if lookback <= article_time <= sentiment_spike_time:
                article_sentiment = article.get("sentiment", 0.0)
                if article_sentiment < -0.3:
                    return True

                title = article.get("title", "").lower()
                negative_keywords = [
                    "lawsuit",
                    "fraud",
                    "investigation",
                    "recall",
                    "scandal",
                    "layoff",
                    "bankruptcy",
                    "downgrade",
                    "warning",
                    "miss",
                    "loss",
                    "decline",
                    "fell",
                    "crash",
                    "drop",
                    "plunge",
                ]
                if any(kw in title for kw in negative_keywords):
                    return True

        return False

    def _generate_recommendation(
        self,
        score: float,
        confidence: str,
        evidence: dict[str, bool],
    ) -> str:
        """Generate actionable recommendation based on analysis."""
        if score > 0.7:
            if evidence.get("sentiment_spike") and evidence.get("no_news_catalyst"):
                return "STRONG_CONTRARIAN_BUY - High manipulation probability with negative sentiment spike and no news catalyst"
            return "CONSIDER_CONTRARIAN_BUY - High manipulation probability detected"
        elif score > 0.5:
            return (
                "MONITOR - Moderate manipulation signals present, wait for confirmation"
            )
        else:
            return "PASS - Insufficient manipulation indicators for contrarian play"

    def get_risk_assessment(
        self,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate risk assessment for a manipulation analysis.

        Args:
            analysis: Result from analyze() method.

        Returns:
            Dict with risk metrics and warnings.
        """
        score = analysis.get("manipulation_score", 0.0)
        evidence = analysis.get("evidence", {})

        warnings = []

        if evidence.get("high_bot_activity"):
            warnings.append("High bot activity may indicate organized campaign")

        if evidence.get("coordination_detected"):
            warnings.append("Coordinated posting detected - proceed with caution")

        if not evidence.get("no_news_catalyst"):
            warnings.append("News catalyst exists - sentiment may be justified")

        if score > 0.8:
            risk_level = "HIGH"
        elif score > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "warnings": warnings,
            "false_positive_risk": 1.0 - analysis.get("bayesian_probability", 0.5),
            "confidence_interval": self.bayesian.get_confidence_interval(
                analysis.get("bayesian_probability", 0.5),
                sample_size=max(1, len(evidence)),
            ),
        }
