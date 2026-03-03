"""
Manipulation detection algorithms for Leprechaun trading bot.

Detects potential market manipulation by analyzing:
- Sentiment-price divergence
- Abnormal volume spikes
- Coordinated posting patterns
- Bot activity indicators
- Vocabulary similarity across posts
"""

import logging
import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ManipulationDetector:
    """
    Detects potential market manipulation through multiple signals.

    The detector combines multiple indicators to produce a manipulation score:
    - Sentiment-price divergence: When sentiment doesn't match price action
    - Volume spikes: Abnormal mention volume increases
    - Coordination: Posts appearing in tight time clusters
    - Bot activity: Accounts with suspicious characteristics
    - Vocabulary similarity: Copy-paste campaigns

    A high manipulation score suggests the sentiment may be artificially
    generated and should be treated with skepticism for trading decisions.
    """

    def __init__(
        self,
        coordination_threshold: float = 0.6,
        bot_threshold: float = 0.4,
        divergence_weight: float = 0.25,
        spike_weight: float = 0.20,
        coordination_weight: float = 0.25,
        bot_weight: float = 0.15,
        vocabulary_weight: float = 0.15,
    ):
        """
        Initialize manipulation detector with configurable thresholds.

        Args:
            coordination_threshold: Score above this indicates coordination (0-1).
            bot_threshold: Probability above this indicates likely bot (0-1).
            divergence_weight: Weight for divergence in final score.
            spike_weight: Weight for volume spike in final score.
            coordination_weight: Weight for coordination in final score.
            bot_weight: Weight for bot activity in final score.
            vocabulary_weight: Weight for vocabulary similarity in final score.
        """
        self.coordination_threshold = coordination_threshold
        self.bot_threshold = bot_threshold

        self.divergence_weight = divergence_weight
        self.spike_weight = spike_weight
        self.coordination_weight = coordination_weight
        self.bot_weight = bot_weight
        self.vocabulary_weight = vocabulary_weight

    def calculate_divergence_score(
        self,
        sentiment_score: float,
        price_change: float,
        volume_ratio: float = 1.0,
    ) -> float:
        """
        Calculate sentiment-price divergence.

        High score indicates sentiment doesn't match price action, which may
        indicate manipulation or artificial sentiment.

        Args:
            sentiment_score: Sentiment score (-1 to 1).
            price_change: Price change as percentage (e.g., -5.0 for -5%).
            volume_ratio: Trading volume ratio vs average (>1 = above average).

        Returns:
            Divergence score (0-1). Higher = more divergence.
        """
        if sentiment_score == 0 and price_change == 0:
            return 0.0

        sentiment_direction = (
            1 if sentiment_score > 0.1 else (-1 if sentiment_score < -0.1 else 0)
        )
        price_direction = (
            1 if price_change > 1.0 else (-1 if price_change < -1.0 else 0)
        )

        if sentiment_direction == 0 or price_direction == 0:
            return 0.1

        if sentiment_direction == price_direction:
            return 0.0

        sentiment_magnitude = abs(sentiment_score)
        price_magnitude = min(abs(price_change) / 10.0, 1.0)

        base_divergence = (sentiment_magnitude + price_magnitude) / 2.0

        volume_factor = 1.0
        if volume_ratio < 0.5:
            volume_factor = 1.3
        elif volume_ratio > 2.0:
            volume_factor = 0.8

        divergence = min(base_divergence * volume_factor, 1.0)

        return divergence

    def detect_sentiment_spike(
        self,
        sentiment_history: list[dict],
        current: dict,
        std_threshold: float = 2.0,
    ) -> bool:
        """
        Detect abnormal sentiment spike (>2 std dev from mean).

        Args:
            sentiment_history: List of historical sentiment dicts with 'compound' key.
            current: Current sentiment dict with 'compound' key.
            std_threshold: Number of standard deviations for spike detection.

        Returns:
            True if current sentiment is a spike.
        """
        if len(sentiment_history) < 5:
            return False

        historical_scores = [s.get("compound", 0) for s in sentiment_history]

        mean = sum(historical_scores) / len(historical_scores)
        variance = sum((x - mean) ** 2 for x in historical_scores) / len(
            historical_scores
        )
        std = math.sqrt(variance) if variance > 0 else 0.001

        current_score = current.get("compound", 0)
        z_score = abs(current_score - mean) / std

        return z_score > std_threshold

    def detect_volume_spike(
        self,
        mention_history: list[int],
        current: int,
        std_threshold: float = 2.0,
    ) -> bool:
        """
        Detect abnormal mention volume spike.

        Args:
            mention_history: List of historical mention counts.
            current: Current mention count.
            std_threshold: Number of standard deviations for spike detection.

        Returns:
            True if current volume is a spike.
        """
        if len(mention_history) < 5:
            return False

        mean = sum(mention_history) / len(mention_history)
        variance = sum((x - mean) ** 2 for x in mention_history) / len(mention_history)
        std = math.sqrt(variance) if variance > 0 else 1.0

        z_score = (current - mean) / std if std > 0 else 0

        return z_score > std_threshold

    def analyze_posting_patterns(
        self,
        posts: list[dict],
        cluster_window_minutes: int = 5,
    ) -> dict[str, Any]:
        """
        Analyze temporal patterns in posts for coordination detection.

        Coordinated campaigns often show:
        - Multiple posts within tight time windows
        - Regular intervals between posts
        - Peak activity at specific times

        Args:
            posts: List of post dicts with 'timestamp' key (datetime).
            cluster_window_minutes: Time window for clustering posts.

        Returns:
            Dictionary with:
                - is_coordinated: Whether coordination is detected
                - cluster_score: Score indicating coordination level (0-1)
                - peak_times: List of peak activity timestamps
                - clusters: List of post clusters
        """
        if len(posts) < 3:
            return {
                "is_coordinated": False,
                "cluster_score": 0.0,
                "peak_times": [],
                "clusters": [],
            }

        timestamps = []
        for post in posts:
            ts = post.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except (ValueError, TypeError):
                    continue

        if len(timestamps) < 3:
            return {
                "is_coordinated": False,
                "cluster_score": 0.0,
                "peak_times": [],
                "clusters": [],
            }

        timestamps.sort()

        clusters = []
        current_cluster = [timestamps[0]]
        window = timedelta(minutes=cluster_window_minutes)

        for ts in timestamps[1:]:
            if ts - current_cluster[-1] <= window:
                current_cluster.append(ts)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [ts]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        if not clusters:
            return {
                "is_coordinated": False,
                "cluster_score": 0.0,
                "peak_times": [],
                "clusters": [],
            }

        total_clustered = sum(len(c) for c in clusters)
        cluster_ratio = total_clustered / len(timestamps)

        largest_cluster_ratio = max(len(c) for c in clusters) / len(timestamps)

        cluster_score = (cluster_ratio * 0.6) + (largest_cluster_ratio * 0.4)
        cluster_score = min(cluster_score, 1.0)

        peak_times = []
        for cluster in clusters:
            if len(cluster) >= 3:
                mid_idx = len(cluster) // 2
                peak_times.append(cluster[mid_idx])

        return {
            "is_coordinated": cluster_score > self.coordination_threshold,
            "cluster_score": cluster_score,
            "peak_times": peak_times,
            "clusters": [[ts.isoformat() for ts in c] for c in clusters],
        }

    def analyze_vocabulary_similarity(
        self,
        posts: list[dict],
        min_similarity: float = 0.6,
    ) -> float:
        """
        Detect similar vocabulary across posts (copy-paste campaigns).

        Uses Jaccard similarity on word sets to detect posts that share
        unusually high vocabulary overlap.

        Args:
            posts: List of post dicts with 'text' key.
            min_similarity: Minimum Jaccard similarity to consider similar.

        Returns:
            Similarity score (0-1). Higher = more similar vocabulary.
        """
        if len(posts) < 2:
            return 0.0

        word_sets = []
        for post in posts:
            text = post.get("text", "")
            if not text:
                continue
            words = set(text.lower().split())
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "just",
                "and",
                "but",
                "if",
                "or",
                "because",
                "until",
                "while",
                "it",
                "its",
                "this",
                "that",
                "i",
                "you",
                "he",
                "she",
                "we",
                "they",
                "my",
                "your",
                "his",
                "her",
                "our",
                "their",
                "what",
                "which",
                "who",
                "whom",
            }
            words = words - stop_words
            if len(words) >= 3:
                word_sets.append(words)

        if len(word_sets) < 2:
            return 0.0

        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarity = intersection / union
                    if similarity >= min_similarity:
                        similarities.append(similarity)

        if not similarities:
            return 0.0

        total_pairs = len(word_sets) * (len(word_sets) - 1) / 2
        similar_ratio = len(similarities) / total_pairs

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        return similar_ratio * 0.5 + avg_similarity * 0.5

    def estimate_bot_probability(self, author_data: dict) -> float:
        """
        Estimate probability that author is a bot.

        Based on signals like:
        - Account age (very new accounts)
        - Posting frequency (inhuman rates)
        - Content patterns (repetitive or generic)
        - Username patterns (random characters)

        Args:
            author_data: Dictionary with author information:
                - account_age_days: int
                - posts_per_day: float
                - karma: int (optional, Reddit-specific)
                - username: str (optional)
                - is_verified: bool (optional)

        Returns:
            Bot probability (0-1). Higher = more likely bot.
        """
        score = 0.0
        weights_sum = 0.0

        account_age = author_data.get("account_age_days", 365)
        if account_age < 7:
            score += 0.8
        elif account_age < 30:
            score += 0.5
        elif account_age < 90:
            score += 0.2
        else:
            score += 0.0
        weights_sum += 1.0

        posts_per_day = author_data.get("posts_per_day", 5.0)
        if posts_per_day > 50:
            score += 0.9
        elif posts_per_day > 20:
            score += 0.6
        elif posts_per_day > 10:
            score += 0.3
        else:
            score += 0.0
        weights_sum += 1.0

        karma = author_data.get("karma")
        if karma is not None:
            if karma < 10:
                score += 0.7
            elif karma < 100:
                score += 0.4
            elif karma < 1000:
                score += 0.1
            else:
                score += 0.0
            weights_sum += 1.0

        username = author_data.get("username", "")
        if username:
            import re

            if re.match(r"^[a-zA-Z]+_[a-zA-Z]+_\d+$", username):
                score += 0.6
            elif re.match(r"^[a-zA-Z0-9]{15,}$", username):
                score += 0.4
            elif re.search(r"\d{4,}$", username):
                score += 0.3
            weights_sum += 1.0

        if author_data.get("is_verified", False):
            score -= 0.3
            weights_sum += 0.3

        probability = score / weights_sum if weights_sum > 0 else 0.0
        return max(0.0, min(1.0, probability))

    def calculate_manipulation_score(self, signals: dict) -> float:
        """
        Combine all signals into final manipulation score.

        Args:
            signals: Dictionary with signal values:
                - divergence: float (0-1)
                - spike: bool or float
                - coordination: float (0-1)
                - bot_activity: float (0-1)
                - vocabulary_similarity: float (0-1)

        Returns:
            Manipulation score (0-1). Higher = more likely manipulation.
        """
        score = 0.0
        total_weight = 0.0

        if "divergence" in signals:
            score += signals["divergence"] * self.divergence_weight
            total_weight += self.divergence_weight

        if "spike" in signals:
            spike_value = (
                1.0
                if signals["spike"] is True
                else (0.0 if signals["spike"] is False else float(signals["spike"]))
            )
            score += spike_value * self.spike_weight
            total_weight += self.spike_weight

        if "coordination" in signals:
            score += signals["coordination"] * self.coordination_weight
            total_weight += self.coordination_weight

        if "bot_activity" in signals:
            score += signals["bot_activity"] * self.bot_weight
            total_weight += self.bot_weight

        if "vocabulary_similarity" in signals:
            score += signals["vocabulary_similarity"] * self.vocabulary_weight
            total_weight += self.vocabulary_weight

        if total_weight == 0:
            return 0.0

        normalized_score = score / total_weight

        return max(0.0, min(1.0, normalized_score))

    def analyze_posts(
        self,
        posts: list[dict],
        sentiment_score: Optional[float] = None,
        price_change: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        sentiment_history: Optional[list[dict]] = None,
        mention_history: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """
        Full manipulation analysis on a set of posts.

        Convenience method that runs all detection algorithms and combines
        them into a comprehensive analysis.

        Args:
            posts: List of post dicts with 'text', 'timestamp', 'author' keys.
            sentiment_score: Current aggregate sentiment score.
            price_change: Recent price change percentage.
            volume_ratio: Trading volume vs average ratio.
            sentiment_history: Historical sentiment data for spike detection.
            mention_history: Historical mention counts for volume spike detection.

        Returns:
            Comprehensive analysis dictionary.
        """
        signals = {}
        details = {}

        if sentiment_score is not None and price_change is not None:
            divergence = self.calculate_divergence_score(
                sentiment_score,
                price_change,
                volume_ratio or 1.0,
            )
            signals["divergence"] = divergence
            details["divergence"] = {
                "score": divergence,
                "sentiment": sentiment_score,
                "price_change": price_change,
            }

        if sentiment_history and sentiment_score is not None:
            current = {"compound": sentiment_score}
            spike = self.detect_sentiment_spike(sentiment_history, current)
            signals["spike"] = spike
            details["sentiment_spike"] = spike

        if mention_history and posts:
            volume_spike = self.detect_volume_spike(mention_history, len(posts))
            if "spike" in signals:
                signals["spike"] = signals["spike"] or volume_spike
            else:
                signals["spike"] = volume_spike
            details["volume_spike"] = volume_spike

        posting_analysis = self.analyze_posting_patterns(posts)
        signals["coordination"] = posting_analysis["cluster_score"]
        details["posting_patterns"] = posting_analysis

        vocab_score = self.analyze_vocabulary_similarity(posts)
        signals["vocabulary_similarity"] = vocab_score
        details["vocabulary_similarity"] = vocab_score

        bot_scores = []
        for post in posts:
            author_data = post.get("author_data", post.get("author", {}))
            if isinstance(author_data, dict):
                bot_prob = self.estimate_bot_probability(author_data)
                bot_scores.append(bot_prob)

        if bot_scores:
            avg_bot_activity = sum(bot_scores) / len(bot_scores)
            signals["bot_activity"] = avg_bot_activity
            details["bot_activity"] = {
                "average": avg_bot_activity,
                "high_probability_count": sum(
                    1 for s in bot_scores if s > self.bot_threshold
                ),
                "total_analyzed": len(bot_scores),
            }

        manipulation_score = self.calculate_manipulation_score(signals)

        return {
            "manipulation_score": manipulation_score,
            "is_suspicious": manipulation_score > 0.5,
            "signals": signals,
            "details": details,
            "posts_analyzed": len(posts),
        }
