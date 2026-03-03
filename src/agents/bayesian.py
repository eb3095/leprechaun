"""Bayesian inference engine for manipulation detection.

Uses Bayes' theorem to calculate probability of market manipulation
based on multiple evidence signals.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceWeights:
    """Likelihood ratios for evidence types.

    Each weight represents P(evidence|manipulation) / P(evidence|not_manipulation).
    Values > 1 indicate evidence supports manipulation hypothesis.
    """

    sentiment_spike: float = 3.0
    no_news_catalyst: float = 2.5
    coordination_detected: float = 4.0
    high_bot_activity: float = 3.5
    volume_sentiment_divergence: float = 2.0


class BayesianManipulationDetector:
    """Bayesian inference for manipulation detection.

    Uses Bayes' theorem to update probability of manipulation given evidence.
    P(manipulation | evidence) = P(evidence | manipulation) * P(manipulation)
                                 / P(evidence)
    """

    DEFAULT_PRIOR = 0.1
    MIN_PRIOR = 0.01
    MAX_PRIOR = 0.5

    def __init__(
        self,
        prior_manipulation: float = DEFAULT_PRIOR,
        evidence_weights: Optional[dict[str, float]] = None,
    ):
        """Initialize detector with prior probability and evidence weights.

        Args:
            prior_manipulation: Base rate of manipulation (default 10%).
            evidence_weights: Dict of evidence type to likelihood ratio.
                            If None, uses default weights.
        """
        self.prior_manipulation = max(
            self.MIN_PRIOR, min(self.MAX_PRIOR, prior_manipulation)
        )
        self.evidence_weights = EvidenceWeights()

        if evidence_weights:
            for key, value in evidence_weights.items():
                if hasattr(self.evidence_weights, key):
                    setattr(self.evidence_weights, key, max(0.1, value))

    def calculate_posterior(self, evidence: dict[str, bool]) -> float:
        """Calculate P(manipulation | evidence) using Bayes theorem.

        For multiple independent pieces of evidence, we multiply likelihood ratios.

        Args:
            evidence: Dict mapping evidence type to bool indicating presence.
                     e.g., {"sentiment_spike": True, "no_news_catalyst": True}

        Returns:
            Probability of manipulation (0 to 1).
        """
        if not evidence:
            return self.prior_manipulation

        combined_likelihood_ratio = 1.0

        for evidence_type, is_present in evidence.items():
            weight = getattr(self.evidence_weights, evidence_type, 1.0)

            if is_present:
                combined_likelihood_ratio *= weight
            else:
                combined_likelihood_ratio *= 1.0 / weight

        prior_odds = self.prior_manipulation / (1 - self.prior_manipulation)
        posterior_odds = prior_odds * combined_likelihood_ratio
        posterior = posterior_odds / (1 + posterior_odds)

        return min(0.999, max(0.001, posterior))

    def calculate_log_posterior(self, evidence: dict[str, bool]) -> float:
        """Calculate posterior using log-odds for numerical stability.

        Use this method when dealing with extreme evidence combinations
        that might cause overflow with regular odds multiplication.

        Args:
            evidence: Dict mapping evidence type to bool.

        Returns:
            Probability of manipulation (0 to 1).
        """
        if not evidence:
            return self.prior_manipulation

        log_prior_odds = math.log(
            self.prior_manipulation / (1 - self.prior_manipulation)
        )

        log_likelihood_ratio = 0.0
        for evidence_type, is_present in evidence.items():
            weight = getattr(self.evidence_weights, evidence_type, 1.0)

            if is_present:
                log_likelihood_ratio += math.log(weight)
            else:
                log_likelihood_ratio -= math.log(weight)

        log_posterior_odds = log_prior_odds + log_likelihood_ratio

        log_posterior_odds = max(-10, min(10, log_posterior_odds))
        posterior = 1 / (1 + math.exp(-log_posterior_odds))

        return posterior

    def update_priors(self, historical_data: list[dict]) -> dict[str, float]:
        """Update prior and likelihood ratios from historical outcomes.

        Analyzes historical cases where manipulation was confirmed or refuted
        to calibrate the model parameters.

        Args:
            historical_data: List of dicts with keys:
                - "was_manipulation": bool - ground truth
                - "evidence": dict - evidence that was observed

        Returns:
            Dict with updated parameters:
                - "prior_manipulation": updated base rate
                - Evidence weights for each type
        """
        if not historical_data:
            return {"prior_manipulation": self.prior_manipulation}

        manipulation_count = sum(
            1 for d in historical_data if d.get("was_manipulation", False)
        )
        total = len(historical_data)

        new_prior = max(
            self.MIN_PRIOR,
            min(
                self.MAX_PRIOR,
                manipulation_count / total if total > 0 else self.DEFAULT_PRIOR,
            ),
        )

        self.prior_manipulation = new_prior

        result = {"prior_manipulation": new_prior}

        evidence_types = [
            "sentiment_spike",
            "no_news_catalyst",
            "coordination_detected",
            "high_bot_activity",
            "volume_sentiment_divergence",
        ]

        for evidence_type in evidence_types:
            manip_with_evidence = sum(
                1
                for d in historical_data
                if d.get("was_manipulation", False)
                and d.get("evidence", {}).get(evidence_type, False)
            )
            manip_without_evidence = manipulation_count - manip_with_evidence

            not_manip_with_evidence = sum(
                1
                for d in historical_data
                if not d.get("was_manipulation", False)
                and d.get("evidence", {}).get(evidence_type, False)
            )
            not_manip_without_evidence = (
                total - manipulation_count
            ) - not_manip_with_evidence

            p_evidence_given_manip = (manip_with_evidence + 1) / (
                manipulation_count + 2
            )
            p_evidence_given_not_manip = (not_manip_with_evidence + 1) / (
                total - manipulation_count + 2
            )

            if p_evidence_given_not_manip > 0:
                new_weight = p_evidence_given_manip / p_evidence_given_not_manip
                new_weight = max(0.5, min(10.0, new_weight))
                setattr(self.evidence_weights, evidence_type, new_weight)
                result[evidence_type] = new_weight

        return result

    def get_confidence_interval(
        self, posterior: float, sample_size: int, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate credible interval for the posterior.

        Uses normal approximation for Beta distribution (appropriate for
        binomial data with sufficient sample size).

        Args:
            posterior: Calculated posterior probability.
            sample_size: Number of observations (affects certainty).
            confidence: Desired confidence level (default 95%).

        Returns:
            Tuple of (lower_bound, upper_bound) for credible interval.
        """
        if sample_size < 1:
            return (0.0, 1.0)

        if confidence not in {0.90, 0.95, 0.99}:
            confidence = 0.95

        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        variance = (posterior * (1 - posterior)) / sample_size
        std_dev = math.sqrt(variance)

        margin = z * std_dev

        lower = max(0.0, posterior - margin)
        upper = min(1.0, posterior + margin)

        return (lower, upper)

    def get_evidence_contribution(self, evidence: dict[str, bool]) -> dict[str, float]:
        """Get individual contribution of each piece of evidence.

        Useful for explaining which evidence had the most impact on the
        manipulation probability.

        Args:
            evidence: Dict mapping evidence type to bool.

        Returns:
            Dict mapping evidence type to its contribution to log-odds.
        """
        contributions = {}

        for evidence_type, is_present in evidence.items():
            weight = getattr(self.evidence_weights, evidence_type, 1.0)

            if is_present:
                contribution = math.log(weight)
            else:
                contribution = -math.log(weight)

            contributions[evidence_type] = contribution

        return contributions

    def explain_posterior(self, evidence: dict[str, bool]) -> dict:
        """Generate human-readable explanation of posterior calculation.

        Args:
            evidence: Dict mapping evidence type to bool.

        Returns:
            Dict with:
                - "posterior": calculated probability
                - "prior": prior probability used
                - "contributions": per-evidence contributions
                - "interpretation": text explanation
        """
        posterior = self.calculate_posterior(evidence)
        contributions = self.get_evidence_contribution(evidence)

        sorted_contributions = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        if posterior > 0.7:
            interpretation = "High probability of manipulation"
        elif posterior > 0.5:
            interpretation = "Moderate probability of manipulation"
        elif posterior > 0.3:
            interpretation = "Low probability of manipulation"
        else:
            interpretation = "Manipulation unlikely"

        top_factors = []
        for evidence_type, contrib in sorted_contributions[:3]:
            direction = "supports" if contrib > 0 else "refutes"
            top_factors.append(f"{evidence_type} {direction} manipulation")

        return {
            "posterior": posterior,
            "prior": self.prior_manipulation,
            "contributions": contributions,
            "interpretation": interpretation,
            "top_factors": top_factors,
        }

    def get_weights(self) -> dict[str, float]:
        """Get current evidence weights as dictionary."""
        return {
            "sentiment_spike": self.evidence_weights.sentiment_spike,
            "no_news_catalyst": self.evidence_weights.no_news_catalyst,
            "coordination_detected": self.evidence_weights.coordination_detected,
            "high_bot_activity": self.evidence_weights.high_bot_activity,
            "volume_sentiment_divergence": self.evidence_weights.volume_sentiment_divergence,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set evidence weights from dictionary."""
        for key, value in weights.items():
            if hasattr(self.evidence_weights, key):
                setattr(self.evidence_weights, key, max(0.1, value))
