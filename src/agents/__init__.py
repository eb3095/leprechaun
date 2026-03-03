"""Agentic AI components for Leprechaun trading bot.

This module provides the agent framework for autonomous trading decisions:

- BayesianManipulationDetector: Bayesian inference for manipulation probability
- DecisionLogger: Audit trail for all trading decisions
- SentimentAgent: Aggregates sentiment from multiple sources
- ManipulationAgent: Detects potential market manipulation
- TradingAgent: Makes final trading decisions
- AgentOrchestrator: Coordinates all agents in analysis cycles
"""

from src.agents.bayesian import BayesianManipulationDetector, EvidenceWeights
from src.agents.decision_log import Decision, DecisionLogger
from src.agents.manipulation_agent import ManipulationAgent
from src.agents.orchestrator import AgentOrchestrator
from src.agents.sentiment_agent import SentimentAgent, SentimentResult, SourceWeight
from src.agents.trading_agent import RiskManager, TradingAgent, TradingStrategy

__all__ = [
    "AgentOrchestrator",
    "BayesianManipulationDetector",
    "Decision",
    "DecisionLogger",
    "EvidenceWeights",
    "ManipulationAgent",
    "RiskManager",
    "SentimentAgent",
    "SentimentResult",
    "SourceWeight",
    "TradingAgent",
    "TradingStrategy",
]
