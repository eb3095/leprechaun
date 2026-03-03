"""
Sentiment analysis using VADER and FinBERT for Leprechaun trading bot.

VADER provides fast sentiment scoring optimized for social media.
FinBERT provides accurate financial sentiment analysis when available.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

FINANCIAL_LEXICON = {
    "bullish": 3.0,
    "bearish": -3.0,
    "moon": 2.5,
    "mooning": 2.5,
    "rocket": 2.0,
    "rockets": 2.0,
    "dump": -2.5,
    "dumping": -2.5,
    "short": -1.5,
    "shorting": -1.5,
    "squeeze": 2.0,
    "squeezing": 2.0,
    "diamond": 2.0,
    "hands": 0.5,
    "paper": -1.5,
    "tendies": 2.0,
    "gains": 2.0,
    "loss": -2.0,
    "losses": -2.0,
    "bag": -1.5,
    "bagholder": -2.0,
    "bagholding": -2.0,
    "pump": 1.5,
    "pumping": 1.5,
    "fomo": 1.0,
    "fud": -1.5,
    "yolo": 1.5,
    "ape": 1.5,
    "apes": 1.5,
    "hodl": 1.5,
    "hold": 0.5,
    "holding": 0.5,
    "sell": -0.5,
    "selling": -0.5,
    "buy": 0.5,
    "buying": 0.5,
    "calls": 1.0,
    "puts": -1.0,
    "options": 0.0,
    "undervalued": 2.0,
    "overvalued": -2.0,
    "oversold": 1.5,
    "overbought": -1.5,
    "dip": 0.5,
    "btfd": 1.5,
    "crash": -3.0,
    "crashing": -3.0,
    "rally": 2.5,
    "rallying": 2.5,
    "breakout": 2.0,
    "breakdown": -2.0,
    "support": 1.0,
    "resistance": 0.5,
    "reversal": 1.0,
    "recovery": 1.5,
    "rebound": 1.5,
    "plunge": -2.5,
    "plunging": -2.5,
    "surge": 2.5,
    "surging": 2.5,
    "soar": 2.5,
    "soaring": 2.5,
    "tank": -2.5,
    "tanking": -2.5,
    "rip": 2.0,
    "ripping": 2.0,
    "drill": -2.0,
    "drilling": -2.0,
    "green": 1.5,
    "red": -1.5,
    "profit": 2.0,
    "profitable": 2.0,
    "earnings": 0.5,
    "beat": 2.0,
    "miss": -2.0,
    "missed": -2.0,
    "guidance": 0.5,
    "upgrade": 2.0,
    "downgrade": -2.0,
    "outperform": 2.0,
    "underperform": -2.0,
    "target": 0.5,
    "accumulate": 1.5,
    "distribute": -1.0,
    "insider": 0.5,
    "manipulation": -1.0,
    "fraud": -3.0,
    "scam": -3.0,
    "bankrupt": -3.0,
    "bankruptcy": -3.0,
    "debt": -1.0,
    "dilution": -2.0,
    "offering": -1.0,
    "lawsuit": -1.5,
    "sec": -0.5,
    "investigation": -1.5,
}

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
TICKER_PATTERN_NO_DOLLAR = re.compile(r"\b([A-Z]{2,5})\b")


class SentimentAnalyzer:
    """
    Sentiment analyzer using NLTK's VADER with financial lexicon extensions.

    VADER is optimized for social media text and handles emojis, slang, and
    emphasis (ALL CAPS, exclamation marks) appropriately.
    """

    def __init__(self):
        """Initialize VADER sentiment analyzer with financial lexicon."""
        self.vader = None
        self._initialized = False
        self._init_vader()

    def _init_vader(self) -> None:
        """Initialize VADER, downloading data if needed."""
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                logger.info("Downloading VADER lexicon...")
                nltk.download("vader_lexicon", quiet=True)

            self.vader = SentimentIntensityAnalyzer()
            self._add_financial_lexicon()
            self._initialized = True
            logger.info("VADER sentiment analyzer initialized")
        except ImportError:
            logger.warning("NLTK not available, sentiment analysis disabled")
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")

    def _add_financial_lexicon(self) -> None:
        """Add financial terms to VADER lexicon."""
        if self.vader is None:
            return

        for term, score in FINANCIAL_LEXICON.items():
            self.vader.lexicon[term] = score
            self.vader.lexicon[term.upper()] = score * 1.2

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary with keys:
                - compound: Overall sentiment score (-1 to 1)
                - positive: Proportion of positive sentiment (0 to 1)
                - negative: Proportion of negative sentiment (0 to 1)
                - neutral: Proportion of neutral sentiment (0 to 1)
        """
        if not self._initialized or self.vader is None:
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

        if not text or not text.strip():
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

        scores = self.vader.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyze multiple texts efficiently.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of sentiment dictionaries.
        """
        return [self.analyze_text(text) for text in texts]

    def extract_tickers(
        self, text: str, require_dollar_sign: bool = False
    ) -> list[str]:
        """
        Extract stock ticker symbols from text.

        Looks for patterns like $AAPL, $TSLA, or plain uppercase symbols.

        Args:
            text: Text to extract tickers from.
            require_dollar_sign: If True, only extract tickers with $ prefix.

        Returns:
            List of unique ticker symbols (without $ prefix).
        """
        if not text:
            return []

        tickers = set()

        dollar_matches = TICKER_PATTERN.findall(text)
        tickers.update(dollar_matches)

        if not require_dollar_sign:
            plain_matches = TICKER_PATTERN_NO_DOLLAR.findall(text)
            common_words = {
                "THE",
                "AND",
                "FOR",
                "ARE",
                "BUT",
                "NOT",
                "YOU",
                "ALL",
                "CAN",
                "HAD",
                "HER",
                "WAS",
                "ONE",
                "OUR",
                "OUT",
                "HAS",
                "HIS",
                "HOW",
                "ITS",
                "MAY",
                "NEW",
                "NOW",
                "OLD",
                "SEE",
                "WAY",
                "WHO",
                "BOY",
                "DID",
                "GET",
                "HIM",
                "LET",
                "PUT",
                "SAY",
                "SHE",
                "TOO",
                "USE",
                "CEO",
                "CFO",
                "IPO",
                "ETF",
                "NYSE",
                "USA",
                "USD",
                "EUR",
                "GDP",
                "FBI",
                "SEC",
                "FDA",
                "CEO",
                "CTO",
                "COO",
                "CEO",
                "IMO",
                "LOL",
                "OMG",
                "WTF",
                "BTW",
                "FYI",
                "TBH",
                "SMH",
                "LMAO",
                "ROFL",
                "EOD",
                "EOM",
            }
            for match in plain_matches:
                if match not in common_words:
                    tickers.add(match)

        return sorted(tickers)

    def classify_sentiment(self, compound: float) -> str:
        """
        Classify compound score as positive, negative, or neutral.

        Uses standard VADER thresholds:
        - compound >= 0.05: positive
        - compound <= -0.05: negative
        - otherwise: neutral

        Args:
            compound: Compound sentiment score (-1 to 1).

        Returns:
            Classification string: 'positive', 'negative', or 'neutral'.
        """
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"


class FinBERTAnalyzer:
    """
    FinBERT analyzer for accurate financial sentiment analysis.

    FinBERT is a BERT model fine-tuned on financial text. It provides more
    accurate sentiment analysis for financial documents compared to VADER.

    This implementation gracefully degrades to VADER when FinBERT is not
    available (missing dependencies or model loading fails).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize FinBERT analyzer.

        Args:
            model_path: Path to local model or HuggingFace model name.
                       Defaults to 'ProsusAI/finbert'.
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or "ProsusAI/finbert"
        self._initialized = False
        self._fallback = SentimentAnalyzer()
        self._load_model()

    def _load_model(self) -> None:
        """Load FinBERT model if available, otherwise work in fallback mode."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading FinBERT model from {self.model_path}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.eval()

            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("FinBERT loaded on CUDA")
            else:
                logger.info("FinBERT loaded on CPU")

            self._initialized = True

        except ImportError as e:
            logger.warning(
                f"FinBERT dependencies not available ({e}), using VADER fallback"
            )
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}, using VADER fallback")

    @property
    def is_available(self) -> bool:
        """Check if FinBERT model is loaded and ready."""
        return self._initialized and self.model is not None

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment using FinBERT or fallback to VADER.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary with keys:
                - compound: Overall sentiment score (-1 to 1)
                - positive: Positive probability (0 to 1)
                - negative: Negative probability (0 to 1)
                - neutral: Neutral probability (0 to 1)
                - model: 'finbert' or 'vader' indicating which model was used
        """
        if not self._initialized:
            result = self._fallback.analyze_text(text)
            result["model"] = "vader"
            return result

        if not text or not text.strip():
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "model": "finbert",
            }

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            if torch.cuda.is_available() and self.model is not None:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()[0]

            positive = float(probs[0])
            negative = float(probs[1])
            neutral = float(probs[2])

            compound = positive - negative

            return {
                "compound": compound,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "model": "finbert",
            }

        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}, falling back to VADER")
            result = self._fallback.analyze_text(text)
            result["model"] = "vader"
            return result

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """
        Batch analysis for efficiency.

        Args:
            texts: List of texts to analyze.
            batch_size: Number of texts to process at once.

        Returns:
            List of sentiment dictionaries.
        """
        if not self._initialized:
            return [
                {**self._fallback.analyze_text(text), "model": "vader"}
                for text in texts
            ]

        if not texts:
            return []

        results = []

        try:
            import torch

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                valid_texts = [t if t and t.strip() else "neutral" for t in batch]

                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )

                if torch.cuda.is_available() and self.model is not None:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    probs = probs.cpu().numpy()

                for j, text in enumerate(batch):
                    if not text or not text.strip():
                        results.append(
                            {
                                "compound": 0.0,
                                "positive": 0.0,
                                "negative": 0.0,
                                "neutral": 1.0,
                                "model": "finbert",
                            }
                        )
                    else:
                        positive = float(probs[j][0])
                        negative = float(probs[j][1])
                        neutral = float(probs[j][2])
                        results.append(
                            {
                                "compound": positive - negative,
                                "positive": positive,
                                "negative": negative,
                                "neutral": neutral,
                                "model": "finbert",
                            }
                        )

        except Exception as e:
            logger.warning(f"FinBERT batch failed: {e}, falling back to VADER")
            for text in texts:
                result = self._fallback.analyze_text(text)
                result["model"] = "vader"
                results.append(result)

        return results

    def classify_sentiment(self, compound: float) -> str:
        """
        Classify compound score as positive, negative, or neutral.

        Args:
            compound: Compound sentiment score (-1 to 1).

        Returns:
            Classification string: 'positive', 'negative', or 'neutral'.
        """
        if compound >= 0.1:
            return "positive"
        elif compound <= -0.1:
            return "negative"
        else:
            return "neutral"
