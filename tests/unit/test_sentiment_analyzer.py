"""Unit tests for sentiment analyzer module."""

import pytest

from src.core.sentiment.analyzer import (
    FINANCIAL_LEXICON,
    FinBERTAnalyzer,
    SentimentAnalyzer,
)


def nltk_available():
    """Check if NLTK is available."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        return True
    except ImportError:
        return False


class TestSentimentAnalyzer:
    """Tests for VADER-based sentiment analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        return SentimentAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initializes (may be disabled if NLTK unavailable)."""
        if nltk_available():
            assert analyzer._initialized is True
            assert analyzer.vader is not None
        else:
            assert analyzer._initialized is False

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_financial_lexicon_loaded(self, analyzer):
        """Test financial terms are added to VADER lexicon."""
        assert "bullish" in analyzer.vader.lexicon
        assert "bearish" in analyzer.vader.lexicon
        assert "moon" in analyzer.vader.lexicon

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_analyze_positive_text(self, analyzer):
        """Test positive sentiment detection."""
        result = analyzer.analyze_text("This stock is amazing! Great earnings!")

        assert "compound" in result
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert result["compound"] > 0

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_analyze_negative_text(self, analyzer):
        """Test negative sentiment detection."""
        result = analyzer.analyze_text("This stock is terrible. Huge losses expected.")

        assert result["compound"] < 0

    def test_analyze_neutral_text(self, analyzer):
        """Test neutral sentiment detection."""
        result = analyzer.analyze_text("The stock closed at 150.")

        assert -0.2 < result["compound"] < 0.2

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_analyze_financial_terms_bullish(self, analyzer):
        """Test bullish financial terminology."""
        result = analyzer.analyze_text("Super bullish on AAPL, rocket to the moon!")

        assert result["compound"] > 0.5

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_analyze_financial_terms_bearish(self, analyzer):
        """Test bearish financial terminology."""
        result = analyzer.analyze_text("Very bearish, this stock will dump hard")

        assert result["compound"] < -0.5

    def test_analyze_empty_text(self, analyzer):
        """Test empty text returns neutral sentiment."""
        result = analyzer.analyze_text("")

        assert abs(result["compound"]) < 0.01
        assert abs(result["neutral"] - 1.0) < 0.01

    def test_analyze_none_text(self, analyzer):
        """Test None text returns neutral sentiment."""
        result = analyzer.analyze_text(None)

        assert abs(result["compound"]) < 0.01

    @pytest.mark.skipif(not nltk_available(), reason="NLTK not installed")
    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = [
            "Great stock, very bullish!",
            "Terrible performance, sell now",
            "The market closed today",
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert results[0]["compound"] > 0
        assert results[1]["compound"] < 0

    def test_analyze_batch_empty(self, analyzer):
        """Test empty batch returns empty list."""
        results = analyzer.analyze_batch([])

        assert results == []

    def test_extract_tickers_with_dollar_sign(self, analyzer):
        """Test ticker extraction with $ prefix."""
        text = "Buying $AAPL and $TSLA today!"

        tickers = analyzer.extract_tickers(text, require_dollar_sign=True)

        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_extract_tickers_without_dollar_sign(self, analyzer):
        """Test ticker extraction without $ prefix."""
        text = "Looking at AAPL and MSFT for my portfolio"

        tickers = analyzer.extract_tickers(text, require_dollar_sign=False)

        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_extract_tickers_filters_common_words(self, analyzer):
        """Test common words are filtered out."""
        text = "THE stock AND market ARE doing well FOR now"

        tickers = analyzer.extract_tickers(text, require_dollar_sign=False)

        assert "THE" not in tickers
        assert "AND" not in tickers
        assert "ARE" not in tickers
        assert "FOR" not in tickers

    def test_extract_tickers_empty_text(self, analyzer):
        """Test empty text returns empty list."""
        tickers = analyzer.extract_tickers("")

        assert tickers == []

    def test_classify_sentiment_positive(self, analyzer):
        """Test positive classification."""
        assert analyzer.classify_sentiment(0.5) == "positive"
        assert analyzer.classify_sentiment(0.05) == "positive"

    def test_classify_sentiment_negative(self, analyzer):
        """Test negative classification."""
        assert analyzer.classify_sentiment(-0.5) == "negative"
        assert analyzer.classify_sentiment(-0.05) == "negative"

    def test_classify_sentiment_neutral(self, analyzer):
        """Test neutral classification."""
        assert analyzer.classify_sentiment(0.0) == "neutral"
        assert analyzer.classify_sentiment(0.04) == "neutral"
        assert analyzer.classify_sentiment(-0.04) == "neutral"


class TestFinBERTAnalyzer:
    """Tests for FinBERT sentiment analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create FinBERT analyzer instance (will use fallback if model not available)."""
        return FinBERTAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initializes (may fallback to VADER)."""
        assert analyzer._fallback is not None

    def test_is_available_property(self, analyzer):
        """Test is_available property."""
        assert isinstance(analyzer.is_available, bool)

    def test_analyze_text_returns_required_keys(self, analyzer):
        """Test analysis returns all required keys."""
        result = analyzer.analyze_text("Great earnings report!")

        assert "compound" in result
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "model" in result

    def test_analyze_text_model_indicator(self, analyzer):
        """Test model indicator is included."""
        result = analyzer.analyze_text("Test text")

        assert result["model"] in ["finbert", "vader"]

    def test_analyze_empty_text(self, analyzer):
        """Test empty text handling."""
        result = analyzer.analyze_text("")

        assert result["compound"] == 0.0
        assert result["neutral"] == 1.0

    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = ["Positive news!", "Negative outlook", "Neutral statement"]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        for result in results:
            assert "compound" in result
            assert "model" in result

    def test_analyze_batch_empty(self, analyzer):
        """Test empty batch."""
        results = analyzer.analyze_batch([])

        assert results == []

    def test_classify_sentiment(self, analyzer):
        """Test sentiment classification."""
        assert analyzer.classify_sentiment(0.5) == "positive"
        assert analyzer.classify_sentiment(-0.5) == "negative"
        assert analyzer.classify_sentiment(0.0) == "neutral"


class TestFinancialLexicon:
    """Tests for financial lexicon constants."""

    def test_lexicon_not_empty(self):
        """Test lexicon has entries."""
        assert len(FINANCIAL_LEXICON) > 0

    def test_lexicon_has_bullish_terms(self):
        """Test lexicon includes bullish terms."""
        assert "bullish" in FINANCIAL_LEXICON
        assert FINANCIAL_LEXICON["bullish"] > 0

    def test_lexicon_has_bearish_terms(self):
        """Test lexicon includes bearish terms."""
        assert "bearish" in FINANCIAL_LEXICON
        assert FINANCIAL_LEXICON["bearish"] < 0

    def test_lexicon_scores_are_floats(self):
        """Test all scores are numeric."""
        for term, score in FINANCIAL_LEXICON.items():
            assert isinstance(score, (int, float))

    def test_lexicon_scores_in_range(self):
        """Test scores are in reasonable range."""
        for term, score in FINANCIAL_LEXICON.items():
            assert -5 <= score <= 5
