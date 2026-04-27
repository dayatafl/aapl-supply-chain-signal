"""
test_sentiment.py
-----------------
Unit tests for the sentiment scoring pipeline.
"""

import pandas as pd
import pytest

from src.features.sentiment_scorer import FinBERTScorer


@pytest.fixture(scope="module")
def scorer():
    """Load FinBERT once for all tests in the module."""
    return FinBERTScorer()


class TestFinBERTScorer:

    def test_positive_headline(self, scorer):
        texts = ["TSMC reports record profits, expands Apple chip capacity significantly"]
        scores = scorer.score_batch(texts)
        assert len(scores) == 1
        assert scores[0] > 0, "Expected positive score for bullish headline"

    def test_negative_headline(self, scorer):
        texts = ["TSMC halts production due to severe earthquake damage in Taiwan"]
        scores = scorer.score_batch(texts)
        assert len(scores) == 1
        assert scores[0] < 0, "Expected negative score for bearish headline"

    def test_score_range(self, scorer):
        texts = ["Samsung announces new display panel", "TSMC cuts guidance", "Apple supply chain normal"]
        scores = scorer.score_batch(texts)
        for s in scores:
            assert -1.0 <= s <= 1.0, f"Score {s} out of [-1, 1] range"

    def test_batch_length(self, scorer):
        texts = [f"Headline number {i}" for i in range(50)]
        scores = scorer.score_batch(texts)
        assert len(scores) == 50

    def test_score_dataframe(self, scorer):
        df = pd.DataFrame({
            "title": ["TSMC beats earnings", "Samsung faces chip shortage crisis"],
            "supplier": ["tsmc", "samsung"],
        })
        result = scorer.score_dataframe(df, text_col="title")
        assert "sentiment_score" in result.columns
        assert len(result) == 2

    def test_empty_string_handled(self, scorer):
        scores = scorer.score_batch([""])
        assert len(scores) == 1
        assert isinstance(scores[0], float)
