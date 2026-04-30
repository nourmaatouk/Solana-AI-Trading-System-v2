import pytest
import torch
from unittest.mock import MagicMock
from src.models.finbert_wrapper import FinBertSentimentAnalyzer

@pytest.fixture
def mocked_analyzer(monkeypatch):
    """
    Initializes a FinBertSentimentAnalyzer with transformers libraries mocked entirely.
    This safely guarantees tests do not download large LLMs.
    """

    class MockTokenizer:
        def __call__(self, text, *args, **kwargs):
            return {"input_ids": [1, 2, 3]}

    class MockModelOutputs:
        # Mocking Logits to heavily skew positive (e.g [positive, negative, neutral])
        logits = torch.tensor([[2.0, -1.0, 0.5]])

    class MockModel:
        def eval(self):
            pass
        def __call__(self, **kwargs):
            return MockModelOutputs()

    # Pre-emptively intercept HuggingFace imports before FinBert setup connects externally.
    monkeypatch.setattr("src.models.finbert_wrapper.AutoTokenizer.from_pretrained", lambda x: MockTokenizer())
    monkeypatch.setattr("src.models.finbert_wrapper.AutoModelForSequenceClassification.from_pretrained", lambda x: MockModel())
    
    return FinBertSentimentAnalyzer()

def test_empty_text_returns_neutral(mocked_analyzer):
    """Ensure baseline mapping fires upon empty sequence parsing."""
    result = mocked_analyzer.get_sentiment("")
    assert result["neutral_prob"] == 1.0
    assert result["sentiment_score"] == 0.0
    assert result["sentiment_label"] == "neutral"
    
def test_none_text_returns_neutral(mocked_analyzer):
    """Ensure baseline mapping fires upon None type safely bypassing tokenizer."""
    result = mocked_analyzer.get_sentiment(None)
    assert result["neutral_prob"] == 1.0

def test_valid_text_expected_schema(mocked_analyzer):
    """Ensure output correctly constructs the necessary fields."""
    result = mocked_analyzer.get_sentiment("This is a valid test string.")
    
    # Assert requested payload schema requirements
    assert "positive_prob" in result
    assert "negative_prob" in result
    assert "neutral_prob" in result
    assert "sentiment_score" in result
    assert "confidence" in result
    assert "sentiment_label" in result
    
def test_probabilities_sum_close_to_one_and_scores_in_range(mocked_analyzer):
    """Calculates probabilities natively ensuring bounding limits and score mappings align."""
    result = mocked_analyzer.get_sentiment("This is a valid test string.")
    
    prob_sum = result["positive_prob"] + result["negative_prob"] + result["neutral_prob"]
    
    # Assert distribution properties bounding mapping
    assert pytest.approx(prob_sum, 0.001) == 1.0
    
    # Validate mathematical bounding requirements 
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    
def test_label_and_confidence_alignment(mocked_analyzer):
    """Ensures maximum confidence maps structurally out to proper label names correctly."""
    result = mocked_analyzer.get_sentiment("This is a valid test string.")
    
    # Assert alignment rules correctly triggered
    assert result["sentiment_label"] == "positive"
    assert result["confidence"] == result["positive_prob"]
    assert result["sentiment_score"] > 0.0
