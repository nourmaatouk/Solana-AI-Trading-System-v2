import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional


class FinBertSentimentAnalyzer:
    """
    Wrapper around ProsusAI/finbert to extract sentiment scores from financial and crypto text.
    Handles neutral edgecases and formats outputs safely.
    Automatically uses CUDA if available, falling back to CPU.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        # ProsusAI/finbert class labels mapping natively:
        # 0 -> positive, 1 -> negative, 2 -> neutral
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _empty_sentiment(self) -> Dict:
        """Returns baseline neutral outputs when provided with empty or invalid text."""
        return {
            "negative_prob": 0.0,
            "neutral_prob": 1.0,
            "positive_prob": 0.0,
            "sentiment_score": 0.0,
            "confidence": 1.0,
            "sentiment_label": "neutral"
        }

    def get_sentiment(self, text: Optional[str]) -> Dict:
        if not text or not str(text).strip():
            return self._empty_sentiment()

        # Tokenize the target text sequence
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply softmax across classification dimensions and flatten sequence list
            probs = F.softmax(logits, dim=-1).squeeze().tolist()

        # Ensure single sequence inference structures out securely.
        if type(probs) is not list:
            probs = [probs]

        # Extract according to ProsusAI/finbert output mapping structure [positive, negative, neutral]
        positive_prob = float(probs[0])
        negative_prob = float(probs[1])
        neutral_prob = float(probs[2])

        # Core logic requested: 
        sentiment_score = positive_prob - negative_prob
        confidence = max(positive_prob, negative_prob, neutral_prob)

        # Map to human readable format
        if confidence == positive_prob:
            sentiment_label = "positive"
        elif confidence == negative_prob:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return {
            "negative_prob": negative_prob,
            "neutral_prob": neutral_prob,
            "positive_prob": positive_prob,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "sentiment_label": sentiment_label
        }
