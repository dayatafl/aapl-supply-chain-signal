"""
sentiment_scorer.py
-------------------
Scores each headline using ProsusAI/finbert (finance-tuned BERT).
Sentiment score = P(positive) - P(negative) ∈ [-1, +1]

Input:  data/raw/headlines_raw.parquet
Output: data/processed/headlines_scored.parquet

Usage:
    python src/features/sentiment_scorer.py
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from loguru import logger

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
INPUT_PATH = Path("data/raw/headlines_raw.parquet")
OUTPUT_PATH = Path("data/processed/headlines_scored.parquet")

LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}


class FinBERTScorer:
    """Lightweight wrapper around FinBERT for batch sentiment scoring."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]
        logger.success("FinBERT loaded.")

    def score_batch(self, texts: list[str]) -> list[float]:
        """Return sentiment scores for a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        scores = []
        for p in probs:
            # p[0]=positive, p[1]=negative, p[2]=neutral (FinBERT label order)
            score = float(p[0] - p[1])
            scores.append(round(score, 4))
        return scores

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """Score all rows in a DataFrame."""
        texts = df[text_col].fillna("").tolist()
        all_scores = []

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
            batch = texts[i : i + BATCH_SIZE]
            all_scores.extend(self.score_batch(batch))

        df = df.copy()
        df["sentiment_score"] = all_scores
        return df


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found. Run collect_headlines.py first.")

    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded {len(df)} headlines")

    scorer = FinBERTScorer()
    df_scored = scorer.score_dataframe(df, text_col="title")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(OUTPUT_PATH, index=False)
    logger.success(f"Saved scored headlines to {OUTPUT_PATH}")

    # Quick stats
    print("\nSentiment distribution:")
    print(df_scored["sentiment_score"].describe())
    print(f"\nPositive: {(df_scored['sentiment_score'] > 0.1).sum()}")
    print(f"Negative: {(df_scored['sentiment_score'] < -0.1).sum()}")
    print(f"Neutral:  {(df_scored['sentiment_score'].abs() <= 0.1).sum()}")


if __name__ == "__main__":
    main()
