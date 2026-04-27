"""
train.py
--------
Trains an XGBoost classifier to predict AAPL 7-day return direction.
Uses time-series aware cross-validation (TimeSeriesSplit — no data leakage).

Input:  data/processed/master_features.parquet
Output: models/xgb_model.pkl
        models/feature_importance.csv

Usage:
    python src/models/train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# ── Config ──────────────────────────────────────────────────────────────
INPUT_PATH = Path("data/processed/master_features.parquet")
MODEL_PATH = Path("models/xgb_model.pkl")
IMPORTANCE_PATH = Path("models/feature_importance.csv")

TARGET = "fwd_direction_7d"

FEATURES = [
    # Supplier sentiment (lagged — no lookahead)
    "tsmc_sentiment_idx_lag7d",
    "samsung_sentiment_idx_lag7d",
    "composite_sentiment_idx_lag7d",
    "tsmc_sentiment_idx_lag5d",
    "composite_sentiment_idx_lag5d",
    # Price momentum
    "momentum_5d",
    "momentum_20d",
]

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

N_SPLITS = 5


def load_data(path: Path, features: list[str], target: str):
    df = pd.read_parquet(path)

    # Only require the target to be non-null; fill missing sentiment with 0 (neutral)
    df = df.dropna(subset=[target])
    df[features] = df[features].fillna(0.0)

    # Drop any feature columns that don't exist and warn
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns (will use 0.0): {missing_cols}")
        for col in missing_cols:
            df[col] = 0.0

    X = df[features]
    y = df[target]
    logger.info(f"Dataset: {len(df)} rows, {len(features)} features")
    logger.info(f"Target balance: {y.mean():.2%} positive")
    return X, y, df


def time_series_cv(X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int):
    """Evaluate with TimeSeriesSplit — respects temporal ordering."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fold_scores.append({"fold": fold + 1, "accuracy": acc, "f1": f1})
        logger.info(f"Fold {fold + 1}: accuracy={acc:.3f}, f1={f1:.3f}")

    scores_df = pd.DataFrame(fold_scores)
    logger.info(f"\nCV Summary:\n{scores_df.describe().loc[['mean','std']]}")
    return scores_df


def train_final_model(X: pd.DataFrame, y: pd.Series, params: dict) -> XGBClassifier:
    """Train on the full dataset for deployment."""
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def main():
    X, y, df = load_data(INPUT_PATH, FEATURES, TARGET)

    # Baseline: always predict "up" (market drift)
    baseline_acc = y.mean()
    logger.info(f"Baseline accuracy (always-up): {baseline_acc:.3f}")

    # Cross-validation
    cv_scores = time_series_cv(X, y, XGB_PARAMS, N_SPLITS)

    # Final model on full data
    model = train_final_model(X, y, XGB_PARAMS)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop features:")
    print(importance_df.to_string(index=False))

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "target": TARGET}, f)

    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    logger.success(f"Model saved to {MODEL_PATH}")
    logger.success(f"Feature importance saved to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()