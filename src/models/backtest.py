"""
backtest.py
-----------
Walk-forward backtest — no lookahead bias.
Trains on a rolling window, tests on the next step,
steps forward and repeats. Simulates a realistic trading scenario.

Computes: accuracy, Sharpe ratio, max drawdown, vs. buy-and-hold.

Input:  data/processed/master_features.parquet
        models/xgb_model.pkl
Output: data/processed/backtest_results.parquet
        data/processed/backtest_summary.csv

Usage:
    python src/models/backtest.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ── Config ───────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data/processed/master_features.parquet")
MODEL_PATH   = Path("models/xgb_model.pkl")
RESULTS_PATH = Path("data/processed/backtest_results.parquet")
SUMMARY_PATH = Path("data/processed/backtest_summary.csv")

TRAIN_WINDOW = 40    # ~2 months of trading days (fits 2026 window)
TEST_STEP    = 5     # ~1 week per fold
MIN_TRAIN    = 20    # absolute minimum to attempt a fold


# ── Metrics ───────────────────────────────────────────────────────────────
def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    return float(((equity_curve - peak) / peak).min())


# ── Backtest ──────────────────────────────────────────────────────────────
def walk_forward_backtest(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """
    Roll through time: train on [t-window, t), test on [t, t+step).
    Returns a DataFrame of all test predictions with dates.
    """
    # ── 1. Clean — work on a full independent copy to avoid SettingWithCopyWarning
    df = df.copy()
    df = df.dropna(subset=[target, "Close", "fwd_return_7d"])

    # Add missing feature columns as 0 and fill NaN
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    df[features] = df[features].fillna(0.0)

    total_rows = len(df)
    logger.info(f"Dataset after cleaning: {total_rows} rows")

    # ── 2. Adapt window to available data
    train_window = TRAIN_WINDOW
    test_step    = TEST_STEP

    if total_rows < train_window + test_step:
        train_window = max(MIN_TRAIN, total_rows // 2)
        test_step    = max(5, total_rows - train_window)
        logger.warning(
            f"Dataset too small for default windows. "
            f"Adjusted → train_window={train_window}, test_step={test_step}"
        )

    if total_rows < train_window + test_step:
        logger.error(
            f"Not enough rows ({total_rows}) to run even one fold "
            f"(need at least {train_window + test_step}). "
            f"Make sure index_builder.py completed successfully."
        )
        return pd.DataFrame()

    # ── 3. Walk-forward loop
    results = []
    start   = train_window
    fold    = 0

    while start + test_step <= total_rows:
        train_slice = df.iloc[start - train_window : start]
        test_slice  = df.iloc[start : start + test_step]

        X_train = train_slice[features]
        y_train = train_slice[target]
        X_test  = test_slice[features]
        y_test  = test_slice[target]

        # Skip fold if target has only one class (can happen with tiny slices)
        if y_train.nunique() < 2:
            logger.warning(f"Fold {fold:02d}: skipped — only one class in training target")
            start += test_step
            fold  += 1
            continue

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)

        for i, idx in enumerate(test_slice.index):
            results.append({
                "date":             idx,
                "fold":             fold,
                "prediction":       int(preds[i]),
                "probability":      float(probs[i]),
                "actual_direction": int(y_test.iloc[i]),
                "actual_return_14d": float(test_slice["fwd_return_7d"].iloc[i]),
                "close":            float(test_slice["Close"].iloc[i]),
                "fold_accuracy":    float(acc),
            })

        logger.info(
            f"Fold {fold:02d} | "
            f"train [{start - train_window}:{start}] "
            f"test [{start}:{start + test_step}] | "
            f"acc={acc:.3f}"
        )
        fold  += 1
        start += test_step

    if not results:
        logger.error("No results produced — all folds were skipped.")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.set_index("date")
    logger.success(f"Backtest complete: {fold} folds, {len(out)} prediction rows")
    return out


# ── Strategy returns ───────────────────────────────────────────────────────
def compute_strategy_returns(results: pd.DataFrame) -> pd.DataFrame:
    """Long when prediction=1, flat when prediction=0. Compare to buy-and-hold."""
    results = results.copy()
    results["strategy_return"] = results["prediction"] * results["actual_return_14d"]
    results["buyhold_return"]  = results["actual_return_14d"]
    results["strategy_equity"] = (1 + results["strategy_return"]).cumprod()
    results["buyhold_equity"]  = (1 + results["buyhold_return"]).cumprod()
    return results


# ── Summary ────────────────────────────────────────────────────────────────
def print_summary(results: pd.DataFrame) -> dict:
    strat = results["strategy_return"]
    bh    = results["buyhold_return"]
    acc   = accuracy_score(results["actual_direction"], results["prediction"])

    summary = {
        "Accuracy":                f"{acc:.3f}",
        "Baseline (always-up)":    f"{results['actual_direction'].mean():.3f}",
        "Strategy Sharpe":         f"{sharpe_ratio(strat):.3f}",
        "Buy-Hold Sharpe":         f"{sharpe_ratio(bh):.3f}",
        "Strategy Max Drawdown":   f"{max_drawdown(results['strategy_equity']):.2%}",
        "Buy-Hold Max Drawdown":   f"{max_drawdown(results['buyhold_equity']):.2%}",
        "Strategy Total Return":   f"{results['strategy_equity'].iloc[-1] - 1:.2%}",
        "Buy-Hold Total Return":   f"{results['buyhold_equity'].iloc[-1] - 1:.2%}",
        "Total folds":             f"{results['fold'].nunique()}",
        "Prediction rows":         f"{len(results)}",
    }

    print("\n" + "=" * 52)
    print("  BACKTEST SUMMARY")
    print("=" * 52)
    for k, v in summary.items():
        print(f"  {k:<32} {v}")
    print("=" * 52)
    return summary


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Load feature data
    if not INPUT_PATH.exists():
        logger.error(f"{INPUT_PATH} not found. Run index_builder.py first.")
        return

    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded master features: {df.shape}")

    # Load model artifact for feature list
    if not MODEL_PATH.exists():
        logger.error(f"{MODEL_PATH} not found. Run train.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    features = artifact["features"]
    target   = artifact["target"]
    logger.info(f"Features ({len(features)}): {features}")
    logger.info(f"Target: {target}")

    # Run backtest
    logger.info("Running walk-forward backtest...")
    results = walk_forward_backtest(df, features, target)

    if results.empty:
        logger.error(
            "Backtest produced no results. "
            "Common causes:\n"
            "  1. master_features.parquet has fewer rows than TRAIN_WINDOW + TEST_STEP\n"
            "  2. fwd_return_14d or Close columns are all NaN\n"
            "  3. Target column has only one class\n"
            "Run: python -c \"import pandas as pd; df=pd.read_parquet('data/processed/master_features.parquet'); print(df.shape); print(df[['Close','fwd_return_14d','fwd_direction_14d']].isnull().sum())\""
        )
        return

    results = compute_strategy_returns(results)
    summary = print_summary(results)

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_PATH)
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)
    logger.success(f"Saved backtest results → {RESULTS_PATH}")
    logger.success(f"Saved summary          → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()