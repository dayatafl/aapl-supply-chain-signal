"""
forecast.py
-----------
Generates a forward forecast for AAPL using the trained model.
Predicts direction + probability for the next 7 trading days from today.

How it works:
  1. Loads the trained model and latest master_features row
  2. Predicts direction (up/down) and confidence for the next 7 trading days
  3. Builds a price scenario range (bull / base / bear) using recent volatility
  4. Saves forecast to data/processed/forecast.parquet

Output columns:
  date            — future trading date
  predicted_direction — 1=up, 0=down
  probability     — model confidence (prob of going up)
  scenario        — "bull" / "base" / "bear"
  price_low       — bear scenario price
  price_mid       — base scenario price
  price_high      — bull scenario price

Usage:
    python src/models/forecast.py
"""

import pickle
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH    = Path("models/xgb_model.pkl")
DATA_PATH     = Path("data/processed/master_features.parquet")
OUTPUT_PATH   = Path("data/processed/forecast.parquet")

FORECAST_DAYS = 7     # how many trading days ahead to forecast (matches model's 7d training target)
CONF_INTERVAL = 0.80    # 80% confidence band for price range


def next_trading_days(from_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """Return next N business days (Mon–Fri) from a given date."""
    dates = []
    current = from_date
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:   # 0=Mon ... 4=Fri
            dates.append(current)
    return dates


def compute_price_scenarios(
    last_price: float,
    daily_vol: float,
    horizon_days: int,
    prob_up: float,
    conf_interval: float,
) -> tuple[list, list, list]:
    """
    Build bull / base / bear price paths using random-walk with drift.

    Parameters
    ----------
    last_price   : last known AAPL closing price
    daily_vol    : historical daily return std (annualised ÷ sqrt(252))
    horizon_days : number of future trading days
    prob_up      : model's probability that the direction is up
    conf_interval: width of the confidence band (e.g. 0.80 = 80%)

    Returns
    -------
    price_low, price_mid, price_high  — lists of prices, one per day
    """
    # Drift: positive if model says up, negative if down
    # Scale drift by how confident the model is
    drift_sign  = 1 if prob_up >= 0.5 else -1
    confidence  = abs(prob_up - 0.5) * 2        # rescale 0.5–1.0 → 0–1.0
    annual_ret  = drift_sign * confidence * 0.15  # cap at ±15% annualised
    daily_drift = annual_ret / 252

    z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960}.get(conf_interval, 1.282)

    lows, mids, highs = [], [], []
    price = last_price

    for day in range(1, horizon_days + 1):
        # Base (expected path with drift)
        mid = last_price * np.exp(daily_drift * day)

        # Uncertainty grows with sqrt(time) — standard diffusion
        spread = last_price * daily_vol * np.sqrt(day) * z

        lows.append(round(mid - spread, 2))
        mids.append(round(mid, 2))
        highs.append(round(mid + spread, 2))

    return lows, mids, highs


def main():
    # ── Load model ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        logger.error("Model not found. Run: python src/models/train.py")
        return
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model    = artifact["model"]
    features = artifact["features"]

    # ── Load latest features ──────────────────────────────────────────────
    if not DATA_PATH.exists():
        logger.error("Master features not found. Run the full pipeline first.")
        return
    df = pd.read_parquet(DATA_PATH)

    # Fill any NaN features with 0 (neutral)
    df[features] = df[features].fillna(0.0)
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    last_row   = df.iloc[[-1]]
    last_date  = df.index[-1]
    last_price = float(df["Close"].iloc[-1])

    # ── Predict ───────────────────────────────────────────────────────────
    prob_up   = float(model.predict_proba(last_row[features])[0][1])
    direction = 1 if prob_up >= 0.5 else 0
    label     = "UP" if direction == 1 else "DOWN"
    confidence_pct = max(prob_up, 1 - prob_up) * 100

    logger.info(f"Last known date  : {last_date.date()}")
    logger.info(f"Last known price : ${last_price:.2f}")
    logger.info(f"Predicted 7d dir : {label}  (confidence: {confidence_pct:.1f}%)")
    logger.info(f"P(up) = {prob_up:.4f}   P(down) = {1 - prob_up:.4f}")

    # ── Compute historical volatility ─────────────────────────────────────
    returns    = df["Close"].pct_change().dropna()
    daily_vol  = float(returns.tail(20).std())  # 20-day realised vol
    logger.info(f"20-day daily vol : {daily_vol:.4f}  ({daily_vol * np.sqrt(252):.1%} annualised)")

    # ── Build future dates ────────────────────────────────────────────────
    future_dates = next_trading_days(last_date, FORECAST_DAYS)

    # ── Build price scenarios ─────────────────────────────────────────────
    lows, mids, highs = compute_price_scenarios(
        last_price, daily_vol, FORECAST_DAYS, prob_up, CONF_INTERVAL
    )

    # ── Assemble forecast DataFrame ───────────────────────────────────────
    forecast = pd.DataFrame({
        "date":                 future_dates,
        "day":                  list(range(1, FORECAST_DAYS + 1)),
        "predicted_direction":  [direction] * FORECAST_DAYS,
        "probability_up":       [round(prob_up, 4)] * FORECAST_DAYS,
        "probability_down":     [round(1 - prob_up, 4)] * FORECAST_DAYS,
        "price_bear":           lows,
        "price_base":           mids,
        "price_bull":           highs,
        "last_known_price":     [last_price] * FORECAST_DAYS,
        "last_known_date":      [last_date] * FORECAST_DAYS,
    })
    forecast = forecast.set_index("date")

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(OUTPUT_PATH)
    logger.success(f"Forecast saved → {OUTPUT_PATH}")

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print(f"  AAPL FORWARD FORECAST  ({last_date.date()} → +7 trading days) [MODEL TARGET]")
    print("=" * 58)
    print(f"  Last price   : ${last_price:.2f}")
    print(f"  Direction    : {label}  ({confidence_pct:.1f}% confidence)")
    print(f"  P(up)        : {prob_up:.1%}     P(down): {1-prob_up:.1%}")
    print(f"  Volatility   : {daily_vol * np.sqrt(252):.1%} annualised (20d)")
    print()
    print(f"  {'Date':<14}  {'Bear':>8}  {'Base':>8}  {'Bull':>8}  {'Change':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for d, row in forecast.iterrows():
        chg = (row["price_base"] / last_price - 1) * 100
        chg_str = f"{chg:+.2f}%"
        print(f"  {str(d.date()):<14}  ${row['price_bear']:>7.2f}  ${row['price_base']:>7.2f}  ${row['price_bull']:>7.2f}  {chg_str:>8}")
    print("=" * 58)
    print(f"  80% confidence band: price_bear to price_bull")
    print(f"  Base = expected path given model direction + drift")
    print("=" * 58)
    print()
    print("  DISCLAIMER: This is a model output, not financial advice.")
    print("  Short-window 2026 data = low statistical reliability.")
    print()


if __name__ == "__main__":
    main()