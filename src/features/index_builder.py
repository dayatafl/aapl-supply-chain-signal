"""
index_builder.py
----------------
Constructs the daily Supplier Sentiment Index from scored headlines.

KEY DESIGN:
  - AAPL price history is the backbone (left join) — never truncated
  - Sentiment is merged in where available; filled with 0.0 (neutral) elsewhere
  - This means the model trains on years of price data even if sentiment
    only covers the last 30 days (NewsAPI) or is patchy (GDELT)

Input:  data/processed/headlines_scored.parquet
        data/processed/aapl_prices.parquet
Output: data/processed/master_features.parquet

Usage:
    python src/features/index_builder.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────
HEADLINES_PATH = Path("data/processed/headlines_scored.parquet")
PRICES_PATH    = Path("data/processed/aapl_prices.parquet")
OUTPUT_PATH    = Path("data/processed/master_features.parquet")

EWM_SPAN      = 5    # days — exponential smoothing span (shorter for 2026 window)
ZSCORE_WINDOW = 20   # days — rolling window for z-score (reduced for short history)
LAG_DAYS      = [3, 5, 7, 10]   # sentiment lags matching shorter forward horizons


# ── Sentiment index builder ───────────────────────────────────────────────
def build_daily_index(headlines: pd.DataFrame, supplier: str) -> pd.Series:
    """
    Build one daily sentiment series for a single supplier.
    Steps: daily mean → EWM smooth → rolling z-score.
    Returns a Series indexed by date (datetime).
    """
    sub = headlines[headlines["supplier"] == supplier].copy()
    if sub.empty:
        logger.warning(f"No headlines for supplier='{supplier}' — index will be all-zero")
        return pd.Series(dtype=float, name=f"{supplier}_sentiment_idx")

    sub["date"] = pd.to_datetime(sub["date"])
    daily = sub.groupby("date")["sentiment_score"].mean()

    # EWM smoothing — reduces noise from single-day spikes
    smoothed = daily.ewm(span=EWM_SPAN, adjust=False).mean()

    # Rolling z-score — makes series stationary and comparable across suppliers
    roll_mean = smoothed.rolling(ZSCORE_WINDOW, min_periods=5).mean()
    roll_std  = smoothed.rolling(ZSCORE_WINDOW, min_periods=5).std()
    normalised = (smoothed - roll_mean) / roll_std.replace(0, np.nan)

    normalised.name = f"{supplier}_sentiment_idx"
    logger.info(f"  {supplier}: {len(daily)} headline days, "
                f"{normalised.notna().sum()} non-null index values")
    return normalised


def build_lagged_features(
    df: pd.DataFrame,
    base_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of each sentiment index column."""
    for col in base_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}d"] = df[col].shift(lag)
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── Load inputs ───────────────────────────────────────────────────────
    for path in [HEADLINES_PATH, PRICES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the previous pipeline steps first:\n"
                f"  python src/data/collect_headlines.py\n"
                f"  python src/data/fetch_prices.py\n"
                f"  python src/features/sentiment_scorer.py"
            )

    headlines = pd.read_parquet(HEADLINES_PATH)
    prices    = pd.read_parquet(PRICES_PATH)

    logger.info(f"Headlines: {len(headlines)} rows, "
                f"suppliers={headlines['supplier'].unique().tolist()}, "
                f"date range={headlines['date'].min()} → {headlines['date'].max()}")
    logger.info(f"Prices:    {len(prices)} rows, "
                f"date range={prices.index.min().date()} → {prices.index.max().date()}")

    # ── Build per-supplier daily index ────────────────────────────────────
    suppliers = headlines["supplier"].unique().tolist()
    logger.info(f"Building sentiment indices for: {suppliers}")

    index_series = []
    for supplier in suppliers:
        idx = build_daily_index(headlines, supplier)
        index_series.append(idx)

    # Combine into a single DataFrame, indexed by date
    sentiment_df = pd.concat(index_series, axis=1)
    sentiment_df.index = pd.to_datetime(sentiment_df.index)

    # Composite = equal-weight average across all supplier indices
    supplier_cols = [f"{s}_sentiment_idx" for s in suppliers]
    sentiment_df["composite_sentiment_idx"] = sentiment_df[supplier_cols].mean(axis=1)

    logger.info(f"Sentiment index shape: {sentiment_df.shape}, "
                f"non-null rows: {sentiment_df.notna().any(axis=1).sum()}")

    # ── LEFT JOIN: price history is the backbone ──────────────────────────
    # This is the critical fix — left join keeps ALL price rows.
    # Sentiment columns will be NaN for dates with no headlines,
    # which we forward-fill (max 5 days) then fill remaining with 0.
    prices.index = pd.to_datetime(prices.index)
    master = prices.join(sentiment_df, how="left")

    # Forward-fill sentiment up to 5 trading days (weekend gaps, sparse days)
    sentiment_all_cols = supplier_cols + ["composite_sentiment_idx"]
    for col in sentiment_all_cols:
        if col in master.columns:
            master[col] = master[col].ffill(limit=5).fillna(0.0)
        else:
            master[col] = 0.0

    logger.info(f"After left join: {len(master)} rows "
                f"({master[sentiment_all_cols].ne(0).any(axis=1).sum()} rows with non-zero sentiment)")

    # ── Add lagged sentiment features ─────────────────────────────────────
    all_sentiment_cols = supplier_cols + ["composite_sentiment_idx"]
    master = build_lagged_features(master, all_sentiment_cols, LAG_DAYS)

    # Fill lagged columns too
    lag_cols = [f"{col}_lag{lag}d"
                for col in all_sentiment_cols
                for lag in LAG_DAYS
                if f"{col}_lag{lag}d" in master.columns]
    master[lag_cols] = master[lag_cols].fillna(0.0)

    # ── Drop rows with NaN targets (end of series only) ───────────────────
    # Do NOT drop rows with NaN sentiment — those are filled with 0 above
    before = len(master)
    master = master.dropna(subset=["fwd_direction_7d", "Close", "fwd_return_7d"])
    after  = len(master)
    logger.info(f"Dropped {before - after} rows with NaN target/price → {after} rows remaining")

    if after < 100:
        logger.warning(
            f"Only {after} rows in master features. "
            f"This is likely because aapl_prices.parquet is too short. "
            f"Re-run: python src/data/fetch_prices.py"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_PATH)
    logger.success(f"Saved master_features.parquet → shape={master.shape}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Column overview ──────────────────────────────────────────")
    print(f"  Total columns : {len(master.columns)}")
    print(f"  Total rows    : {len(master)}")
    print(f"  Date range    : {master.index.min().date()} → {master.index.max().date()}")
    print(f"  Sentiment coverage (non-zero rows):")
    for col in supplier_cols + ["composite_sentiment_idx"]:
        n = (master[col] != 0).sum()
        print(f"    {col:<35} {n} / {len(master)} rows  ({n/len(master):.1%})")
    print(f"  NaN in target columns:")
    for col in ["fwd_direction_7d", "fwd_return_7d", "Close"]:
        print(f"    {col:<35} {master[col].isna().sum()} NaN")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()