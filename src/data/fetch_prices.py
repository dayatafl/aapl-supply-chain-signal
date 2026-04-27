"""
fetch_prices.py
---------------
Downloads AAPL daily OHLCV data via yfinance.
Computes forward returns at multiple horizons.
Saves to data/processed/aapl_prices.parquet

Usage:
    python src/data/fetch_prices.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# ── Config ──────────────────────────────────────────────────────────────
TICKER = "AAPL"
START_DATE = "2016-01-01"
END_DATE = None                    # None = today
FORWARD_HORIZONS = [3, 5, 7, 10]  # shorter horizons to fit 2026-only window
OUTPUT_PATH = Path("data/processed/aapl_prices.parquet")


def compute_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add forward return columns and binary direction labels."""
    close = df["Close"]
    for h in horizons:
        ret_col = f"fwd_return_{h}d"
        dir_col = f"fwd_direction_{h}d"
        df[ret_col] = close.shift(-h) / close - 1
        df[dir_col] = (df[ret_col] > 0).astype(int)
    return df


def fetch_aapl(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    logger.info(f"Downloading {ticker} from {start} to {end or 'today'}")
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        raw = pd.DataFrame()

    if raw.empty:
        # Try to use cached data if available
        if OUTPUT_PATH.exists():
            logger.warning(f"Download failed. Using cached data from {OUTPUT_PATH}")
            return pd.read_parquet(OUTPUT_PATH)
        raise RuntimeError(f"No data returned for {ticker}. Check your internet connection or yfinance API status.")

    # Flatten multi-index columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    # Basic momentum feature
    raw["momentum_5d"] = raw["Close"].pct_change(5)
    raw["momentum_20d"] = raw["Close"].pct_change(20)

    raw = compute_forward_returns(raw, FORWARD_HORIZONS)

    logger.info(f"Downloaded {len(raw)} trading days")
    return raw


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_aapl(TICKER, START_DATE, END_DATE)
    df.to_parquet(OUTPUT_PATH)
    logger.success(f"Saved to {OUTPUT_PATH}  shape={df.shape}")
    print(df.tail())


if __name__ == "__main__":
    main()