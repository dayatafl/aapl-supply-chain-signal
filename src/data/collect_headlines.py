"""
python src/data/collect_headlines.py
--------------------
Hybrid pipeline:
  - GDELT (historical, run once)
  - NewsAPI (incremental updates)

Safe against rate limits + scalable.
"""

import argparse
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

OUTPUT_PATH = Path("data/raw/headlines_raw.parquet")

QUERIES = {
    "tsmc": [
        "(TSMC OR \"Taiwan Semiconductor\") AND (chip OR Apple OR production)"
    ],
    "samsung": [
        "(Samsung semiconductor OR Samsung foundry OR Samsung DRAM)"
    ],
}

# ─────────────────────────────────────────
# GDELT CONFIG
# ─────────────────────────────────────────
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250
GDELT_SLEEP = 15


def _gdelt_fetch_one(query: str, start: str, end: str, max_retries=3):
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": GDELT_MAX_RECORDS,
        "startdatetime": f"{start}000000",
        "enddatetime": f"{end}235959",
        "format": "json",
        "sort": "DateDesc",
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=30)

            if r.status_code == 429:
                wait = (2 ** attempt) * 15
                logger.warning(f"429 hit → sleeping {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json().get("articles", [])

        except Exception as e:
            logger.error(f"GDELT error '{query}': {e}")
            time.sleep(2 ** attempt * 5)

    return []

CHECKPOINT_PATH = Path("data/raw/gdelt_checkpoint.parquet")
def fetch_gdelt(queries, start_date, end_date, chunk_days=14):
    logger.info("=== GDELT FETCH (HISTORICAL) ===")

    records = []
    resume_from = start_date

    # Load existing checkpoint
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint_df = pd.read_parquet(CHECKPOINT_PATH)
            if not checkpoint_df.empty:
                records = checkpoint_df.to_dict("records")
                last_date = pd.to_datetime(checkpoint_df["published_at"]).max()
                resume_from = last_date.date() + timedelta(days=1)
                logger.info(
                    f"Resuming from checkpoint: {resume_from} ({len(records)} records already saved)"
                )
        except Exception as e:
            logger.warning(f"Could not load checkpoint, starting fresh: {e}")

    for supplier, query_list in queries.items():
        for query in query_list:
            chunk_start = resume_from

            while chunk_start < end_date:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)

                s = chunk_start.strftime("%Y%m%d")
                e = chunk_end.strftime("%Y%m%d")

                
                logger.info(f"[{supplier}] {s} → {e}")
                articles = _gdelt_fetch_one(query, s, e)

                for art in articles:
                    records.append({
                        "source_api": "gdelt",
                        "supplier": supplier,
                        "query": query,
                        "published_at": art.get("seendate"),
                        "source": art.get("domain"),
                        "title": art.get("title"),
                        "description": "",
                        "url": art.get("url"),
                        "language": art.get("language"),
                    })

                chunk_start = chunk_end + timedelta(days=1)

                time.sleep(GDELT_SLEEP + random.uniform(1, 3))

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])
    df["date"] = df["published_at"].dt.date

    return df


# ─────────────────────────────────────────
# NEWSAPI
# ─────────────────────────────────────────
def fetch_newsapi(queries, lookback_days=29):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("Missing NEWSAPI_KEY")
        return pd.DataFrame()

    from newsapi import NewsApiClient

    client = NewsApiClient(api_key=api_key)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    records = []

    for supplier, query_list in queries.items():
        for query in query_list:
            logger.info(f"[NewsAPI] {supplier}")

            try:
                res = client.get_everything(
                    q=query,
                    from_param=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    language="en",
                    sort_by="publishedAt",
                    page_size=100,
                )

                for art in res.get("articles", []):
                    records.append({
                        "source_api": "newsapi",
                        "supplier": supplier,
                        "query": query,
                        "published_at": art["publishedAt"],
                        "source": art["source"]["name"],
                        "title": art["title"],
                        "description": art.get("description", ""),
                        "url": art["url"],
                        "language": "ENGLISH",
                    })

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"NewsAPI error: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["date"] = df["published_at"].dt.date

    return df


# ─────────────────────────────────────────
# MERGE + CLEAN (FIXED)
# ─────────────────────────────────────────
def merge_and_clean(new_df):
    if new_df.empty:
        return new_df

    if OUTPUT_PATH.exists():
        old_df = pd.read_parquet(OUTPUT_PATH)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    before = len(df)

    # ✅ CRITICAL FIX: remove NaN + empty rows properly
    df = df.dropna(subset=["title", "url", "published_at"])

    df["title"] = df["title"].astype(str)
    df = df[df["title"].str.strip() != ""]

    # Dedup
    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["title", "date"])

    # Sort
    df = df.sort_values("published_at").reset_index(drop=True)

    logger.info(f"Dedup: {before} → {len(df)}")

    return df


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gdelt", action="store_true")
    args = parser.parse_args()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=3 * 365)

    frames = []

    # Only run GDELT once
    if not OUTPUT_PATH.exists() and not args.skip_gdelt:
        gdelt_df = fetch_gdelt(QUERIES, start_date, end_date)
        frames.append(gdelt_df)
    else:
        logger.info("Skipping GDELT (already collected)")

    # Always run NewsAPI
    news_df = fetch_newsapi(QUERIES)
    frames.append(news_df)

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    final_df = merge_and_clean(combined)

    if final_df.empty:
        logger.error("No data collected")
        return

    # ✅ Final safety check (optional but recommended)
    blank_rows = final_df[
        final_df["title"].isna() | (final_df["title"].str.strip() == "")
    ]

    if not blank_rows.empty:
        logger.warning(f"Found {len(blank_rows)} blank rows before saving!")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)

    logger.success(f"Saved {len(final_df)} articles")


if __name__ == "__main__":
    main()