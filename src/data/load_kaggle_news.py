"""
load_kaggle_news.py
-------------------
Ingests the Kaggle AAPL historical news dataset and normalises it
into the same format as collect_headlines.py output.

Dataset: "Apple Stock (AAPL): Historical Financial News Data"
Source : https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data
File   : apple_news_data.csv  (29,752 articles, 2016–2024)

Strategy:
  - ALL 29k articles are kept (they're all AAPL-related)
  - Supplier tag added per article:
      "tsmc"    → title/content mentions TSMC or Taiwan Semiconductor
      "samsung" → title/content mentions Samsung
      "apple"   → everything else (general AAPL news)
  - Existing sentiment columns are DROPPED — FinBERT will re-score
  - Optionally topped up with NewsAPI/GDELT for 2025–2026

Output: data/raw/headlines_raw.parquet  (same schema as collect_headlines.py)

Usage:
    # Basic — Kaggle CSV only
    python src/data/load_kaggle_news.py --csv data/raw/apple_news_data.csv

    # With NewsAPI top-up for recent 30 days
    python src/data/load_kaggle_news.py --csv data/raw/apple_news_data.csv --newsapi

    # With GDELT top-up for 2025-01-01 → today
    python src/data/load_kaggle_news.py --csv data/raw/apple_news_data.csv --gdelt --gdelt-start 2025-01-01
"""

import argparse
import os
import time
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

OUTPUT_PATH = Path("data/raw/headlines_raw.parquet")

# ── Supplier tagging rules ────────────────────────────────────────────────
SUPPLIER_RULES = {
    "tsmc":    ["tsmc", "taiwan semiconductor", "tsm "],
    "samsung": ["samsung"],
}


def tag_supplier(text: str) -> str:
    """Return 'tsmc', 'samsung', or 'apple' based on keyword presence."""
    text_lower = str(text).lower()
    for supplier, keywords in SUPPLIER_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return supplier
    return "apple"


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 1: KAGGLE CSV
# ══════════════════════════════════════════════════════════════════════════

def load_kaggle_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalise the Kaggle AAPL news CSV.

    Input columns used:
        date, title, content, link, symbols

    Dropped (will be re-scored by FinBERT):
        sentiment_polarity, sentiment_neg, sentiment_neu, sentiment_pos, tags
    """
    logger.info(f"Loading Kaggle CSV: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Download from: https://www.kaggle.com/datasets/frankossai/"
            "apple-stock-aapl-historical-financial-news-data\n"
            "Then place it at: data/raw/apple_news_data.csv"
        )

    raw = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  Raw shape: {raw.shape}")

    # ── Parse dates ───────────────────────────────────────────────────────
    raw["published_at"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["published_at"])
    raw["date_only"] = raw["published_at"].dt.date

    # ── Drop existing sentiment scores — FinBERT will replace them ────────
    drop_cols = ["sentiment_polarity","sentiment_neg","sentiment_neu","sentiment_pos","tags"]
    raw = raw.drop(columns=[c for c in drop_cols if c in raw.columns])

    # ── Clean text ────────────────────────────────────────────────────────
    raw["title"]   = raw["title"].fillna("").str.strip()
    raw["content"] = raw["content"].fillna("").str.strip()
    raw = raw[raw["title"].str.len() > 5]   # drop blank/stub titles

    # ── Tag supplier ──────────────────────────────────────────────────────
    # Use title + first 200 chars of content for tagging
    raw["combined_text"] = raw["title"] + " " + raw["content"].str[:200]
    raw["supplier"]      = raw["combined_text"].apply(tag_supplier)

    # ── Normalise to pipeline schema ──────────────────────────────────────
    df = pd.DataFrame({
        "source_api":   "kaggle",
        "supplier":     raw["supplier"],
        "query":        "kaggle_dataset",
        "published_at": raw["published_at"],
        "date":         raw["date_only"],
        "source":       raw["link"].apply(lambda x: _extract_domain(str(x))),
        "title":        raw["title"],
        "description":  raw["content"].str[:500],  # first 500 chars as description
        "url":          raw["link"].fillna(""),
        "language":     "ENGLISH",
    })

    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["title", "date"])
    df = df.sort_values("published_at").reset_index(drop=True)

    # ── Stats ─────────────────────────────────────────────────────────────
    logger.success(f"Kaggle CSV loaded: {len(df)} articles")
    logger.info(f"  Date range : {df['date'].min()} → {df['date'].max()}")
    logger.info(f"  By supplier:")
    for sup, count in df["supplier"].value_counts().items():
        logger.info(f"    {sup:<12} {count:>5} articles  ({count/len(df):.1%})")
    logger.info(f"  By year:")
    df["year"] = pd.to_datetime(df["published_at"]).dt.year
    for yr, count in df.groupby("year").size().items():
        logger.info(f"    {yr}  {count:>5}")
    df = df.drop(columns=["year"])

    return df


def _extract_domain(url: str) -> str:
    """Extract domain name from URL for source field."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url[:40]


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 2: NewsAPI top-up (last 30 days)
# ══════════════════════════════════════════════════════════════════════════

NEWSAPI_QUERIES = {
    "tsmc":    ["TSMC Apple chip", "Taiwan Semiconductor Apple"],
    "samsung": ["Samsung semiconductor Apple", "Samsung Apple supply"],
    "apple":   ["Apple AAPL stock earnings"],
}


def fetch_newsapi_topup(lookback_days: int = 29) -> pd.DataFrame:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI top-up.")
        return pd.DataFrame()
    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.warning("newsapi-python not installed: pip install newsapi-python")
        return pd.DataFrame()

    logger.info("Fetching NewsAPI top-up...")
    client     = NewsApiClient(api_key=api_key)
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    records    = []

    for supplier, queries in NEWSAPI_QUERIES.items():
        for query in queries:
            try:
                resp = client.get_everything(
                    q=query,
                    from_param=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    language="en",
                    sort_by="publishedAt",
                    page_size=100,
                )
                for art in resp.get("articles", []):
                    records.append({
                        "source_api":   "newsapi",
                        "supplier":     supplier,
                        "query":        query,
                        "published_at": pd.to_datetime(art["publishedAt"], utc=True),
                        "date":         pd.to_datetime(art["publishedAt"]).date(),
                        "source":       art["source"]["name"],
                        "title":        art.get("title", ""),
                        "description":  art.get("description", ""),
                        "url":          art.get("url", ""),
                        "language":     "ENGLISH",
                    })
                time.sleep(0.4)
            except Exception as e:
                logger.error(f"NewsAPI error '{query}': {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df[df["title"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["url"])
    logger.success(f"NewsAPI top-up: {len(df)} articles")
    return df


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 3: GDELT top-up (2025 → today)
# ══════════════════════════════════════════════════════════════════════════

GDELT_QUERIES = {
    "tsmc":    ["TSMC Apple chip", "Taiwan Semiconductor Apple"],
    "samsung": ["Samsung semiconductor Apple", "Samsung Apple supply"],
    "apple":   ["Apple AAPL stock earnings"],
}
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Setup a robust session with retries and backoff
def get_robust_session():
    session = requests.Session()
    # Retry strategy: 5 retries, waiting 2s, 4s, 8s... between attempts
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Browsers headers to avoid being flagged as a basic script
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session

def fetch_gdelt_topup(start_date: datetime, end_date: datetime, chunk_days: int = 7) -> pd.DataFrame:
    """
    Improved GDELT fetcher with robust retries and smaller chunking.
    Saves each chunk incrementally to output parquet to prevent data loss.
    """
    session = get_robust_session()
    logger.info(f"Fetching GDELT top-up: {start_date.date()} → {end_date.date()}")
    
    # Load existing parquet if it exists (from Kaggle)
    if OUTPUT_PATH.exists():
        existing_df = pd.read_parquet(OUTPUT_PATH)
        logger.info(f"  Existing parquet has {len(existing_df)} articles")
    else:
        existing_df = pd.DataFrame()
    
    total_gdelt_saved = 0

    for supplier, queries in GDELT_QUERIES.items():
        for query in queries:
            chunk_start = start_date
            while chunk_start < end_date:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)
                params = {
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": 250,
                    "startdatetime": chunk_start.strftime("%Y%m%d") + "000000",
                    "enddatetime": chunk_end.strftime("%Y%m%d") + "235959",
                    "format": "json",
                    "sort": "DateDesc",
                }
                
                try:
                    # INCREASED TIMEOUT: Historical GDELT queries often need >30s
                    r = session.get(GDELT_URL, params=params, timeout=90)
                    r.raise_for_status() # Raise error for bad status codes
                    
                    data = r.json()
                    
                    # Handle different possible response structures
                    if isinstance(data, dict) and "articles" in data:
                        articles = data.get("articles", [])
                    elif isinstance(data, list):
                        articles = data
                    else:
                        logger.warning(f"  Unexpected GDELT response structure: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                        articles = []
                    
                    logger.info(f"  GDELT [{supplier}] '{query}' {chunk_start.date()}→{chunk_end.date()}: Got {len(articles)} articles")
                    
                    if articles:
                        records = []
                        # Debug: check first article structure
                        if records == [] and len(articles) > 0:
                            logger.debug(f"  First article keys: {list(articles[0].keys())}")
                        
                        for art in articles:
                            try:
                                # Debug first article date format
                                seendate_raw = art.get("seendate", "")
                                if not records and len(articles) > 0:
                                    logger.debug(f"    First article seendate value: '{seendate_raw}' (type: {type(seendate_raw).__name__})")
                                
                                pub = _parse_gdelt_date(seendate_raw)
                                if pub is None:
                                    logger.debug(f"    Failed to parse date: '{seendate_raw}'")
                                    continue
                                
                                # Check language — only keep English articles
                                article_language = art.get("language", "").upper()
                                if article_language and article_language not in ("EN", "ENG", "ENGLISH"):
                                    continue
                                
                                # Try multiple field names for title/URL
                                title = art.get("title") or art.get("name") or art.get("headline") or ""
                                url = art.get("url") or art.get("weblink") or ""
                                domain = art.get("domain") or art.get("source") or ""
                                
                                if not url or not title:
                                    logger.debug(f"    Skipped: title='{title[:20] if title else 'EMPTY'}', url='{url[:20] if url else 'EMPTY'}'")
                                    continue
                                
                                records.append({
                                    "source_api": "gdelt",
                                    "supplier": supplier,
                                    "query": query,
                                    "published_at": pub,
                                    "date": pub.date(),
                                    "source": domain,
                                    "title": title,
                                    "description": "",
                                    "url": url,
                                    "language": "ENGLISH",
                                })
                            except KeyError as ke:
                                logger.debug(f"    KeyError on article: {ke}, keys: {list(art.keys())}")
                                continue
                        
                        logger.info(f"    Processed {len(records)} records from {len(articles)} articles")
                        
                        if not records:
                            logger.warning(f"    → No valid records extracted from {len(articles)} articles")
                            continue
                        
                        chunk_df = pd.DataFrame(records)
                        
                        chunk_df = chunk_df[chunk_df["title"].str.strip().astype(bool)]
                        chunk_df = chunk_df.drop_duplicates(subset=["url"])
                        chunk_df = chunk_df.drop_duplicates(subset=["title", "date"])
                        
                        # Remove duplicates with existing data
                        chunk_df = chunk_df[~chunk_df["url"].isin(existing_df["url"])]
                        chunk_df = chunk_df[~chunk_df[["title", "date"]].apply(tuple, axis=1).isin(
                            existing_df[["title", "date"]].apply(tuple, axis=1)
                        )]
                        
                        if not chunk_df.empty:
                            # Append chunk to parquet file
                            existing_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                            existing_df = existing_df.drop_duplicates(subset=["url"], keep='first')
                            existing_df = existing_df.sort_values("published_at").reset_index(drop=True)
                            
                            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                            existing_df.to_parquet(OUTPUT_PATH, index=False)
                            
                            total_gdelt_saved += len(chunk_df)
                            logger.success(f"    ✓ Saved {len(chunk_df)} articles (total in parquet: {len(existing_df)})")
                        else:
                            logger.info(f"    → All {len(records)} records were duplicates or empty")
                    
                except requests.exceptions.Timeout:
                    logger.error(f"  GDELT Timeout on {query} ({chunk_start.date()}). Server is likely overloaded.")
                except KeyError as ke:
                    logger.error(f"  GDELT KeyError on {query} ({chunk_start.date()}): Missing field '{ke}'")
                except Exception as e:
                    logger.error(f"  GDELT error on {query} ({chunk_start.date()}): {type(e).__name__}: {e}")

                # Move to next chunk
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(2) # Polite delay
    
    logger.success(f"GDELT top-up complete: {total_gdelt_saved} new articles saved incrementally")
    return existing_df


def _parse_gdelt_date(s: str):
    """Parse GDELT seendate field. Format: 20241209T234500Z (YYYYMMDDThhmmssZ)"""
    if not s or s == "":
        return None
    
    s_str = str(s).strip()
    
    # GDELT uses ISO 8601 format with T separator: 20241209T234500Z
    # Remove the Z suffix if present
    s_clean = s_str.rstrip('Z')
    
    try:
        # Try full format: 20241209T234500
        if 'T' in s_clean:
            return pd.Timestamp(datetime.strptime(s_clean, "%Y%m%dT%H%M%S"), tz="UTC")
    except (ValueError, TypeError):
        pass
    
    # Fallback: try other formats
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            parsed = datetime.strptime(s_clean[:len(fmt)//4*8], fmt) if len(fmt)//4*8 <= len(s_clean) else None
            if parsed:
                return pd.Timestamp(parsed, tz="UTC")
        except (ValueError, TypeError):
            pass
    
    return None


# ══════════════════════════════════════════════════════════════════════════
# MERGE & SAVE
# ══════════════════════════════════════════════════════════════════════════

def merge_and_save(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        logger.error("All sources empty — nothing to save.")
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)
    before   = len(combined)
    
    logger.info(f"Combined before final dedup: {combined['source_api'].value_counts().to_dict()}")

    combined = combined[combined["title"].str.strip().astype(bool)]
    logger.info(f"  After removing blank titles: {len(combined)} (removed {before - len(combined)})")
    before_url = len(combined)
    
    combined = combined.drop_duplicates(subset=["url"], keep='first')
    logger.info(f"  After URL dedup: {len(combined)} (removed {before_url - len(combined)})")
    before_title = len(combined)
    
    combined = combined.drop_duplicates(subset=["title", "date"], keep='first')
    logger.info(f"  After title+date dedup: {len(combined)} (removed {before_title - len(combined)})")
    
    combined = combined.sort_values("published_at").reset_index(drop=True)
    after    = len(combined)

    logger.info(f"Final consolidated: {before} → {after} total articles")
    logger.info("By source:")
    for src, n in combined["source_api"].value_counts().items():
        logger.info(f"  {src:<12} {n:>6} articles")
    logger.info("By supplier:")
    for sup, n in combined["supplier"].value_counts().items():
        logger.info(f"  {sup:<12} {n:>6} articles")
    logger.info(f"Date range: {combined['date'].min()} → {combined['date'].max()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.success(f"Final consolidated {len(combined)} headlines → {OUTPUT_PATH}")
    return combined


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Load Kaggle AAPL news + optional top-ups")
    p.add_argument("--csv",          default="data/raw/apple_news_data.csv",
                   help="Path to the Kaggle CSV file")
    p.add_argument("--newsapi",      action="store_true",
                   help="Add NewsAPI top-up for last 30 days")
    p.add_argument("--gdelt",        action="store_true",
                   help="Add GDELT top-up for gap between Kaggle end and today")
    p.add_argument("--gdelt-start",  default="2025-01-01",
                   help="GDELT top-up start date (default: 2025-01-01)")
    return p.parse_args()


def main():
    args = parse_args()
    frames = []

    # ── 1. Kaggle CSV (primary source) ────────────────────────────────────
    kaggle_df = load_kaggle_csv(Path(args.csv))
    frames.append(kaggle_df)

    # ── 2. GDELT top-up (saves incrementally to parquet) ─────────────────
    if args.gdelt:
        gdelt_start = datetime.strptime(args.gdelt_start, "%Y-%m-%d")
        gdelt_end   = datetime.today()
        gdelt_df    = fetch_gdelt_topup(gdelt_start, gdelt_end)
        # GDELT is already saved to parquet during fetch, just track it
        frames.append(gdelt_df)
    else:
        logger.info("Skipping GDELT top-up (add --gdelt to enable)")

    # ── 3. NewsAPI top-up (last 30 days, highest quality recent news) ─────
    if args.newsapi:
        newsapi_df = fetch_newsapi_topup()
        frames.append(newsapi_df)
    else:
        logger.info("Skipping NewsAPI top-up (add --newsapi to enable)")

    # ── Merge and save (final consolidation) ───────────────────────────────
    merge_and_save(frames)

    print()
    print("Next step:")
    print("  python src/features/sentiment_scorer.py")
    print("  (FinBERT will score all headlines — existing scores ignored)")


if __name__ == "__main__":
    main()