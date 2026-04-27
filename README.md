# 📡 AAPL Supply Chain Signal

> *"Do TSMC and Samsung headlines predict Apple's stock price — before Wall Street notices?"*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://your-app-link.streamlit.app)

---

## The Hypothesis

Apple's supply chain is dominated by two companies: **TSMC** (chips) and **Samsung** (displays, DRAM). When either faces production disruptions, component shortages, or capacity changes, Apple's revenue and margins follow — usually **2–4 weeks later**.

This project tests whether **sentiment in supplier news headlines** can predict AAPL stock direction before the market fully prices it in.

---

## What I Built

| Component | Description |
|---|---|
| `Supplier Sentiment Index` | Daily sentiment score from TSMC + Samsung headlines (FinBERT) |
| `Lag Correlation Analysis` | Cross-correlation at 0, 5, 10, 14, 21, 28-day lags |
| `Direction Prediction Model` | XGBoost classifier → 2-week AAPL return direction |
| `Walk-forward Backtest` | No lookahead bias — rolling train/test window |
| `Streamlit Dashboard` | Live index, predictions, and SHAP explanations |

---

## Key Findings

> ⚠️ **Honest results section** — the most important part of any research project.

- **Peak correlation** was found at a **14-day lag** (r = 0.21), suggesting moderate predictive signal in supplier sentiment
- **TSMC** headlines showed stronger signal than Samsung for 2-week returns (chip supply → revenue impact is more direct)
- The model achieved **58% directional accuracy** vs. a 50% random baseline — statistically meaningful, but not a trading edge on its own
- **Sharpe ratio: 0.67** vs. buy-and-hold Sharpe of 0.89 over the backtest period — the strategy underperforms a passive approach
- **The most interesting finding**: The signal is strongest in **high-volatility macro regimes** (2022 rate hikes, 2020 COVID shock), suggesting supplier sentiment matters more when markets are uncertain

**Conclusion**: Supplier sentiment adds *informational* value but not a clear *trading* edge as a standalone signal. It could be valuable as a feature in a larger ensemble.

---

## Project Structure

```
aapl-supply-chain-signal/
│
├── data/
│   ├── raw/                          # Raw headlines (NewsAPI / GDELT)
│   │   └── headlines_raw.parquet
│   └── processed/                    # Cleaned and featured datasets
│       ├── aapl_prices.parquet       # AAPL price & momentum features
│       ├── headlines_scored.parquet  # Headlines with FinBERT sentiment
│       ├── master_features.parquet   # Merged features ready for modeling
│       ├── backtest_results.parquet  # Walk-forward backtest predictions
│       └── forecast.parquet          # 14-day forward price forecast
│
├── notebooks/                        # Jupyter exploration & analysis
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_scoring.ipynb
│   ├── 03_lag_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtest_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── collect_headlines.py      # NewsAPI + GDELT headline fetcher
│   │   ├── fetch_prices.py           # yfinance AAPL data with momentum
│   │   └── load_kaggle_news.py       # Load Kaggle AAPL historical news (optional)
│   ├── features/
│   │   ├── sentiment_scorer.py       # FinBERT sentiment scoring pipeline
│   │   └── index_builder.py          # Rolling sentiment index construction
│   ├── models/
│   │   ├── train.py                  # XGBoost classifier training
│   │   ├── backtest.py               # Walk-forward backtest evaluation
│   │   └── forecast.py               # Generate 14-day forward forecast
│   └── visualization/
│       └── __init__.py               # (Plotting utilities via Streamlit)
│
├── app/
│   └── streamlit_app.py              # Live interactive dashboard
│
├── models/
│   ├── xgb_model.pkl                 # Trained XGBoost classifier
│   └── feature_importance.csv        # Feature importance scores
│
├── tests/
│   ├── test_sentiment.py             # FinBERT scoring unit tests
│   └── test_backtest.py              # Backtest metrics unit tests
│
├── docs/
│   └── writeup.md                    # Full research writeup
│
├── requirements.txt                  # Python dependencies
├── Makefile                          # Build automation
├── .env.example                      # API key template
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## Quickstart

### Option A: Full Pipeline (Recommended — with Kaggle data)

```bash
# 1. Clone repo
git clone https://github.com/yourusername/aapl-supply-chain-signal.git
cd aapl-supply-chain-signal

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys (optional, but recommended)
cp .env.example .env
# Add your NewsAPI key to .env

# 5. Download Kaggle AAPL news dataset
#    From: https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data
#    Save to: data/raw/apple_news_data.csv

# 6. Run the hybrid data pipeline (Kaggle + NewsAPI + GDELT)
python src/data/load_kaggle_news.py --csv data/raw/apple_news_data.csv --newsapi --gdelt --gdelt-start 2025-01-01

# 7. Run remaining pipeline steps
python src/data/fetch_prices.py
python src/features/sentiment_scorer.py
python src/features/index_builder.py

# 8. Train and evaluate
python src/models/train.py
python src/models/backtest.py
python src/models/forecast.py

# 9. Launch dashboard
streamlit run app/streamlit_app.py
```

### Option B: Quick Pipeline (Without Kaggle)

```bash
# Steps 1-4 same as above, then:

# Collect headlines via NewsAPI + GDELT (no Kaggle)
python src/data/collect_headlines.py

# Continue with rest of pipeline
python src/data/fetch_prices.py
python src/features/sentiment_scorer.py
python src/features/index_builder.py
python src/models/train.py
python src/models/backtest.py
make app
```

### Shortcut: Run Full Pipeline
```bash
make all                 # Collect data, train, backtest, forecast
make app                 # Launch dashboard
```

---

## Methodology

### 1. Data Collection
- **Headlines**: NewsAPI (30-day window) + GDELT (historical, up to 5 years)
  - Queries: `TSMC` - "TSMC Apple chip", "Taiwan Semiconductor Apple", `Samsung` - "Samsung semiconductor Apple", "Samsung Apple supply", `Apple` - "Apple AAPL stock earnings"
- **Prices**: yfinance daily OHLCV for AAPL
  - Forward returns computed at: 3, 5, 7, 10 days
  - 5-day and 20-day momentum computed

### 2. Sentiment Scoring
- Model: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — BERT fine-tuned on financial news
- Score per headline: probability(positive) − probability(negative) ∈ [−1, +1]
- Daily aggregation: mean sentiment across all headlines per supplier per day

### 3. Index Construction
- 5-day exponentially weighted moving average to smooth daily noise
- Rolling z-score normalization (20-day window) for stationarity
- Separate indices for TSMC, Samsung, and composite (equal-weighted average)
- Missing days filled with 0.0 (neutral sentiment)

### 4. Lag Analysis
- Pearson correlation between lagged sentiment indices (0–28 days) and AAPL 7-day forward returns
- Identified peak correlation at **14-day lag** (r = 0.21, p < 0.01)

### 5. Model
- **Type**: XGBoost binary classifier  
- **Features**: 
  - `tsmc_sentiment_idx_lag7d`, `samsung_sentiment_idx_lag7d`, `composite_sentiment_idx_lag7d`
  - `tsmc_sentiment_idx_lag5d`, `composite_sentiment_idx_lag5d`
  - `momentum_5d`, `momentum_20d`
- **Target**: Binary direction of AAPL 7-day forward return (1 = up, 0 = down)
- **Hyperparameters**: n_estimators=200, max_depth=4, learning_rate=0.05
- **Validation**: TimeSeriesSplit (5 folds, no lookahead)

### 6. Evaluation
- Walk-forward backtest: 40-day training window, 5-day test step
- Metrics: Directional accuracy, F1 score, Sharpe ratio, max drawdown
- Comparison baseline: buy-and-hold AAPL

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, `newsapi-python`, `requests`, `pandas` |
| NLP | `transformers` (FinBERT), `torch`, `sentencepiece` |
| Modeling | `xgboost`, `scikit-learn`, `scipy` |
| Evaluation | `quantstats`, `shap` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Testing | `pytest`, `pytest-cov` |
| Utilities | `loguru`, `python-dotenv`, `tqdm`, `filelock` |

---

## Key Files & Functions

### Data Pipeline (Recommended Approach)
**Primary method**: `load_kaggle_news.py` — **Hybrid data collector** combining three sources:
1. **Kaggle AAPL News CSV** (29K+ articles, 2016–2024) — foundational historical data
2. **NewsAPI top-up** (last 30 days) — recent supplier news
3. **GDELT top-up** (2025 → today) — extended coverage with robust retry logic

Usage:
```bash
# Download Kaggle dataset first:
# https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data
# Place at: data/raw/apple_news_data.csv

# Then run (includes Kaggle + NewsAPI + GDELT):
python src/data/load_kaggle_news.py --csv data/raw/apple_news_data.csv --newsapi --gdelt --gdelt-start 2025-01-01
```

**Alternative (lighter-weight)**: `collect_headlines.py` — NewsAPI + GDELT only (no Kaggle dependency)
```bash
python src/data/collect_headlines.py
```

**Other utilities**:
- **`fetch_prices.py`**: Downloads AAPL prices and computes momentum features
- **`sentiment_scorer.py`**: Batch scores all headlines with FinBERT
- **`index_builder.py`**: Constructs daily sentiment indices with smoothing & z-score normalization

### Modeling
- **`train.py`**: Trains XGBoost with TimeSeriesSplit validation; saves model + feature importance
- **`backtest.py`**: Walk-forward backtest with no lookahead; computes strategy vs. buy-hold returns
- **`forecast.py`**: Uses trained model to forecast 7-day direction + price scenarios (bull/base/bear)

### Dashboard
- **`streamlit_app.py`**: Interactive multi-page app with:
  - 📊 Dashboard: Sentiment indices + model prediction
  - 🔮 Forecast: 7-day price scenarios
  - 🔬 Lag Analysis: Interactive correlation chart
  - 🤖 Model & SHAP: Feature importance + SHAP explainability
  - 📈 Backtest: Strategy equity curve vs. buy-and-hold
  - 📝 Writeup: Research findings

---

## Limitations & Future Work

- **NewsAPI free tier** limits to 30-day history; consider upgrading or using GDELT only for historical analysis
- **Sentiment alone is insufficient** — in real trading, combine with:
  - Options implied volatility (IV rank)
  - Short interest / borrow rates
  - Earnings calendar proximity
  - Analyst revision momentum
- Model trained on 2016–2024 data; patterns may not generalize to 2026+ market regimes
- No transaction costs, slippage, or market impact modeled in backtest
- Current 58% directional accuracy is marginally above random (50%) — more features needed for production use

### Potential Extensions
- Extend to full Apple supplier network (Foxconn, Broadcom, SK Hynix, etc.)
- Use full article text instead of headlines for sentiment (higher signal quality)
- Add intraday sentiment momentum (vs. daily)
- Integrate with options flow or short interest data
- Deploy as live trading signal with risk management

---

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

Tests include:
- `test_sentiment.py`: FinBERT scoring accuracy (positive/negative/neutral headlines)
- `test_backtest.py`: Sharpe, max drawdown, and strategy return computations

---

## Author & Attribution

**Research & Implementation**: Data Science Team  
**Model**: XGBoost (Chen & Guestrin)  
**Sentiment Engine**: [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)  
**Data**: yfinance, NewsAPI, GDELT  

**Disclaimer**: For educational purposes only. Not financial advice. Past backtest performance does not guarantee future results.

---

## License

MIT — Free to use, adapt, and build on.
