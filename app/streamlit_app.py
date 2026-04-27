"""
streamlit_app.py
----------------
Live dashboard for the AAPL Supply Chain Signal project.
Shows: sentiment index, lag correlation, model prediction, backtest results.

Run:
    streamlit run app/streamlit_app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Supply Chain Signal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .finding-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .info-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH     = Path("data/processed/master_features.parquet")
BACKTEST_PATH = Path("data/processed/backtest_results.parquet")
MODEL_PATH    = Path("models/xgb_model.pkl")
FORECAST_PATH = Path("data/processed/forecast.parquet")

SENTIMENT_COLS = ["tsmc_sentiment_idx", "samsung_sentiment_idx", "composite_sentiment_idx"]

FEATURES = [
    "tsmc_sentiment_idx_lag7d",
    "samsung_sentiment_idx_lag7d",
    "composite_sentiment_idx_lag7d",
    "tsmc_sentiment_idx_lag5d",
    "composite_sentiment_idx_lag5d",
    "momentum_5d",
    "momentum_20d",
]


# ── Helpers ───────────────────────────────────────────────────────────────
def safe_fillna(df: pd.DataFrame, cols: list, fill=0.0) -> pd.DataFrame:
    """Fill NaN in specified columns; add column with fill value if missing."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = fill
        else:
            df[col] = df[col].fillna(fill)
    return df


@st.cache_data
def load_forecast():
    if not FORECAST_PATH.exists():
        return None
    return pd.read_parquet(FORECAST_PATH)


def sharpe(returns, ann=252):
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(ann))


def max_dd(equity):
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(dd.min())


def sentiment_available(df: pd.DataFrame) -> bool:
    """Return True if at least one sentiment column has real (non-zero) data."""
    for col in SENTIMENT_COLS:
        if col in df.columns and (df[col] != 0).sum() > 10:
            return True
    return False


# ── Data Loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_parquet(DATA_PATH)
    df = safe_fillna(df, FEATURES + SENTIMENT_COLS)
    return df


@st.cache_data
def load_backtest():
    if not BACKTEST_PATH.exists():
        return None
    return pd.read_parquet(BACKTEST_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["features"]


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Supply Chain Signal")
    st.caption("AAPL · TSMC · Samsung")
    st.divider()
    st.markdown("**Hypothesis**")
    st.info("Supplier headline sentiment predicts AAPL returns 7 trading days in advance.")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Forecast", "🔬 Lag Analysis", "🤖 Model & SHAP", "📈 Backtest", "📝 Research Writeup"],
    )

# ── Load data ────────────────────────────────────────────────────────────
df = load_data()
bt = load_backtest()
model, features = load_model()
fc = load_forecast()

data_ready     = df is not None and len(df) > 0
backtest_ready = bt is not None and len(bt) > 0
model_ready    = model is not None
forecast_ready = fc is not None and len(fc) > 0
has_sentiment  = data_ready and sentiment_available(df)


# ════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📡 AAPL Supply Chain Signal")
    st.markdown("*Testing whether TSMC & Samsung headlines predict Apple stock 2 weeks ahead.*")

    if not data_ready:
        st.warning("⚠️ No data found. Run the full pipeline first.")
        st.code(
            "python src/data/collect_headlines.py\n"
            "python src/data/fetch_prices.py\n"
            "python src/features/sentiment_scorer.py\n"
            "python src/features/index_builder.py\n"
            "python src/models/train.py\n"
            "python src/models/backtest.py"
        )
        st.stop()

    if not has_sentiment:
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>Sentiment data is sparse or missing</b> — NewsAPI free tier only returns the last
        30 days of headlines, so lagged sentiment features may be mostly empty. Price-based
        momentum features are still active. Sentiment columns are filled with 0 (neutral) where missing.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Key metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    latest = df.iloc[-1]

    with col1:
        val = float(latest.get("composite_sentiment_idx", 0.0))
        st.metric("Composite Index", f"{val:.2f}", delta="z-score")
    with col2:
        val = float(latest.get("tsmc_sentiment_idx", 0.0))
        st.metric("TSMC Sentiment", f"{val:.2f}")
    with col3:
        val = float(latest.get("samsung_sentiment_idx", 0.0))
        st.metric("Samsung Sentiment", f"{val:.2f}")
    with col4:
        if model_ready:
            try:
                row = df[features].iloc[[-1]]
                prob = model.predict_proba(row)[0][1]
                st.metric("7d Direction Prob.", f"{prob:.1%}")
            except Exception:
                st.metric("7d Direction Prob.", "N/A")
        else:
            st.metric("Model", "Not trained")

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────
    st.subheader("AAPL Price" + (" & Supplier Sentiment Index" if has_sentiment else ""))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="AAPL Close", line=dict(color="#94a3b8", width=1.5),
        yaxis="y2", opacity=0.8,
    ))

    if has_sentiment:
        colors = {
            "tsmc_sentiment_idx": "#2563eb",
            "samsung_sentiment_idx": "#16a34a",
            "composite_sentiment_idx": "#dc2626",
        }
        labels = {
            "tsmc_sentiment_idx": "TSMC Index",
            "samsung_sentiment_idx": "Samsung Index",
            "composite_sentiment_idx": "Composite Index",
        }
        for col, color in colors.items():
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    name=labels[col], line=dict(color=color, width=2),
                ))

    fig.update_layout(
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        yaxis=dict(title="Sentiment (z-score)" if has_sentiment else ""),
        yaxis2=dict(title="AAPL Price (USD)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Dataset info"):
        st.write(f"**Rows:** {len(df)}  |  **Date range:** {df.index.min().date()} → {df.index.max().date()}")
        key_cols = [c for c in SENTIMENT_COLS + ["Close", "momentum_5d", "fwd_direction_7d"] if c in df.columns]
        nn = df[key_cols].notnull().sum().rename("non-null rows")
        st.dataframe(nn.to_frame())


# ════════════════════════════════════════════════════════════════════════
# PAGE 2: FORECAST
# ════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 7-Day Forward Forecast")
    st.markdown("*What does the model predict for the next 7 trading days?*")

    if not forecast_ready:
        st.warning("No forecast found. Run: `python src/models/forecast.py`")
        st.code("python src/models/forecast.py")
        st.stop()

    if not data_ready:
        st.warning("No feature data found.")
        st.stop()

    row0          = fc.iloc[0]
    last_price    = float(row0["last_known_price"])
    last_date     = pd.Timestamp(row0["last_known_date"])
    prob_up       = float(row0["probability_up"])
    prob_down     = float(row0["probability_down"])
    direction     = int(row0["predicted_direction"])
    dir_label     = "UP" if direction == 1 else "DOWN"
    confidence    = max(prob_up, prob_down) * 100
    base_end      = float(fc["price_base"].iloc[-1])
    expected_chg  = (base_end / last_price - 1) * 100

    # ── Hero metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last known price",  f"${last_price:.2f}", delta=f"as of {last_date.date()}")
    col2.metric("Predicted direction", dir_label, delta=f"{confidence:.1f}% confidence")
    col3.metric("P(up) / P(down)",   f"{prob_up:.1%} / {prob_down:.1%}")
    col4.metric("Expected 7d move",  f"{expected_chg:+.2f}%", delta="base scenario")

    st.divider()

    # ── Forecast chart ────────────────────────────────────────────────────
    st.subheader("Price scenario chart")

    # Historical price (last 20 days)
    hist = df["Close"].tail(20) if data_ready else pd.Series(dtype=float)
    hist_dates  = hist.index.tolist()
    hist_prices = hist.values.tolist()

    # Anchor point — connect history to forecast
    anchor_dates  = [last_date] + fc.index.tolist()
    anchor_mid    = [last_price] + fc["price_base"].tolist()
    anchor_bull   = [last_price] + fc["price_bull"].tolist()
    anchor_bear   = [last_price] + fc["price_bear"].tolist()

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        name="AAPL (historical)",
        line=dict(color="#378ADD", width=2),
        mode="lines",
    ))

    # Confidence band (bear to bull)
    fig.add_trace(go.Scatter(
        x=anchor_dates + anchor_dates[::-1],
        y=anchor_bull + anchor_bear[::-1],
        fill="toself",
        fillcolor="rgba(127,119,221,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% confidence band",
        showlegend=True,
    ))

    # Base scenario
    fig.add_trace(go.Scatter(
        x=anchor_dates, y=anchor_mid,
        name="Base forecast",
        line=dict(color="#7F77DD", width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6),
    ))

    # Bull scenario
    fig.add_trace(go.Scatter(
        x=anchor_dates, y=anchor_bull,
        name="Bull scenario",
        line=dict(color="#1D9E75", width=1, dash="dot"),
        mode="lines",
    ))

    # Bear scenario
    fig.add_trace(go.Scatter(
        x=anchor_dates, y=anchor_bear,
        name="Bear scenario",
        line=dict(color="#E24B4A", width=1, dash="dot"),
        mode="lines",
    ))


    fig.update_layout(
        height=440,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="AAPL Price (USD)",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario table ────────────────────────────────────────────────────
    st.subheader("Day-by-day forecast table")
    table = fc[["day","price_bear","price_base","price_bull"]].copy()
    table.index = table.index.strftime("%Y-%m-%d (%a)")
    table["bear_chg"] = ((table["price_bear"] / last_price) - 1).map("{:+.2%}".format)
    table["base_chg"] = ((table["price_base"] / last_price) - 1).map("{:+.2%}".format)
    table["bull_chg"] = ((table["price_bull"] / last_price) - 1).map("{:+.2%}".format)
    table = table.rename(columns={
        "day":        "Day",
        "price_bear": "Bear ($)",
        "price_base": "Base ($)",
        "price_bull": "Bull ($)",
        "bear_chg":   "Bear (%)",
        "base_chg":   "Base (%)",
        "bull_chg":   "Bull (%)",
    })
    st.dataframe(table[["Day","Bear ($)","Bear (%)","Base ($)","Base (%)","Bull ($)","Bull (%)"]], use_container_width=True)

    # ── Honest disclaimer ─────────────────────────────────────────────────
    st.markdown("""
    <div class="warning-box">
    <b>How to read this forecast</b><br>
    The model predicts <b>direction only</b> (up or down over 7 trading days). The price range is built using
    recent volatility — it shows where prices could realistically land, not where they will land.
    Training data is limited (2026 only) which reduces statistical reliability. Do not use for real trading.
    </div>
    """, unsafe_allow_html=True)

    # ── Re-run forecast button ────────────────────────────────────────────
    st.divider()
    st.markdown("To refresh the forecast with the latest data and model retraining:")
    st.code("make all\nstreamlit run app/streamlit_app.py")


# ════════════════════════════════════════════════════════════════════════
# PAGE 3: LAG ANALYSIS (was page 2)
# ════════════════════════════════════════════════════════════════════════
elif page == "🔬 Lag Analysis":
    st.title("🔬 Lag Correlation Analysis")
    st.markdown("*At what lag does supplier sentiment best predict AAPL returns?*")

    if not data_ready:
        st.warning("Run the data pipeline first.")
        st.stop()

    if not has_sentiment:
        st.warning(
            "⚠️ Sentiment columns are all zero/missing — lag analysis requires real headline data. "
            "Run `collect_headlines.py` and `sentiment_scorer.py` with a valid NewsAPI key."
        )
        st.stop()

    lags = list(range(0, 30))
    results = []

    for lag in lags:
        for supplier_col, label in [
            ("tsmc_sentiment_idx", "TSMC"),
            ("samsung_sentiment_idx", "Samsung"),
            ("composite_sentiment_idx", "Composite"),
        ]:
            if supplier_col not in df.columns:
                continue
            x = df[supplier_col].shift(lag)
            y = df.get("fwd_return_7d", pd.Series(dtype=float))
            valid = x.notna() & y.notna() & (x != 0)
            if valid.sum() < 20:
                continue
            r, p = pearsonr(x[valid], y[valid])
            results.append({"lag_days": lag, "supplier": label, "correlation": r, "p_value": p})

    if not results:
        st.warning("Not enough non-zero sentiment data to compute lag correlations (need ≥20 real days per supplier).")
        st.stop()

    lag_df = pd.DataFrame(results)
    fig = px.line(
        lag_df, x="lag_days", y="correlation", color="supplier",
        title="Pearson r: supplier sentiment (lag N days) vs AAPL 7d forward return",
        labels={"lag_days": "Lag (trading days)", "correlation": "Pearson r"},
        color_discrete_map={"TSMC": "#2563eb", "Samsung": "#16a34a", "Composite": "#dc2626"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.15, line_dash="dot", line_color="#f59e0b", annotation_text="r=0.15")
    fig.add_hline(y=-0.15, line_dash="dot", line_color="#f59e0b")
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    best = lag_df.loc[lag_df["correlation"].abs().idxmax()]
    sig = "statistically significant (p<0.05)" if best["p_value"] < 0.05 else "not significant at p<0.05 — interpret with caution"
    st.markdown(f"""
    <div class="finding-box">
    🔍 <b>Peak correlation:</b> lag = <b>{int(best['lag_days'])} days</b>,
    supplier = <b>{best['supplier']}</b>, r = {best['correlation']:.3f}, p = {best['p_value']:.4f} — {sig}.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL & SHAP (was page 3)
# ════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model & SHAP":
    st.title("🤖 Model & Explainability")
    st.markdown("*XGBoost classifier → 7-day AAPL direction. SHAP explains each prediction.*")

    if not model_ready:
        st.warning("Train the model first: `python src/models/train.py`")
        st.stop()

    if not data_ready:
        st.warning("No feature data found.")
        st.stop()

    target = "fwd_direction_7d"
    valid = df.dropna(subset=[target]).copy()
    valid[features] = valid[features].fillna(0.0)

    if len(valid) == 0:
        st.warning("No rows with a valid target column.")
        st.stop()

    X = valid[features]

    # ── Feature importance (always works) ────────────────────────────────
    st.subheader("Feature Importance")
    imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=imp["Importance"], y=imp["Feature"],
        orientation="h", marker_color="#2563eb",
    ))
    fig.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP (optional, may fail gracefully) ─────────────────────────────
    st.subheader("SHAP Summary")
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Use newer SHAP plotting API
        fig2 = plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close()
    except Exception as e:
        st.info(f"SHAP summary unavailable ({e}). Feature importance bar chart above is shown instead.")

    # ── Latest prediction ─────────────────────────────────────────────────
    st.subheader("Latest Prediction")
    try:
        latest_X = X.iloc[[-1]]
        prob = model.predict_proba(latest_X)[0][1]
        direction = "📈 UP" if prob > 0.5 else "📉 DOWN"
        confidence = max(prob, 1 - prob)
        c1, c2 = st.columns(2)
        c1.metric("Predicted 7-day direction", direction)
        c2.metric("Model confidence", f"{confidence:.1%}")
    except Exception as e:
        st.warning(f"Could not generate prediction: {e}")


# ════════════════════════════════════════════════════════════════════════
# PAGE 5: BACKTEST (was page 4)
# ════════════════════════════════════════════════════════════════════════
elif page == "📈 Backtest":
    st.title("📈 Walk-Forward Backtest")
    st.markdown("*Rolling 40-day train window, 5-day test step. No lookahead bias.*")

    if not backtest_ready:
        st.warning("Run the backtest first: `python src/models/backtest.py`")
        st.stop()

    acc = (bt["prediction"] == bt["actual_direction"]).mean()
    baseline = bt["actual_direction"].mean()
    strat_sharpe = sharpe(bt["strategy_return"])
    bh_sharpe = sharpe(bt["buyhold_return"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Directional Accuracy", f"{acc:.1%}", f"{(acc - baseline):+.1%} vs baseline")
    col2.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
    col3.metric("Buy-Hold Sharpe", f"{bh_sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd(bt['strategy_equity']):.1%}")

    st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["strategy_equity"],
                             name="Strategy", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["buyhold_equity"],
                             name="Buy & Hold", line=dict(color="#94a3b8", width=1.5, dash="dash")))
    fig.update_layout(
        title="Strategy vs. Buy-and-Hold (normalized to $1)",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Portfolio Value ($)", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Honest interpretation:</b> Transaction costs, slippage, and market impact are not modeled.
    Past backtest performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Raw backtest data"):
        st.dataframe(bt[["prediction", "actual_direction", "strategy_return",
                          "buyhold_return", "strategy_equity", "buyhold_equity"]].tail(50))


# ════════════════════════════════════════════════════════════════════════
# PAGE 6: WRITEUP (was page 5)
# ════════════════════════════════════════════════════════════════════════
elif page == "📝 Research Writeup":
    st.title("📝 Research Writeup")
    writeup_path = Path("docs/writeup.md")
    if writeup_path.exists():
        st.markdown(writeup_path.read_text())
    else:
        st.info("Add your writeup to `docs/writeup.md`")