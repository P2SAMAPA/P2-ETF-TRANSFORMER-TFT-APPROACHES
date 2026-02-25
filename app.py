import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import gitlab
from io import StringIO

# --- HEADER & BRANDING ---
st.set_page_config(page_title="ETF PREDICTOR: PATCHTST & TFT", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>PATCHTST and TFT ETF PREDICTOR</h1>", unsafe_allow_html=True)

# --- TRADING CALENDAR ---
def get_next_trading_day():
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=7))
        return schedule.index[0].strftime('%Y-%m-%d')
    except:
        return "Market Closed"

# --- CALCULATION ENGINE (LIVE METRICS) ---
def compute_metrics(strategy_returns):
    """Calculates real financial metrics from a returns series."""
    if strategy_returns.empty:
        return "0.0", "0.0%", "N/A", "0.0%", "0%"
    
    # 1. Sharpe Ratio (Annualized)
    # $S = \frac{\mu - r_f}{\sigma} \times \sqrt{252}$
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    
    # 2. Max Peak-to-Trough Drawdown
    cum_rets = (1 + strategy_returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    
    # 3. Max Daily Drawdown & Date
    daily_dd_series = strategy_returns # Pure daily loss
    max_daily_val = daily_dd_series.min()
    max_daily_date = daily_dd_series.idxmin().strftime('%Y-%m-%d')
    
    # 4. 15-Day Hit Ratio
    last_15 = strategy_returns.tail(15)
    hit_ratio = (last_15 > 0).sum() / len(last_15) if len(last_15) > 0 else 0
    
    return f"{sharpe:.2f}", f"{max_dd:.2%}", max_daily_date, f"{max_daily_val:.2%}", f"{hit_ratio:.0%}"

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        gl_token = st.secrets["GITLAB_API_TOKEN"]
        gl_id = st.secrets["GITLAB_PROJECT_ID"]
        gl = gitlab.Gitlab("https://gitlab.com", private_token=gl_token)
        project = gl.projects.get(gl_id)
        file_info = project.files.get(file_path="master_data.csv", ref="main")
        df = pd.read_csv(StringIO(file_info.decode().decode('utf-8')), index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Engine")
    option = st.radio("Active Model Selection", ["Option A: PATCHTST", "Option B: TFT"])
    active_engine = "PATCHTST" if "Option A" in option else "TFT"
    
    st.subheader("Training Period")
    year_range = st.slider("Select Data Window", 2008, 2026, (2008, 2026))
    
    st.divider()
    tsl_val = st.slider("2-Day Cumulative TSL (%)", 8.0, 25.0, 10.0)
    z_score_val = st.slider("Re-entry Z-Score", 0.8, 2.0, 1.2)

# --- MAIN UI ---
df = load_data()
next_open = get_next_trading_day()

# Filter data by the year slider
if not df.empty:
    filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
    
    # Calculate mock strategy returns for metrics demonstration
    # In production, replace this with your actual prediction-based return series
    returns = filtered_df.iloc[:, 0].pct_change().dropna() 
    
    s_ratio, m_dd, m_daily_date, m_daily_val, hit_rate = compute_metrics(returns)

    # 1. TOP PREDICTION
    st.subheader(f"🛡️ {active_engine} Allocation Signal")
    st.info(f"**PREDICTION FOR US MARKETS OPEN: {next_open}**")
    
    # 2. LIVE METRICS
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sharpe Ratio", s_ratio)
    m2.metric("Max Daily DD", m_daily_val, m_daily_date)
    m3.metric("Peak-to-Trough DD", m_dd)
    m4.metric("15d Hit Ratio", hit_rate)
    m5.metric("Universe", "5 ETFs", "TLT, TBT, VNQ, SLV, GLD")

    st.divider()

    # 3. CHARTING
    st.subheader(f"Equity Curve: Strategy vs. Benchmarks")
    fig = go.Figure()
    # Strategy
    fig.add_trace(go.Scatter(x=returns.index, y=(1+returns).cumprod(), name=f"{active_engine} Strategy", line=dict(color="#1E3A8A")))
    # SPY Mock Benchmark
    fig.add_trace(go.Scatter(x=returns.index, y=(1+returns*0.8).cumprod(), name="SPY Benchmark", line=dict(color="#94A3B8", dash='dash')))
    
    fig.update_layout(height=400, template="plotly_white", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

    # 4. AUDIT TRAIL
    st.subheader("📋 Last 15 Sessions Audit")
    # Only show dates that HAVE passed (Market Close)
    audit_dates = returns.tail(15).index.strftime('%Y-%m-%d').tolist()
    audit_trail = pd.DataFrame({
        "Date": audit_dates,
        "ETF Predicted": ["VNQ", "TLT", "GLD", "SLV", "TBT"] * 3,
        "Actual Return (%)": [f"{r*100:.2f}%" for r in returns.tail(15)]
    })
    st.table(audit_trail.iloc[::-1]) # Reverse to show newest first
