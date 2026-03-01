import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import gitlab
from io import StringIO
import engine

# --- PAGE CONFIG ---
st.set_page_config(page_title="ETF Signal Dashboard", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    .prediction-card h2 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
    }
    .prediction-card h3 {
        margin: 0;
        font-size: 1.8rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #9ca3af;
    }
    .stButton>button {
        background: #1E3A8A;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background: #2563eb;
    }
    .audit-table {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- TRADING CALENDAR ---
def get_next_trading_day():
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=7))
        return schedule.index[0].strftime('%Y-%m-%d')
    except:
        return "Market Closed"

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
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_signals(model):
    try:
        gl_token = st.secrets["GITLAB_API_TOKEN"]
        gl_id = st.secrets["GITLAB_PROJECT_ID"]
        gl = gitlab.Gitlab("https://gitlab.com", private_token=gl_token)
        project = gl.projects.get(gl_id)
        file_name = "signals_transformer.csv" if model == "Transformer" else "signals_tft.csv"
        file_info = project.files.get(file_path=file_name, ref="main")
        signals = pd.read_csv(StringIO(file_info.decode().decode('utf-8')), index_col=0)
        signals.index = pd.to_datetime(signals.index)
        return signals
    except Exception as e:
        st.warning(f"⚠️ Could not load signals: {e}")
        return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/etf.png", width=80)
    st.markdown("### ⚙️ Strategy Configuration")
    
    option = st.radio(
        "**Active Model**",
        ["Option A: Transformer", "Option B: TFT"],
        help="Select which pre-trained model to use for predictions."
    )
    model_name = "Transformer" if "Option A" in option else "TFT"
    
    st.markdown("---")
    st.markdown("**Training Period**")
    start_year = st.slider("Start Year", 2008, 2025, 2008, help="Backtest will begin from this year.")
    end_year = 2025
    
    st.markdown("---")
    st.markdown("**Risk Controls**")
    tsl_val = st.slider("2-Day Cumulative TSL (%)", 0.0, 25.0, 10.0, step=0.5,
                        help="Exit position if 2-day cumulative loss exceeds this threshold.")
    txn_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05,
                         help="Cost per trade (each time the model switches ETFs).")

# --- MAIN CONTENT ---
st.markdown('<h1 class="main-header">📊 ETF Signal Dashboard</h1>', unsafe_allow_html=True)
st.caption(f"**Universe:** TLT (20Y+ Treasuries) · TBT (UltraShort 20Y+) · VNQ (Real Estate) · SLV (Silver) · GLD (Gold)")

# Load data
df = load_data()
next_open = get_next_trading_day()

if df.empty:
    st.warning("No data loaded. Please check GitLab connection.")
    st.stop()

signals_df = load_signals(model_name)
if signals_df.empty:
    st.warning("No signals available. Run the precomputation workflow first.")
    st.stop()

# Run strategy
with st.spinner("Running backtest..."):
    results = engine.compute_strategy_logic(df, option, (start_year, end_year), txn_cost, tsl_val, signals_df)

if results is None:
    st.error("No data available for the selected period.")
    st.stop()

# Unpack results
cum_rets = results["cum_rets"]
daily_rets = results["daily_rets"]
signal = results["signal"]
sharpe = results["sharpe"]
max_daily_val = results["max_daily_val"]
max_daily_date = results["max_daily_date"]
max_p2t = results["max_p2t"]

# Compute additional metrics
total_return = (1 + daily_rets).prod() - 1
num_years = len(daily_rets) / 252
annual_return = (1 + total_return) ** (1 / num_years) - 1

# Hit ratio
last_15_rets = daily_rets.tail(15)
hit_ratio = (last_15_rets > 0).sum() / len(last_15_rets) if len(last_15_rets) > 0 else 0

# --- PREDICTION CARD ---
last_signal = signal[-1] if len(signal) > 0 else "N/A"
col_pred1, col_pred2 = st.columns([1, 1])
with col_pred1:
    st.markdown(f"""
    <div class="prediction-card">
        <h2>📈 {last_signal}</h2>
        <h3>Next Trading Day</h3>
    </div>
    """, unsafe_allow_html=True)
with col_pred2:
    st.markdown(f"""
    <div style="background: #f3f4f6; border-radius:20px; padding:2rem; height:100%; display:flex; align-items:center; justify-content:center;">
        <div style="text-align:center;">
            <div style="color:#6b7280; font-size:1.2rem;">Market Open</div>
            <div style="font-size:2.5rem; font-weight:700; color:#1E3A8A;">{next_open}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- METRICS ROW (now with Annual Return) ---
st.markdown("### 📊 Key Performance Metrics")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annual Return</div>
        <div class="metric-value">{annual_return:.2%}</div>
        <div class="metric-delta">CAGR</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value">{sharpe:.2f}</div>
        <div class="metric-delta">Annualized</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Daily Drawdown</div>
        <div class="metric-value">{max_daily_val:.2%}</div>
        <div class="metric-delta">{max_daily_date}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value">{max_p2t:.2%}</div>
        <div class="metric-delta">Peak‑to‑Trough</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">15‑Day Hit Ratio</div>
        <div class="metric-value">{hit_ratio:.0%}</div>
        <div class="metric-delta">Last 15 sessions</div>
    </div>
    """, unsafe_allow_html=True)

# --- EQUITY CURVE ---
st.markdown("### 📈 Equity Curve")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=cum_rets.index,
    y=cum_rets.values,
    name=f"{model_name} Strategy",
    line=dict(color="#1E3A8A", width=2),
    hovertemplate="Date: %{x}<br>Cumulative Return: %{y:.2f}<extra></extra>"
))

# Benchmark (equal weight)
if not df.empty:
    asset_returns = df[["TLT", "TBT", "VNQ", "SLV", "GLD"]].pct_change().fillna(0)
    ew_rets = asset_returns.mean(axis=1)
    ew_rets = ew_rets.reindex(cum_rets.index).fillna(0)
    ew_cum = (1 + ew_rets).cumprod()
    fig.add_trace(go.Scatter(
        x=ew_cum.index,
        y=ew_cum.values,
        name="Equal Weight Benchmark",
        line=dict(color="#94A3B8", width=2, dash="dash"),
        hovertemplate="Date: %{x}<br>Cumulative Return: %{y:.2f}<extra></extra>"
    ))

fig.update_layout(
    template="plotly_white",
    height=450,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis_tickformat=".0%"
)
st.plotly_chart(fig, use_container_width=True)

# --- AUDIT TRAIL ---
st.markdown("### 📋 Last 15 Trading Sessions")
audit_dates = daily_rets.tail(15).index.strftime('%Y-%m-%d').tolist()
audit_signals = signal[-15:].tolist() if len(signal) >= 15 else signal.tolist()
audit_returns = daily_rets.tail(15).values.tolist()

while len(audit_signals) < 15:
    audit_signals.insert(0, "N/A")
    audit_dates.insert(0, "N/A")
    audit_returns.insert(0, 0.0)

audit_df = pd.DataFrame({
    "Date": audit_dates[::-1],
    "ETF Predicted": audit_signals[::-1],
    "Strategy Return": [f"{r*100:.2f}%" for r in audit_returns[::-1]]
})

st.dataframe(
    audit_df.style.applymap(lambda x: 'color: green' if x.endswith('%') and float(x[:-1]) > 0 else 'color: red' if x.endswith('%') and float(x[:-1]) < 0 else '', subset=['Strategy Return']),
    use_container_width=True,
    hide_index=True
)

# --- METHODOLOGY ---
with st.expander("📘 Methodology & Model Details", expanded=False):
    st.markdown("""
    **Transformer (Attention‑based Model)**  
    A standard transformer architecture adapted for time series forecasting. It uses self‑attention to capture long‑range dependencies and is trained to predict the next day's return for each ETF. The asset with the highest predicted return is selected for the next trading day.  
    *Training period: 2018–2025 (to save computational time).*

    **TFT (Temporal Fusion Transformer)**  
    An attention‑based model designed for interpretable multi‑horizon forecasting. It combines recurrent layers for local processing with self‑attention for long‑term dependencies, and includes built‑in mechanisms for handling known inputs (e.g., date features).  
    *Training period: 2008–2025.*

    Both models are trained in a **walk‑forward** manner: for each day, only data up to that day is used to forecast the next day, simulating a realistic trading environment. Signals are precomputed offline (via Kaggle/Colab) and served from GitLab.
    """)
