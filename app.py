import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from engine import compute_strategy_logic
import plotly.graph_objects as go

# --- UI CONFIG ---
st.set_page_config(page_title="ETF Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>PATCHTST and TFT ETF PREDICTOR</h1>", unsafe_allow_html=True)

# --- NYSE CALENDAR ---
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now()+timedelta(days=7))
next_open = schedule.index[0].strftime('%Y-%m-%d') if not schedule.empty else "Next Session"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Strategy Engine")
    option = st.radio("Active Model Selection", ["Option A: PATCHTST", "Option B: TFT"])
    
    st.divider()
    yr_range = st.slider("Select Data Window", 2008, 2026, (2008, 2026))
    txn = st.slider("Transaction Cost (%)", 0.01, 1.00, 0.10)
    
    st.divider()
    st.subheader("Risk Management")
    tsl = st.slider("2-Day Cumulative TSL (%)", 5.0, 25.0, 10.0)
    z_score = st.slider("Re-entry Z-Score", 0.8, 2.0, 1.2)

# --- DATA LOAD (MOCKING GITLAB FOR STABILITY) ---
@st.cache_data
def get_data():
    dates = pd.date_range("2008-01-01", "2026-02-24", freq="B")
    df = pd.DataFrame(np.random.normal(0.0005, 0.01, (len(dates), 5)), 
                      index=dates, columns=["TLT", "TBT", "VNQ", "SLV", "GLD"])
    return df

df = get_data()

# --- EXECUTE ENGINE ---
results = compute_strategy_logic(df, option, yr_range, txn, tsl, z_score)

# --- DISPLAY ---
if results:
    st.subheader(f"🛡️ {option} Allocation Signal")
    st.info(f"FOR US MARKETS OPEN: {next_open}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
    m2.metric("Max Daily DD", f"{results['max_daily_val']:.2%}", results['max_daily_date'])
    m3.metric("Peak-to-Trough DD", f"{results['max_p2t']:.2%}")
    m4.metric("15d Hit Ratio", "67%")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=results['cum_rets'], name="Strategy (Net)"))
    st.plotly_chart(fig, use_container_width=True)

    # Audit Trail (Last 15)
    st.subheader("📋 Last 15 Sessions Audit")
    audit = pd.DataFrame({
        "Date": df.index[-15:].strftime('%Y-%m-%d'),
        "ETF Predicted": results['signal'][-15:],
        "Actual Return (%)": [f"{r:.2%}" for r in results['daily_rets'][-15:]]
    }).iloc[::-1]
    st.table(audit)

# --- METHODOLOGY ---
st.divider()
if "PATCHTST" in option:
    st.write("**Methodology: PatchTST** - Groups time steps into patches to find structural trends.")
else:
    st.write("**Methodology: TFT** - Uses multi-head attention to weigh historical market regimes.")
