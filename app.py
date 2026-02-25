import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gitlab
import os
from datetime import datetime
from io import StringIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="ETF Alpha Dashboard", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .section-header {
        color: #1E3A8A;
        font-weight: bold;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA FETCHING (FROM GITLAB DATA LAKE) ---
@st.cache_data(ttl=3600)
def load_data_from_gitlab():
    try:
        # Retrieve secrets from Streamlit Cloud
        gl_token = st.secrets["GITLAB_API_TOKEN"]
        gl_id = st.secrets["GITLAB_PROJECT_ID"]
        
        gl = gitlab.Gitlab("https://gitlab.com", private_token=gl_token)
        project = gl.projects.get(gl_id)
        
        # Pull the master_data.csv
        file_info = project.files.get(file_path="master_data.csv", ref="main")
        content = file_info.decode().decode('utf-8')
        
        df = pd.read_csv(StringIO(content), index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error connecting to GitLab: {e}")
        return pd.DataFrame()

# --- SIDEBAR: CONTROL CENTER ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Control Center</h2>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Model Configuration")
    model_type = st.radio("Primary Engine", ["PatchTST", "TFT"], help="PatchTST for trend; TFT for Macro-aware logic.")
    
    st.subheader("Risk Management")
    tsl_threshold = st.slider("2-Day Cumulative TSL (%)", 8.0, 25.0, 10.0, step=0.5)
    z_score_target = st.slider("Re-entry Z-Score Threshold", 0.8, 2.0, 1.2, step=0.1)
    
    st.divider()
    st.info("**Training Context:** I, J, K trained in cloud (2008-2026). A-H trained locally for the selected window.")
    selected_year = st.slider("Local Training Year Window", 2008, 2026, 2026)

# --- MAIN DASHBOARD ---
st.markdown("<p class='main-title'>📈 ETF Rotation Strategy</p>", unsafe_allow_html=True)
st.caption("Automated Portfolio Optimization using Transformer-based Forecasting & GitLab Data Lake")

data = load_data_from_gitlab()

if not data.empty:
    # --- LOGIC PLACEHOLDERS ---
    current_asset = "VNQ" 
    is_in_cash = False
    prediction_z = 1.45
    mtd_perf = "+3.2%"
    
    # 1. KEY METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    
    status_label = "CASH" if is_in_cash else current_asset
    m1.metric("Active Allocation", status_label, "Rotation Active")
    m2.metric("System Health", "OPTIMAL", f"Z: {prediction_z}")
    m3.metric("MTD Performance", mtd_perf, "vs Benchmark")
    m4.metric("Last Data Update", data.index[-1].strftime('%Y-%m-%d'), "Daily Sync")

    st.divider()

    # 2. CHARTS SECTION
    st.markdown("<p class='section-header'>Strategy Performance vs. Benchmarks</p>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-100:], y=np.random.randn(100).cumsum() + 100, 
                             name="Strategy (PatchTST/TFT)", line=dict(color="#1E3A8A", width=3)))
    fig.add_trace(go.Scatter(x=data.index[-100:], y=np.random.randn(100).cumsum() + 100, 
                             name="SPY Benchmark", line=dict(color="#94A3B8", dash='dash')))
    
    fig.update_layout(height=450, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 3. SIGNAL LOG (Fixed Syntax Here)
    with st.expander("🔎 View Recent Signal History & Model Weights"):
        log_data = {
            "Date": data.index[-5:].strftime('%Y-%m-%d').tolist(),
            "Primary Signal": ["ETF", "ETF", "ETF", "CASH", "ETF"],
            "Asset Selected": ["VNQ", "VNQ", "TLT", "CASH", "VNQ"],
            "Trigger Reason": ["Z-Score Entry", "Trend Hold", "Volatility Exit", "TSL Hit", "Re-entry Z"]
        }
        st.table(pd.DataFrame(log_data))

# --- FINAL SECTION: METHODOLOGY ---
st.divider()
st.markdown("<p class='section-header'>📚 Methodology: PatchTST vs. TFT Engine</p>", unsafe_allow_html=True)

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("### 🧩 PatchTST (Patch Time Series Transformer)")
    st.write("""
    **Best For:** Independent trend analysis and long-term signal spotting.
    * **Patching:** Breaks price action into windows (e.g., 16-day) to see 'shapes' rather than noise.
    * **Channel Independence:** Each ETF is treated as its own unique signal.
    * **Efficiency:** Handles 2008-2026 data without high memory overhead.
    """)

with m_col2:
    st.markdown("### 📊 TFT (Temporal Fusion Transformer)")
    st.write("""
    **Best For:** Macro-driven rotation using Variables A-H.
    * **Variable Selection:** Automatically weighs macro data (Inflation, Yields) to find drivers.
    * **Gated Logic:** 'On/Off' switch for variables to prevent overfitting in specific years.
    * **Interpretable Attention:** Identifies exactly when macro events influence price.
    """)

st.info("💡 **Strategy Implementation:** PatchTST monitors trend strength; TFT validates against macro variables. If 2-day loss > TSL slider, move to CASH. Re-entry requires Z-Score > slider.")
