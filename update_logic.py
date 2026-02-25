import os
import gitlab
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import StringIO

# --- CONFIGURATION (Loaded from GitHub Secrets) ---
GITLAB_URL = "https://gitlab.com"
PROJECT_ID = os.getenv('GITLAB_PROJECT_ID')
GL_TOKEN = os.getenv('GITLAB_API_TOKEN')
FILE_NAME = "master_data.csv"

# The assets we track for rotation + Benchmarks
SYMBOLS = ["TLT", "TBT", "VNQ", "GLD", "SLV", "SPY", "AGG", "^IRX"]

def update_data_lake():
    print(f"🚀 Starting daily sync for Project ID: {PROJECT_ID}")
    
    if not GL_TOKEN or not PROJECT_ID:
        print("❌ Error: Missing GITLAB_API_TOKEN or GITLAB_PROJECT_ID in environment.")
        return

    # 1. Connect to GitLab
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)

    # 2. Pull the existing master_data.csv from GitLab
    try:
        file_info = project.files.get(file_path=FILE_NAME, ref='main')
        existing_content = file_info.decode().decode('utf-8')
        df_old = pd.read_csv(StringIO(existing_content), index_col=0)
        df_old.index = pd.to_datetime(df_old.index)
        print(f"📊 Existing data loaded: {len(df_old)} rows.")
    except Exception as e:
        print(f"❌ Could not retrieve existing file: {e}")
        return

    # 3. Fetch the latest data (last 5 days to ensure we catch any missed closes)
    print("📡 Fetching latest market data...")
    df_new = yf.download(SYMBOLS, period="5d")['Close']
    df_new.index = pd.to_datetime(df_new.index)

    # 4. Merge New Data with Old Data
    # 'combine_first' fills in the latest gaps without duplicating existing rows
    df_updated = df_new.combine_first(df_old)
    
    # Sort and remove any duplicates by index
    df_updated = df_updated[~df_updated.index.duplicated(keep='last')].sort_index()

    # 5. Convert back to CSV string
    csv_buffer = StringIO()
    df_updated.to_csv(csv_buffer)
    updated_content = csv_buffer.getvalue()

    # 6. Push the updated file back to GitLab
    try:
        file_info.content = updated_content
        file_info.save(branch='main', commit_message=f"Daily Auto-Update: {datetime.now().strftime('%Y-%m-%d')}")
        print("✅ SUCCESS: GitLab Data Lake updated with latest closes.")
    except Exception as e:
        print(f"❌ Failed to push update to GitLab: {e}")

if __name__ == "__main__":
    update_data_lake()
