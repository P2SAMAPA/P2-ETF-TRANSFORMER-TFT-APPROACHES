import os
import gitlab
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import StringIO

GITLAB_URL = "https://gitlab.com"
PROJECT_ID = os.getenv('GITLAB_PROJECT_ID')
GL_TOKEN = os.getenv('GITLAB_API_TOKEN')
FILE_NAME = "master_data.csv"
SYMBOLS = ["TLT", "TBT", "VNQ", "GLD", "SLV", "SPY", "AGG", "^IRX"]

def update_data_lake():
    print(f"🚀 Starting daily sync for Project ID: {PROJECT_ID}")
    
    if not GL_TOKEN or not PROJECT_ID:
        print("❌ Error: Missing GITLAB_API_TOKEN or GITLAB_PROJECT_ID in environment.")
        return

    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)

    # 1. Pull existing data
    try:
        file_info = project.files.get(file_path=FILE_NAME, ref='main')
        existing_content = file_info.decode().decode('utf-8')
        df_old = pd.read_csv(StringIO(existing_content), index_col=0)
        df_old.index = pd.to_datetime(df_old.index)
        print(f"📊 Existing data loaded: {len(df_old)} rows (last date: {df_old.index[-1].date()}).")
    except gitlab.exceptions.GitlabGetError:
        print("❌ master_data.csv not found. Run seeder.py first.")
        return

    # 2. Fetch latest data (use a generous window to cover holidays)
    print("📡 Fetching latest market data...")
    df_new = yf.download(SYMBOLS, period="10d")['Close']  # 10 days to be safe
    df_new.index = pd.to_datetime(df_new.index)

    # 3. Merge new with old
    df_updated = df_new.combine_first(df_old)
    df_updated = df_updated[~df_updated.index.duplicated(keep='last')].sort_index()

    # 4. Check if there are actual changes
    if df_updated.equals(df_old):
        print("✅ No new data – nothing to commit.")
        return

    # 5. Convert to CSV and push
    csv_buffer = StringIO()
    df_updated.to_csv(csv_buffer)
    updated_content = csv_buffer.getvalue()

    try:
        file_info.content = updated_content
        file_info.save(branch='main', commit_message=f"Daily Auto-Update: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"✅ SUCCESS: GitLab Data Lake updated. New rows: {len(df_updated) - len(df_old)}")
    except Exception as e:
        print(f"❌ Failed to push update to GitLab: {e}")

if __name__ == "__main__":
    update_data_lake()
