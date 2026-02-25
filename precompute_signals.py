import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel, PatchTSTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import gitlab
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

GITLAB_URL = "https://gitlab.com"
PROJECT_ID = os.getenv('GITLAB_PROJECT_ID')
GL_TOKEN = os.getenv('GITLAB_API_TOKEN')
DATA_FILE = "master_data.csv"
TFT_SIGNALS_FILE = "signals_tft.csv"
PATCHTST_SIGNALS_FILE = "signals_patchtst.csv"
TICKERS = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

def fetch_data_from_gitlab():
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    file_info = project.files.get(file_path=DATA_FILE, ref='main')
    df = pd.read_csv(StringIO(file_info.decode().decode('utf-8')), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def prepare_series(df, target_ticker):
    series = TimeSeries.from_dataframe(df, value_cols=target_ticker, fill_missing_dates=True, freq='B')
    # Simple time covariates (month and dayofweek)
    covariates = datetime_attribute_timeseries(series, attribute="month", cyclic=True)
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="dayofweek", cyclic=True))
    return series, covariates

def walk_forward_signals(model_class, model_params, df, tickers, start_date=None):
    """
    Walk‑forward prediction. If start_date is given, only run from that date onward.
    """
    if start_date is not None:
        df = df.loc[df.index >= start_date]
    
    dates = df.index
    min_train = 252 * 2  # at least 2 years
    signals = []
    
    for i in range(min_train, len(dates)):
        train_end = dates[i-1]
        test_date = dates[i]
        
        train_df = df.loc[:train_end]
        pred_returns = {}
        
        for ticker in tickers:
            series, covariates = prepare_series(train_df, ticker)
            scaler = Scaler()
            scaled_series = scaler.fit_transform(series)
            scaled_covariates = scaler.transform(covariates)
            
            model = model_class(**model_params)
            model.fit(scaled_series, past_covariates=scaled_covariates, verbose=False)
            
            pred = model.predict(n=1, series=scaled_series, past_covariates=scaled_covariates)
            pred = scaler.inverse_transform(pred)
            pred_price = pred.values()[0][0]
            
            last_price = train_df.loc[train_end, ticker]
            pred_return = (pred_price - last_price) / last_price
            pred_returns[ticker] = pred_return
        
        best_ticker = max(pred_returns, key=pred_returns.get)
        signals.append((test_date, best_ticker))
        print(f"{test_date.date()}: {best_ticker}")
    
    return pd.DataFrame(signals, columns=['date', 'signal']).set_index('date')

def upload_to_gitlab(file_name, content):
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    try:
        f = project.files.get(file_path=file_name, ref='main')
        f.content = content
        f.save(branch='main', commit_message=f"Update {file_name} - {datetime.now().date()}")
    except:
        project.files.create({
            'file_path': file_name,
            'branch': 'main',
            'content': content,
            'commit_message': f"Add {file_name}"
        })

def main():
    print("📥 Fetching data from GitLab...")
    df = fetch_data_from_gitlab()
    print(f"Data shape: {df.shape}")
    
    # Reduce model complexity for GitHub free tier
    tft_params = {
        'input_chunk_length': 30,        # shorter lookback
        'output_chunk_length': 1,
        'hidden_size': 32,                # smaller hidden size
        'lstm_layers': 1,
        'num_attention_heads': 2,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 5,                    # fewer epochs
        'optimizer_kwargs': {'lr': 1e-3},
        'add_relative_index': True,
        'force_reset': True,
        'random_state': 42
    }
    
    patchtst_params = {
        'input_chunk_length': 30,
        'output_chunk_length': 1,
        'num_encoder_layers': 2,
        'num_attention_heads': 2,
        'd_model': 64,
        'd_ff': 128,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 5,
        'optimizer_kwargs': {'lr': 1e-3},
        'force_reset': True,
        'random_state': 42
    }
    
    # Optionally limit the date range to reduce runtime (e.g., last 5 years)
    # start_date = "2018-01-01"  # uncomment to restrict
    start_date = None
    
    print("\n🔮 Generating TFT signals...")
    tft_signals = walk_forward_signals(TFTModel, tft_params, df, TICKERS, start_date)
    
    print("\n🔮 Generating PatchTST signals...")
    patchtst_signals = walk_forward_signals(PatchTSTModel, patchtst_params, df, TICKERS, start_date)
    
    print("\n📡 Uploading to GitLab...")
    upload_to_gitlab(TFT_SIGNALS_FILE, tft_signals.to_csv())
    upload_to_gitlab(PATCHTST_SIGNALS_FILE, patchtst_signals.to_csv())
    print("✅ Done.")

if __name__ == "__main__":
    main()
