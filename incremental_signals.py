import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import TFTModel, TransformerModel
import gitlab
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
GITLAB_URL = "https://gitlab.com"
PROJECT_ID = os.getenv('GITLAB_PROJECT_ID')
GL_TOKEN = os.getenv('GITLAB_API_TOKEN')
DATA_FILE = "master_data.csv"
TFT_SIGNALS_FILE = "signals_tft.csv"
TRANSFORMER_SIGNALS_FILE = "signals_transformer.csv"
TICKERS = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

def fetch_file_from_gitlab(file_name):
    """Download a file from GitLab and return its content as string."""
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    file_info = project.files.get(file_path=file_name, ref='main')
    return file_info.decode().decode('utf-8')

def upload_to_gitlab(file_name, content):
    """Upload content to GitLab, overwriting if exists."""
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    try:
        f = project.files.get(file_path=file_name, ref='main')
        f.content = content
        f.save(branch='main', commit_message=f"Incremental update {file_name} - {datetime.now().date()}")
    except:
        project.files.create({
            'file_path': file_name,
            'branch': 'main',
            'content': content,
            'commit_message': f"Add {file_name}"
        })

def prepare_series(df, target_ticker):
    """Create target series and covariates."""
    series = TimeSeries.from_dataframe(df, value_cols=target_ticker, fill_missing_dates=True, freq='B')
    covariates = datetime_attribute_timeseries(series, attribute="month", cyclic=True)
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="dayofweek", cyclic=True))
    return series, covariates

def predict_next_day(train_df, ticker, model_class, model_params):
    """Train model on train_df and predict next day's return for a single ticker."""
    series, covariates = prepare_series(train_df, ticker)
    
    target_scaler = Scaler()
    scaled_series = target_scaler.fit_transform(series)
    
    covariate_scaler = Scaler()
    scaled_covariates = covariate_scaler.fit_transform(covariates)
    
    model = model_class(**model_params)
    model.fit(scaled_series, past_covariates=scaled_covariates, verbose=False)
    
    pred = model.predict(n=1, series=scaled_series, past_covariates=scaled_covariates)
    pred = target_scaler.inverse_transform(pred)
    pred_price = pred.values()[0][0]
    
    last_price = train_df.loc[train_df.index[-1], ticker]
    pred_return = (pred_price - last_price) / last_price
    return pred_return

def update_signals(model_name, model_class, model_params, df, existing_signals_df):
    """
    Append new signals for dates not already in existing_signals_df.
    Returns updated DataFrame.
    """
    if existing_signals_df.empty:
        print(f"No existing signals for {model_name}. Skipping update.")
        return existing_signals_df
    
    last_signal_date = existing_signals_df.index[-1]
    print(f"Last signal date for {model_name}: {last_signal_date.date()}")
    
    # Get all dates in df after last_signal_date that have data for all tickers
    all_dates = df.index[df.index > last_signal_date]
    if len(all_dates) == 0:
        print(f"No new dates for {model_name}.")
        return existing_signals_df
    
    new_signals = []
    for test_date in all_dates:
        # Training data ends the day before test_date
        train_end = df.index[df.index < test_date][-1]
        train_df = df.loc[:train_end]
        
        print(f"Processing {test_date.date()} for {model_name}...")
        pred_returns = {}
        for ticker in TICKERS:
            pred_return = predict_next_day(train_df, ticker, model_class, model_params)
            pred_returns[ticker] = pred_return
        
        best_ticker = max(pred_returns, key=pred_returns.get)
        new_signals.append((test_date, best_ticker))
        print(f"  -> {best_ticker}")
    
    new_signals_df = pd.DataFrame(new_signals, columns=['date', 'signal']).set_index('date')
    updated = pd.concat([existing_signals_df, new_signals_df])
    updated = updated[~updated.index.duplicated(keep='last')].sort_index()
    return updated

def main():
    print("📥 Fetching master data from GitLab...")
    df_content = fetch_file_from_gitlab(DATA_FILE)
    df = pd.read_csv(StringIO(df_content), index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"Data shape: {df.shape}, last date: {df.index[-1].date()}")
    
    # Model parameters (same as in precompute_signals.py)
    tft_params = {
        'input_chunk_length': 30,
        'output_chunk_length': 1,
        'hidden_size': 32,
        'lstm_layers': 1,
        'num_attention_heads': 2,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 5,
        'optimizer_kwargs': {'lr': 1e-3},
        'add_relative_index': True,
        'force_reset': True,
        'random_state': 42
    }
    
    transformer_params = {
        'input_chunk_length': 30,
        'output_chunk_length': 1,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 5,
        'optimizer_kwargs': {'lr': 1e-3},
        'force_reset': True,
        'random_state': 42
    }
    
    # Update TFT signals
    print("\n🔮 Updating TFT signals...")
    try:
        tft_content = fetch_file_from_gitlab(TFT_SIGNALS_FILE)
        tft_signals = pd.read_csv(StringIO(tft_content), index_col=0)
        tft_signals.index = pd.to_datetime(tft_signals.index)
    except:
        print(f"Could not fetch {TFT_SIGNALS_FILE}. Skipping TFT update.")
        tft_signals = pd.DataFrame()
    
    if not tft_signals.empty:
        updated_tft = update_signals("TFT", TFTModel, tft_params, df, tft_signals)
        if len(updated_tft) > len(tft_signals):
            upload_to_gitlab(TFT_SIGNALS_FILE, updated_tft.to_csv())
            print(f"✅ TFT signals updated. New rows: {len(updated_tft) - len(tft_signals)}")
        else:
            print("No new TFT signals.")
    
    # Update Transformer signals
    print("\n🔮 Updating Transformer signals...")
    try:
        trans_content = fetch_file_from_gitlab(TRANSFORMER_SIGNALS_FILE)
        trans_signals = pd.read_csv(StringIO(trans_content), index_col=0)
        trans_signals.index = pd.to_datetime(trans_signals.index)
    except:
        print(f"Could not fetch {TRANSFORMER_SIGNALS_FILE}. Skipping Transformer update.")
        trans_signals = pd.DataFrame()
    
    if not trans_signals.empty:
        updated_trans = update_signals("Transformer", TransformerModel, transformer_params, df, trans_signals)
        if len(updated_trans) > len(trans_signals):
            upload_to_gitlab(TRANSFORMER_SIGNALS_FILE, updated_trans.to_csv())
            print(f"✅ Transformer signals updated. New rows: {len(updated_trans) - len(trans_signals)}")
        else:
            print("No new Transformer signals.")
    
    print("🎉 Incremental update complete.")

if __name__ == "__main__":
    main()
