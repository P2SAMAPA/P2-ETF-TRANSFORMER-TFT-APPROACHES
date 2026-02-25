import pandas as pd
import numpy as np
import torch
from darts import TimeSeries
from darts.models import PatchTSTModel, TFTModel
from darts.dataprocessing.transformers import Scaler
import logging

# Disable unnecessary logs to keep the Streamlit console clean
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

class RotationEngine:
    def __init__(self, model_choice="PatchTST"):
        self.model_choice = model_choice
        self.scaler = Scaler()
        self.model = None

    def prepare_series(self, df, target_col):
        """Converts a dataframe column into a Darts TimeSeries."""
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Create series and scale it (Transformers are sensitive to scale)
        series = TimeSeries.from_dataframe(df.reset_index(), 'Date', target_col)
        series_scaled = self.scaler.fit_transform(series)
        return series_scaled

    def run_prediction(self, df, target_asset):
        """
        Trains the selected model on 2008-2026 data and predicts next-day return.
        """
        # 1. Prepare Data
        series_scaled = self.prepare_series(df, target_asset)
        
        # 2. Initialize Model based on choice
        if self.model_choice == "PatchTST":
            # Best for spotting 'shapes' in trend (High-Frequency rotation)
            self.model = PatchTSTModel(
                input_chunk_length=30,  # Look back 30 days
                output_chunk_length=1,  # Predict 1 day ahead
                n_epochs=20,            # Balanced for Streamlit speed/accuracy
                patch_len=16,           # Group days into 16-day patches
                random_state=42
            )
        else:
            # Best for Macro-aware logic (TFT)
            self.model = TFTModel(
                input_chunk_length=30,
                output_chunk_length=1,
                n_epochs=20,
                add_relative_index=True,
                random_state=42
            )

        # 3. Fit and Predict
        self.model.fit(series_scaled)
        forecast_scaled = self.model.predict(n=1)
        
        # 4. Inverse Scale to get actual price prediction
        forecast = self.scaler.inverse_transform(forecast_scaled)
        
        # 5. Logic: Calculate expected 1-day % Return
        current_price = df[target_asset].iloc[-1]
        predicted_price = forecast.values()[0][0]
        expected_return = (predicted_price - current_price) / current_price
        
        return expected_return

    def get_z_score(self, return_series, current_prediction):
        """
        Calculates how 'strong' the current prediction is relative to history.
        Used for the Re-entry Slider logic (0.8 - 2.0).
        """
        mean_ret = return_series.mean()
        std_ret = return_series.std()
        
        if std_ret == 0:
            return 0
            
        z_score = (current_prediction - mean_ret) / std_ret
        return z_score
