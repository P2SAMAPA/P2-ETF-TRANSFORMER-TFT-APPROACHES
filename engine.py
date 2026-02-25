import pandas as pd
from darts import TimeSeries
from darts.models import PatchTSTModel, TFTModel
from darts.dataprocessing.transformers import Scaler

class RotationEngine:
    def __init__(self, model_choice="PatchTST"):
        self.model_choice = model_choice
        self.scaler = Scaler()

    def train_and_predict(self, df, target_col, covariates_df=None):
        series = TimeSeries.from_dataframe(df, 'Date', target_col)
        series_scaled = self.scaler.fit_transform(series)
        
        if self.model_choice == "TFT" and covariates_df is not None:
            cov = TimeSeries.from_dataframe(covariates_df, 'Date')
            model = TFTModel(input_chunk_length=30, output_chunk_length=7, n_epochs=20)
            model.fit(series_scaled, future_covariates=cov)
            pred = model.predict(n=7, future_covariates=cov)
        else:
            model = PatchTSTModel(input_chunk_length=30, output_chunk_length=7, n_epochs=20)
            model.fit(series_scaled)
            pred = model.predict(n=7)
            
        return self.scaler.inverse_transform(pred).last_value()
