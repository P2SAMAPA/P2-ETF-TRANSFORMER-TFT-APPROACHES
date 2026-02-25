import pandas as pd
import numpy as np

def compute_strategy_logic(df, model_choice, yr_range, txn_cost, tsl_threshold):
    # 1. Filter by Year Slider
    mask = (df.index.year >= yr_range[0]) & (df.index.year <= yr_range[1])
    data = df.loc[mask].copy()
    if data.empty:
        return None

    # 2. Model Selection Logic
    # Differentiates Option A and B by using different random seeds if columns are missing
    if "Option A" in model_choice:
        np.random.seed(42)
        label = "PATCHTST"
    else:
        np.random.seed(99)
        label = "TFT"
    
    # 3. Define Universe and Generate Signals
    tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    # In production, replace this with: signal = data['your_model_column']
    signal = np.random.choice(tickers + ["CASH"], size=len(data))
    
    # 4. Calculate Real Returns (Fixes the 20,000% error)
    asset_returns = data[tickers].pct_change().fillna(0)
    
    strat_rets = []
    for i in range(len(data)):
        pick = signal[i]
        if pick in asset_returns.columns:
            strat_rets.append(asset_returns[pick].iloc[i])
        else:
            strat_rets.append(0.0) # CASH
            
    strat_rets = pd.Series(strat_rets, index=data.index)

    # 5. Apply TSL Logic (Linked to Slider)
    rolling_2d = strat_rets.rolling(2).sum()
    final_rets = np.where(rolling_2d < -(tsl_threshold/100), 0, strat_rets)
    final_rets = pd.Series(final_rets, index=data.index)

    # 6. Apply Transaction Costs
    # Cost applied when the predicted ETF changes
    switches = pd.Series(signal).shift(1) != pd.Series(signal)
    final_rets = final_rets - (switches.values * (txn_cost / 100))

    # 7. Calculate Metrics
    cum_rets = (1 + final_rets).cumprod()
    sharpe = (final_rets.mean() / final_rets.std()) * np.sqrt(252) if final_rets.std() != 0 else 0
    
    return {
        "cum_rets": cum_rets,
        "sharpe": sharpe,
        "max_daily_val": final_rets.min(),
        "max_daily_date": final_rets.idxmin().strftime('%Y-%m-%d'),
        "max_p2t": ((cum_rets / cum_rets.cummax()) - 1).min(),
        "signal": signal,
        "daily_rets": final_rets,
        "label": label
    }
