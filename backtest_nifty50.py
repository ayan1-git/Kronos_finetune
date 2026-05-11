import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from config import Config
from model.kronos import KronosTokenizer, Kronos, KronosPredictor

def load_benchmark(config):
    """Loads the NIFTY 50 index as the benchmark."""
    benchmark_path = os.path.join("Data /", "NIFTY 50_30minute.csv")
    df = pd.read_csv(benchmark_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["close"]

def calculate_metrics(preds, actuals):
    """Calculates IC, Rank IC, and Hit Rate."""
    if len(preds) < 2:
        return 0.0, 0.0, 0.5
    
    # Pearson IC
    ic = np.corrcoef(preds, actuals)[0, 1]
    if np.isnan(ic): ic = 0.0
    
    # Spearman Rank IC
    rank_ic, _ = spearmanr(preds, actuals)
    if np.isnan(rank_ic): rank_ic = 0.0
    
    # Directional Accuracy (Hit Rate)
    # Only count where actual move is non-zero to be fair
    mask = actuals != 0
    if np.any(mask):
        hit_rate = np.mean(np.sign(preds[mask]) == np.sign(actuals[mask]))
    else:
        hit_rate = 0.5
        
    return ic, rank_ic, hit_rate

def run_backtest():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running backtest on {device}...")

    # 1. Load Models (Base models from Hugging Face)
    token = os.getenv("HF_TOKEN")
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path, token=token)
    model = Kronos.from_pretrained(config.pretrained_predictor_path, token=token)
    predictor = KronosPredictor(model, tokenizer, device=device)

    # 2. Load Test Data
    test_data_path = os.path.join(config.dataset_path, "test_data.pkl")
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    # 3. Load Benchmark
    benchmark_series = load_benchmark(config)
    
    # 4. Prepare Backtest Window
    # We rebalance every 'predict_window' steps (10)
    symbols = list(test_data.keys())
    # Find all common timestamps across symbols in the backtest range
    all_dates = []
    for sym in symbols:
        df = test_data[sym]
        dates = df[(df.index >= config.backtest_time_range[0]) & 
                   (df.index <= config.backtest_time_range[1])].index
        all_dates.extend(dates)
    
    unique_dates = sorted(list(set(all_dates)))
    print(f"Total timestamps in backtest window: {len(unique_dates)}")

    # We skip the first 'lookback' steps to have enough history
    # and we skip the last 'predict' steps because we need ground truth to evaluate
    step = config.predict_window
    rebalance_dates = unique_dates[::step]
    
    results = []
    portfolio_returns = []
    benchmark_returns = []

    # 5. Iterative Evaluation
    for i in tqdm(range(len(rebalance_dates) - 1), desc="Backtesting"):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        batch_dfs = []
        batch_x_stamps = []
        batch_y_stamps = []
        valid_symbols = []
        actual_returns = []

        for sym in symbols:
            df = test_data[sym]
            
            # Ensure we have enough lookback
            if current_date not in df.index:
                continue
            
            idx = df.index.get_loc(current_date)
            if idx < config.lookback_window:
                continue
            
            # Extract historical window
            hist_df = df.iloc[idx - config.lookback_window + 1 : idx + 1]
            x_stamp = df.index[idx - config.lookback_window + 1 : idx + 1]
            
            # Extract future window for ground truth
            # Check if there are enough future bars
            if idx + config.predict_window >= len(df):
                continue
                
            future_df = df.iloc[idx + 1 : idx + config.predict_window + 1]
            y_stamp = df.index[idx + 1 : idx + config.predict_window + 1]
            
            # Target: Log return over the next 10 steps
            actual_ret = np.log(future_df["close"].iloc[-1] / hist_df["close"].iloc[-1])
            
            batch_dfs.append(hist_df)
            batch_x_stamps.append(x_stamp)
            batch_y_stamps.append(y_stamp)
            valid_symbols.append(sym)
            actual_returns.append(actual_ret)

        if not valid_symbols:
            continue

        # Batch Prediction
        pred_dfs = predictor.predict_batch(
            batch_dfs, batch_x_stamps, batch_y_stamps, 
            pred_len=config.predict_window, verbose=False
        )
        
        predicted_returns = []
        for j, p_df in enumerate(pred_dfs):
            # Predicted log return over the next 10 steps
            pred_ret = np.log(p_df["close"].iloc[-1] / batch_dfs[j]["close"].iloc[-1])
            predicted_returns.append(pred_ret)
        
        # Calculate Stats for this step
        predicted_returns = np.array(predicted_returns)
        actual_returns = np.array(actual_returns)
        
        ic, rank_ic, hit_rate = calculate_metrics(predicted_returns, actual_returns)
        results.append({
            "date": current_date,
            "ic": ic,
            "rank_ic": rank_ic,
            "hit_rate": hit_rate
        })

        # Trading Strategy: Top-K (Top 10 stocks)
        top_k = min(10, len(valid_symbols))
        top_indices = np.argsort(predicted_returns)[-top_k:]
        step_return = np.mean(actual_returns[top_indices])
        portfolio_returns.append(step_return)
        
        # Benchmark Return
        if current_date in benchmark_series.index and next_date in benchmark_series.index:
            bench_ret = np.log(benchmark_series.loc[next_date] / benchmark_series.loc[current_date])
            benchmark_returns.append(bench_ret)
        else:
            benchmark_returns.append(0.0)

    # 6. Aggregate Results
    res_df = pd.DataFrame(results)
    print("\n" + "="*30)
    print("BACKTEST RESULTS Summary")
    print("="*30)
    print(f"Average IC:      {res_df['ic'].mean():.4f}")
    print(f"Average Rank IC: {res_df['rank_ic'].mean():.4f}")
    print(f"Average Hit Rate: {res_df['hit_rate'].mean():.4f}")
    
    # Portfolio Performance
    port_rets = np.array(portfolio_returns)
    bench_rets = np.array(benchmark_returns)
    excess_rets = port_rets - bench_rets
    
    total_port_ret = np.exp(np.sum(port_rets)) - 1
    total_bench_ret = np.exp(np.sum(bench_rets)) - 1
    
    # Annualization factor (30-minute intervals, 10 steps per trade)
    # Approx 375 minutes per trading day in NSE (9:15 to 15:30)
    # 30-min bars: 12.5 bars per day. 10 steps = ~0.8 days.
    # Trading days per year = 252.
    steps_per_day = 12.5 / config.predict_window
    annual_factor = steps_per_day * 252
    
    ann_excess_ret = np.mean(excess_rets) * annual_factor
    sharpe = (np.mean(excess_rets) / (np.std(excess_rets) + 1e-8)) * np.sqrt(annual_factor)
    
    print(f"Cumulative Portfolio Return: {total_port_ret:.2%}")
    print(f"Cumulative Benchmark Return: {total_bench_ret:.2%}")
    print(f"Annualized Excess Return:    {ann_excess_ret:.2%}")
    print(f"Information Ratio (Sharpe):  {sharpe:.4f}")
    
    # Max Drawdown
    cum_wealth = np.exp(np.cumsum(port_rets))
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (cum_wealth - peak) / peak
    print(f"Max Drawdown:                {np.min(drawdown):.2%}")
    print("="*30)

    # Save detailed results
    save_dir = config.backtest_result_path
    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, "backtest_metrics.csv"), index=False)
    pd.DataFrame({"port_ret": port_rets, "bench_ret": bench_rets}).to_csv(os.path.join(save_dir, "equity_curve.csv"), index=False)
    print(f"Detailed results saved to {save_dir}")

if __name__ == "__main__":
    run_backtest()
