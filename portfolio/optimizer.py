import numpy as np
import pandas as pd
from typing import Tuple
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions, CLA, HRPOpt

def get_expected_returns(prices: pd.DataFrame, frequency: int = 252) -> pd.Series:
    # mean historical returns (daily) and annualize using pypfopt helper
    mu = expected_returns.mean_historical_return(prices, frequency=frequency)
    return mu

def get_cov_matrix(prices: pd.DataFrame, frequency: int = 252, method: str = "ledoit_wolf") -> pd.DataFrame:
    # use sample_cov or shrunk covariances
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf() if method == "ledoit_wolf" else risk_models.sample_cov(prices)
    return S

def max_sharpe_weights(prices: pd.DataFrame, risk_free_rate: float = 0.02) -> Tuple[np.ndarray, dict]:
    mu = get_expected_returns(prices)
    S = get_cov_matrix(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.0)
    raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance()
    return cleaned, {"performance": perf}

def min_volatility_weights(prices: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    mu = get_expected_returns(prices)
    S = get_cov_matrix(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
    raw = ef.min_volatility()
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance()
    return cleaned, {"performance": perf}

def efficient_frontier_curve(prices, points=50, method="sample_cov"):
    mu = expected_returns.mean_historical_return(prices)
    S = (
        risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        if method == "ledoit_wolf"
        else risk_models.sample_cov(prices)
    )
    ef = EfficientFrontier(mu, S)

    # Sweep across target returns
    target_returns = np.linspace(mu.min(), mu.max(), points)
    rets, vols = [], []
    for tr in target_returns:
        try:
            ef.efficient_return(target_return=tr)
            r, v, _ = ef.portfolio_performance()
            rets.append(r)
            vols.append(v)
        except Exception:
            continue
    return rets, vols

def hrp_weights(prices: pd.DataFrame):
    S = get_cov_matrix(prices)
    hrp = HRPOpt(prices)
    w = hrp.optimize()
    return w

def get_rebalance_weights(prices: pd.DataFrame, lookback_days=252, rebalance_months=1, strategy='max_sharpe', risk_free_rate=0.0):
    prices = prices.dropna()
    dates = prices.index
    start = dates[0] + pd.Timedelta(days=lookback_days)
    rebalance_dates = pd.date_range(start=start, end=dates[-1], freq=pd.DateOffset(months=rebalance_months))
    
    all_weights = []

    for current_date in rebalance_dates:
        window_start = current_date - pd.Timedelta(days=lookback_days)
        window_prices = prices.loc[window_start:current_date]
        if window_prices.shape[0] < 30:
            continue
        if strategy == 'max_sharpe':
            try:
                w, _ = max_sharpe_weights(window_prices, risk_free_rate=risk_free_rate)
            except ValueError:
                w, _ = min_volatility_weights(window_prices)
        else:
            w, _ = min_volatility_weights(window_prices)
        current_weights = pd.Series(w, index=prices.columns, name=current_date)
        all_weights.append(current_weights)

    weights_df = pd.DataFrame(all_weights)
    return weights_df
