import yfinance as yf
import pandas as pd
from typing import List

def fetch_adjusted_close(tickers: List[str], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch adjusted close prices for given tickers. Returns a DataFrame with ticker columns."""
    data = yf.download(tickers, period=period, interval=interval, progress=False, threads=True, auto_adjust=True)
    # yfinance may return different structures depending on single/multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.levels[0]:
            adj = data['Close']
        else:
            adj = data
    else:
        # single ticker
        adj = data['Close'] if 'Close' in data.columns else data
    adj = adj.dropna(how='all')
    return adj

def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        returns = (prices / prices.shift(1)).apply(lambda x: np.log(x)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns
