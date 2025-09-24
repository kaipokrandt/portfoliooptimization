import streamlit as st
import pandas as pd
import numpy as np
from portfolio.data import fetch_adjusted_close, compute_returns
from portfolio.optimizer import max_sharpe_weights, min_volatility_weights, efficient_frontier_curve
from portfolio.visuals import plot_frontier_and_random, pie_weights
from pypfopt import expected_returns, risk_models
import datetime

st.set_page_config(page_title='Portfolio Optimizer', layout='wide')
st.title('Portfolio Optimization & Backtesting')

with st.sidebar:
    st.header('Settings')
    tickers_input = st.text_input('Tickers (comma separated):', value='AAPL,MSFT,GOOG,AMZN,TSLA')
    period = st.selectbox('Historical period:', ['1y','2y','5y','10y'], index=2)
    n_random = st.slider('Number of random Monte Carlo portfolios:', 1000, 10000, 3000, step=500)
    risk_free = st.number_input('Risk-free rate (annual, decimal):', value=0.02, step=0.005)
    rebalance_months = st.slider('Rebalance every N months (backtest):', 1, 6, 1)
    run = st.button('Run')

def monte_carlo_random(prices: pd.DataFrame, n_portfolios: int = 3000, risk_free: float = 0.02):
    returns = prices.pct_change().dropna()
    mean = returns.mean() * 252
    cov = returns.cov() * 252
    n = prices.shape[1]
    results = []
    weights = []
    for _ in range(n_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        p_ret = np.dot(w, mean)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        p_sharpe = (p_ret - risk_free)/p_vol if p_vol>0 else 0
        results.append((p_ret, p_vol, p_sharpe))
        weights.append(w)
    df = pd.DataFrame(results, columns=['return','volatility','sharpe'])
    return df, np.array(weights)

def backtest_rolling(prices: pd.DataFrame, lookback_days: int = 252, rebalance_months: int = 1, strategy: str = 'max_sharpe'):
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    dates = returns.index
    
    start = dates[0] + pd.Timedelta(days=lookback_days)
    rebalance_dates = pd.date_range(start=start, end=dates[-1], freq=pd.DateOffset(months=rebalance_months))
    
    portfolio_daily = pd.Series(1.0, index=dates)
    current_value = 1.0
    current_weights = None
    all_weights = []

    for i, current_date in enumerate(dates):
        if current_date in rebalance_dates:
            window_start = current_date - pd.Timedelta(days=lookback_days)
            window_prices = prices.loc[window_start:current_date]
            if window_prices.shape[0] < 30:
                continue
            if strategy == 'max_sharpe':
                try:
                    w, _ = max_sharpe_weights(window_prices, risk_free_rate=risk_free)
                except ValueError:
                    w, _ = min_volatility_weights(window_prices)
            else:
                w, _ = min_volatility_weights(window_prices)
            current_weights = pd.Series(w, index=prices.columns)
            current_weights.name = current_date
            all_weights.append(current_weights)

        if current_weights is not None:
            daily_ret = (returns.loc[current_date] * current_weights).sum()
            current_value *= (1 + daily_ret)
            portfolio_daily.loc[current_date] = current_value
        else:
            portfolio_daily.loc[current_date] = current_value

    weights_df = pd.DataFrame(all_weights)
    return portfolio_daily.fillna(method='ffill').fillna(1.0), weights_df

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    prices = fetch_adjusted_close(tickers, period=period)
    if prices.shape[1] < 2:
        st.error('Please provide at least two valid tickers.')
    else:
        st.subheader('Data Overview')
        st.write(prices.tail())

        # Random portfolios
        df_rand, _ = monte_carlo_random(prices, n_portfolios=n_random, risk_free=risk_free)

        # Optimal portfolios
        ms_w, ms_meta = max_sharpe_weights(prices, risk_free_rate=risk_free)
        mv_w, mv_meta = min_volatility_weights(prices)
        frontier_rets, frontier_vols = efficient_frontier_curve(prices, points=50)

        optimal_points = {
            'Max Sharpe': {'ret': ms_meta['performance'][0], 'vol': ms_meta['performance'][1]},
            'Min Vol': {'ret': mv_meta['performance'][0], 'vol': mv_meta['performance'][1]}
        }

        st.plotly_chart(plot_frontier_and_random(df_rand, frontier_vols, frontier_rets, optimal_points), use_container_width=True)

        st.subheader('Optimal Weights')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Max Sharpe Weights')
            ms_series = pd.Series(ms_w)
            st.dataframe(ms_series.to_frame('weight').style.format('{:.2%}'))
            st.plotly_chart(pie_weights(ms_series, title='Max Sharpe Weights'))
        with col2:
            st.markdown('### Min Vol Weights')
            mv_series = pd.Series(mv_w)
            st.dataframe(mv_series.to_frame('weight').style.format('{:.2%}'))
            st.plotly_chart(pie_weights(mv_series, title='Min Vol Weights'))

        # Backtest
        st.subheader('Backtest (Rolling Rebalance)')
        bt_max, weights_max = backtest_rolling(prices, lookback_days=252, rebalance_months=rebalance_months, strategy='max_sharpe')
        bt_min, weights_min = backtest_rolling(prices, lookback_days=252, rebalance_months=rebalance_months, strategy='min_volatility')
        bt_df = pd.DataFrame({'max_sharpe': bt_max, 'min_vol': bt_min})
        st.line_chart(bt_df.fillna(method='ffill').apply(lambda x: x / x.iloc[0]))

        # CSV Download Buttons
        st.download_button(
            label="Download Backtest Portfolio Values CSV",
            data=bt_df.to_csv().encode('utf-8'),
            file_name="backtest_portfolio_values.csv",
            mime="text/csv"
        )

        weights_all = pd.concat([weights_max, weights_min], axis=1)
        weights_all.columns = [f"{col}_max_sharpe" for col in weights_max.columns] + [f"{col}_min_vol" for col in weights_min.columns]
        st.download_button(
            label="Download Portfolio Weights CSV",
            data=weights_all.to_csv().encode('utf-8'),
            file_name="portfolio_weights.csv",
            mime="text/csv"
        )

        st.subheader('Portfolio Weights Preview')
        st.dataframe(weights_all.tail())
