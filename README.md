# Portfolio Optimization & Backtesting App

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green)
![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-Enabled-orange)

---

## Overview

This project helps you figure out **how to split your money among multiple stocks** to either maximize returns or minimize risk.  

It uses **past stock prices** to calculate:

- **Max Sharpe Portfolio:** The split that aims to give the highest reward for the risk you take.  
- **Min Volatility Portfolio:** The split that is safest and has the least fluctuations.  

It also simulates how these strategies would have performed in the past — this is called a **rolling backtest**.

---

## Key Concepts

- **Expected Return:** How much a stock might go up in the future.  
- **Risk / Volatility:** How much a stock’s price jumps up and down.  
- **Efficient Frontier:** A map showing the best combinations of stocks for different levels of risk.  
- **Backtesting:** A simulation to see: “If I had used this strategy in the past, how would it have performed?”

---

## Features

- Fetches real stock data from **Yahoo Finance**.  
- Calculates optimal portfolios using **PyPortfolioOpt**.  
- Plots the **efficient frontier** to visualize risk vs. return trade-offs.  
- Runs a **rolling-window backtest** with monthly rebalancing.  
- Interactive **Streamlit app** for exploring different stocks, strategies, and time periods.

---

## How It Works

1. Pick the stocks you want to analyze.  
2. Choose a strategy: **Max Sharpe** (aggressive) or **Min Volatility** (safe).  
3. The app calculates the best way to split your money among the selected stocks.  
4. The rolling backtest simulates portfolio performance over time.  
5. View results in interactive graphs showing portfolio growth and risk/return trade-offs.

---

## Requirements

- Python 3.11  
- Packages (install via `requirements.txt`):

```text
streamlit
pandas
numpy
scipy
yfinance
plotly
matplotlib
pypfopt
scikit-learn
```

- Clone the Repository
```
git clone <https://github.com/kaipokrandt/portfoliooptimization/tree/main>

cd portfoliooptimization

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```
