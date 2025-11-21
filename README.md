# Systematic FX Mean Reversion Model

A robust, risk-controlled trading strategy implemented in Python

This project implements a complete systematic trading model based on mean reversion signals applied to EUR/USD daily data from 2015 to 2024.  
The objective is to design a transparent, interpretable and well-risk-managed trading system following professional quantitative research standards, rather than to maximize returns through overfitting.

## Overview

The project includes all core components of a quantitative trading pipeline:  
data import and cleaning, signal generation, volatility-adjusted risk management, position sizing, backtesting and performance evaluation.  
All results are obtained using real market data from Yahoo Finance and the backtesting.py framework.

## Strategy Logic

The strategy follows a simple mean reversion idea: price tends to revert toward its recent average after moving too far away from it.  
The model uses a rolling mean and standard deviation to construct dynamic upper and lower bands.  
When price falls below the lower band, the system opens a long position; when price rises above the upper band, it opens a short position.  
Positions are closed once price returns to the rolling mean.  
This structure aims to exploit short-term dislocations while avoiding excessive trading.

## Risk Management

Risk control is the central element of the strategy.  
Stop-loss levels are based on the Average True Range (ATR) to ensure that risk adjusts to market volatility.  
Position size is determined as a fixed fraction of equity, allowing the system to maintain consistent risk exposure over time.  
A global maximum drawdown limit stops trading entirely if cumulative losses exceed a predefined threshold.  
Basic data cleaning is applied to remove abnormal price spikes and guarantee stable calculations.

## Backtesting Framework

Backtesting is carried out with backtesting.py, which manages order execution, performance tracking, commissions, drawdown analysis and summary statistics.  
The entire pipeline is implemented in a single Python file to keep the structure clear and easy to review.

## Performance Summary (EUR/USD, 2015–2024)

The strategy delivers a stable equity curve with limited drawdowns and a small but positive total return over the full sample.  
This outcome is consistent with the efficiency of major FX markets, where simple mean reversion models rarely generate large excess returns without additional filters or multi-factor components.

Typical metrics observed in the baseline test:

- Total return close to zero but positive  
- CAGR close to zero  
- Max drawdown around 0.3%  
- Win rate around 65%  
- Expectancy slightly positive  
- Very low volatility and stable exposure  

The emphasis is on robustness rather than performance enlargement.

## Interpretation

The results reflect the reality of FX markets: predictability is limited and excess returns are hard to extract without more complex modelling.  
The strategy behaves in a stable and controlled manner, does not blow up, and does not rely on curve-fitted parameters.  
This makes it a useful baseline model and a good foundation for further research, as opposed to an unrealistic or over-optimized system.

## How to Run

Required packages: yfinance, backtesting.py, pandas, numpy and matplotlib.  
After installing the dependencies, the model can be executed by calling the backtest function in the main script or notebook.

## Possible Extensions

Potential improvements include adding a trend or volatility regime filter, testing multiple assets, combining mean reversion with momentum signals, or incorporating simple machine learning classifiers to detect favorable environments.
