#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install yfinance backtesting pandas numpy matplotlib')


# In[18]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy

# ================== ATR FATTO A MANO ==================

def ATR(high, low, close, period=14):
    """
    Average True Range manuale.
    Restituisce un array numpy con l'ATR.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr.values

# ================== CONFIG ==================

SYMBOL = "EURUSD=X"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

RAW_INTERVAL = "1d"   # DAILY

LOOKBACK = 20
BAND_WIDTH = 2.0
ATR_PERIOD = 14
ATR_MULTIPLIER = 3.0

INITIAL_CAPITAL = 10000
RISK_FRACTION = 0.02          # 2% dell'equity per trade
MAX_DRAWDOWN_LIMIT = 0.20     # stop a -20% dal massimo

# ================== DATA IMPORT ==================

def download_raw_data():
    df = yf.download(
        SYMBOL,
        start=START_DATE,
        end=END_DATE,
        interval=RAW_INTERVAL,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"Nessun dato per {SYMBOL}")

    # FIX MultiIndex: alcune versioni di yfinance danno colonne a 2 livelli
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano queste colonne nei dati scaricati: {missing}")

    df = df[required_cols].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def get_clean_data():
    data = download_raw_data()
    # filtro spike assurdi >10% giornalieri
    returns = data["Close"].pct_change().abs()
    data = data[returns < 0.10].dropna()
    return data

# ================== STRATEGY ==================

def rolling_mean_std(series, window):
    s = pd.Series(series)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return mean.values, std.values


class FXMeanReversionStrategy(Strategy):
    """
    Mean Reversion su EURUSD:
    - Bande attorno alla media (tipo Bollinger)
    - Long sotto banda inferiore, short sopra banda superiore
    - Stop dinamico con ATR
    - Position sizing: frazione dell'equity
    - Hard stop su max drawdown di portafoglio
    """

    lookback = LOOKBACK
    band_width = BAND_WIDTH
    atr_period = ATR_PERIOD
    atr_mult = ATR_MULTIPLIER
    risk_fraction = RISK_FRACTION
    max_dd_limit = MAX_DRAWDOWN_LIMIT

    def init(self):
        price = self.data.Close

        # media e deviazione standard rolling
        self.ma, self.std = self.I(
            rolling_mean_std, price, self.lookback, name="ma_std"
        )

        # ATR rolling
        self.atr = self.I(
            ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
        )

        # tracking equity peak e drawdown
        self.equity_peak = self.equity
        self.max_dd_seen = 0.0

    def next(self):
        price = self.data.Close[-1]

        # aggiorna drawdown
        self.equity_peak = max(self.equity_peak, self.equity)
        dd = (self.equity - self.equity_peak) / self.equity_peak
        self.max_dd_seen = min(self.max_dd_seen, dd)

        # se superiamo il max drawdown, chiudi tutto e smetti
        if abs(self.max_dd_seen) >= self.max_dd_limit:
            if self.position:
                self.position.close()
            return

        ma = self.ma[-1]
        std = self.std[-1]
        atr = self.atr[-1]

        if np.isnan(ma) or np.isnan(std) or np.isnan(atr):
            return

        upper_band = ma + self.band_width * std
        lower_band = ma - self.band_width * std

        # size come frazione dell'equity (tra 0 e 1, richiesto da backtesting.py)
        size = float(self.risk_fraction)

        # ingressi
        if not self.position:
            if price < lower_band:
                stop_price = price - self.atr_mult * atr
                self.buy(size=size, sl=stop_price)
            elif price > upper_band:
                stop_price = price + self.atr_mult * atr
                self.sell(size=size, sl=stop_price)
        else:
            # uscite: ritorno verso la media
            if self.position.is_long and price >= ma:
                self.position.close()
            elif self.position.is_short and price <= ma:
                self.position.close()

# ================== METRICS (usiamo quelle built-in di backtesting.py) ==================

def compute_performance_metrics(stats):
    """
    stats è una pd.Series con un sacco di metriche già pronte.
    Usiamo quelle, invece di ricalcolare Sharpe, Sortino ecc.
    """
    return {
        "Total Return %":       stats["Return [%]"],
        "CAGR %":               stats.get("CAGR [%]", np.nan),
        "Sharpe":               stats["Sharpe Ratio"],
        "Sortino":              stats["Sortino Ratio"],
        "Max Drawdown %":       stats["Max. Drawdown [%]"],
        "Win Rate %":           stats["Win Rate [%]"],
        "Expectancy %":         stats["Expectancy [%]"],
        "Num Trades":           stats["# Trades"],
        "Equity Final [$]":     stats["Equity Final [$]"],
    }


def print_performance_report(metrics):
    print("===== PERFORMANCE REPORT =====")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


def run_backtest():
    data = get_clean_data()

    bt = Backtest(
        data,
        FXMeanReversionStrategy,
        cash=INITIAL_CAPITAL,
        commission=0.0001,      # costo transazione/swap simulato
        trade_on_close=True,
        finalize_trades=True,   # chiude i trade aperti a fine backtest
    )

    stats = bt.run()
    # Plot interattivo (equity curve, prezzi, trades)
    bt.plot(filename=None)

    perf = compute_performance_metrics(stats)
    print_performance_report(perf)


# In[19]:


run_backtest ()


# In[21]:


from IPython.display import display

def run_backtest():
    data = get_clean_data()

    bt = Backtest(
        data,
        FXMeanReversionStrategy,
        cash=INITIAL_CAPITAL,
        commission=0.0001,      # costo transazione/swap simulato
        trade_on_close=True,
        finalize_trades=True,   # chiude i trade aperti a fine backtest
    )

    stats = bt.run()

    # Mostra la TABELLA BELLA (DataFrame)
    display(stats)

    # Mostra anche la versione text (utile se salvi in file)
    print("\n===== RAW STATS =====")
    print(stats)

    # Grafico completo (equity, trades, DD)
    bt.plot(filename=None)

    # Metriche riepilogative (Sharpe, Sortino, ecc.)
    perf = compute_performance_metrics(stats)
    print("\n===== PERFORMANCE REPORT =====")
    print_performance_report(perf)


# In[22]:


run_backtest()


# In[ ]:




