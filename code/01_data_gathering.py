"""
STEP 1: DATA GATHERING
=======================
Pulls daily OHLCV data for semiconductor-sector tickers + a sector benchmark
(SOXX ETF) and the S&P 500 (^GSPC, used as the market factor for the event
study's market-model regression). Saves a tidy long-format CSV that every
later script reads from, so the whole pipeline only hits the network once.

Run this locally (yfinance needs to reach Yahoo Finance, which this sandbox
can't access) — everything downstream in this project reads from the CSV
it produces.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# --- Config -----------------------------------------------------------
# Firms chosen to vary in exposure to US-China chip export controls:
# TSM/ASML = high exposure (fab equipment/foundry directly targeted by
#            controls); NVDA = high exposure (advanced AI chips banned
#            from export to China); AMD = moderate; INTC = lower
#            (more US-domestic fab footprint, less China-reliant revenue)
TICKERS = ["NVDA", "TSM", "ASML", "AMD", "INTC"]
BENCHMARK = "SOXX"      # sector ETF, for descriptive comparison
MARKET = "^GSPC"        # S&P 500, used as market factor in event study
START = "2022-01-01"
END = "2026-08-01"

OUT_PATH = Path("data/prices_long.csv")
OUT_PATH.parent.mkdir(exist_ok=True)


def fetch_prices(tickers, start, end):
    """Download adjusted close prices for a list of tickers, return wide df."""
    all_tickers = tickers + [BENCHMARK, MARKET]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
    # auto_adjust=True already gives split/dividend-adjusted 'Close'
    prices = raw["Close"]
    return prices


def to_long_format(prices_wide):
    """
    Reshape wide (date x ticker) price matrix into long (tidy) format:
    columns = [date, ticker, price, ret, log_ret]
    Tidy format is what statsmodels/pandas groupby operations want, and
    it's the format the event-study and GARCH scripts expect.
    """
    long = prices_wide.reset_index().melt(id_vars="Date", var_name="ticker", value_name="price")
    long = long.sort_values(["ticker", "Date"]).reset_index(drop=True)
    import numpy as np
    long["ret"] = long.groupby("ticker")["price"].pct_change()
    long["log_ret"] = long.groupby("ticker")["price"].transform(lambda s: np.log(s / s.shift(1)))
    return long.dropna(subset=["ret"])


if __name__ == "__main__":
    wide = fetch_prices(TICKERS, START, END)
    long = to_long_format(wide)
    long.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(long):,} rows to {OUT_PATH}")
    print(long.head())
