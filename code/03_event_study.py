"""
STEP 2: EXPLORATORY ANALYSIS  +  STEP 3: EVENT STUDY (market-model CARs)
==========================================================================

EXPLORATORY ANALYSIS first (always do this before modeling — you want to
SEE volatility clustering, obvious level shifts, or data errors before you
trust a regression on top of them):
  - summary stats per ticker
  - rolling 30-day volatility
  - rolling correlation with the S&P 500

EVENT STUDY METHODOLOGY (the core econometric model):
  For each ticker and each event date, we estimate a "market model" —
      r_it = alpha_i + beta_i * r_mt + e_it
  over a CLEAN ESTIMATION WINDOW (250 to 30 trading days BEFORE the event,
  deliberately excluding the event window itself so the event doesn't
  contaminate the "normal return" benchmark).

  We then use alpha_i and beta_i to predict what the stock SHOULD have
  returned on each day in the EVENT WINDOW (event day -5 to +5), and the
  difference between actual and predicted is the Abnormal Return (AR):
      AR_it = r_it - (alpha_i + beta_i * r_mt)

  Summing AR over the event window gives the Cumulative Abnormal Return
  (CAR) — the standard measure of "how much extra return (positive or
  negative) is attributable to this event, after stripping out normal
  market movement."

  We test CAR significance with a simple t-test against the estimation-
  window residual standard deviation (the standard, if simplified,
  event-study test statistic).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import ast
from pathlib import Path

EST_WINDOW = (250, 30)   # (start, end) trading days before event, e.g. day -250 to day -30
EVENT_WINDOW = (-5, 5)   # trading days relative to event date


def load_data():
    prices = pd.read_csv("data/prices_long.csv", parse_dates=["Date"])
    events = pd.read_csv("data/events.csv", parse_dates=["date"])
    return prices, events


def exploratory_summary(prices: pd.DataFrame):
    """Basic descriptive stats + rolling vol/correlation. Prints, doesn't plot
    (charts happen in 05_charts.py) — keep EDA numeric-first so you actually
    read the numbers instead of skimming a picture."""
    summary = prices.groupby("ticker")["ret"].agg(
        mean="mean", std="std", skew="skew", min="min", max="max"
    )
    summary["ann_vol"] = summary["std"] * np.sqrt(252)
    print("=== Descriptive statistics (daily returns) ===")
    print(summary.round(4))
    return summary


def get_market_series(prices, market_ticker="^GSPC"):
    m = prices[prices.ticker == market_ticker].set_index("Date")["ret"]
    return m


def market_model_car(prices, events, market_ticker="^GSPC"):
    """
    Core event-study loop. For each (ticker, event) pair:
    1. Fit market model on the estimation window
    2. Predict expected returns in the event window
    3. Compute AR and CAR
    4. Compute a t-stat for CAR significance
    Returns a tidy dataframe of results — one row per ticker-event.
    """
    market = get_market_series(prices, market_ticker)
    results = []

    for _, ev in events.iterrows():
        exposed = ast.literal_eval(ev["exposed_tickers"]) if isinstance(ev["exposed_tickers"], str) else ev["exposed_tickers"]
        for ticker in exposed:
            tdf = prices[prices.ticker == ticker].set_index("Date")["ret"]
            merged = pd.concat([tdf, market], axis=1, keys=["r_i", "r_m"]).dropna()
            merged = merged.sort_index()

            # locate event date's integer position in the trading-day index
            if ev["date"] not in merged.index:
                # if event fell on a non-trading day, use the next trading day
                future_dates = merged.index[merged.index >= ev["date"]]
                if len(future_dates) == 0:
                    continue
                event_pos = merged.index.get_loc(future_dates[0])
            else:
                event_pos = merged.index.get_loc(ev["date"])

            est_start = event_pos - EST_WINDOW[0]
            est_end = event_pos - EST_WINDOW[1]
            if est_start < 0:
                continue  # not enough history before this event

            est_data = merged.iloc[est_start:est_end]

            X = sm.add_constant(est_data["r_m"])
            y = est_data["r_i"]
            model = sm.OLS(y, X).fit()
            alpha, beta = model.params["const"], model.params["r_m"]
            resid_std = model.resid.std()

            ev_start = event_pos + EVENT_WINDOW[0]
            ev_end = event_pos + EVENT_WINDOW[1] + 1
            event_data = merged.iloc[max(ev_start, 0):ev_end]

            predicted = alpha + beta * event_data["r_m"]
            ar = event_data["r_i"] - predicted
            car = ar.sum()

            n_days = len(ar)
            car_std = resid_std * np.sqrt(n_days)
            t_stat = car / car_std if car_std > 0 else np.nan

            results.append({
                "ticker": ticker,
                "event_date": ev["date"],
                "event_label": ev["label"],
                "beta": beta,
                "car": car,
                "t_stat": t_stat,
                "n_event_days": n_days,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    prices, events = load_data()
    exploratory_summary(prices)
    car_results = market_model_car(prices, events)
    car_results.to_csv("data/event_study_results.csv", index=False)
    print("\n=== Event study results (CAR by ticker x event) ===")
    numeric_cols = car_results.select_dtypes(include="number").columns
    print(car_results.assign(**{c: car_results[c].round(4) for c in numeric_cols}).to_string(index=False))
