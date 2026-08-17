"""
STEP 6: CHARTS
================
Three chart types, each answering a different question:

1. Event-window price path (±10 days around each event) — the "what
   actually happened" visual, easy to explain to a non-technical reader.
2. Cumulative Abnormal Return (CAR) bar chart by ticker x event — the
   direct visualization of the event-study results.
3. Rolling 30-day volatility over time, with event dates marked as
   vertical lines — shows whether volatility regime shifts are visible
   to the eye, complementing the GARCH numbers.

Saved as PNGs in outputs/ so they can be dropped straight into a slide
deck or report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
plt.rcParams["figure.dpi"] = 120


def plot_event_window_prices(prices, events, window=10):
    for _, ev in events.iterrows():
        exposed = ast.literal_eval(ev["exposed_tickers"]) if isinstance(ev["exposed_tickers"], str) else ev["exposed_tickers"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for ticker in exposed:
            tdf = prices[prices.ticker == ticker].set_index("Date")["price"].sort_index()
            if ev["date"] not in tdf.index:
                future = tdf.index[tdf.index >= ev["date"]]
                if len(future) == 0:
                    continue
                pos = tdf.index.get_loc(future[0])
            else:
                pos = tdf.index.get_loc(ev["date"])
            window_data = tdf.iloc[max(0, pos - window):pos + window + 1]
            # Normalize to 100 at event day so different-priced stocks
            # are comparable on one axis
            normalized = window_data / window_data.iloc[min(window, len(window_data)-1)] * 100
            rel_days = np.arange(-min(window, pos), len(window_data) - min(window, pos))
            ax.plot(rel_days, normalized.values, marker="o", markersize=3, label=ticker)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{ev['label']} ({ev['date'].date()})")
        ax.set_xlabel("Trading days relative to event")
        ax.set_ylabel("Price (indexed = 100 at event day)")
        ax.legend()
        fig.tight_layout()
        fname = f"outputs/event_{ev['date'].date()}.png"
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")


def plot_car_bars(car_results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = car_results["ticker"] + "\n" + car_results["event_date"].astype(str)
    colors = ["#c0392b" if c < 0 else "#27ae60" for c in car_results["car"]]
    ax.bar(labels, car_results["car"] * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative Abnormal Return (%)")
    ax.set_title("CAR by ticker x event (±5 trading days)")
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/car_bars.png")
    plt.close(fig)
    print("Saved outputs/car_bars.png")


def plot_rolling_vol(prices, events, tickers):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ticker in tickers:
        tdf = prices[prices.ticker == ticker].set_index("Date")["ret"].sort_index()
        rolling_vol = tdf.rolling(30).std() * np.sqrt(252) * 100  # annualized %
        ax.plot(rolling_vol.index, rolling_vol.values, label=ticker, linewidth=1)
    for _, ev in events.iterrows():
        ax.axvline(ev["date"], color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("30-day rolling annualized volatility (%)")
    ax.set_title("Volatility over time, with policy events marked")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/rolling_vol.png")
    plt.close(fig)
    print("Saved outputs/rolling_vol.png")


if __name__ == "__main__":
    prices = pd.read_csv("data/prices_long.csv", parse_dates=["Date"])
    events = pd.read_csv("data/events.csv", parse_dates=["date"])
    car_results = pd.read_csv("data/event_study_results.csv", parse_dates=["event_date"])

    plot_event_window_prices(prices, events)
    plot_car_bars(car_results)
    plot_rolling_vol(prices, events, tickers=["NVDA", "TSM", "ASML", "AMD", "INTC"])
