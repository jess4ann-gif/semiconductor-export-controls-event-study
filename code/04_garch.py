"""
STEP 3b: GARCH(1,1) VOLATILITY MODEL
======================================
The event study (step above) asks "did the LEVEL of returns move?"
GARCH asks a complementary question: "did the VOLATILITY REGIME shift?"
This matters for the business conclusion — a firm can have a small mean
return reaction but a lasting jump in volatility, which is exactly the
kind of thing a risk manager / portfolio hedger cares about.

MODEL: r_t = mu + e_t,  e_t = sigma_t * z_t,  z_t ~ N(0,1)
       sigma_t^2 = omega + alpha * e_(t-1)^2 + beta * sigma_(t-1)^2

We fit GARCH(1,1) separately on a PRE-event window and a POST-event
window for each exposed ticker, then compare persistence (alpha+beta)
and unconditional variance (omega / (1 - alpha - beta)) before vs after.
A rise in unconditional variance = the market now prices this stock as
structurally riskier post-shock, not just reacting to one day's news.
"""

import pandas as pd
import numpy as np
from arch import arch_model

WINDOW_DAYS = 120  # trading days before/after each event to fit GARCH on


def fit_garch(returns_pct):
    """
    arch_model expects returns roughly on a 0-100 scale (not decimals) for
    numerical stability of the optimizer -- a common gotcha. We handle the
    x100 scaling here so callers can just pass raw decimal returns.
    """
    am = arch_model(returns_pct * 100, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res


def compare_pre_post(prices, events):
    results = []
    for _, ev in events.iterrows():
        import ast
        exposed = ast.literal_eval(ev["exposed_tickers"]) if isinstance(ev["exposed_tickers"], str) else ev["exposed_tickers"]
        for ticker in exposed:
            tdf = prices[prices.ticker == ticker].set_index("Date")["ret"].sort_index()
            if ev["date"] not in tdf.index:
                future = tdf.index[tdf.index >= ev["date"]]
                if len(future) == 0:
                    continue
                event_pos = tdf.index.get_loc(future[0])
            else:
                event_pos = tdf.index.get_loc(ev["date"])

            pre = tdf.iloc[max(0, event_pos - WINDOW_DAYS):event_pos]
            post = tdf.iloc[event_pos:event_pos + WINDOW_DAYS]

            if len(pre) < 60 or len(post) < 60:
                continue  # GARCH needs a reasonable sample to converge

            try:
                pre_fit = fit_garch(pre)
                post_fit = fit_garch(post)
            except Exception as e:
                print(f"GARCH failed to converge for {ticker} @ {ev['date']}: {e}")
                continue

            def uncond_var(fit):
                omega = fit.params["omega"]
                a = fit.params["alpha[1]"]
                b = fit.params["beta[1]"]
                persistence = a + b
                denom = 1 - persistence
                # Guard against near-unit-root persistence (denom -> 0), which
                # makes the unconditional variance formula numerically
                # unstable/meaningless -- flag it as NaN rather than report
                # a spurious near-zero or huge number.
                if denom < 0.02:
                    return np.nan, persistence
                return omega / denom, persistence

            pre_var, pre_persist = uncond_var(pre_fit)
            post_var, post_persist = uncond_var(post_fit)

            results.append({
                "ticker": ticker,
                "event_date": ev["date"],
                "event_label": ev["label"],
                "pre_uncond_vol_pct": np.sqrt(pre_var) if pre_var == pre_var else np.nan,
                "post_uncond_vol_pct": np.sqrt(post_var) if post_var == post_var else np.nan,
                "pre_persistence": pre_persist,
                "post_persistence": post_persist,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    prices = pd.read_csv("data/prices_long.csv", parse_dates=["Date"])
    events = pd.read_csv("data/events.csv", parse_dates=["date"])
    garch_results = compare_pre_post(prices, events)
    garch_results.to_csv("data/garch_results.csv", index=False)
    numeric_cols = garch_results.select_dtypes(include="number").columns
    print(garch_results.assign(**{c: garch_results[c].round(4) for c in numeric_cols}).to_string(index=False))
