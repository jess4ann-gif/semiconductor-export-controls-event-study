# Semiconductor Export Controls & Equity Returns: An LLM-Assisted Event Study

## Project purpose
Demonstrate an efficient, critically-supervised Python + LLM workflow applied
to a live trade-policy question: **how do US-China semiconductor export
controls move the stocks of firms with different exposure to those
controls?**

## Pipeline 
1. `01_data_gathering.py` — pulls daily prices (NVDA, TSM, ASML, AMD, INTC,
   SOXX, ^GSPC) via yfinance, reshapes to tidy long format.
2. `02_event_list.py` — human-verified list of export-control policy events
   (LLM-assisted first draft, hand-checked against primary sources — see
   in-file notes on what the LLM got wrong).
3. `03_event_study.py` — exploratory stats + market-model event study
   (CAR, t-stats) per ticker x event.
4. `04_garch.py` — GARCH(1,1) pre/post-event volatility regime comparison.
5. `05_charts.py` — event-window price charts, CAR bar chart, rolling
   volatility chart.
6. `06_llm_sentiment.py` — LLM-based news headline sentiment classification
   for each event, with a manual spot-check protocol.

## Methodology summary
- **Event study**: market-model abnormal returns, estimation window
  [-250,-30] trading days, event window [-5,+5]. Standard Fama-style
  event-study design.
- **GARCH(1,1)**: separately fit pre- and post-event to test whether the
  *volatility regime* (not just the mean return) shifted.
- **LLM use**: (a) first-draft event-date research, hand-verified;
  (b) news headline sentiment classification via the Anthropic API.

## Critical evaluation of LLM output
Document specifically:
- this project had two real code bugs worth citing as an example: an `eval()` security
  anti-pattern in event-list parsing, replaced with `ast.literal_eval`;
  and a GARCH unconditional-variance calculation that produced a
  nonsensical near-zero value when persistence was close to 1, fixed by
  adding a numerical-stability guard.

## Business/investment conclusion 
Structure:
1. **Which firms showed statistically significant CAR around export
   control events, and in which direction?** (pulled numbers from
   `data/event_study_results.csv`)
2. **Did volatility regimes shift, and for whom?** (from
   `data/garch_results.csv`) — a firm can show no CAR but a lasting vol
   increase, which matters for hedging/options pricing even if the
   direction of stock movement is unclear.
3. **Practical takeaway**: ASML shows the clearest post-event volatility regime shift — estimated persistence rose from 0.15 to 0.82 following the December 2024 equipment/HBM restrictions, even though its CAR over that window was positive (+3.2%). Nvidia shows the opposite pattern: a sharply negative CAR (-12.8%) but falling persistence (0.97 → 0.83). This decoupling between price direction and volatility persistence is the key risk-management signal — a concentrated semiconductor book can't rely on CAR sign alone to anticipate risk around BIS announcement dates, since the company with the calmer initial price reaction was the one that entered the more persistently volatile regime. Options-based hedges (which pay off on volatility itself) are likely more efficient than simple directional bets given this decoupling.
4. **Caveat**: small event count (n=4-6 events) limits statistical power.
