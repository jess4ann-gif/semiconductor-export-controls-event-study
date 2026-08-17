"""
STEP 4: LLM-ASSISTED RESEARCH — news headline classification (batch version)
================================================================================
Reads data/headlines.csv (event_date, ticker, headline columns -- you fill
this in yourself with real headlines you've gathered), groups headlines by
(event_date, ticker), and calls the LLM once per group to classify sentiment.
Saves everything to one combined results file.

WHY THIS DESIGN: one call per (event, ticker) group rather than one call per
headline keeps API costs and rate limits sane, and gives the LLM the useful
context of "these headlines are all about the same event" when judging tone.

CRITICAL EVALUATION STEP (do this, and document it in your report):
  After running, hand-check a RANDOM SAMPLE of ~15-20 rows in the output
  yourself. LLM sentiment classifiers are prone to two specific failure
  modes worth naming in your report:
    1. Missing sarcasm/hedged language ("not expected to be as bad as
       feared" often gets misclassified as negative because of the
       negation).
    2. Conflating "this is important/big news" with "this is bad news" --
       a neutral-but-major regulatory headline sometimes gets scored
       negative purely because it sounds serious.
  Report your hand-check agreement rate (e.g. "18/20 = 90% agreement")
  as evidence you evaluated rather than blindly trusted the LLM output.

HOW TO USE:
  1. Open data/headlines.csv and replace the example rows with real
     headlines you've gathered (Google News / Reuters / Bloomberg) for
     each event_date + ticker combination that's in your events.csv.
  2. Set your API key: $env:ANTHROPIC_API_KEY="your-key-here"
  3. Run: python 06_llm_sentiment.py
  4. Results land in data/sentiment_results.csv
"""

import os
import json
import time
import anthropic
import pandas as pd

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a financial news sentiment classifier for equity
research. For each headline, classify sentiment specifically FOR THE NAMED
TICKER's stock price outlook as one of: positive, negative, neutral.
Respond with ONLY a JSON array of objects: [{"headline": "...", "sentiment": "...", "reason": "..."}]
No preamble, no markdown fences, just the JSON array."""


def classify_headlines(headlines: list[str], ticker: str) -> pd.DataFrame:
    user_prompt = f"Ticker: {ticker}\nHeadlines:\n" + "\n".join(f"- {h}" for h in headlines)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # LLMs occasionally wrap JSON in markdown fences despite instructions
        # not to -- strip and retry rather than silently failing.
        cleaned = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
    return pd.DataFrame(parsed)


def run_batch(headlines_path="data/headlines.csv"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Run:\n"
            '  $env:ANTHROPIC_API_KEY="your-key-here"\n'
            "in this PowerShell session before running this script."
        )

    df = pd.read_csv(headlines_path)
    required_cols = {"event_date", "ticker", "headline"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{headlines_path} must have columns: {required_cols}")

    all_results = []
    groups = df.groupby(["event_date", "ticker"])
    print(f"Found {len(groups)} event/ticker groups to classify...")

    for (event_date, ticker), group in groups:
        headlines = group["headline"].tolist()
        print(f"  Classifying {len(headlines)} headline(s) for {ticker} @ {event_date}...")
        try:
            result = classify_headlines(headlines, ticker)
        except Exception as e:
            print(f"    FAILED for {ticker} @ {event_date}: {e}")
            continue
        result["event_date"] = event_date
        result["ticker"] = ticker
        all_results.append(result)
        time.sleep(0.5)  # small pause to stay comfortably under rate limits

    if not all_results:
        print("No results produced -- check your headlines.csv and API key.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[["event_date", "ticker", "headline", "sentiment", "reason"]]
    return combined


if __name__ == "__main__":
    results = run_batch()
    if not results.empty:
        results.to_csv("data/sentiment_results.csv", index=False)
        print(f"\nSaved {len(results)} classified headlines to data/sentiment_results.csv\n")
        print(results.to_string(index=False))

        print("\n--- MANUAL SPOT-CHECK REMINDER ---")
        print("Pull a random sample of ~15-20 rows from data/sentiment_results.csv")
        print("and hand-check the sentiment label against the headline yourself.")
        print("Report your agreement rate in your write-up.")
