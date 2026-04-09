"""
34_market_benchmark.py

Compute log-loss, Brier score, and accuracy for Kalshi and Polymarket
pre-tipoff implied probabilities on 2025 WNBA games.

For each game we extract:
  - Kalshi: last 1-min candle mid-price (bid+ask)/2 for the team_a YES market,
            strictly before game tipoff.
  - Polymarket: last price-history tick for team_a token, strictly before tipoff.

Both metrics use P(team_a wins) vs. actual team_a_won outcome.

Usage:
  python 34_market_benchmark.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

EPS = 1e-6  # clip for log-loss stability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(df, prob_col, outcome_col, label):
    sub = df[[prob_col, outcome_col]].dropna()
    sub = sub.copy()
    sub[prob_col] = sub[prob_col].clip(EPS, 1 - EPS)
    y_true = sub[outcome_col].values
    y_prob = sub[prob_col].values
    y_pred = (y_prob >= 0.5).astype(int)

    ll  = log_loss(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    n   = len(sub)
    return {"source": label, "n": n, "log_loss": ll, "brier": bs, "accuracy": acc}


# ---------------------------------------------------------------------------
# 1. Game outcomes + tipoff timestamps
# ---------------------------------------------------------------------------
print("Loading gold data...")
gold = pd.read_csv(
    DATA / "gold/game_xgboost_input_2025_REGPST.csv",
    usecols=["game_id", "game_ts"],
)
gold["game_ts"] = pd.to_datetime(gold["game_ts"], utc=True)


# ---------------------------------------------------------------------------
# 2. Kalshi pre-tipoff odds
# ---------------------------------------------------------------------------
print("Building Kalshi pre-tipoff odds...")

k_matched = pd.read_csv(DATA / "kalshi/wnba_2025_game_markets_matched.csv")
k_matched = k_matched[k_matched["winner"].notna()].copy()
k_matched["team_a_won"] = (k_matched["winner"] == k_matched["team_a"]).astype(int)
k_matched = k_matched.merge(gold, on="game_id", how="left")

# find market_ticker for team_a YES market
k_markets = pd.read_csv(
    DATA / "kalshi/kalshi_markets.csv",
    usecols=["market_ticker", "event_ticker", "yes_sub_title"],
)
k_game_market = (
    k_matched
    .merge(k_markets, on="event_ticker", how="left")
    .query("yes_sub_title == team_a")
    [["game_id", "event_ticker", "market_ticker", "game_ts", "team_a_won"]]
    .drop_duplicates("game_id")
)

# load candles, compute midpoint
print("  Loading Kalshi candles...")
k_candles = pd.read_csv(
    DATA / "kalshi/kalshi_candles_1m.csv",
    usecols=["market_ticker", "end_period_ts", "yes_bid_close", "yes_ask_close"],
)
k_candles["end_period_ts"] = pd.to_datetime(k_candles["end_period_ts"], utc=True)
k_candles["mid"] = (k_candles["yes_bid_close"] + k_candles["yes_ask_close"]) / 2

# restrict to relevant tickers and drop rows with no midpoint
valid_k = set(k_game_market["market_ticker"].dropna())
k_candles = k_candles[k_candles["market_ticker"].isin(valid_k) & k_candles["mid"].notna()]
k_candles = k_candles.sort_values("end_period_ts")

# merge_asof: last candle before each game tipoff
k_game_sorted = k_game_market.dropna(subset=["game_ts", "market_ticker"]).sort_values("game_ts")
kalshi_pre = pd.merge_asof(
    k_game_sorted,
    k_candles[["market_ticker", "end_period_ts", "mid"]],
    left_on="game_ts",
    right_on="end_period_ts",
    by="market_ticker",
    direction="backward",
)
kalshi_pre["mins_before_tipoff"] = (
    (kalshi_pre["game_ts"] - kalshi_pre["end_period_ts"]).dt.total_seconds() / 60
)

print(f"  Kalshi games with pre-tipoff price: {kalshi_pre['mid'].notna().sum()} / {len(kalshi_pre)}")
print(f"  Median mins before tipoff: {kalshi_pre['mins_before_tipoff'].median():.1f}")


# ---------------------------------------------------------------------------
# 3. Polymarket pre-tipoff odds
# ---------------------------------------------------------------------------
print("Building Polymarket pre-tipoff odds...")

p_matched = pd.read_csv(DATA / "polymarket/wnba_2025_game_markets_matched.csv")
p_matched = p_matched[p_matched["winner"].notna()].copy()
p_matched["team_a_won"] = (p_matched["winner"] == p_matched["team_a"]).astype(int)
p_matched = p_matched.merge(gold, on="game_id", how="left")

# load price history, restrict to team_a tokens
print("  Loading Polymarket price history...")
p_prices = pd.read_csv(DATA / "polymarket/polymarket_prices_history.csv")
p_prices["ts"] = pd.to_datetime(p_prices["ts"], utc=True)

valid_p = set(p_matched["team_a_token_id"].dropna())
p_prices = p_prices[p_prices["token_id"].isin(valid_p)].sort_values("ts")

p_game_sorted = (
    p_matched[["game_id", "team_a_token_id", "game_ts", "team_a_won"]]
    .drop_duplicates("game_id")
    .dropna(subset=["game_ts", "team_a_token_id"])
    .sort_values("game_ts")
)
poly_pre = pd.merge_asof(
    p_game_sorted,
    p_prices[["token_id", "ts", "price"]],
    left_on="game_ts",
    right_on="ts",
    left_by="team_a_token_id",
    right_by="token_id",
    direction="backward",
)
poly_pre["mins_before_tipoff"] = (
    (poly_pre["game_ts"] - poly_pre["ts"]).dt.total_seconds() / 60
)

print(f"  Polymarket games with pre-tipoff price: {poly_pre['price'].notna().sum()} / {len(poly_pre)}")
print(f"  Median mins before tipoff: {poly_pre['mins_before_tipoff'].median():.1f}")


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MARKET BENCHMARK: Pre-Tipoff Implied Probability Accuracy")
print("=" * 60)

results = []
results.append(compute_metrics(kalshi_pre,  "mid",   "team_a_won", "Kalshi"))
results.append(compute_metrics(poly_pre,    "price", "team_a_won", "Polymarket"))

# common games (both platforms have a pre-tipoff price)
common = (
    kalshi_pre[["game_id", "mid",   "team_a_won"]]
    .merge(
        poly_pre[["game_id", "price"]],
        on="game_id", how="inner"
    )
    .dropna(subset=["mid", "price"])
)
if len(common) > 0:
    results.append(compute_metrics(common, "mid",   "team_a_won", "Kalshi (common)"))
    results.append(compute_metrics(common, "price", "team_a_won", "Polymarket (common)"))

summary = pd.DataFrame(results)
print()
print(summary.to_string(index=False, float_format="%.4f"))


# ---------------------------------------------------------------------------
# 5. Per-game detail (optional diagnostic)
# ---------------------------------------------------------------------------
print("\n--- Kalshi sample (first 10 games) ---")
print(
    kalshi_pre[["game_id", "market_ticker", "game_ts", "mid", "team_a_won", "mins_before_tipoff"]]
    .head(10)
    .to_string(index=False)
)

print("\n--- Polymarket sample (first 10 games) ---")
print(
    poly_pre[["game_id", "team_a_token_id", "game_ts", "price", "team_a_won", "mins_before_tipoff"]]
    .head(10)
    .to_string(index=False)
)
