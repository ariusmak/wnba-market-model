"""Market-data loaders shared across model-vs-market and trading notebooks.

Two shapes are supported:

  1. Pre-tipoff snapshot (one probability per game) — for model-vs-market
     forecasting comparisons.
  2. Pre-tipoff candle stream + historical-trades stream — for trading
     backtests (entry windows, fill simulation).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KALSHI_DIR_DEFAULT = PROJECT_ROOT / "data" / "kalshi"
POLY_DIR_DEFAULT = PROJECT_ROOT / "data" / "polymarket"


# --------------------------------------------------------------------------- #
# Snapshot loaders (one row per game, P(home win))                            #
# --------------------------------------------------------------------------- #

def load_kalshi_pretipoff_probs(kalshi_dir: Path = KALSHI_DIR_DEFAULT,
                                cutoff_hour: int = 23) -> pd.DataFrame:
    """Return DataFrame with columns: game_id, kalshi_prob (P(home_win)).

    Deduplicated to one row per `game_id` (drops rows missing a `game_id` and
    averages any remaining duplicates so a noisy match file does not silently
    inflate downstream merges).
    """
    matched = pd.read_csv(kalshi_dir / "wnba_2025_game_markets_matched.csv")
    markets = pd.read_csv(kalshi_dir / "kalshi_markets.csv")
    candles = pd.read_csv(kalshi_dir / "kalshi_candles_1m.csv")

    mkt_team = dict(zip(markets["market_ticker"], markets["yes_sub_title"]))
    candles["event_ticker"] = candles["market_ticker"].str.rsplit("-", n=1).str[0]
    candles["ts"] = pd.to_datetime(candles["end_period_ts"], utc=True)

    matched["cutoff"] = pd.to_datetime(matched["game_date"]) + pd.Timedelta(hours=cutoff_hour)
    matched["cutoff"] = matched["cutoff"].dt.tz_localize("UTC")
    cutoff_map = dict(zip(matched["event_ticker"], matched["cutoff"]))
    candles["cutoff"] = candles["event_ticker"].map(cutoff_map)

    pre = candles[candles["ts"] <= candles["cutoff"]].copy()
    pre["mid_price"] = (pre["yes_bid_close"] + pre["yes_ask_close"]) / 2
    last = pre.sort_values("ts").groupby("event_ticker").tail(1)
    last["mkt_team"] = last["market_ticker"].map(mkt_team)

    odds = last[["event_ticker", "mid_price", "mkt_team"]].merge(
        matched[["event_ticker", "game_id", "team_a", "team_b"]],
        on="event_ticker",
    )
    # team_b == home team in the matched file
    odds["kalshi_prob"] = np.where(
        odds["mkt_team"] == odds["team_b"],
        odds["mid_price"], 1 - odds["mid_price"],
    )
    odds = odds[["game_id", "kalshi_prob"]].dropna(subset=["game_id"])
    return odds.groupby("game_id", as_index=False)["kalshi_prob"].mean()


def load_polymarket_pretipoff_probs(poly_dir: Path = POLY_DIR_DEFAULT) -> pd.DataFrame:
    """Return DataFrame with columns: game_id, poly_prob (P(home_win)).

    Polymarket sometimes has multiple condition_ids resolving to the same
    `game_id` (re-listed markets, etc.). Deduplicated to one row per
    `game_id` by averaging — otherwise a left-join against per-game model
    predictions silently fans out and double-counts those games downstream.
    """
    matched = pd.read_csv(poly_dir / "wnba_2025_game_markets_matched.csv")
    markets = pd.read_csv(poly_dir / "polymarket_markets.csv")
    prices = pd.read_csv(poly_dir / "polymarket_prices_history.csv")

    gst = dict(zip(markets["condition_id"], markets["game_start_ts"]))
    matched["game_start_ts"] = pd.to_datetime(matched["condition_id"].map(gst), utc=True)

    team_a_tokens = set(matched["team_a_token_id"].astype(str))
    prices["ts"] = pd.to_datetime(prices["ts"], utc=True)
    a = prices[prices["token_id"].astype(str).isin(team_a_tokens)].copy()

    tok_to_cid = dict(zip(matched["team_a_token_id"].astype(str), matched["condition_id"]))
    a["condition_id"] = a["token_id"].astype(str).map(tok_to_cid)
    cid_to_gst = dict(zip(matched["condition_id"], matched["game_start_ts"]))
    a["game_start_ts"] = a["condition_id"].map(cid_to_gst)

    pre = a[a["ts"] <= a["game_start_ts"]].copy()
    last = pre.sort_values("ts").groupby("condition_id").tail(1)
    odds = last[["condition_id", "price"]].merge(
        matched[["condition_id", "game_id"]], on="condition_id",
    )
    # team_a is away — last['price'] = P(away win) → flip
    odds["poly_prob"] = 1 - odds["price"]
    odds = odds[["game_id", "poly_prob"]].dropna(subset=["game_id"])
    return odds.groupby("game_id", as_index=False)["poly_prob"].mean()


# --------------------------------------------------------------------------- #
# Trading-window structures                                                   #
# --------------------------------------------------------------------------- #

def build_kalshi_trading_index(
    signals: pd.DataFrame,
    kalshi_dir: Path = KALSHI_DIR_DEFAULT,
    label_col: str = "home_win",
    pred_cols: tuple[str, ...] = ("p_full_model", "p_elo"),
):
    """Build the per-ticker info / pre-tipoff candle / window-trades structures
    used by trading and liquidity notebooks.

    `signals` must include: game_id, game_ts, home_team_id, away_team_id, label,
    and every column in `pred_cols`.

    Returns a dict with keys:
      ticker_info  (market_ticker → info dict, including entry_start_half)
      pretip       (market_ticker → DataFrame[ts, yes_bid_close, yes_ask_close])
      wt           (DataFrame of historical trades inside the entry window)
      home_tickers (DataFrame of home-side tickers per game)
    """
    matched = pd.read_csv(kalshi_dir / "wnba_2025_game_markets_matched.csv")
    markets = pd.read_csv(
        kalshi_dir / "kalshi_markets.csv",
        usecols=["market_ticker", "event_ticker", "yes_sub_title", "result"],
    )
    settle = pd.read_csv(
        kalshi_dir / "kalshi_settlements.csv",
        usecols=["market_ticker", "result", "settlement_value_dollars"],
    )

    gm = (
        matched[["event_ticker", "game_id", "team_a", "team_a_id", "team_b", "team_b_id"]]
        .merge(markets, on="event_ticker", how="inner")
    )
    gm = gm.merge(
        signals[["game_id", "home_team_id", "away_team_id", "game_ts", label_col,
                 *pred_cols]],
        on="game_id", how="inner",
    )
    gm["ticker_team_id"] = np.where(
        gm["yes_sub_title"] == gm["team_a"], gm["team_a_id"], gm["team_b_id"],
    )
    gm["is_home_ticker"] = gm["ticker_team_id"] == gm["home_team_id"]
    home = gm[gm["is_home_ticker"]].drop_duplicates("game_id").copy()
    home = home.merge(
        settle[["market_ticker", "result"]].rename(columns={"result": "settle_result"}),
        on="market_ticker", how="left",
    )

    candles = pd.read_csv(
        kalshi_dir / "kalshi_candles_1m.csv",
        usecols=["market_ticker", "end_period_ts", "yes_bid_close", "yes_ask_close"],
    )
    candles["ts"] = pd.to_datetime(candles["end_period_ts"], utc=True)
    candles = candles[candles["market_ticker"].isin(set(home["market_ticker"]))].copy()
    candles = candles.sort_values(["market_ticker", "ts"]).reset_index(drop=True)

    matched_ts = pd.read_csv(
        kalshi_dir / "wnba_2025_game_markets_matched.csv",
        usecols=["event_ticker", "open_time"],
    )
    matched_ts["open_time"] = pd.to_datetime(matched_ts["open_time"], utc=True)
    open_times = matched_ts.set_index("event_ticker")["open_time"].to_dict()
    home["open_time"] = home["event_ticker"].map(open_times)

    info_cols = ["game_id", "game_ts", "open_time", *pred_cols, label_col]
    ticker_info = home.set_index("market_ticker")[info_cols].to_dict("index")
    for tkr, info in ticker_info.items():
        lifespan = info["game_ts"] - info["open_time"]
        info["entry_start_half"] = info["open_time"] + lifespan * 0.50
        info["entry_start_twothirds"] = info["open_time"] + lifespan * (2 / 3)
        info["lifespan_h"] = lifespan.total_seconds() / 3600

    pretip: dict[str, pd.DataFrame] = {}
    for tkr, info in ticker_info.items():
        sub = candles[
            (candles["market_ticker"] == tkr)
            & (candles["ts"] >= info["entry_start_half"])
            & (candles["ts"] <= info["game_ts"])
        ]
        if not sub.empty:
            pretip[tkr] = sub[["ts", "yes_bid_close", "yes_ask_close"]].reset_index(drop=True)

    trades_raw = pd.read_csv(
        kalshi_dir / "kalshi_trades.csv",
        usecols=["market_ticker", "trade_ts", "yes_price", "no_price", "count", "taker_side"],
    )
    trades_raw["ts"] = pd.to_datetime(trades_raw["trade_ts"], utc=True)
    trades_raw = trades_raw[trades_raw["market_ticker"].isin(set(ticker_info))].copy()
    window_chunks = []
    for tkr, info in ticker_info.items():
        sub = trades_raw[
            (trades_raw["market_ticker"] == tkr)
            & (trades_raw["ts"] >= info["entry_start_half"])
            & (trades_raw["ts"] <= info["game_ts"])
        ].copy()
        if not sub.empty:
            sub["game_id"] = info["game_id"]
            window_chunks.append(sub)
    wt = pd.concat(window_chunks, ignore_index=True) if window_chunks else pd.DataFrame()

    return {
        "ticker_info":  ticker_info,
        "pretip":       pretip,
        "wt":           wt,
        "home_tickers": home,
    }
