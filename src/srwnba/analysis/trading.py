"""Trading engine helpers: entry collection, sizing, fill simulation, edge buckets.

Pulled from `trading_results2.ipynb` and `return_investigation.ipynb` so the
trading-strategy, return-decomposition, liquidity, and significance notebooks
share one implementation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_PIPE = Path(__file__).resolve().parents[3] / "pipelines" / "05_modeling"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from final_model import LABEL_COL, kalshi_taker_fee  # noqa: E402

LABEL = LABEL_COL
BANKROLL_INIT = 100.0


# --------------------------------------------------------------------------- #
# Entry collection                                                            #
# --------------------------------------------------------------------------- #

def _entry_key(window: str) -> str:
    if window == "half_life":
        return "entry_start_half"
    if window == "two_thirds_life":
        return "entry_start_twothirds"
    raise ValueError(f"unknown entry_window: {window}")


def collect_entries(ticker_info, pretip, model_col, edge_min, norm_edge_min,
                    entry_window, label_col=LABEL):
    """Scan candles every 15 min; return first qualifying snapshot per game."""
    key = _entry_key(entry_window)
    out = []
    for tkr, info in ticker_info.items():
        if tkr not in pretip:
            continue
        p, hw = info[model_col], info[label_col]
        eligible = pretip[tkr][pretip[tkr]["ts"] >= info[key]]
        for _, row in eligible.iterrows():
            if row["ts"].minute % 15 != 0:
                continue
            yb, ya = row["yes_bid_close"], row["yes_ask_close"]
            if not (0 < yb < 1 and 0 < ya < 1):
                continue
            q_yes, q_no = ya, 1 - yb
            edge_yes, edge_no = p - q_yes, (1 - p) - q_no
            if edge_yes >= edge_no:
                side, entry_px, edge, p_side = "YES", q_yes, edge_yes, p
            else:
                side, entry_px, edge, p_side = "NO", q_no, edge_no, 1 - p
            if edge < edge_min:
                continue
            norm_edge = edge / entry_px if entry_px > 0 else 0
            if norm_edge_min > 0 and norm_edge < norm_edge_min:
                continue
            out.append({
                "game_id": info["game_id"], "game_ts": info["game_ts"],
                "side": side, "entry_px": entry_px, "entry_ts": row["ts"],
                "edge": edge, "norm_edge": norm_edge,
                "p_side": p_side, "p_model": p, "home_win": hw,
            })
            break
    return out


def collect_all_snapshots(ticker_info, pretip, model_col, edge_min, norm_edge_min,
                          entry_window, label_col=LABEL):
    """All qualifying snapshots per game (not just the first)."""
    key = _entry_key(entry_window)
    out = []
    for tkr, info in ticker_info.items():
        if tkr not in pretip:
            continue
        p, hw = info[model_col], info[label_col]
        eligible = pretip[tkr][pretip[tkr]["ts"] >= info[key]]
        for _, row in eligible.iterrows():
            if row["ts"].minute % 15 != 0:
                continue
            yb, ya = row["yes_bid_close"], row["yes_ask_close"]
            if not (0 < yb < 1 and 0 < ya < 1):
                continue
            q_yes, q_no = ya, 1 - yb
            edge_yes, edge_no = p - q_yes, (1 - p) - q_no
            if edge_yes >= edge_no:
                side, entry_px, edge, p_side = "YES", q_yes, edge_yes, p
            else:
                side, entry_px, edge, p_side = "NO", q_no, edge_no, 1 - p
            if edge < edge_min:
                continue
            norm_edge = edge / entry_px if entry_px > 0 else 0
            if norm_edge_min > 0 and norm_edge < norm_edge_min:
                continue
            out.append({
                "game_id": info["game_id"], "game_ts": info["game_ts"],
                "side": side, "entry_px": entry_px, "entry_ts": row["ts"],
                "edge": edge, "norm_edge": norm_edge,
                "p_side": p_side, "p_model": p, "home_win": hw,
            })
    return out


# --------------------------------------------------------------------------- #
# Sizing engines                                                              #
# --------------------------------------------------------------------------- #

def _won(side, home_win):
    return 1.0 if (side == "YES" and home_win == 1) or (side == "NO" and home_win == 0) else 0.0


def run_fixed_risk(entries, risk_per_trade=1.0, bankroll_init=BANKROLL_INIT):
    """Risk a constant dollar amount per trade, no compounding."""
    out = []
    bankroll = bankroll_init
    for e in sorted(entries, key=lambda x: x["entry_ts"]):
        wager = risk_per_trade
        n = wager / e["entry_px"]
        fee = kalshi_taker_fee(n, e["entry_px"])
        pay = _won(e["side"], e["home_win"])
        pnl = n * pay - wager - fee
        bankroll += pnl
        out.append({**e, "wager": wager, "n_contracts": n, "fee": fee,
                    "pnl": pnl, "won": int(pnl > 0), "bankroll": bankroll})
    return out


def run_kelly_ideal(entries, fraction, bankroll_init=BANKROLL_INIT):
    out = []
    bankroll = bankroll_init
    for e in sorted(entries, key=lambda x: x["entry_ts"]):
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager = kf * fraction * bankroll
        if wager < 0.01:
            continue
        n = wager / e["entry_px"]
        fee = kalshi_taker_fee(n, e["entry_px"])
        pay = _won(e["side"], e["home_win"])
        pnl = n * pay - wager - fee
        bankroll += pnl
        out.append({**e, "kelly_f": kf, "wager_ideal": wager, "wager": wager,
                    "n_ideal": n, "n_contracts": n, "fill_pct": 1.0,
                    "entry_px_actual": e["entry_px"],
                    "fee": fee, "pnl": pnl, "won": int(pnl > 0), "bankroll": bankroll})
    return out


def run_kelly_sweep(entries, fraction, trade_data, bankroll_init=BANKROLL_INIT):
    """Liquidity-constrained sweep: fill against historical trades ≤ entry price."""
    out = []
    bankroll = bankroll_init
    for e in sorted(entries, key=lambda x: x["entry_ts"]):
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_ideal = kf * fraction * bankroll
        if wager_ideal < 0.01:
            continue
        n_ideal = wager_ideal / e["entry_px"]
        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty:
            continue
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        qual = gtr[gtr["our_price"] <= e["entry_px"]].sort_values("our_price")
        if qual.empty:
            continue
        filled, cost = 0.0, 0.0
        for _, t in qual.iterrows():
            take = min(t["count"], n_ideal - filled)
            filled += take
            cost += take * t["our_price"]
            if filled >= n_ideal:
                break
        if filled == 0:
            continue
        vwap = cost / filled
        n_actual = filled
        wager_actual = cost
        fee = kalshi_taker_fee(n_actual, vwap)
        pay = _won(e["side"], e["home_win"])
        pnl = n_actual * pay - wager_actual - fee
        bankroll += pnl
        out.append({**e, "kelly_f": kf,
                    "wager_ideal": wager_ideal, "wager": wager_actual,
                    "n_ideal": n_ideal, "n_contracts": n_actual,
                    "fill_pct": n_actual / n_ideal if n_ideal > 0 else 0,
                    "entry_px_actual": vwap,
                    "fee": fee, "pnl": pnl, "won": int(pnl > 0), "bankroll": bankroll})
    return out


def run_kelly_limit(entries, fraction, trade_data, bankroll_init=BANKROLL_INIT):
    """Limit order: fills only against historical volume at exactly entry price."""
    out = []
    bankroll = bankroll_init
    for e in sorted(entries, key=lambda x: x["entry_ts"]):
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_ideal = kf * fraction * bankroll
        if wager_ideal < 0.01:
            continue
        n_ideal = wager_ideal / e["entry_px"]
        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty:
            continue
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        vol_at = gtr.loc[gtr["our_price"] == e["entry_px"], "count"].sum()
        if vol_at == 0:
            continue
        n_actual = min(n_ideal, vol_at)
        wager_actual = n_actual * e["entry_px"]
        fee = kalshi_taker_fee(n_actual, e["entry_px"])
        pay = _won(e["side"], e["home_win"])
        pnl = n_actual * pay - wager_actual - fee
        bankroll += pnl
        out.append({**e, "kelly_f": kf,
                    "wager_ideal": wager_ideal, "wager": wager_actual,
                    "n_ideal": n_ideal, "n_contracts": n_actual,
                    "fill_pct": n_actual / n_ideal if n_ideal > 0 else 0,
                    "entry_px_actual": e["entry_px"],
                    "fee": fee, "pnl": pnl, "won": int(pnl > 0), "bankroll": bankroll})
    return out


def fill_diagnostics(trades_df: pd.DataFrame, trade_data: pd.DataFrame) -> pd.DataFrame:
    """Per-trade fill diagnostics at the wager sizes in `trades_df`."""
    rows = []
    for _, t in trades_df.iterrows():
        gid = t["game_id"]; side = t["side"]; entry_px = t["entry_px"]
        n_needed = t["n_contracts"]
        gtr = trade_data[trade_data["game_id"] == gid].copy()
        if gtr.empty:
            rows.append({
                "game_id": gid, "side": side, "entry_px": entry_px,
                "wager": t["wager"], "n_needed": n_needed,
                "total_volume": 0, "volume_at_or_below": 0,
                "vol_at_price": 0, "vwap": np.nan, "fill_pct": 0,
                "limit_fill_pct": 0,
            })
            continue
        gtr["our_price"] = gtr["yes_price"] if side == "YES" else gtr["no_price"]
        total_vol = gtr["count"].sum()
        qual = gtr[gtr["our_price"] <= entry_px]
        vol_below = qual["count"].sum()
        vol_at = gtr.loc[gtr["our_price"] == entry_px, "count"].sum()
        qual_sorted = qual.sort_values("our_price")
        filled, cost = 0.0, 0.0
        for _, r in qual_sorted.iterrows():
            take = min(r["count"], n_needed - filled)
            filled += take
            cost += take * r["our_price"]
            if filled >= n_needed:
                break
        vwap = cost / filled if filled > 0 else np.nan
        rows.append({
            "game_id": gid, "side": side, "entry_px": entry_px,
            "wager": t["wager"], "n_needed": n_needed,
            "total_volume": total_vol, "volume_at_or_below": vol_below,
            "vol_at_price": vol_at, "vwap": vwap,
            "fill_pct": min(filled / n_needed, 1.0) if n_needed > 0 else 0,
            "limit_fill_pct": min(vol_at / n_needed, 1.0) if n_needed > 0 else 0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bucket / partition helpers                                                  #
# --------------------------------------------------------------------------- #

def edge_bucket(df: pd.DataFrame, col: str = "edge",
                bins: Iterable[float] = (0, 0.05, 0.10, 0.15, 0.25, 1.0),
                labels: Iterable[str] = ("0-5%", "5-10%", "10-15%", "15-25%", "25%+"),
                ) -> pd.DataFrame:
    """Add a categorical 'edge_bucket' column to `df` and return a copy."""
    out = df.copy()
    out["edge_bucket"] = pd.cut(out[col], bins=list(bins), labels=list(labels), right=False)
    return out


def bucket_summary(df: pd.DataFrame, bucket_col: str,
                   metric_cols: Iterable[str] = ("won", "log_ret", "pnl")) -> pd.DataFrame:
    rows = []
    for b, sub in df.groupby(bucket_col, observed=True):
        row = {"bucket": str(b), "n": len(sub)}
        for col in metric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_trade_returns(tdf: pd.DataFrame, *, bankroll_init: float = BANKROLL_INIT) -> pd.DataFrame:
    """Append `bankroll_before` and `log_ret` columns to a trades DataFrame."""
    out = tdf.copy()
    out["bankroll_before"] = out["bankroll"] - out["pnl"]
    out["log_ret"] = np.log(out["bankroll"] / out["bankroll_before"])
    return out


def equity_by_payout(tdf: pd.DataFrame, *,
                     bankroll_init: float = BANKROLL_INIT,
                     ts_col: str = "game_ts") -> pd.DataFrame:
    """Build a clean equity curve sorted by settlement (payout) time.

    Engine bankrolls compound in entry-timestamp order. When trades enter on
    different days than the games settle, plotting bankroll vs. game_date can
    "loop back" and look like an error. This helper sorts trades by settlement
    time and recomputes a strictly chronological cumulative-PnL trajectory.

    Per-trade pnl values are unchanged; only the display order and the running
    bankroll are recomputed. Returns a copy of `tdf` with a new
    `display_bankroll` column, sorted ascending on `ts_col`.
    """
    if ts_col not in tdf.columns:
        raise KeyError(f"trades DataFrame is missing '{ts_col}' — re-run "
                       "collect_entries so each entry carries game_ts.")
    out = tdf.sort_values(ts_col).reset_index(drop=True).copy()
    out["display_bankroll"] = bankroll_init + out["pnl"].cumsum()
    return out
