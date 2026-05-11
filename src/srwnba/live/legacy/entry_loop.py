"""
src/srwnba/live/legacy/entry_loop.py
─────────────────────────────
Runtime that sweeps WNBA moneyline markets continuously from half-life
(~17h pre-tipoff) through tipoff, per the user's directive:

    > run taker-sweep constantly whenever a wnba moneyline market opens,
    > starting at halflife and until tipoff

One `EntryLoop` instance manages one game (one pair of YES markets — one
ticker per team). Given:
    - a `LivePredictor` (loaded once)
    - an `AuthedKalshiClient`
    - a `GameContext` naming home/away tickers + tipoff + feature row

the loop polls each side's orderbook on a cadence, prices both sides via
the frozen predictor, asks `plan_sweep` whether to act, and fires a
limit-IOC taker sweep when there's edge. Fills and plans are appended
to a JSONL log. Total exposure per side is capped at the half-Kelly
target; unfilled remainders simply drop.

Intended use is one process per game scheduled around tipoff; this module
is side-effect-heavy (network, disk, sleep) so keep it thin and keep all
pricing logic in `trader.py`.

Hold-to-settlement
------------------
Once filled, we hold until Kalshi settles the market. The loop never submits
a close order; settlement handles PnL.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...util.final_model import FinalModel
from .trader import (
    MarketSide,
    OrderbookSnapshot,
    SweepPlan,
    TradeConfig,
    estimate_kalshi_fee_dollars,
    plan_sweep,
    snapshot_from_kalshi,
)

log = logging.getLogger("srwnba.live.legacy.entry_loop")


# ──────────────────────────────────────────────────────────────────────────
# Context + state
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GameContext:
    """Everything we need to price and trade one WNBA game.

    `feature_row` is a single-row DataFrame matching LivePredictor's input
    schema (raw gold columns, pre-scoring). `home_ticker` and `away_ticker`
    are the Kalshi tickers for the home-YES and away-YES markets
    respectively — on Kalshi each team in a two-way market gets its own
    YES ticker.
    """
    game_id: str
    home_ticker: str
    away_ticker: str
    tipoff_ts_s: int                      # unix seconds
    feature_row: pd.DataFrame
    home_team_label: str = ""
    away_team_label: str = ""


@dataclass
class SideState:
    """Running count of contracts we've filled on one side this game."""
    ticker: str
    side_label: str                       # "home" | "away"
    filled_contracts: int = 0
    total_cost_cents: int = 0             # Σ fill_price · fill_size
    total_fee_dollars: float = 0.0
    last_poll_ts: float = 0.0
    orders: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Entry loop
# ──────────────────────────────────────────────────────────────────────────

class EntryLoop:
    """One instance per game. Call `.run()` to block until tipoff.

    Side effects:
      - prints/logs each poll
      - writes a JSONL event stream to `log_path`
      - submits taker-IOC orders via the authed client
    """

    def __init__(
        self,
        predictor: FinalModel,
        client: Any,                      # AuthedKalshiClient — avoid hard dep
        ctx: GameContext,
        cfg: Optional[TradeConfig] = None,
        log_path: Optional[Path] = None,
        poll_interval_s: float = 5.0,
        dry_run: bool = False,
    ) -> None:
        self.predictor = predictor
        self.client = client
        self.ctx = ctx
        self.cfg = cfg or TradeConfig()
        self.log_path = Path(log_path) if log_path else None
        self.poll_interval_s = float(poll_interval_s)
        self.dry_run = bool(dry_run)

        # Price the game once — features are pregame and don't update.
        # FinalModel.predict returns {"p_home": [float, ...], "p_raw": [...], ...}
        self.p_home = float(predictor.predict(ctx.feature_row)["p_home"][0])
        self.p_away = 1.0 - self.p_home

        self.home = SideState(ticker=ctx.home_ticker, side_label="home")
        self.away = SideState(ticker=ctx.away_ticker, side_label="away")

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, wall_clock_now_s=None) -> None:
        """Poll until tipoff (or until interrupted). Blocks this process."""
        now_fn = wall_clock_now_s or (lambda: time.time())
        log.info(
            "[entry_loop] game=%s home=%s (p=%.4f) away=%s (p=%.4f) tipoff=%d dry_run=%s",
            self.ctx.game_id, self.ctx.home_ticker, self.p_home,
            self.ctx.away_ticker, self.p_away, self.ctx.tipoff_ts_s, self.dry_run,
        )
        self._write_event({"evt": "start",
                           "game_id": self.ctx.game_id,
                           "p_home": self.p_home, "p_away": self.p_away,
                           "tipoff_ts_s": self.ctx.tipoff_ts_s})

        while True:
            now = now_fn()
            if now >= self.ctx.tipoff_ts_s:
                log.info("[entry_loop] reached tipoff, stopping")
                self._write_event({"evt": "tipoff_stop", "ts_s": now})
                return
            try:
                self._poll_once()
            except Exception as exc:
                log.exception("[entry_loop] poll failed: %s", exc)
                self._write_event({"evt": "poll_error", "err": repr(exc)})
            time.sleep(self.poll_interval_s)

    # ------------------------------------------------------------------
    # One poll across both sides
    # ------------------------------------------------------------------

    def _poll_once(self) -> None:
        for state, side in [
            (self.home, MarketSide(self.ctx.home_ticker, "home", self.p_home)),
            (self.away, MarketSide(self.ctx.away_ticker, "away", self.p_away)),
        ]:
            book = self._fetch_book(state.ticker)
            state.last_poll_ts = time.time()
            plan = plan_sweep(
                side=side,
                book=book,
                already_held=state.filled_contracts,
                cfg=self.cfg,
            )
            self._write_event({
                "evt": "plan",
                "ticker": state.ticker,
                "side_label": side.side_label,
                "p_model": side.p_model,
                "best_ask": plan.best_ask_cents,
                "max_price": plan.max_price_cents,
                "target": plan.target_contracts,
                "fillable": plan.fillable_size,
                "count": plan.count,
                "skip": plan.skip,
                "skip_reason": plan.skip_reason,
                "held": state.filled_contracts,
            })
            if plan.skip or plan.count < self.cfg.min_contracts:
                continue
            self._execute(state, plan)

    # ------------------------------------------------------------------
    # Orderbook fetch + cent-level conversion
    # ------------------------------------------------------------------

    def _fetch_book(self, ticker: str) -> OrderbookSnapshot:
        payload = self.client.get_orderbook(ticker)
        ts_ms = int(time.time() * 1000)
        return snapshot_from_kalshi(ticker, payload, ts_ms)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _execute(self, state: SideState, plan: SweepPlan) -> None:
        """Submit a limit-IOC buy-YES at max_price_cents for plan.count."""
        client_order_id = f"wnba-{self.ctx.game_id}-{state.side_label}-{uuid.uuid4().hex[:10]}"
        if self.dry_run:
            log.info(
                "[dry_run] would buy %d %s YES @ ≤%d¢  (target=%d, fillable=%d)",
                plan.count, state.ticker, plan.max_price_cents,
                plan.target_contracts, plan.fillable_size,
            )
            self._write_event({
                "evt": "dry_order",
                "client_order_id": client_order_id,
                "ticker": state.ticker,
                "count": plan.count,
                "yes_price_cents": plan.max_price_cents,
            })
            return

        try:
            resp = self.client.create_order(
                ticker=state.ticker,
                action="buy",
                side="yes",
                count=plan.count,
                order_type="limit",
                client_order_id=client_order_id,
                yes_price_cents=plan.max_price_cents,
                time_in_force="IOC",
            )
        except Exception as exc:
            log.exception("[entry_loop] order submit failed: %s", exc)
            self._write_event({
                "evt": "order_error",
                "client_order_id": client_order_id,
                "ticker": state.ticker, "count": plan.count,
                "yes_price_cents": plan.max_price_cents,
                "err": repr(exc),
            })
            return

        order = (resp or {}).get("order") or {}
        state.orders.append({
            "client_order_id": client_order_id,
            "order_id": order.get("order_id"),
            "status": order.get("status"),
            "requested_count": plan.count,
            "yes_price_cents": plan.max_price_cents,
            "response": order,
        })
        self._write_event({
            "evt": "order_submitted",
            "client_order_id": client_order_id,
            "order_id": order.get("order_id"),
            "ticker": state.ticker,
            "requested_count": plan.count,
            "yes_price_cents": plan.max_price_cents,
            "status": order.get("status"),
        })

        # Update state from filled portion (IOC may partially fill).
        filled = int(order.get("filled_count") or order.get("fill_count") or 0)
        taker_fill_cost = int(order.get("taker_fill_cost") or 0)  # cents
        # Fee estimate uses actual size-weighted avg fill price when we
        # have it; falls back to the cap if the response omits cost.
        if filled > 0 and taker_fill_cost > 0:
            avg_fill_cents = max(1, min(99, round(taker_fill_cost / filled)))
        else:
            avg_fill_cents = plan.max_price_cents
        fee_dollars = estimate_kalshi_fee_dollars(
            count=filled,
            price_cents=avg_fill_cents,
            fee_rate=self.cfg.fee_rate,
        )
        state.filled_contracts += filled
        state.total_cost_cents += taker_fill_cost
        state.total_fee_dollars += fee_dollars
        self._write_event({
            "evt": "fill",
            "order_id": order.get("order_id"),
            "ticker": state.ticker,
            "filled": filled,
            "cost_cents": taker_fill_cost,
            "fee_dollars": fee_dollars,
            "cumulative_filled": state.filled_contracts,
        })

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def _write_event(self, payload: Dict[str, Any]) -> None:
        payload = {"ts_ms": int(time.time() * 1000), **payload}
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        if payload.get("evt") in {"order_submitted", "fill", "order_error", "start", "tipoff_stop"}:
            log.info("[event] %s", json.dumps(payload))
