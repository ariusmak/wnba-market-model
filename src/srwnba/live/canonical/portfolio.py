"""Kalshi portfolio sizing helpers for canonical execution."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PortfolioSizingSnapshot:
    ts_ms: int
    kalshi_cash_dollars: Optional[float]
    kalshi_portfolio_value_dollars: Optional[float]
    sizing_bankroll_dollars: float
    sizing_bankroll_source: str
    sizing_bankroll_override_dollars: Optional[float]
    available_cash_dollars: Optional[float]
    available_cash_source: str
    available_cash_override_dollars: Optional[float]
    raw_balance_payload: Mapping[str, Any]

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "portfolio_ts_ms": self.ts_ms,
            "kalshi_cash_dollars": self.kalshi_cash_dollars,
            "kalshi_portfolio_value_dollars": self.kalshi_portfolio_value_dollars,
            "sizing_bankroll_dollars": self.sizing_bankroll_dollars,
            "sizing_bankroll_source": self.sizing_bankroll_source,
            "sizing_bankroll_override_dollars": self.sizing_bankroll_override_dollars,
            "available_cash_dollars": self.available_cash_dollars,
            "available_cash_source": self.available_cash_source,
            "available_cash_override_dollars": self.available_cash_override_dollars,
            "raw_balance_payload": dict(self.raw_balance_payload or {}),
        }


def resolve_portfolio_sizing(
    client: Any,
    *,
    sizing_bankroll_override_dollars: Optional[float] = None,
    available_cash_override_dollars: Optional[float] = None,
) -> PortfolioSizingSnapshot:
    """Resolve sizing bankroll and cash cap from Kalshi plus optional overrides.

    Default production behavior:
      - sizing bankroll follows Kalshi `portfolio_value` when present
      - fallback sizing bankroll is Kalshi `balance`
      - available cash follows Kalshi `balance`

    Override behavior:
      - `sizing_bankroll_override_dollars` replaces the bankroll used for
        Kelly/cap math, even if above or below Kalshi wealth
      - `available_cash_override_dollars` replaces the cash cap used for
        order-feasibility checks
    """
    if client is None or not hasattr(client, "get_balance"):
        raise RuntimeError("Kalshi portfolio sizing requires a client with get_balance()")

    payload = client.get_balance()
    cash = _money_field(payload, "balance")
    portfolio_value = _money_field(payload, "portfolio_value")

    if sizing_bankroll_override_dollars is not None:
        sizing = _positive_money(sizing_bankroll_override_dollars, "sizing bankroll override")
        sizing_source = "override"
    elif portfolio_value is not None and portfolio_value > 0:
        sizing = _positive_money(portfolio_value, "Kalshi portfolio_value")
        sizing_source = "kalshi_portfolio_value"
    elif cash is not None and cash > 0:
        sizing = _positive_money(cash, "Kalshi balance")
        sizing_source = "kalshi_balance_fallback"
    else:
        raise RuntimeError(
            "Kalshi balance response did not include portfolio_value or balance; "
            f"keys={sorted((payload or {}).keys())}"
        )

    if available_cash_override_dollars is not None:
        available_cash = _nonnegative_money(available_cash_override_dollars, "available cash override")
        available_source = "override"
    else:
        if cash is None:
            raise RuntimeError(
                "Kalshi balance response did not include balance; cannot enforce available-cash cap"
            )
        available_cash = cash
        available_source = "kalshi_balance"

    return PortfolioSizingSnapshot(
        ts_ms=int(time.time() * 1000),
        kalshi_cash_dollars=cash,
        kalshi_portfolio_value_dollars=portfolio_value,
        sizing_bankroll_dollars=sizing,
        sizing_bankroll_source=sizing_source,
        sizing_bankroll_override_dollars=sizing_bankroll_override_dollars,
        available_cash_dollars=available_cash,
        available_cash_source=available_source,
        available_cash_override_dollars=available_cash_override_dollars,
        raw_balance_payload=payload or {},
    )


def _money_field(payload: Mapping[str, Any], key: str) -> Optional[float]:
    if not payload or payload.get(key) is None:
        return None
    return _cents_to_dollars(payload.get(key))


def _cents_to_dollars(value: Any) -> Optional[float]:
    try:
        out = float(value) / 100.0
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _positive_money(value: float, label: str) -> float:
    out = _nonnegative_money(value, label)
    if out <= 0:
        raise ValueError(f"{label} must be > 0, got {value}")
    return out


def _nonnegative_money(value: float, label: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{label} must be finite and >= 0, got {value}")
    return out
