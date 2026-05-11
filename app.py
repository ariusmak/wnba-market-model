from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

import supabase_io as db

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from srwnba.live.canonical.operator_control import (  # noqa: E402
    load_game_override,
    load_global_control,
    resolve_operator_decision,
    save_game_override,
    save_global_control,
)


st.set_page_config(page_title="WNBA Control Plane", page_icon="WNBA", layout="wide")


def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: 0.75rem;
        }
        h2, h3 {
            line-height: 1.2;
        }
        div[data-testid="stButton"] > button {
            min-height: 2.75rem;
            border-radius: 0.45rem;
            font-weight: 700;
            white-space: normal;
        }
        div[data-testid="stTextInput"] input {
            min-height: 2.6rem;
        }
        .mobile-stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
            gap: 0.55rem;
            margin: 0.55rem 0 0.8rem 0;
        }
        .mobile-stat {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.45rem;
            padding: 0.65rem 0.7rem;
            background: rgba(15, 23, 42, 0.03);
            min-width: 0;
        }
        .mobile-stat-label {
            color: rgb(100, 116, 139);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            overflow-wrap: anywhere;
        }
        .mobile-stat-value {
            font-size: 1.05rem;
            font-weight: 800;
            margin-top: 0.22rem;
            overflow-wrap: anywhere;
        }
        .market-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.7rem;
            margin-bottom: 0.55rem;
        }
        .market-title {
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.2;
        }
        .market-subtitle {
            color: rgb(100, 116, 139);
            font-size: 0.86rem;
            margin-top: 0.18rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            justify-content: flex-end;
        }
        .pill {
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 800;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(148, 163, 184, 0.12);
            color: rgb(51, 65, 85);
            white-space: nowrap;
        }
        .pill.good {
            background: rgba(22, 163, 74, 0.12);
            border-color: rgba(22, 163, 74, 0.30);
            color: rgb(21, 128, 61);
        }
        .pill.warn {
            background: rgba(217, 119, 6, 0.13);
            border-color: rgba(217, 119, 6, 0.32);
            color: rgb(180, 83, 9);
        }
        .pill.bad {
            background: rgba(220, 38, 38, 0.12);
            border-color: rgba(220, 38, 38, 0.30);
            color: rgb(185, 28, 28);
        }
        .meta-line {
            color: rgb(71, 85, 105);
            font-size: 0.88rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .mode-banner {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.5rem;
            padding: 0.8rem 0.9rem;
            margin: 0.45rem 0 0.7rem 0;
            background: rgba(15, 23, 42, 0.035);
        }
        .mode-banner-title {
            font-size: 0.78rem;
            font-weight: 800;
            color: rgb(100, 116, 139);
            text-transform: uppercase;
        }
        .mode-banner-value {
            font-size: 1.35rem;
            font-weight: 900;
            line-height: 1.15;
            margin-top: 0.18rem;
        }
        .mode-banner-detail {
            color: rgb(71, 85, 105);
            font-size: 0.9rem;
            margin-top: 0.3rem;
            overflow-wrap: anywhere;
        }
        .mode-banner.conservative {
            background: rgba(217, 119, 6, 0.12);
            border-color: rgba(217, 119, 6, 0.32);
        }
        .mode-banner.killed {
            background: rgba(220, 38, 38, 0.12);
            border-color: rgba(220, 38, 38, 0.32);
        }
        .mode-banner.normal {
            background: rgba(22, 163, 74, 0.10);
            border-color: rgba(22, 163, 74, 0.28);
        }
        .freshness-banner {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.5rem;
            padding: 0.75rem 0.85rem;
            margin: 0.45rem 0 0.8rem 0;
            background: rgba(15, 23, 42, 0.035);
        }
        .freshness-banner.good {
            background: rgba(22, 163, 74, 0.10);
            border-color: rgba(22, 163, 74, 0.28);
        }
        .freshness-banner.warn {
            background: rgba(217, 119, 6, 0.12);
            border-color: rgba(217, 119, 6, 0.32);
        }
        .freshness-banner.bad {
            background: rgba(220, 38, 38, 0.12);
            border-color: rgba(220, 38, 38, 0.32);
        }
        .freshness-title {
            font-size: 0.78rem;
            font-weight: 850;
            color: rgb(100, 116, 139);
            text-transform: uppercase;
        }
        .freshness-value {
            font-size: 1.12rem;
            font-weight: 900;
            margin-top: 0.18rem;
        }
        .recent-command {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 0.42rem;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.42rem;
            background: rgba(15, 23, 42, 0.025);
        }
        .recent-command-main {
            font-weight: 850;
            overflow-wrap: anywhere;
        }
        .recent-command-sub {
            color: rgb(71, 85, 105);
            font-size: 0.84rem;
            margin-top: 0.15rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 720px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-top: 0.55rem;
            }
            h1 {
                font-size: 1.45rem;
            }
            h2 {
                font-size: 1.2rem;
            }
            h3 {
                font-size: 1.05rem;
            }
            .mobile-stat-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.45rem;
            }
            .mobile-stat {
                padding: 0.55rem 0.58rem;
            }
            .mobile-stat-value {
                font-size: 0.98rem;
            }
            .market-head {
                display: block;
            }
            .pill-row {
                justify-content: flex-start;
                margin-top: 0.5rem;
            }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
            div[data-testid="stHorizontalBlock"] {
                gap: 0.35rem;
            }
            div[data-testid="stButton"] > button {
                min-height: 3rem;
                font-size: 0.95rem;
            }
        }
        @media (max-width: 390px) {
            .mobile-stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_mobile_css()


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def signed_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.1%}"


def price(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{100 * float(value):.1f}c"


def dollars(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"${float(value):,.0f}"


def decimal(value: Any, places: int = 6) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{places}f}"


def model_odds_change_label(row: dict[str, Any]) -> str:
    delta = row.get("model_prob_change_t20_to_t8")
    if delta is None or pd.isna(delta):
        return "-"
    moved = bool(row.get("model_prob_changed_t20_to_t8"))
    marker = "moved" if moved else "flat"
    return f"{signed_pct(delta)} {marker}"


def short_ts(value: Any) -> str:
    if not value:
        return "-"
    try:
        return pd.to_datetime(value).strftime("%b %-d %I:%M %p")
    except ValueError:
        return pd.to_datetime(value).strftime("%b %#d %I:%M %p")
    except Exception:
        return str(value)


def df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def safe(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return escape(str(value))


def to_utc_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def age_minutes(value: Any) -> float | None:
    ts = to_utc_ts(value)
    if ts is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (now - ts).total_seconds() / 60.0)


def age_label(value: Any) -> str:
    minutes = age_minutes(value)
    if minutes is None:
        return "-"
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


def age_tone(minutes: float | None, warn_min: float, bad_min: float) -> str:
    if minutes is None or minutes >= bad_min:
        return "bad"
    if minutes >= warn_min:
        return "warn"
    return "good"


def worst_tone(tones: list[str]) -> str:
    if "bad" in tones:
        return "bad"
    if "warn" in tones:
        return "warn"
    return "good"


def latest_heartbeat(heartbeat: list[dict[str, Any]]) -> dict[str, Any]:
    return heartbeat[0] if heartbeat else {}


def max_market_age(markets: list[dict[str, Any]], field: str) -> float | None:
    ages = [age_minutes(row.get(field)) for row in markets if row.get(field)]
    ages = [age for age in ages if age is not None]
    return max(ages) if ages else None


def max_market_age_label(markets: list[dict[str, Any]], field: str) -> str:
    ages = [(age_minutes(row.get(field)), row.get(field)) for row in markets if row.get(field)]
    ages = [(age, value) for age, value in ages if age is not None]
    if not ages:
        return "-"
    return age_label(max(ages, key=lambda item: item[0])[1])


def pill_tone(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"eligible", "active", "normal", "clear", "enabled", "shadow"}:
        return "good"
    if any(part in text for part in ("kill", "blocked", "cancel", "gate", "late", "stale")):
        return "bad"
    if any(part in text for part in ("pause", "no_edge", "conservative", "disabled")):
        return "warn"
    return ""


def render_stat_grid(items: list[tuple[str, Any]]) -> None:
    cards = []
    for label, value in items:
        cards.append(
            "<div class='mobile-stat'>"
            f"<div class='mobile-stat-label'>{safe(label)}</div>"
            f"<div class='mobile-stat-value'>{safe(value)}</div>"
            "</div>"
        )
    st.markdown("<div class='mobile-stat-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_market_header(row: dict[str, Any]) -> None:
    title = f"{row.get('away_team') or '-'} @ {row.get('home_team') or '-'}"
    phase = row.get("phase") or "-"
    status = row.get("trading_status") or "-"
    gate = row.get("expansion_gate_status") or "-"
    st.markdown(
        "<div class='market-head'>"
        "<div>"
        f"<div class='market-title'>{safe(title)}</div>"
        f"<div class='market-subtitle'>Tipoff {safe(short_ts(row.get('tipoff_ts')))}"
        f" | Selected {safe(row.get('selected_team') or '-')}</div>"
        "</div>"
        "<div class='pill-row'>"
        f"<span class='pill {pill_tone(phase)}'>{safe(phase)}</span>"
        f"<span class='pill {pill_tone(status)}'>{safe(status)}</span>"
        f"<span class='pill {pill_tone(gate)}'>{safe(gate)}</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_mode_banner(control: dict[str, Any]) -> None:
    mode = str(control.get("mode") or "-")
    burst = "on" if control.get("allow_burst_mode") else "off"
    trading = "enabled" if control.get("trading_enabled") else "disabled"
    entries = "new entries on" if control.get("allow_new_entries") else "new entries off"
    shadow = "shadow on" if control.get("shadow_mode_enabled") else "shadow off"
    tone = "killed" if control.get("kill_switch_active") else mode.lower()
    st.markdown(
        f"<div class='mode-banner {safe(tone)}'>"
        "<div class='mode-banner-title'>Current Mode</div>"
        f"<div class='mode-banner-value'>{safe(mode.upper())}</div>"
        f"<div class='mode-banner-detail'>Max exposure {safe(pct(control.get('max_market_exposure_pct')))}"
        f" | Burst {safe(burst)} | Trading {safe(trading)} | {safe(entries)} | {safe(shadow)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_operator_banner(control: dict[str, Any]) -> None:
    auto = bool(control.get("auto_trade_enabled", True))
    risk = str(control.get("risk_mode") or "normal").lower()
    mode = "TRADE BY DEFAULT" if auto and risk != "kill" else "BLOCKED BY DEFAULT"
    if risk == "kill":
        mode = "KILL"
    tone = "killed" if risk == "kill" or not auto else ("conservative" if risk == "conservative" else "normal")
    st.markdown(
        f"<div class='mode-banner {safe(tone)}'>"
        "<div class='mode-banner-title'>Execution Default</div>"
        f"<div class='mode-banner-value'>{safe(mode)}</div>"
        f"<div class='mode-banner-detail'>Risk mode {safe(risk.upper())}"
        f" | Auto-trade default {'on' if auto else 'off'}"
        f" | {safe(control.get('reason') or '-')}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def worker_obedience(control: dict[str, Any], heartbeat: list[dict[str, Any]]) -> tuple[str, str, str]:
    hb = latest_heartbeat(heartbeat)
    control_ts = to_utc_ts(control.get("updated_at"))
    seen_ts = to_utc_ts(hb.get("last_control_seen_at") or hb.get("last_seen_at"))
    if control_ts is None:
        return "unknown", "bad", "control timestamp missing"
    if seen_ts is None:
        return "not observed", "bad", "worker heartbeat missing"
    lag_seconds = (seen_ts - control_ts).total_seconds()
    if lag_seconds >= -5:
        return "observed", "good", f"seen {age_label(seen_ts)} ago"
    return "not observed", "bad", f"control newer than worker by {abs(lag_seconds):.0f}s"


def freshness_status(state: dict[str, Any]) -> None:
    markets = state["markets"]
    heartbeat = state["heartbeat"]
    latest_equity = state.get("latest_equity")
    hb = latest_heartbeat(heartbeat)

    checks = [
        ("Heartbeat", age_minutes(hb.get("last_seen_at")), 3, 10, age_label(hb.get("last_seen_at"))),
        ("Account", age_minutes((latest_equity or {}).get("ts")), 10, 30, age_label((latest_equity or {}).get("ts"))),
        ("Market data", max_market_age(markets, "market_data_ts"), 3, 10, max_market_age_label(markets, "market_data_ts")),
        ("Model", max_market_age(markets, "model_snapshot_ts"), 10, 30, max_market_age_label(markets, "model_snapshot_ts")),
        ("Injuries", max_market_age(markets, "injury_data_ts"), 240, 720, max_market_age_label(markets, "injury_data_ts")),
        ("Orderbook", max_market_age(markets, "orderbook_ts"), 3, 10, max_market_age_label(markets, "orderbook_ts")),
    ]
    tones = [age_tone(minutes, warn, bad) for _label, minutes, warn, bad, _age in checks]
    ack_status, ack_tone, ack_detail = worker_obedience(state["control"], heartbeat)
    tones.append(ack_tone)
    overall = worst_tone(tones)
    label = {"good": "Fresh", "warn": "Needs attention", "bad": "Stale or unconfirmed"}[overall]

    st.markdown(
        f"<div class='freshness-banner {overall}'>"
        "<div class='freshness-title'>Freshness / Worker Obedience</div>"
        f"<div class='freshness-value'>{safe(label)}</div>"
        f"<div class='mode-banner-detail'>Worker command state: {safe(ack_status)} ({safe(ack_detail)})</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_stat_grid(
        [(label, age) for label, _minutes, _warn, _bad, age in checks]
        + [("Worker saw controls", ack_status)]
    )


def sample_data() -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "operator_control": load_global_control(),
        "control": {
            "trading_enabled": False,
            "kill_switch_active": False,
            "mode": "demo",
            "max_market_exposure_pct": 0.15,
            "allow_new_entries": True,
            "allow_ioc_orders": True,
            "allow_passive_orders": True,
            "allow_burst_mode": True,
            "shadow_mode_enabled": True,
            "updated_at": now,
            "reason": "Read-only demo mode: configure Supabase to enable controls.",
        },
        "markets": [
            {
                "game_id": "DEMO-ATL-NYL",
                "home_team": "Atlanta Dream",
                "away_team": "New York Liberty",
                "selected_team": "New York Liberty",
                "opponent_team": "Atlanta Dream",
                "tipoff_ts": now,
                "time_to_tipoff_minutes": 960,
                "phase": "ACTIVE",
                "trading_status": "eligible",
                "expansion_gate_status": "clear",
                "model_prob": 0.66,
                "model_prob_t20": 0.64,
                "model_prob_latest_pre_t8": 0.66,
                "model_prob_change_t20_to_t8": 0.02,
                "model_prob_changed_t20_to_t8": True,
                "model_prob_last_refresh_at": now,
                "market_prob": 0.53,
                "abs_edge": 0.13,
                "norm_edge": 0.245,
                "q_max_price": 0.528,
                "q_exec_all_in_price": 0.525,
                "filled_position_dollars": 140,
                "target_position_now_dollars": 690,
                "remaining_position_dollars": 550,
                "target_position_binder": "half_kelly",
                "execution_binder": "cumulative",
                "available_cash_after_buffer_dollars": 100,
                "cash_limited_mode": True,
                "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
                "cash_priority_rank": 2,
                "cash_priority_rank_total": 2,
                "cash_priority_score": 0.000074,
                "expected_log_growth_next_child": 0.00555,
                "cash_priority_candidate_child_dollars": 75,
                "cash_priority_allocated_child_dollars": 0,
                "skipped_due_to_cash": True,
                "q_current_position": 0.528,
                "q_avg_after_child": 0.526,
                "last_action": "shadow_ioc_planned",
                "last_reject_reason": None,
                "market_data_ts": now,
                "model_snapshot_ts": now,
                "injury_data_ts": now,
                "orderbook_ts": now,
            },
            {
                "game_id": "DEMO-TOR-SEA",
                "home_team": "Toronto Tempo",
                "away_team": "Seattle Storm",
                "selected_team": "Seattle Storm",
                "opponent_team": "Toronto Tempo",
                "tipoff_ts": now,
                "time_to_tipoff_minutes": 1860,
                "phase": "BLOCKED",
                "trading_status": "expansion_gate",
                "expansion_gate_status": "blocked_14_prior_games_required",
                "model_prob": 0.58,
                "model_prob_t20": 0.58,
                "model_prob_latest_pre_t8": 0.58,
                "model_prob_change_t20_to_t8": 0.0,
                "model_prob_changed_t20_to_t8": False,
                "model_prob_last_refresh_at": now,
                "market_prob": 0.45,
                "abs_edge": 0.13,
                "norm_edge": 0.289,
                "q_max_price": 0.464,
                "q_exec_all_in_price": 0.456,
                "filled_position_dollars": 0,
                "target_position_now_dollars": 598,
                "remaining_position_dollars": 598,
                "target_position_binder": "half_kelly",
                "execution_binder": "cold_start",
                "available_cash_after_buffer_dollars": 100,
                "cash_limited_mode": False,
                "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
                "cash_priority_rank": None,
                "cash_priority_rank_total": 2,
                "cash_priority_score": None,
                "expected_log_growth_next_child": None,
                "cash_priority_candidate_child_dollars": 0,
                "cash_priority_allocated_child_dollars": 0,
                "skipped_due_to_cash": False,
                "q_current_position": None,
                "q_avg_after_child": None,
                "last_action": "skip_expansion_gate",
                "last_reject_reason": "expansion_gate",
                "market_data_ts": now,
                "model_snapshot_ts": now,
                "injury_data_ts": now,
                "orderbook_ts": now,
            },
        ],
        "routes": [],
        "orders": [],
        "commands": [],
        "controls": [],
        "closed": [],
        "equity": [],
        "latest_equity": {
            "ts": now,
            "equity_dollars": 5000,
            "cash_dollars": 4860,
            "open_position_value_dollars": 140,
            "realized_pnl_dollars": 0,
            "drawdown_dollars": 0,
            "total_markets_observed": 2,
            "entered_markets": 1,
        },
        "heartbeat": [
            {
                "bot_id": "demo-worker",
                "status": "demo",
                "last_seen_at": now,
                "last_control_seen_at": now,
                "current_mode": "demo",
                "kalshi_connected": False,
                "market_data_connected": False,
                "database_connected": False,
                "open_orders_count": 0,
                "open_positions_count": 0,
                "last_error": "Supabase not configured.",
            }
        ],
        "alerts": [],
    }


def require_auth() -> tuple[bool, bool]:
    configured = db.config_snapshot()
    st.sidebar.caption("Control-plane v1")
    if configured["supabase_configured"] and not configured["dashboard_password_configured"]:
        st.sidebar.error("Set DASHBOARD_PASSWORD before enabling connected mode.")
        st.stop()

    if not configured["supabase_configured"]:
        st.sidebar.warning("Demo dashboard data. Local execution controls remain available.")
        return False, True

    if st.session_state.get("authenticated"):
        return True, False

    st.title("WNBA Control Plane")
    st.caption("Secure dashboard access")
    password = st.text_input("Dashboard password", type="password")
    if not db.dashboard_password_matches(password):
        st.warning("Enter dashboard password.")
        st.stop()
    st.session_state["authenticated"] = True
    return True, False


def load_state(connected: bool) -> dict[str, Any]:
    if not connected:
        return sample_data()
    return {
        "operator_control": load_global_control(),
        "control": db.read_control_state() or {},
        "markets": db.list_live_market_snapshots(),
        "routes": db.list_route_snapshots(),
        "orders": db.list_order_events(),
        "commands": db.list_control_commands(),
        "controls": db.list_market_controls(),
        "closed": db.list_closed_market_summaries(),
        "equity": db.list_equity_curve(),
        "latest_equity": db.read_latest_equity_snapshot(),
        "heartbeat": db.list_bot_heartbeat(),
        "alerts": db.list_system_alerts(),
    }


def run_command(label: str, func, connected: bool) -> None:
    if not connected:
        st.toast("Controls are disabled in read-only demo mode.")
        return
    try:
        func()
        st.toast(f"{label} applied.")
        st.rerun()
    except Exception as exc:
            st.error(f"{label} failed: {exc}")


def run_local_command(label: str, func) -> None:
    try:
        func()
        st.toast(f"{label} applied.")
        st.rerun()
    except Exception as exc:
        st.error(f"{label} failed: {exc}")


def global_status(control: dict[str, Any], heartbeat: list[dict[str, Any]]) -> None:
    latest_heartbeat = heartbeat[0] if heartbeat else {}
    render_mode_banner(control)
    render_stat_grid(
        [
            ("Trading", "Enabled" if control.get("trading_enabled") else "Disabled"),
            ("Kill Switch", "Active" if control.get("kill_switch_active") else "Clear"),
            ("Mode", str(control.get("mode") or "-")),
            ("Max Exposure", pct(control.get("max_market_exposure_pct"))),
            ("Heartbeat", str(latest_heartbeat.get("status") or "-")),
        ]
    )
    st.caption(f"Last control update: {short_ts(control.get('updated_at'))} | {control.get('reason') or '-'}")


def operator_status(control: dict[str, Any]) -> None:
    render_operator_banner(control)
    render_stat_grid(
        [
            ("Trade default", "ON" if control.get("auto_trade_enabled", True) else "OFF"),
            ("Risk mode", str(control.get("risk_mode") or "normal").upper()),
            ("Control file", "operator_control.json"),
            ("Updated", short_ts(control.get("updated_at_utc"))),
        ]
    )


def account_status(latest_equity: dict[str, Any] | None, markets: list[dict[str, Any]]) -> None:
    source = "equity_curve"
    snapshot = latest_equity or {}
    account_age = age_minutes(snapshot.get("ts"))
    account_tone = age_tone(account_age, 10, 30)
    if not latest_equity:
        source = "live-market fallback"
        deployed = sum(float(row.get("filled_position_dollars") or 0) for row in markets)
        snapshot = {
            "equity_dollars": None,
            "cash_dollars": None,
            "open_position_value_dollars": deployed,
            "realized_pnl_dollars": None,
            "drawdown_dollars": None,
            "ts": None,
        }
        account_tone = "warn"

    st.subheader("Kalshi Account")
    st.markdown(
        f"<div class='freshness-banner {account_tone}'>"
        f"<div class='freshness-title'>Account Snapshot</div>"
        f"<div class='freshness-value'>{safe('Fresh' if account_tone == 'good' else 'Needs account update')}</div>"
        f"<div class='mode-banner-detail'>Source: {safe(source)} | Age: {safe(age_label(snapshot.get('ts')))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_stat_grid(
        [
            ("Total NAV", dollars(snapshot.get("equity_dollars"))),
            ("Cash", dollars(snapshot.get("cash_dollars"))),
            ("Deployed", dollars(snapshot.get("open_position_value_dollars"))),
            ("Realized P&L", dollars(snapshot.get("realized_pnl_dollars"))),
            ("Drawdown", dollars(snapshot.get("drawdown_dollars"))),
        ]
    )
    st.caption(f"Last account update: {short_ts(snapshot.get('ts'))}")


def cash_priority_status(markets: list[dict[str, Any]]) -> None:
    cash_limited = any(
        bool(row.get("cash_limited_mode"))
        or str(row.get("execution_binder") or "").lower() == "available_cash_after_buffer"
        for row in markets
    )
    ranked = [row for row in markets if row.get("cash_priority_rank") is not None]
    ranked.sort(key=lambda row: _rank_sort_value(row.get("cash_priority_rank")))
    top = ranked[0] if ranked else {}
    rule = top.get("cash_priority_rule") or "marginal_expected_log_growth_per_dollar"
    render_stat_grid(
        [
            ("Cash-limited mode", "ACTIVE" if cash_limited else "Clear"),
            ("Priority rule", rule),
            ("Cash binder", "yes" if cash_limited else "no"),
            ("Top cash rank", f"#{top.get('cash_priority_rank')}" if top else "-"),
        ]
    )


def _rank_sort_value(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 999999


def global_controls(connected: bool) -> None:
    st.subheader("Global Controls")
    reason = st.text_input("Reason", value="Manual dashboard command", key="global_reason")
    confirm_kill = st.text_input("Type KILL to enable kill switch", value="", key="confirm_kill")
    confirm_pause = st.checkbox("Unlock pause/cancel controls", key="confirm_pause_controls")
    confirm_launch = st.checkbox("Unlock launch/resume controls", key="confirm_launch_controls")
    if st.button("KILL BOT", type="primary", use_container_width=True, disabled=confirm_kill.strip().upper() != "KILL"):
        run_command("KILL BOT", lambda: db.apply_global_command("KILL_BOT", reason=reason), connected)

    rows = [
        [("Launch", "LAUNCH_BOT", confirm_launch), ("Resume", "RESUME_ALL", confirm_launch)],
        [("Pause Entries", "PAUSE_ALL_NEW_ENTRIES", confirm_pause), ("Cancel Passives", "CANCEL_ALL_PASSIVES", confirm_pause)],
        [("Conservative", "CONSERVATIVE_MODE", True), ("Normal Risk", "NORMAL_RISK_MODE", True)],
    ]
    for row in rows:
        cols = st.columns(len(row))
        for col, (label, command_type, enabled) in zip(cols, row):
            if col.button(label, use_container_width=True, disabled=not enabled):
                run_command(
                    label,
                    lambda c=command_type: db.apply_global_command(c, reason=reason),
                    connected,
                )

    st.subheader("Execution Default / Risk Mode")
    local_control = load_global_control()
    local_reason = st.text_input(
        "Execution control reason",
        value="Manual dashboard execution-control update",
        key="local_execution_reason",
    )
    auto_enabled = st.toggle(
        "Trade all eligible games by default",
        value=bool(local_control.get("auto_trade_enabled", True)),
        help="On means every eligible game may be traded unless it has an explicit game abort override.",
    )
    risk_mode = st.selectbox(
        "Risk mode",
        ["normal", "conservative", "kill"],
        index=["normal", "conservative", "kill"].index(str(local_control.get("risk_mode") or "normal").lower()),
        help="Kill blocks all live execution. Conservative is visible to the route loop and logged as a brake mode.",
    )
    if st.button("Save Execution Mode", use_container_width=True):
        run_local_command(
            "Execution mode",
            lambda: save_global_control(
                auto_trade_enabled=auto_enabled,
                risk_mode=risk_mode,
                reason=local_reason,
                updated_by="streamlit_webapp",
            ),
        )


def market_action_buttons(row: dict[str, Any], connected: bool) -> None:
    game_id = row["game_id"]
    reason = st.session_state.get("market_reason", "Manual dashboard command")
    overrides_unlocked = bool(st.session_state.get("confirm_market_overrides"))
    row_one = st.columns(2)
    if row_one[0].button("Open Detail", key=f"detail_{game_id}", use_container_width=True):
        st.session_state["selected_game_id"] = game_id
        st.session_state["page"] = "Live Market Detail"
        st.rerun()
    if row_one[1].button("Pause Market", key=f"pause_{game_id}", use_container_width=True, disabled=not overrides_unlocked):
        run_command(
            "Pause Market",
            lambda gid=game_id: db.apply_market_command(gid, "PAUSE_MARKET", reason),
            connected,
        )

    row_two = st.columns(2)
    if row_two[0].button("Cancel Entry", key=f"cancel_{game_id}", use_container_width=True, disabled=not overrides_unlocked):
        run_command(
            "Cancel Entry",
            lambda gid=game_id: db.apply_market_command(gid, "CANCEL_ENTRY", reason),
            connected,
        )
    if row_two[1].button("Block Game", key=f"block_{game_id}", use_container_width=True, disabled=not overrides_unlocked):
        run_command(
            "Block Game",
            lambda gid=game_id: db.apply_market_command(gid, "BLOCK_GAME", reason),
            connected,
        )
    st.button(
        "Cancel Passives",
        key=f"cancel_passives_{game_id}",
        use_container_width=True,
        disabled=True,
        help="Per-market passive cancellation needs the worker command contract before this button is enabled.",
    )
    decision = resolve_operator_decision(game_id)
    override = load_game_override(game_id)
    st.caption(
        f"Execution override: {decision.game_decision.upper()} | "
        f"allowed={decision.trade_allowed} | risk={decision.risk_mode.upper()}"
    )
    row_three = st.columns(2)
    if row_three[0].button(
        "Abort Game",
        key=f"local_abort_{game_id}",
        use_container_width=True,
        disabled=not overrides_unlocked,
    ):
        run_local_command(
            "Abort Game",
            lambda gid=game_id, r=reason: save_game_override(
                game_id=gid,
                decision="abort",
                reason=r,
                updated_by="streamlit_webapp",
            ),
        )
    if row_three[1].button(
        "Clear Abort",
        key=f"local_clear_abort_{game_id}",
        use_container_width=True,
        disabled=not overrides_unlocked or override.get("decision") != "abort",
    ):
        run_local_command(
            "Clear Abort",
            lambda gid=game_id, r=reason: save_game_override(
                game_id=gid,
                decision="default",
                reason=r,
                updated_by="streamlit_webapp",
            ),
        )


def render_market_meta(row: dict[str, Any]) -> None:
    st.markdown(
        "<div class='meta-line'>"
        f"Target binder: <b>{safe(row.get('target_position_binder') or '-')}</b>"
        f" | Execution binder: <b>{safe(row.get('execution_binder') or '-')}</b>"
        f" | Last action: <b>{safe(row.get('last_action') or '-')}</b>"
        "</div>",
        unsafe_allow_html=True,
    )


def market_card(row: dict[str, Any], connected: bool) -> None:
    with st.container(border=True):
        render_market_header(row)
        render_stat_grid(
            [
                ("Model", pct(row.get("model_prob"))),
                ("T-20 -> T-8", model_odds_change_label(row)),
                ("Market", price(row.get("market_prob"))),
                ("Edge", pct(row.get("abs_edge"))),
                ("Norm", pct(row.get("norm_edge"))),
                (
                    "Cash Rank",
                    f"#{row.get('cash_priority_rank')} / {row.get('cash_priority_rank_total')}"
                    if row.get("cash_priority_rank") is not None
                    else "-",
                ),
                ("Priority", decimal(row.get("cash_priority_score"))),
                ("q max", price(row.get("q_max_price"))),
                (
                    "Filled / Target",
                    f"{dollars(row.get('filled_position_dollars'))} / {dollars(row.get('target_position_now_dollars'))}",
                ),
            ]
        )
        render_market_meta(row)
        market_action_buttons(row, connected)


def recent_command_strip(commands: list[dict[str, Any]]) -> None:
    st.subheader("Recent Commands")
    if not commands:
        st.info("No control commands logged yet.")
        return
    for command in commands[:5]:
        st.markdown(
            "<div class='recent-command'>"
            f"<div class='recent-command-main'>{safe(command.get('command_type'))}"
            f" | {safe(command.get('status'))}</div>"
            f"<div class='recent-command-sub'>{safe(short_ts(command.get('received_at')))}"
            f" | {safe(command.get('requested_via'))}"
            f" | {safe(command.get('reason'))}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def filtered_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    st.subheader("Live Markets")
    query = st.text_input("Search markets", value="", key="market_search")
    view = st.selectbox(
        "Market filter",
        ["All", "Eligible", "Has position", "Edge present", "Paused/blocked/cancelled", "Stale"],
        key="market_filter",
    )
    query_lower = query.strip().lower()
    out = []
    for row in markets:
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ["game_id", "home_team", "away_team", "selected_team", "opponent_team", "phase", "trading_status"]
        ).lower()
        if query_lower and query_lower not in haystack:
            continue
        status = str(row.get("trading_status") or "").lower()
        phase = str(row.get("phase") or "").lower()
        if view == "Eligible" and status != "eligible":
            continue
        if view == "Has position" and float(row.get("filled_position_dollars") or 0) <= 0:
            continue
        if view == "Edge present" and float(row.get("abs_edge") or 0) <= 0:
            continue
        if view == "Paused/blocked/cancelled" and not any(
            token in f"{status} {phase}" for token in ["pause", "block", "cancel", "kill", "gate"]
        ):
            continue
        if view == "Stale" and all(
            age_tone(age_minutes(row.get(field)), 3, 10) != "bad"
            for field in ["market_data_ts", "orderbook_ts"]
        ):
            continue
        out.append(row)
    st.caption(f"Showing {len(out)} of {len(markets)} markets.")
    return out


def control_room(state: dict[str, Any], connected: bool) -> None:
    st.title("WNBA Trading Control Room")
    freshness_status(state)
    global_status(state["control"], state["heartbeat"])
    operator_status(state.get("operator_control") or load_global_control())
    account_status(state.get("latest_equity"), state["markets"])
    cash_priority_status(state["markets"])
    global_controls(connected)
    recent_command_strip(state["commands"])
    st.text_input("Market command reason", value="Manual market dashboard command", key="market_reason")
    st.checkbox("Unlock market override buttons", key="confirm_market_overrides")
    markets = filtered_markets(state["markets"])
    if not markets:
        st.info("No live market snapshots yet.")
    for row in markets:
        market_card(row, connected)


def selected_game(markets: list[dict[str, Any]]) -> str | None:
    ids = [row["game_id"] for row in markets if row.get("game_id")]
    if not ids:
        return None
    current = st.session_state.get("selected_game_id")
    if current not in ids:
        current = ids[0]
    return st.selectbox("Game", ids, index=ids.index(current))


def live_market_detail(state: dict[str, Any], connected: bool) -> None:
    st.title("Live Market Detail")
    game_id = selected_game(state["markets"])
    if not game_id:
        st.info("No live markets available.")
        return
    st.session_state["selected_game_id"] = game_id
    market = next(row for row in state["markets"] if row["game_id"] == game_id)
    st.subheader(f"{market.get('away_team')} @ {market.get('home_team')}")
    freshness_status(state)
    global_status(state["control"], state["heartbeat"])
    operator_status(state.get("operator_control") or load_global_control())
    account_status(state.get("latest_equity"), state["markets"])
    render_stat_grid(
        [
            ("Model Prob", pct(market.get("model_prob"))),
            ("T-20 -> T-8", model_odds_change_label(market)),
            ("Exec Price", price(market.get("q_exec_all_in_price"))),
            ("Abs Edge", pct(market.get("abs_edge"))),
            ("Remaining", dollars(market.get("remaining_position_dollars"))),
        ]
    )

    st.subheader("Route Comparison")
    routes = [row for row in state["routes"] if row.get("game_id") == game_id]
    if routes:
        st.dataframe(df(routes), use_container_width=True, hide_index=True)
    else:
        st.info("No route snapshots for this game yet.")

    st.subheader("Sizing Breakdown")
    sizing_cols = [
        "bankroll_for_sizing_dollars",
        "available_cash_after_buffer_dollars",
        "half_kelly_target_dollars",
        "portfolio_cap_dollars",
        "cash_cap_dollars",
        "target_position_now_dollars",
        "filled_position_dollars",
        "reserved_open_order_dollars",
        "remaining_position_dollars",
    ]
    render_stat_grid([(col.replace("_", " "), dollars(market.get(col))) for col in sizing_cols])

    st.subheader("Liquidity Caps")
    liquidity_cols = [
        "visible_depth_cap_dollars",
        "recent_volume_cap_dollars",
        "cold_start_cap_dollars",
        "rolling_liquidity_cap_dollars",
        "cumulative_cap_remaining_dollars",
        "allowed_to_try_now_dollars",
        "next_child_order_dollars",
        "execution_binder",
    ]
    render_stat_grid(
        [
            (col.replace("_", " "), dollars(market.get(col)) if col.endswith("_dollars") else market.get(col))
            for col in liquidity_cols
        ]
    )

    st.subheader("Cash Priority")
    rank_label = (
        f"{market.get('cash_priority_rank')} / {market.get('cash_priority_rank_total')}"
        if market.get("cash_priority_rank") is not None
        else "-"
    )
    render_stat_grid(
        [
            ("Cash-limited mode", "ACTIVE" if market.get("cash_limited_mode") else "Clear"),
            ("Cash priority rank", rank_label),
            ("Priority score", decimal(market.get("cash_priority_score"))),
            ("Expected log growth of next child", decimal(market.get("expected_log_growth_next_child"))),
            ("Available cash cap", dollars(market.get("available_cash_after_buffer_dollars"))),
            ("Skipped due to cash", "yes" if market.get("skipped_due_to_cash") else "no"),
        ]
    )

    st.subheader("Fill / Order History")
    orders = [row for row in state["orders"] if row.get("game_id") == game_id]
    st.dataframe(df(orders), use_container_width=True, hide_index=True)

    st.subheader("Diagnostics")
    diag_cols = ["market_data_ts", "model_snapshot_ts", "injury_data_ts", "orderbook_ts", "last_reject_reason"]
    render_stat_grid([(col.replace("_", " "), short_ts(market.get(col)) if col.endswith("_ts") else market.get(col)) for col in diag_cols])

    st.subheader("Market Controls")
    reason = st.text_input("Detail command reason", value="Manual detail dashboard command")
    unlock_detail = st.checkbox("Unlock detail market controls", key=f"unlock_detail_{game_id}")
    cols = st.columns(4)
    if cols[0].button("Pause Market", use_container_width=True, disabled=not unlock_detail):
        run_command("Pause Market", lambda: db.apply_market_command(game_id, "PAUSE_MARKET", reason), connected)
    if cols[1].button("Unpause Market", use_container_width=True, disabled=not unlock_detail):
        run_command("Unpause Market", lambda: db.apply_market_command(game_id, "UNPAUSE_MARKET", reason), connected)
    if cols[2].button("Cancel Entry", use_container_width=True, disabled=not unlock_detail):
        run_command("Cancel Entry", lambda: db.apply_market_command(game_id, "CANCEL_ENTRY", reason), connected)
    if cols[3].button("Force Conservative", use_container_width=True, disabled=not unlock_detail):
        run_command("Force Conservative", lambda: db.apply_market_command(game_id, "FORCE_CONSERVATIVE_MARKET", reason), connected)
    decision = resolve_operator_decision(game_id)
    st.caption(
        f"Local execution override: {decision.game_decision.upper()} | "
        f"allowed={decision.trade_allowed} | reason={decision.reason} | risk={decision.risk_mode.upper()}"
    )
    local_cols = st.columns(2)
    if local_cols[0].button("Abort Game Locally", use_container_width=True, disabled=not unlock_detail):
        run_local_command(
            "Abort Game",
            lambda gid=game_id, r=reason: save_game_override(
                game_id=gid,
                decision="abort",
                reason=r,
                updated_by="streamlit_webapp",
            ),
        )
    if local_cols[1].button("Clear Local Abort", use_container_width=True, disabled=not unlock_detail):
        run_local_command(
            "Clear Abort",
            lambda gid=game_id, r=reason: save_game_override(
                game_id=gid,
                decision="default",
                reason=r,
                updated_by="streamlit_webapp",
            ),
        )


def historical_performance(state: dict[str, Any]) -> None:
    st.title("Historical Performance")
    equity = df(state["equity"])
    if not equity.empty and {"ts", "equity_dollars"}.issubset(equity.columns):
        st.plotly_chart(px.line(equity, x="ts", y="equity_dollars", title="Equity"), use_container_width=True)
        if "drawdown_dollars" in equity:
            st.plotly_chart(px.line(equity, x="ts", y="drawdown_dollars", title="Drawdown"), use_container_width=True)
    else:
        st.info("No equity curve rows yet.")

    closed = df(state["closed"])
    if not closed.empty:
        st.dataframe(closed, use_container_width=True, hide_index=True)
    else:
        st.info("No closed market summaries yet.")


def historical_market_detail(state: dict[str, Any]) -> None:
    st.title("Historical Market Detail")
    closed = state["closed"]
    if not closed:
        st.info("No closed market summaries yet.")
        return
    ids = [row["game_id"] for row in closed if row.get("game_id")]
    game_id = st.selectbox("Closed game", ids)
    row = next(item for item in closed if item["game_id"] == game_id)
    render_stat_grid(
        [
            ("Status", row.get("status") or "-"),
            ("Entered", "Yes" if row.get("did_enter") else "No"),
            ("P&L", dollars(row.get("pnl_dollars"))),
            ("Return", pct(row.get("total_return"))),
        ]
    )
    st.dataframe(df([row]), use_container_width=True, hide_index=True)


def ops_audit(state: dict[str, Any]) -> None:
    st.title("Ops / Audit")
    st.subheader("Bot Heartbeat")
    st.dataframe(df(state["heartbeat"]), use_container_width=True, hide_index=True)
    st.subheader("Command Log")
    st.dataframe(df(state["commands"]), use_container_width=True, hide_index=True)
    st.subheader("Market Controls")
    st.dataframe(df(state["controls"]), use_container_width=True, hide_index=True)
    st.subheader("Order Events")
    st.dataframe(df(state["orders"]), use_container_width=True, hide_index=True)
    st.subheader("System Alerts")
    st.dataframe(df(state["alerts"]), use_container_width=True, hide_index=True)
    st.subheader("Current Config Snapshot")
    st.json(db.config_snapshot())


def main() -> None:
    connected, _demo = require_auth()
    if st.button("Refresh", use_container_width=True):
        st.rerun()
    pages = [
        "Control Room",
        "Live Market Detail",
        "Historical Performance",
        "Historical Market Detail",
        "Ops / Audit",
    ]
    if "page" not in st.session_state:
        st.session_state["page"] = pages[0]
    page = st.selectbox("View", pages, index=pages.index(st.session_state["page"]))
    st.session_state["page"] = page
    state = load_state(connected)

    if page == "Control Room":
        control_room(state, connected)
    elif page == "Live Market Detail":
        live_market_detail(state, connected)
    elif page == "Historical Performance":
        historical_performance(state)
    elif page == "Historical Market Detail":
        historical_market_detail(state)
    else:
        ops_audit(state)


if __name__ == "__main__":
    main()
