"""
00_check_kalshi_auth.py
=======================

Safe Kalshi auth smoke test. It loads `.env`, verifies that the configured
private key file exists and can sign requests, then calls `/portfolio/balance`.

This script never prints API keys or private-key material.

Usage:
    python pipelines/07_live/00_check_kalshi_auth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.kalshi_authed_client import AuthedKalshiClient, KalshiAuthConfig  # noqa: E402


def _money_from_cents(cents: int | float | None) -> str:
    if cents is None:
        return "unknown"
    return f"${float(cents) / 100.0:,.2f}"


def main() -> None:
    cfg = KalshiAuthConfig.from_env(REPO_ROOT / ".env")
    print("Kalshi env check")
    print(f"  base_url: {cfg.base_url}")
    print(f"  access_key: {'set' if cfg.access_key else 'missing'}")
    print(f"  private_key_path: {cfg.private_key_path}")
    print(f"  private_key_exists: {cfg.private_key_path.exists()}")
    print(f"  trading_enabled: {cfg.trading_enabled}")

    client = AuthedKalshiClient(cfg)
    balance = client.get_balance()
    raw_balance = balance.get("balance")
    raw_portfolio = balance.get("portfolio_value")
    print("Auth OK")
    print(f"  balance: {_money_from_cents(raw_balance)}")
    if raw_portfolio is not None:
        print(f"  portfolio_value: {_money_from_cents(raw_portfolio)}")


if __name__ == "__main__":
    main()
