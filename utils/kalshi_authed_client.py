"""
utils/kalshi_authed_client.py
─────────────────────────────
Authenticated Kalshi trade-API client for live trading.

Kalshi's authenticated endpoints (orderbook, portfolio, orders) require
RSA-PSS SHA-256 signatures in request headers.  This module handles the
signing + retry logic and exposes the endpoints the live entry loop
needs:

  - get_orderbook(ticker)             -> full bid/ask depth
  - get_market(ticker)                -> market metadata + last-trade px
  - create_order(...)                 -> place a new order (market or limit)
  - cancel_order(order_id)            -> cancel a resting order
  - get_order(order_id)               -> single order status
  - get_orders(...)                   -> list of our orders (filterable)
  - get_fills(...)                    -> realized trade fills
  - get_positions(...)                -> open positions
  - get_balance()                     -> cash balance

Credentials are loaded from env (via python-dotenv, .env in repo root):
    KALSHI_ACCESS_KEY         e.g. 7f3a1b0c-...
    KALSHI_PRIVATE_KEY_PATH   path to RSA private key PEM
    KALSHI_ENV                "prod" (default) or "demo"
    KALSHI_BASE_URL           optional explicit override, including /trade-api/v2
    KALSHI_TRADING_ENABLED    must be exactly "true" to allow write requests

Auth doc: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
"""
from __future__ import annotations

import base64
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dotenv import load_dotenv

BASE_URL_PROD = "https://external-api.kalshi.com/trade-api/v2"
BASE_URL_DEMO = "https://external-api.demo.kalshi.co/trade-api/v2"


@dataclass(frozen=True)
class KalshiAuthConfig:
    access_key: str
    private_key_path: Path
    base_url: str = BASE_URL_PROD
    request_delay_s: float = 0.10
    max_retries: int = 5
    base_sleep_s: float = 0.8
    timeout_s: int = 15
    trading_enabled: bool = False

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "KalshiAuthConfig":
        if dotenv_path is not None:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()
        access_key = os.environ.get("KALSHI_ACCESS_KEY")
        priv = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        env = os.environ.get("KALSHI_ENV", "prod").lower()
        base_url_override = os.environ.get("KALSHI_BASE_URL")
        trading_enabled = os.environ.get("KALSHI_TRADING_ENABLED", "").strip().lower() == "true"
        if not access_key or not priv:
            raise RuntimeError(
                "KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY_PATH must be set in env/.env"
            )
        base_url = base_url_override or (BASE_URL_DEMO if env == "demo" else BASE_URL_PROD)
        base_url = base_url.rstrip("/")
        return cls(
            access_key=access_key,
            private_key_path=Path(priv).expanduser().resolve(),
            base_url=base_url,
            trading_enabled=trading_enabled,
        )


class KalshiSigner:
    """Loads the RSA private key once and signs request strings."""

    def __init__(self, private_key_path: Path) -> None:
        if not private_key_path.exists():
            raise FileNotFoundError(f"Kalshi private key not found at {private_key_path}")
        with private_key_path.open("rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError(f"{private_key_path} is not an RSA private key")
        self._key = key

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        payload = (timestamp_ms + method.upper() + path).encode("utf-8")
        sig = self._key.sign(
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("ascii")


class AuthedKalshiClient:
    """Signed Kalshi trade-API client. Reuse one instance across a session."""

    def __init__(self, cfg: Optional[KalshiAuthConfig] = None) -> None:
        self.cfg = cfg or KalshiAuthConfig.from_env()
        self._signer = KalshiSigner(self.cfg.private_key_path)
        self._session = requests.Session()
        self._last_ts: float = 0.0

    # ------------------------------------------------------------------
    # Market data (authed)
    # ------------------------------------------------------------------

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        """Return full orderbook for a market.

        Response has `orderbook.yes` and `orderbook.no`: each is a list of
        [price_cents, size] pairs for resting orders on that side.
        Kalshi quotes bids — to buy we lift the opposite side.
        """
        params = {"depth": depth} if depth is not None else None
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}")

    def get_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        if max_ts is not None:
            params["max_ts"] = int(max_ts)
        return self._paginated("GET", "/markets/trades", params, key="trades")

    def list_markets(
        self,
        *,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        return self._paginated("GET", "/markets", params, key="markets")

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/events/{event_ticker}")

    # ------------------------------------------------------------------
    # Portfolio / orders
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def get_positions(
        self,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        settlement_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if settlement_status:
            params["settlement_status"] = settlement_status
        return self._paginated("GET", "/portfolio/positions", params, key="market_positions")

    def get_fills(
        self,
        ticker: Optional[str] = None,
        order_id: Optional[str] = None,
        min_ts: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts is not None:
            params["min_ts"] = min_ts
        return self._paginated("GET", "/portfolio/fills", params, key="fills")

    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self._paginated("GET", "/portfolio/orders", params, key="orders")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def create_order(
        self,
        *,
        ticker: str,
        action: str,                # "buy" | "sell"
        side: str,                  # "yes" | "no"
        count: int,
        order_type: str,            # "market" | "limit"
        client_order_id: str,       # idempotency key, caller-provided
        yes_price_cents: Optional[int] = None,
        no_price_cents: Optional[int] = None,
        time_in_force: Optional[str] = None,   # e.g. "IOC" for taker sweep
        expiration_ts: Optional[int] = None,
        post_only: Optional[bool] = None,
        cancel_order_on_pause: Optional[bool] = True,
        self_trade_prevention_type: Optional[str] = "taker_at_cross",
    ) -> Dict[str, Any]:
        """Place an order. Always pass `client_order_id` for idempotency.

        For a taker sweep:  order_type="market", time_in_force="IOC".
        For a resting limit: order_type="limit", yes_price_cents=<int>.

        Prices are integer cents (0..100). Kalshi's API field names are
        `yes_price` / `no_price` (cents). Only the side matching `side`
        needs to be set on a buy (we pay that side's price).
        """
        if action not in {"buy", "sell"}:
            raise ValueError(f"action must be buy/sell, got {action}")
        if side not in {"yes", "no"}:
            raise ValueError(f"side must be yes/no, got {side}")
        if order_type not in {"market", "limit"}:
            raise ValueError(f"order_type must be market/limit, got {order_type}")

        body: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": int(count),
            "type": order_type,
            "client_order_id": client_order_id,
        }
        if order_type == "limit":
            if yes_price_cents is None and no_price_cents is None:
                raise ValueError("limit order requires yes_price_cents or no_price_cents")
            if yes_price_cents is not None:
                body["yes_price"] = int(yes_price_cents)
            if no_price_cents is not None:
                body["no_price"] = int(no_price_cents)
        if time_in_force:
            body["time_in_force"] = time_in_force
        if expiration_ts is not None:
            body["expiration_ts"] = int(expiration_ts)
        if post_only is not None:
            body["post_only"] = bool(post_only)
        if cancel_order_on_pause is not None:
            body["cancel_order_on_pause"] = bool(cancel_order_on_pause)
        if self_trade_prevention_type:
            body["self_trade_prevention_type"] = self_trade_prevention_type

        return self._request("POST", "/portfolio/orders", json_body=body)

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _paginated(
        self,
        method: str,
        path: str,
        params: Dict[str, Any],
        key: str,
        max_pages: int = 50,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            resp = self._request(method, path, params=p)
            items.extend(resp.get(key, []) or [])
            cursor = resp.get("cursor")
            if not cursor:
                break
        return items

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        method = method.upper()
        assert path.startswith("/"), path
        if method in {"POST", "PUT", "PATCH", "DELETE"} and not self.cfg.trading_enabled:
            raise RuntimeError(
                f"{method} {path} blocked: set KALSHI_TRADING_ENABLED=true "
                "to allow authenticated write requests."
            )
        sign_path = "/trade-api/v2" + path
        url = self.cfg.base_url + path
        if params:
            params = {k: v for k, v in params.items() if v is not None}
            if params:
                url = url + "?" + urlencode(params)

        for attempt in range(self.cfg.max_retries + 1):
            self._throttle()
            ts_ms = str(int(time.time() * 1000))
            signature = self._signer.sign(ts_ms, method, sign_path)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "KALSHI-ACCESS-KEY": self.cfg.access_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            }
            try:
                resp = self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body if json_body is not None else None,
                    timeout=self.cfg.timeout_s,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(f"{method} {path} failed: {exc}") from exc
            finally:
                self._last_ts = time.monotonic()

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._retry_sleep(resp, attempt))
                    continue
                raise RuntimeError(
                    f"{method} {path} -> {resp.status_code}: {resp.text[:300]}"
                )
            if not resp.ok:
                raise RuntimeError(
                    f"{method} {path} -> {resp.status_code}: {resp.text[:500]}"
                )
            if not resp.content:
                return {}
            return resp.json()
        raise RuntimeError(f"{method} {path}: exhausted retries")

    def _throttle(self) -> None:
        gap = self.cfg.request_delay_s - (time.monotonic() - self._last_ts)
        if gap > 0:
            time.sleep(gap)

    def _backoff(self, attempt: int) -> float:
        base = self.cfg.base_sleep_s * (2 ** attempt)
        return min(base + random.random() * 0.25, 30.0)

    def _retry_sleep(self, resp: requests.Response, attempt: int) -> float:
        try:
            return float(resp.headers.get("Retry-After", self._backoff(attempt)))
        except (ValueError, TypeError):
            return self._backoff(attempt)
