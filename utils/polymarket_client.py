"""
utils/polymarket_client.py
──────────────────────────
Thin HTTP client for the three Polymarket REST APIs with:
  - Per-API base URL configuration
  - Conservative throttling (stays well within published rate limits)
  - Exponential-backoff retry on 429 / 5xx

Auth note:
  All endpoints used here are public (no API key required).
  CLOB /trades requires L2 auth — that endpoint is NOT used here;
  public trades are fetched from the Data API instead.

Published rate limits (requests per 10 seconds):
  Gamma /markets         300 req/10 s   → limit to 1 req/0.05 s = 200/10 s
  Gamma /events          500 req/10 s   → same conservative throttle
  CLOB  /book /price     1500 req/10 s  → same conservative throttle
  Data  /trades          200 req/10 s   → limit to 1 req/0.06 s = 167/10 s

A single shared throttle of 0.05 s is fine for all Gamma and CLOB calls.
The Data API trades endpoint is inherently serial (one market at a time),
so 0.06 s is used there via the request_delay_data_s parameter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# ── Base URLs ─────────────────────────────────────────────────────────────────
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL  = "https://clob.polymarket.com"
DATA_URL  = "https://data-api.polymarket.com"


@dataclass(frozen=True)
class PolymarketConfig:
    gamma_url: str          = GAMMA_URL
    clob_url: str           = CLOB_URL
    data_url: str           = DATA_URL
    # Minimum seconds between successive requests (conservative rate-limit guard).
    request_delay_s: float  = 0.05   # Gamma + CLOB: safe at 300/10s limit
    request_delay_data_s: float = 0.06  # Data API /trades: 200/10s limit
    max_retries: int        = 6
    base_sleep_s: float     = 1.0
    timeout_s: int          = 30


class PolymarketClient:
    """
    Stateful HTTP client for Gamma, CLOB, and Data APIs.
    Reuse a single instance across the session.
    """

    def __init__(self, cfg: Optional[PolymarketConfig] = None) -> None:
        self.cfg = cfg or PolymarketConfig()
        self._session = requests.Session()
        self._last_ts: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def gamma_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """GET https://gamma-api.polymarket.com{path}"""
        return self._get(self.cfg.gamma_url, path, params, self.cfg.request_delay_s)

    def clob_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """GET https://clob.polymarket.com{path}"""
        return self._get(self.cfg.clob_url, path, params, self.cfg.request_delay_s)

    def data_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """GET https://data-api.polymarket.com{path}"""
        return self._get(self.cfg.data_url, path, params, self.cfg.request_delay_data_s)

    # ── private helpers ───────────────────────────────────────────────────────

    def _get(
        self,
        base: str,
        path: str,
        params: Optional[Dict[str, Any]],
        delay: float,
    ) -> Any:
        url = f"{base}{path}"
        for attempt in range(self.cfg.max_retries + 1):
            self._throttle(delay)
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.timeout_s)
                self._last_ts = time.monotonic()

                if resp.status_code == 429:
                    sleep_s = self._retry_sleep(resp, attempt)
                    time.sleep(sleep_s)
                    continue

                if resp.status_code >= 500:
                    if attempt < self.cfg.max_retries:
                        time.sleep(self._backoff(attempt))
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.ConnectionError as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(
                    f"Connection failed after {attempt + 1} tries: {exc}"
                ) from exc

            except requests.exceptions.Timeout as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(
                    f"Timed out after {attempt + 1} tries: {exc}"
                ) from exc

        raise RuntimeError(f"Request to {url} failed after {self.cfg.max_retries} retries")

    def _throttle(self, delay: float) -> None:
        elapsed = time.monotonic() - self._last_ts
        gap = delay - elapsed
        if gap > 0:
            time.sleep(gap)

    def _backoff(self, attempt: int) -> float:
        return min(self.cfg.base_sleep_s * (2 ** attempt), 60.0)

    def _retry_sleep(self, resp: requests.Response, attempt: int) -> float:
        try:
            return float(resp.headers.get("Retry-After", self._backoff(attempt)))
        except (ValueError, TypeError):
            return self._backoff(attempt)
