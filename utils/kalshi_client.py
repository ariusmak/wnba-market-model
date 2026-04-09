"""
utils/kalshi_client.py
──────────────────────
Thin HTTP client for the Kalshi REST API with:
  - Configurable base URL and request throttle
  - Exponential-backoff retry on 429 / 5xx
  - No authentication required for public read endpoints

Auth note:
  The live orderbook endpoint (GET /markets/{ticker}/orderbook) requires
  RSA-PSS signed headers.  All other endpoints used here are public.

Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiConfig:
    base_url: str = BASE_URL
    # Minimum seconds between successive requests (conservative rate-limit guard).
    # Kalshi does not publish explicit rate limits for public endpoints.
    request_delay_s: float = 0.25
    max_retries: int = 6
    base_sleep_s: float = 1.0
    timeout_s: int = 30


class KalshiClient:
    """Stateful HTTP client.  Reuse a single instance across the session."""

    def __init__(self, cfg: Optional[KalshiConfig] = None) -> None:
        self.cfg = cfg or KalshiConfig()
        self._session = requests.Session()
        self._last_ts: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        GET {base_url}{path} with retry/backoff.

        Parameters
        ----------
        path   : relative path, e.g. "/historical/markets"
        params : query-string parameters dict (values are URL-encoded)

        Returns
        -------
        Parsed JSON dict.

        Raises
        ------
        RuntimeError after max_retries exhausted.
        """
        url = f"{self.cfg.base_url}{path}"

        for attempt in range(self.cfg.max_retries + 1):
            self._throttle()
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
                raise RuntimeError(f"Connection failed after {attempt + 1} tries: {exc}") from exc

            except requests.exceptions.Timeout as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(f"Timed out after {attempt + 1} tries: {exc}") from exc

        raise RuntimeError(f"Request to {url} failed after {self.cfg.max_retries} retries")

    # ── private helpers ───────────────────────────────────────────────────────

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_ts
        gap = self.cfg.request_delay_s - elapsed
        if gap > 0:
            time.sleep(gap)

    def _backoff(self, attempt: int) -> float:
        return min(self.cfg.base_sleep_s * (2 ** attempt), 60.0)

    def _retry_sleep(self, resp: requests.Response, attempt: int) -> float:
        try:
            return float(resp.headers.get("Retry-After", self._backoff(attempt)))
        except (ValueError, TypeError):
            return self._backoff(attempt)
