from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from .config import SRConfig


class SportradarClient:
    def __init__(self, cfg: SRConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        max_retries: int = 6,
        base_sleep_s: float = 1.0,
    ) -> Dict[str, Any]:
        """
        GET JSON with basic retry handling.
        - Retries 429 and 5xx with exponential backoff
        - Respects Retry-After header when present
        """
        params = dict(params or {})
        params["api_key"] = self.cfg.api_key

        last_err: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            r = None
            try:
                t0 = time.time()
                r = self.session.get(url, params=params, timeout=self.cfg.timeout_s)
                _elapsed = time.time() - t0

                if r.status_code < 400:
                    return r.json()

                # 429 / 5xx => retry
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = base_sleep_s * (2 ** attempt)
                    else:
                        sleep_s = base_sleep_s * (2 ** attempt)

                    # small cap so it doesn't go crazy
                    sleep_s = min(sleep_s, 60.0)

                    if attempt < max_retries:
                        time.sleep(sleep_s)
                        continue

                # Non-retryable (or out of retries)
                raise RuntimeError(f"HTTP {r.status_code} for {r.url} :: {r.text[:300]}")

            except Exception as e:
                last_err = e
                # network errors etc: retry with backoff
                if attempt < max_retries:
                    time.sleep(min(base_sleep_s * (2 ** attempt), 60.0))
                    continue
                raise

        # should never reach here
        raise last_err if last_err else RuntimeError("Unknown error")