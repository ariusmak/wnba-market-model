import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4


SECRET_KEYS = {
    "api_key",
    "key",
    "token",
    "access_key",
    "private_key",
    "signature",
    "authorization",
}


def save_bronze(
    payload: Dict[str, Any],
    out_dir: str,
    name: str,
    *,
    source: str | None = None,
    endpoint: str | None = None,
    method: str = "GET",
    request_url: str | None = None,
    request_params: Mapping[str, Any] | None = None,
    response_status: int | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    canonical_dir = Path(out_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{ts}_{uuid4().hex[:10]}"
    run_dir = canonical_dir.parent / "bronze_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    staged_path = run_dir / "payload.json"
    staged_path.write_text(body, encoding="utf-8")

    accepted, validation_errors = _validate_payload(payload)
    canonical_path = canonical_dir / f"{name}__{ts}.json"
    manifest = {
        "run_id": run_id,
        "name": name,
        "source": source,
        "endpoint": endpoint,
        "method": method.upper(),
        "request_url": _sanitize_url(request_url),
        "request_params": _sanitize_mapping(request_params or {}),
        "response_status": response_status,
        "pulled_at_utc": ts,
        "sha256": digest,
        "payload_bytes": len(body.encode("utf-8")),
        "staged_path": str(staged_path),
        "canonical_dir": str(canonical_dir),
        "canonical_path": str(canonical_path) if accepted else None,
        "context": _sanitize_mapping(context or {}),
        "accepted": accepted,
        "validation_errors": validation_errors,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not accepted:
        raise ValueError(f"bronze payload failed validation for {name}: {validation_errors}")

    canonical_path.write_text(body, encoding="utf-8")
    return str(canonical_path)


def _validate_payload(payload: Dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("payload_not_json_object")
    elif len(payload) == 0:
        errors.append("payload_empty_object")
    return len(errors) == 0, errors


def _sanitize_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        k = str(key)
        if _is_secret_key(k):
            out[k] = "***REDACTED***"
        elif isinstance(value, Mapping):
            out[k] = _sanitize_mapping(value)
        elif isinstance(value, (list, tuple)):
            out[k] = [
                _sanitize_mapping(v) if isinstance(v, Mapping) else v
                for v in value
            ]
        else:
            out[k] = value
    return out


def _sanitize_url(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "***REDACTED***" if _is_secret_key(key) else value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SECRET_KEYS or any(part in lowered for part in SECRET_KEYS)
