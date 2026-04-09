import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict


def save_bronze(payload: Dict[str, Any], out_dir: str, name: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) / f"{name}__{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)