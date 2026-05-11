"""Per-game process locks for canonical route loops."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GameLockStatus:
    locked: bool
    path: Path
    pid: Optional[int] = None
    running: bool = False
    metadata: Mapping[str, Any] | None = None


class GameProcessLock:
    def __init__(self, *, game_id: str, lock_dir: Path, metadata: Mapping[str, Any] | None = None) -> None:
        self.game_id = str(game_id)
        self.lock_dir = Path(lock_dir)
        self.path = self.lock_dir / f"{_safe_name(self.game_id)}.lock"
        self.metadata = dict(metadata or {})
        self.acquired = False

    def acquire(self) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "game_id": self.game_id,
            "pid": os.getpid(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_ts_s": time.time(),
            "metadata": self.metadata,
        }
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        while True:
            try:
                with self.path.open("x", encoding="utf-8") as f:
                    f.write(body)
                self.acquired = True
                return
            except FileExistsError:
                status = read_game_lock_status(self.path)
                if status.locked and status.running:
                    raise RuntimeError(
                        f"route loop already running for game_id={self.game_id} "
                        f"pid={status.pid} lock={self.path}"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"stale route lock exists and could not be removed: {self.path}"
                    ) from exc

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            status = read_game_lock_status(self.path)
            if status.pid == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False

    def __enter__(self) -> "GameProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def read_game_lock_status(path: Path) -> GameLockStatus:
    path = Path(path)
    if not path.exists():
        return GameLockStatus(locked=False, path=path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return GameLockStatus(locked=True, path=path, pid=None, running=False, metadata={})
    pid = _int_or_none(payload.get("pid"))
    return GameLockStatus(
        locked=True,
        path=path,
        pid=pid,
        running=pid_is_running(pid or 0),
        metadata=payload,
    )


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _int_or_none(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
