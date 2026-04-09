import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from srwnba.config import load_config
from srwnba.client import SportradarClient
from srwnba.endpoints import EndpointConfig, daily_injuries
from srwnba.storage.bronze import save_bronze

def load_latest(pattern: str) -> dict:
    files = sorted(Path("data/bronze").glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing {pattern} in data/bronze")
    return json.loads(files[-1].read_text(encoding="utf-8"))

def parse_game_dates(schedule: dict) -> list[datetime]:
    out = []
    for g in schedule.get("games", []):
        s = g.get("scheduled")
        if not s:
            continue
        out.append(datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc))
    return out

def daterange(d0: datetime, d1: datetime):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def main():
    reg = load_latest("schedule_2019_REG__*.json")
    pst = load_latest("schedule_2019_PST__*.json")

    all_dates = parse_game_dates(reg) + parse_game_dates(pst)
    if not all_dates:
        raise RuntimeError("No game dates found in schedules")

    start = min(all_dates).date()
    end = max(all_dates).date()

    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=7)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=7)

    print("2019 window (UTC):", start_dt.date(), "to", end_dt.date())

    cfg = load_config()
    client = SportradarClient(cfg)
    ep = EndpointConfig(access_level="trial")

    ok = 0
    fail = 0

    for i, d in enumerate(daterange(start_dt, end_dt), start=1):
        y, m, day = d.year, d.month, d.day

        if i % 10 == 0:
            print(f"[{i}] fetching {y:04d}-{m:02d}-{day:02d} (ok={ok} fail={fail})")

        try:
            data = client.get_json(daily_injuries(ep, y, m, day))
            save_bronze(data, "data/bronze", f"daily_injuries__{y:04d}-{m:02d}-{day:02d}")
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[{i}] {y:04d}-{m:02d}-{day:02d} FAIL :: {e}")

        time.sleep(1.25)  # QPS=1 safety

    print("DONE ok=", ok, "fail=", fail)

if __name__ == "__main__":
    main()