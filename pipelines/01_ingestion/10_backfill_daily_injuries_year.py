import argparse
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
    out: list[datetime] = []
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


def main(year: int, access_level: str = "trial", buffer_days: int = 7, sleep_s: float = 1.25):
    reg = load_latest(f"schedule_{year}_REG__*.json")
    pst = load_latest(f"schedule_{year}_PST__*.json")

    all_dates = parse_game_dates(reg) + parse_game_dates(pst)
    if not all_dates:
        raise RuntimeError(f"No game dates found in schedules for {year}")

    start = min(all_dates).date()
    end = max(all_dates).date()

    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=buffer_days)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=buffer_days)

    existing = {p.name.split("__")[1] for p in Path("data/bronze").glob(f"daily_injuries__{year}-*__*.json")}
    #existing will contain strings like "YYYY-MM-DD"

    print(f"{year} window (UTC): {start_dt.date()} to {end_dt.date()}  (buffer_days={buffer_days})")

    cfg = load_config()
    client = SportradarClient(cfg)
    ep = EndpointConfig(access_level=access_level)

    ok = 0
    fail = 0

    for i, d in enumerate(daterange(start_dt, end_dt), start=1):
        y, m, day = d.year, d.month, d.day

        
        date_key = f"{y:04d}-{m:02d}-{day:02d}"
        if date_key in existing:
            continue
        
        if i % 10 == 0:
            print(f"[{i}] fetching {y:04d}-{m:02d}-{day:02d} (ok={ok} fail={fail})")

        try:
            data = client.get_json(daily_injuries(ep, y, m, day))  # /league/YYYY/MM/DD/daily_injuries.json
            save_bronze(data, "data/bronze", f"daily_injuries__{y:04d}-{m:02d}-{day:02d}")
            ok += 1
        except Exception as e:
            msg = str(e)
            # Stop immediately on quota / access style errors so you can swap keys
            if ("HTTP 403" in msg) or ("quota" in msg.lower()) or ("calls" in msg.lower() and "exceed" in msg.lower()):
                print(f"[{i}] {y:04d}-{m:02d}-{day:02d} QUOTA/ACCESS STOP :: {msg}")
                break
            fail += 1
            print(f"[{i}] {y:04d}-{m:02d}-{day:02d} FAIL :: {msg}")

        time.sleep(sleep_s)

    print("DONE ok=", ok, "fail=", fail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--access-level", type=str, default="trial")
    ap.add_argument("--buffer-days", type=int, default=7)
    ap.add_argument("--sleep-s", type=float, default=1.25)
    args = ap.parse_args()

    main(args.year, access_level=args.access_level, buffer_days=args.buffer_days, sleep_s=args.sleep_s)