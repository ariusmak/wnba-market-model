import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
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


def parse_pull_date_from_bronze_name(path: Path) -> date | None:
    parts = path.name.split("__")
    if len(parts) < 3:
        return None
    pulled = parts[-1].removesuffix(".json")
    try:
        return datetime.strptime(pulled, "%Y%m%dT%H%M%SZ").date()
    except ValueError:
        return None


def accepted_existing_daily_injury_dates(
    year: int,
    force_refresh_on_or_after: date | None = None,
) -> set[str]:
    """
    Count an existing daily-injury file as reusable only if it was pulled on
    or after the injury date. This prevents empty future placeholder files from
    blocking the real date once it arrives.
    """
    out: set[str] = set()
    for p in Path("data/bronze").glob(f"daily_injuries__{year}-*__*.json"):
        parts = p.name.split("__")
        if len(parts) < 3:
            continue
        date_key = parts[1]
        try:
            injury_date = datetime.strptime(date_key, "%Y-%m-%d").date()
        except ValueError:
            continue
        if force_refresh_on_or_after is not None and injury_date >= force_refresh_on_or_after:
            continue
        pull_date = parse_pull_date_from_bronze_name(p)
        if pull_date is not None and pull_date >= injury_date:
            out.add(date_key)
    return out


def main(
    year: int,
    access_level: str = "trial",
    buffer_days: int = 7,
    sleep_s: float = 1.25,
    allow_future: bool = False,
    refresh_lookback_days: int = 2,
):
    reg = load_latest(f"schedule_{year}_REG__*.json")
    pst = load_latest(f"schedule_{year}_PST__*.json")

    all_dates = parse_game_dates(reg) + parse_game_dates(pst)
    if not all_dates:
        raise RuntimeError(f"No game dates found in schedules for {year}")

    start = min(all_dates).date()
    end = max(all_dates).date()

    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=buffer_days)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=buffer_days)
    today_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if not allow_future and end_dt > today_dt:
        end_dt = today_dt

    lookback = max(0, int(refresh_lookback_days))
    force_refresh_on_or_after = None
    if lookback > 0:
        force_refresh_on_or_after = today_dt.date() - timedelta(days=lookback - 1)

    existing = accepted_existing_daily_injury_dates(
        year,
        force_refresh_on_or_after=force_refresh_on_or_after,
    )
    # existing contains reusable strings like "YYYY-MM-DD"

    refresh_msg = (
        f"; force-refresh >= {force_refresh_on_or_after}"
        if force_refresh_on_or_after is not None
        else "; no force-refresh lookback"
    )
    print(
        f"{year} window (UTC): {start_dt.date()} to {end_dt.date()}  "
        f"(buffer_days={buffer_days}{refresh_msg})"
    )

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
            url = daily_injuries(ep, y, m, day)
            data = client.get_json(url)  # /league/YYYY/MM/DD/daily_injuries.json
            save_bronze(
                data,
                "data/bronze",
                f"daily_injuries__{y:04d}-{m:02d}-{day:02d}",
                source="sportradar",
                endpoint="daily_injuries",
                request_url=url,
                request_params={
                    "year": y,
                    "month": m,
                    "day": day,
                    "access_level": access_level,
                    "refresh_lookback_days": refresh_lookback_days,
                },
            )
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
    ap.add_argument(
        "--refresh-lookback-days",
        type=int,
        default=2,
        help="Always re-pull this many recent injury dates, inclusive of today.",
    )
    ap.add_argument(
        "--allow-future",
        action="store_true",
        help="Permit fetching future schedule dates. Production default is false.",
    )
    args = ap.parse_args()

    main(
        args.year,
        access_level=args.access_level,
        buffer_days=args.buffer_days,
        sleep_s=args.sleep_s,
        allow_future=args.allow_future,
        refresh_lookback_days=args.refresh_lookback_days,
    )
