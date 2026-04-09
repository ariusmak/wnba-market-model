import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from srwnba.config import load_config
from srwnba.client import SportradarClient
from srwnba.endpoints import EndpointConfig, game_summary


def load_latest_schedule(year: int, season_type: str) -> dict:
    season_type = season_type.upper()
    pattern = f"schedule_{year}_{season_type}__*.json"
    files = sorted(Path("data/bronze").glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Missing {pattern} in data/bronze. Run 00_backfill_schedule_year.py first."
        )
    return json.loads(files[-1].read_text(encoding="utf-8"))


def already_fetched_game_ids() -> set[str]:
    # We consider a game fetched if any bronze file exists for that game_id.
    # Filenames: game_summary__{gid}__{ts}.json
    out = set()
    for p in Path("data/bronze").glob("game_summary__*__*.json"):
        name = p.name
        # split on "__" -> ["game_summary", "{gid}", "{ts}.json"]
        parts = name.split("__")
        if len(parts) >= 3:
            out.add(parts[1])
    return out


def save_bronze_direct(data: dict, prefix: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{prefix}__{ts}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def is_quota_or_access_error(msg: str) -> bool:
    m = msg.lower()
    return (
        ("http 403" in m)
        or ("forbidden" in m)
        or ("unauthorized" in m)
        or ("api key" in m and ("invalid" in m or "unauthor" in m))
        or ("quota" in m)
        or ("call" in m and "exceed" in m)
        or ("subscription" in m and ("expired" in m or "invalid" in m))
    )


def main(
    year: int,
    season_type: str,
    access_level: str = "trial",
    sleep_s: float = 1.25,
    only_closed: bool = False,
):
    season_type = season_type.upper().strip()
    if season_type not in {"REG", "PST", "PRE"}:
        raise ValueError("season_type must be one of: REG, PST, PRE")

    print("cwd:", os.getcwd())
    sched = load_latest_schedule(year, season_type)

    games = sched.get("games", [])
    if only_closed:
        games = [g for g in games if (g.get("status") or "").lower() == "closed"]

    game_ids = [g["id"] for g in games if g.get("id")]
    print(f"{year} {season_type}: schedule games={len(sched.get('games', []))} using={len(game_ids)}")

    fetched = already_fetched_game_ids()
    print(f"already have summaries for {len(fetched)} games (any year/type)")

    cfg = load_config()
    client = SportradarClient(cfg)
    ep = EndpointConfig(access_level=access_level)

    ok = 0
    skip = 0
    fail = 0
    stop_reason = None

    for i, gid in enumerate(game_ids, start=1):
        if gid in fetched:
            skip += 1
            continue

        if i % 25 == 0:
            print(f"[{i}/{len(game_ids)}] (ok={ok} skip={skip} fail={fail})")

        try:
            data = client.get_json(game_summary(ep, gid))  # /games/{game_id}/summary.json
            out = save_bronze_direct(data, f"game_summary__{gid}", Path("data/bronze"))
            ok += 1
            fetched.add(gid)

            # Light sanity: show first few saves
            if ok <= 3:
                print("saved", out)
        except Exception as e:
            msg = str(e)
            if is_quota_or_access_error(msg):
                stop_reason = msg
                print(f"[{i}/{len(game_ids)}] STOP (quota/access): {msg}")
                break

            fail += 1
            print(f"[{i}/{len(game_ids)}] FAIL {gid}: {msg}")

        time.sleep(sleep_s)

    print("DONE year=", year, "type=", season_type, "ok=", ok, "skip=", skip, "fail=", fail)
    if stop_reason:
        print("STOP_REASON:", stop_reason)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--season-type", type=str, required=True, help="REG, PST, or PRE")
    ap.add_argument("--access-level", type=str, default="trial")
    ap.add_argument("--sleep-s", type=float, default=1.25)
    ap.add_argument(
        "--only-closed",
        action="store_true",
        help="Only fetch games marked status=closed in schedule (optional).",
    )
    args = ap.parse_args()

    main(
        args.year,
        args.season_type,
        access_level=args.access_level,
        sleep_s=args.sleep_s,
        only_closed=args.only_closed,
    )