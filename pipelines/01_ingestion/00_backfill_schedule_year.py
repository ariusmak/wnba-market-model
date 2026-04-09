import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from srwnba.config import load_config
from srwnba.client import SportradarClient
from srwnba.endpoints import EndpointConfig, season_schedule


def main(year: int, season_type: str, access_level: str = "trial"):
    season_type = season_type.upper().strip()
    print("STEP 0: starting")
    print("cwd:", os.getcwd())
    print("args:", year, season_type, access_level)

    # write a marker immediately so we know the script actually executed
    Path("data/bronze").mkdir(parents=True, exist_ok=True)
    marker = Path("data/bronze") / "SCHEDULE_SCRIPT_RAN.marker.txt"
    marker.write_text(f"ran at {datetime.now(timezone.utc).isoformat()}Z\n", encoding="utf-8")
    print("STEP 1: wrote marker:", marker.resolve())

    if season_type not in {"REG", "PST", "PRE"}:
        raise ValueError("season_type must be one of: REG, PST, PRE")

    print("STEP 2: loading config")
    cfg = load_config()

    print("STEP 3: creating client")
    client = SportradarClient(cfg)
    ep = EndpointConfig(access_level=access_level)

    url = season_schedule(ep, year, season_type)
    print("STEP 4: url:", url)

    print("STEP 5: fetching")
    data = client.get_json(url)
    print("STEP 6: fetched keys:", list(data.keys())[:20])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path("data/bronze") / f"schedule_{year}_{season_type}__{ts}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("STEP 7: saved:", out_path.resolve())
    print("games:", len(data.get("games", [])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--season-type", type=str, required=True)
    ap.add_argument("--access-level", type=str, default="trial")
    args = ap.parse_args()
    main(args.year, args.season_type, access_level=args.access_level)