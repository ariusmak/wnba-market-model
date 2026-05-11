import argparse
import os
from pathlib import Path

from srwnba.config import load_config
from srwnba.client import SportradarClient
from srwnba.endpoints import EndpointConfig, season_schedule
from srwnba.storage.bronze import save_bronze


def main(year: int, season_type: str, access_level: str = "trial"):
    season_type = season_type.upper().strip()
    print("STEP 0: starting")
    print("cwd:", os.getcwd())
    print("args:", year, season_type, access_level)

    Path("data/bronze").mkdir(parents=True, exist_ok=True)

    if season_type not in {"REG", "PST", "PRE"}:
        raise ValueError("season_type must be one of: REG, PST, PRE")

    print("STEP 1: loading config")
    cfg = load_config()

    print("STEP 2: creating client")
    client = SportradarClient(cfg)
    ep = EndpointConfig(access_level=access_level)

    url = season_schedule(ep, year, season_type)
    print("STEP 3: url:", url)

    print("STEP 4: fetching")
    data = client.get_json(url)
    print("STEP 5: fetched keys:", list(data.keys())[:20])

    out_path = save_bronze(
        data,
        "data/bronze",
        f"schedule_{year}_{season_type}",
        source="sportradar",
        endpoint="season_schedule",
        request_url=url,
        request_params={
            "year": year,
            "season_type": season_type,
            "access_level": access_level,
        },
    )

    print("STEP 6: saved:", Path(out_path).resolve())
    print("games:", len(data.get("games", [])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--season-type", type=str, required=True)
    ap.add_argument("--access-level", type=str, default="trial")
    args = ap.parse_args()
    main(args.year, args.season_type, access_level=args.access_level)
