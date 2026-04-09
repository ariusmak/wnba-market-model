import json
import time
from pathlib import Path

from srwnba.config import load_config
from srwnba.client import SportradarClient
from srwnba.endpoints import EndpointConfig, game_summary
from srwnba.storage.bronze import save_bronze


def main():
    # Load latest schedule file
    bronze_dir = Path("data/bronze")
    sched_files = sorted(bronze_dir.glob("schedule_2025_REG__*.json"))
    if not sched_files:
        raise FileNotFoundError("No schedule_2025_REG__*.json found in data/bronze")

    schedule_path = sched_files[-1]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    games = schedule.get("games", [])
    game_ids = [g["id"] for g in games if "id" in g]

    cfg = load_config()
    client = SportradarClient(cfg)
    ep_cfg = EndpointConfig(access_level="trial")

    out_dir = "data/bronze"

    print(f"Loaded schedule: {schedule_path}")
    print(f"Found games: {len(game_ids)}")

    ok = 0
    fail = 0

    for i, gid in enumerate(game_ids, start=1):
        try:
            data = client.get_json(game_summary(ep_cfg, gid))
            save_bronze(data, out_dir, f"game_summary__{gid}")
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(game_ids)}] FAIL {gid} :: {e}")
            # stop on first error so we don't spam requests
            break

        if i % 25 == 0:
            print(f"[{i}/{len(game_ids)}] ok={ok} fail={fail}")

        # gentle pacing (safe default)
        time.sleep(0.25)

    print(f"DONE ok={ok} fail={fail}")


if __name__ == "__main__":
    main()