import argparse
from pathlib import Path
import pandas as pd


def to_dt(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def build_episode_ids(evidence_df: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """
    evidence_df must have: player_id, evidence_dt, source
    Returns evidence_df with episode_id assigned.
    """
    evidence_df = evidence_df.sort_values(["player_id", "evidence_dt"]).copy()

    # compute gap within each player
    evidence_df["prev_dt"] = evidence_df.groupby("player_id")["evidence_dt"].shift(1)
    evidence_df["gap_days"] = (evidence_df["evidence_dt"] - evidence_df["prev_dt"]).dt.total_seconds() / 86400.0

    # new episode if first or gap > threshold
    evidence_df["new_episode"] = (evidence_df["prev_dt"].isna()) | (evidence_df["gap_days"] > gap_days)

    # episode counter per player
    evidence_df["episode_num"] = evidence_df.groupby("player_id")["new_episode"].cumsum().astype(int)

    # stable episode_id
    evidence_df["episode_id"] = (
        evidence_df["player_id"].astype(str)
        + f"_{evidence_df['episode_num'].astype(str)}"
    )

    return evidence_df.drop(columns=["prev_dt", "gap_days", "new_episode", "episode_num"])


def main(year: int, gap_days: int = 10):
    upd_path = Path(f"data/silver/injury_updates_clean_{year}.csv")
    dnp_path = Path(f"data/silver/injury_dnp_evidence_{year}.csv")

    if not upd_path.exists():
        raise FileNotFoundError(upd_path)
    if not dnp_path.exists():
        raise FileNotFoundError(dnp_path)

    upd = pd.read_csv(upd_path)
    dnp = pd.read_csv(dnp_path)

    # keep only medical issues for episode building
    upd_med = upd[upd["issue_class"].isin(["injury_medical", "illness"])].copy()
    dnp_med = dnp[dnp["issue_class"].isin(["injury_medical", "illness"])].copy()

    # evidence dates
    upd_med["evidence_dt"] = to_dt(upd_med["event_date"])
    dnp_med["evidence_dt"] = to_dt(dnp_med["game_date"])

    upd_med["source"] = "daily_update"
    dnp_med["source"] = "dnp_injury"

    # Combine evidence stream
    ev = pd.concat([
        upd_med[["player_id", "evidence_dt", "source", "desc", "status", "comment", "injury_id", "issue_class"]],
        dnp_med[["player_id", "evidence_dt", "source", "not_playing_reason", "not_playing_description", "matched_to_update", "matched_update_event_date", "issue_class"]],
    ], ignore_index=True, sort=False)

    ev = ev.dropna(subset=["player_id", "evidence_dt"]).copy()
    ev = ev.sort_values(["player_id", "evidence_dt", "source"], kind="stable")

    # Assign episodes by time-gap clustering per player
    ev_ep = build_episode_ids(ev, gap_days=gap_days)

    # Build episode summary table
    ep = (ev_ep.groupby(["player_id", "episode_id"], as_index=False)
          .agg(
              episode_start=("evidence_dt", "min"),
              episode_end=("evidence_dt", "max"),
              n_evidence=("evidence_dt", "size"),
              n_updates=("source", lambda s: int((s == "daily_update").sum())),
              n_dnp_games=("source", lambda s: int((s == "dnp_injury").sum())),
              episode_class=("issue_class", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
          ))

    # Save episode table
    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out_ep = Path(f"data/silver/injury_episodes_{year}.csv")
    ep.to_csv(out_ep, index=False)

    # Attach episode_id back onto dnp evidence and updates
    # For dnp: merge on player_id + game_date
    dnp_med_out = dnp_med.copy()
    dnp_med_out["evidence_dt"] = to_dt(dnp_med_out["game_date"])
    dnp_med_out = dnp_med_out.merge(
        ev_ep[ev_ep["source"] == "dnp_injury"][["player_id", "evidence_dt", "episode_id"]],
        on=["player_id", "evidence_dt"],
        how="left",
    )
    out_dnp = Path(f"data/silver/injury_dnp_evidence_{year}_with_episode.csv")
    dnp_med_out.to_csv(out_dnp, index=False)

    upd_med_out = upd_med.copy()
    upd_med_out = upd_med_out.merge(
        ev_ep[ev_ep["source"] == "daily_update"][["player_id", "evidence_dt", "episode_id"]],
        on=["player_id", "evidence_dt"],
        how="left",
    )
    out_upd = Path(f"data/silver/injury_updates_{year}_with_episode.csv")
    upd_med_out.to_csv(out_upd, index=False)

    print(f"{year}: episodes={len(ep)} gap_days={gap_days}")
    print("wrote:", out_ep)
    print("wrote:", out_dnp)
    print("wrote:", out_upd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gap-days", type=int, default=10)
    args = ap.parse_args()
    main(args.year, gap_days=args.gap_days)