import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------
# Configurable tokenization
# -------------------------
BODY_TOKENS = [
    "knee", "ankle", "foot", "toe", "leg", "calf", "hamstring", "quad", "groin", "hip",
    "back", "neck", "shoulder", "arm", "elbow", "wrist", "hand", "finger", "thumb",
    "head", "concussion", "illness", "sick", "flu", "covid", "respiratory", "eye"
]

REST_COACH_PATTERNS = [
    r"\brest\b",
    r"\bcoach\b",
    r"coach.?s decision",
    r"\bdnp\b.*\brest\b",
    r"\bdnd\b.*\brest\b",
    r"\bdnp\b.*\bcoach\b",
    r"\bdnd\b.*\bcoach\b",
]

INJURY_ILLNESS_PATTERNS = [
    r"\binjury\b",
    r"\billness\b",
    r"\bsick\b",
    r"\bconcussion\b",
]

PERSONAL_PATTERNS = [
    r"\bpersonal\b",
    r"\bfamily\b",
    r"\bbereavement\b",
]


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def extract_tokens(*fields) -> set:
    text = " ".join([norm_text(f) for f in fields])
    toks = set()
    for t in BODY_TOKENS:
        if t in text:
            toks.add(t)
    # special cases
    if "ill" in text:
        toks.add("illness")
    return toks


def classify_issue(desc: str, comment: str, npr: str, npd: str) -> str:
    """
    A light taxonomy: medical vs nonmedical.
    """
    text = " ".join([norm_text(desc), norm_text(comment), norm_text(npr), norm_text(npd)])

    if any(re.search(p, text) for p in REST_COACH_PATTERNS):
        return "rest_or_coach"

    if any(re.search(p, text) for p in INJURY_ILLNESS_PATTERNS):
        # medical; try to bucket illness vs injury body-part
        if "illness" in text or "sick" in text or "flu" in text or "covid" in text:
            return "illness"
        return "injury_medical"

    if any(re.search(p, text) for p in PERSONAL_PATTERNS):
        return "personal"

    # if desc itself is a body part (common in daily updates: "Knee", "Back", etc.)
    if extract_tokens(desc, comment):
        return "injury_medical"

    return "unknown"


def make_synth_event_id(player_id: str, game_id: str) -> str:
    return f"DNP_{player_id}_{game_id}"


def main(year: int, lookback_days: int = 2):
    inj_path = Path(f"data/silver/injury_events_{year}.csv")
    ava_path = Path(f"data/silver/game_availability_{year}_REGPST.csv")
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")

    if not inj_path.exists():
        raise FileNotFoundError(inj_path)
    if not ava_path.exists():
        raise FileNotFoundError(ava_path)
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    inj = pd.read_csv(inj_path)
    ava = pd.read_csv(ava_path)
    played = pd.read_csv(played_path)

    # -------------------------
    # Clean / normalize injuries
    # -------------------------
    inj["asof_date"] = inj["asof_date"].astype("string")
    inj["event_date"] = inj["asof_date"]  # treat updates as events at asof_date

    inj["desc"] = inj["desc"].astype("string")
    inj["comment"] = inj["comment"].astype("string")
    inj["status"] = inj["status"].astype("string")

    inj["issue_class"] = inj.apply(
        lambda r: classify_issue(r.get("desc"), r.get("comment"), "", ""),
        axis=1
    )
    inj["tokens"] = inj.apply(lambda r: "|".join(sorted(extract_tokens(r.get("desc"), r.get("comment")))), axis=1)

    # keep one row per player_id + event_date + injury_id (still event stream)
    inj_clean = inj.copy()

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out_updates = Path(f"data/silver/injury_updates_clean_{year}.csv")
    inj_clean.to_csv(out_updates, index=False)

    # -------------------------
    # Prepare availability DNP evidence
    # -------------------------
    played_ids = set(played["game_id"].unique())
    ava = ava[ava["game_id"].isin(played_ids)].copy()

    ava["scheduled"] = pd.to_datetime(ava["scheduled"], utc=True, errors="coerce")
    ava["game_date"] = ava["scheduled"].dt.date.astype("string")

    ava["not_playing_reason"] = ava["not_playing_reason"].fillna("").astype("string")
    ava["not_playing_description"] = ava["not_playing_description"].fillna("").astype("string")

    # Only consider rows with a stated reason
    dnp_all = ava[
        ava["not_playing_reason"].str.strip().ne("")
    ].copy()

    # classify DNP reason types
    dnp_all["issue_class"] = dnp_all.apply(
        lambda r: classify_issue("", "", r.get("not_playing_reason"), r.get("not_playing_description")),
        axis=1
    )
    dnp_all["tokens"] = dnp_all.apply(
        lambda r: "|".join(sorted(extract_tokens(r.get("not_playing_reason"), r.get("not_playing_description")))),
        axis=1
    )

    # Split predictable (medical) vs nonpredictable (rest/coach) for diagnostics
    nonpredictable = dnp_all[dnp_all["issue_class"] == "rest_or_coach"].copy()
    out_nonpred = Path(f"data/silver/nonpredictable_dnp_{year}.csv")
    nonpredictable.to_csv(out_nonpred, index=False)

    # Keep only medical DNP evidence (injury/illness); exclude rest/coach
    dnp_med = dnp_all[dnp_all["issue_class"].isin(["injury_medical", "illness"])].copy()

    # -------------------------
    # Match DNP evidence to injury updates (backward window only)
    # -------------------------
    inj_clean["event_dt"] = pd.to_datetime(inj_clean["event_date"], utc=True, errors="coerce")
    dnp_med["game_dt"] = pd.to_datetime(dnp_med["game_date"], utc=True, errors="coerce")

    # Sort for efficient group filtering
    inj_clean = inj_clean.sort_values(["player_id", "event_dt"])
    dnp_med = dnp_med.sort_values(["player_id", "game_dt"])

    # Pre-index injury events by player for fast lookup
    inj_by_player = {}
    for pid, grp in inj_clean.groupby("player_id", sort=False):
        inj_by_player[pid] = grp

    matched_rows = []
    unmatched_synth_events = []

    W = pd.Timedelta(days=lookback_days)

    for _, r in dnp_med.iterrows():
        pid = r["player_id"]
        gdt = r["game_dt"]
        if pd.isna(gdt) or pid not in inj_by_player:
            # no injury updates for player at all → unmatched
            matched = False
            best = None
        else:
            grp = inj_by_player[pid]
            lo = gdt - W
            hi = gdt

            window = grp[(grp["event_dt"] >= lo) & (grp["event_dt"] <= hi)]
            if len(window) == 0:
                matched = False
                best = None
            else:
                # Token-aware match if possible; otherwise date-only
                dnp_toks = set(str(r.get("tokens", "")).split("|")) if str(r.get("tokens", "")).strip() else set()
                if dnp_toks:
                    window = window.copy()
                    window["tok_overlap"] = window["tokens"].fillna("").apply(
                        lambda s: len(dnp_toks.intersection(set(str(s).split("|"))))
                    )
                    # prefer higher overlap, then most recent event_dt
                    window = window.sort_values(["tok_overlap", "event_dt"], ascending=[False, False])
                    best = window.iloc[0]
                    matched = (best["tok_overlap"] > 0) or True  # if we found any window row, treat as matched
                else:
                    # date-only: choose most recent update in window
                    window = window.sort_values("event_dt", ascending=False)
                    best = window.iloc[0]
                    matched = True

        out = {
            "season_year": year,
            "player_id": pid,
            "player_name": r.get("player_name"),
            "team_id": r.get("team_id"),
            "game_id": r.get("game_id"),
            "scheduled": r.get("scheduled"),
            "game_date": r.get("game_date"),
            "not_playing_reason": r.get("not_playing_reason"),
            "not_playing_description": r.get("not_playing_description"),
            "issue_class": r.get("issue_class"),
            "tokens": r.get("tokens"),
            "matched_to_update": bool(matched),
            "matched_update_event_date": (best["event_date"] if best is not None else pd.NA),
            "matched_injury_id": (best["injury_id"] if best is not None else pd.NA),
            "matched_desc": (best["desc"] if best is not None else pd.NA),
            "matched_status": (best["status"] if best is not None else pd.NA),
        }
        matched_rows.append(out)

        # If unmatched, add a synthetic injury-update-style event at the game date
        if not matched:
            synth = {
                "asof_date": r.get("game_date"),
                "event_date": r.get("game_date"),
                "team_id": r.get("team_id"),
                "team_name": pd.NA,
                "player_id": pid,
                "player_name": r.get("player_name"),
                "injury_id": make_synth_event_id(pid, r.get("game_id")),
                "desc": r.get("not_playing_description") if str(r.get("not_playing_description","")).strip() else "Injury/Illness",
                "status": "Out",  # conservative
                "comment": f"SYNTH_FROM_DNP: {r.get('not_playing_reason')} :: {r.get('not_playing_description')}",
                "start_date": r.get("game_date"),
                "update_date": r.get("game_date"),
                "bronze_file": pd.NA,
                "issue_class": r.get("issue_class"),
                "tokens": r.get("tokens"),
                "source": "dnp_injury_unmatched",
            }
            unmatched_synth_events.append(synth)

    dnp_evidence = pd.DataFrame(matched_rows)
    out_dnp = Path(f"data/silver/injury_dnp_evidence_{year}.csv")
    dnp_evidence.to_csv(out_dnp, index=False)

    # -------------------------
    # Build augmented injury events (updates + unmatched DNP synth events)
    # -------------------------
    inj_aug = inj_clean.copy()
    inj_aug["source"] = "daily_update"
    # ensure tokens/issue_class columns exist
    if "tokens" not in inj_aug.columns:
        inj_aug["tokens"] = ""
    if "issue_class" not in inj_aug.columns:
        inj_aug["issue_class"] = "unknown"

    if unmatched_synth_events:
        synth_df = pd.DataFrame(unmatched_synth_events)
        # align columns
        for col in inj_aug.columns:
            if col not in synth_df.columns:
                synth_df[col] = pd.NA
        for col in synth_df.columns:
            if col not in inj_aug.columns:
                inj_aug[col] = pd.NA

        inj_aug = pd.concat([inj_aug, synth_df[inj_aug.columns]], ignore_index=True)

    inj_aug = inj_aug.sort_values(["player_id", "event_date", "update_date"], kind="stable")

    out_aug = Path(f"data/silver/injury_events_augmented_{year}.csv")
    inj_aug.to_csv(out_aug, index=False)

    print(f"{year}: daily_updates={len(inj_clean)}")
    print(f"{year}: dnp_reason_rows_total={len(dnp_all)}  nonpredictable_restcoach={len(nonpredictable)}  dnp_medical={len(dnp_med)}")
    print(f"{year}: dnp_med_matched={int(dnp_evidence['matched_to_update'].sum())}  dnp_med_unmatched={int((~dnp_evidence['matched_to_update']).sum())}")
    print("wrote:", out_updates)
    print("wrote:", out_nonpred)
    print("wrote:", out_dnp)
    print("wrote:", out_aug)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lookback-days", type=int, default=2)
    args = ap.parse_args()
    main(args.year, lookback_days=args.lookback_days)