"""
Production artifact validation.

This is intentionally small and hard-edged. It checks the contract that matters
for live trading:
  - silver daily player state is separate from game-wise player feature store
  - full game-wise player feature store can track all listed players
  - gold model-input tables are exactly metadata + 160 model features
  - no stale p8-p12/debug player columns leak into model input
  - live-year player priors are not accidentally cold-zeroed
  - future daily-injury placeholders are not present in canonical bronze
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.util.model_schema import FEAT_COLS, GOLD_MODEL_INPUT_COLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--today", default=date.today().isoformat())
    ap.add_argument("--start-year", type=int, default=2015)
    args = ap.parse_args()

    today = date.fromisoformat(args.today)
    errors: list[str] = []
    gold: pd.DataFrame | None = None

    gold_path = REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.year}_REGPST.csv"
    combined_path = REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.start_year}_{args.year}_REGPST.csv"
    state_path = REPO_ROOT / "data" / "silver" / f"player_state_history_{args.year}.csv"
    player_path = REPO_ROOT / "data" / "silver_plus" / f"game_team_player_{args.year}_REGPST.csv"
    legacy_player_path = REPO_ROOT / "data" / "silver" / f"game_team_player_{args.year}_REGPST.csv"
    legacy_state_paths = list((REPO_ROOT / "data" / "silver_plus").glob("player_state_history_*.csv"))
    bronze_dir = REPO_ROOT / "data" / "bronze"
    injury_events_path = REPO_ROOT / "data" / "silver" / f"injury_events_{args.year}.csv"

    marker_files = []
    for layer in ("bronze", "silver", "silver_plus", "gold"):
        layer_dir = REPO_ROOT / "data" / layer
        if layer_dir.exists():
            marker_files.extend(p.relative_to(REPO_ROOT).as_posix() for p in layer_dir.rglob("*.marker.txt"))
    if marker_files:
        errors.append(f"debug marker files found in canonical data layers: {marker_files[:10]}")

    if len(FEAT_COLS) != 160:
        errors.append(f"FEAT_COLS expected 160, got {len(FEAT_COLS)}")

    yearly_gold: dict[int, pd.DataFrame] = {}
    for y in range(args.start_year, args.year + 1):
        yearly_path = REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{y}_REGPST.csv"
        yearly, yearly_errors = validate_gold_table(yearly_path)
        errors.extend(yearly_errors)
        if yearly is not None:
            yearly_gold[y] = yearly

    holdout_baseline_path = REPO_ROOT / "data" / "gold" / "game_xgboost_input_2015_2024_REGPST.csv"
    if holdout_baseline_path.exists():
        _, holdout_errors = validate_gold_table(holdout_baseline_path)
        errors.extend(holdout_errors)

    gold = yearly_gold.get(args.year)
    if gold is not None:
        for col in ("home_p1_m_ewma_pre", "away_p1_m_ewma_pre"):
            if col in gold.columns and (pd.to_numeric(gold[col], errors="coerce") == 0).all():
                errors.append(f"{col} is all zero in {gold_path.name}")

    if legacy_player_path.exists():
        errors.append(f"game-wise player feature table must live in silver_plus, not silver: {legacy_player_path}")
    if legacy_state_paths:
        errors.append(
            "daily player_state_history belongs in silver, not silver_plus: "
            + ", ".join(p.name for p in legacy_state_paths[:5])
        )

    if not state_path.exists():
        errors.append(f"missing daily player state: {state_path}")
    else:
        state = pd.read_csv(state_path, nrows=5)
        needed_state = {"player_id", "asof_ts", "m_ewma", "q", "strength"}
        missing_state = sorted(needed_state - set(state.columns))
        if missing_state:
            errors.append(f"{state_path.name} missing columns: {missing_state}")

    if injury_events_path.exists():
        injuries = pd.read_csv(injury_events_path)
        if len(injuries):
            injury_key = ["asof_date", "team_id", "player_id", "injury_id", "status", "start_date", "update_date"]
            missing_key = [col for col in injury_key if col not in injuries.columns]
            if missing_key:
                errors.append(f"{injury_events_path.name} missing injury event key columns: {missing_key}")
            elif injuries.duplicated(subset=injury_key).any():
                errors.append(f"{injury_events_path.name} has duplicate injury event keys")

    if not player_path.exists():
        errors.append(f"missing full player store: {player_path}")
    else:
        players = pd.read_csv(player_path)
        needed = {"game_id", "team_id", "player_id", "m_ewma_pre", "q_pre", "strength_pre"}
        missing = sorted(needed - set(players.columns))
        if missing:
            errors.append(f"{player_path.name} missing columns: {missing}")
        if len(players) > 0:
            if pd.to_numeric(players["q_pre"], errors="coerce").fillna(0).max() <= 0:
                errors.append(f"{player_path.name} q_pre is all non-positive")
            max_players_team_game = players.groupby(["game_id", "team_id"])["player_id"].nunique().max()
            if max_players_team_game < 8:
                errors.append(
                    f"{player_path.name} does not look like a full player store; max players/team/game={max_players_team_game}"
                )

    future_files = []
    if bronze_dir.exists():
        for path in bronze_dir.glob(f"daily_injuries__{args.year}-*__*.json"):
            match = re.search(r"daily_injuries__(\d{4}-\d{2}-\d{2})__", path.name)
            if match and date.fromisoformat(match.group(1)) > today:
                future_files.append(path.name)
    if future_files:
        errors.append(f"future daily-injury files in canonical bronze: {future_files[:10]}")

    errors.extend(validate_bronze_run_manifests())
    errors.extend(validate_franchise_continuity(args.year, gold))

    if args.year >= 2026:
        if not combined_path.exists():
            errors.append(f"missing production combined training file: {combined_path}")
        else:
            combined, combined_errors = validate_gold_table(combined_path)
            errors.extend(combined_errors)

            expected_rows = 0
            missing_from_combined: list[str] = []
            combined_ids = set(combined["game_id"].astype(str)) if combined is not None and "game_id" in combined.columns else set()
            for y in range(args.start_year, args.year + 1):
                yearly = yearly_gold.get(y)
                if yearly is None:
                    continue
                expected_rows += len(yearly)
                missing_from_combined.extend(
                    sorted(set(yearly["game_id"].astype(str)) - combined_ids)
                )
            if combined is not None and len(combined) != expected_rows:
                errors.append(
                    f"{combined_path.name} row count stale: expected {expected_rows} from yearly gold, got {len(combined)}"
                )
            if missing_from_combined:
                errors.append(
                    f"{combined_path.name} missing yearly gold game_ids: {missing_from_combined[:10]}"
                )

    if errors:
        print("[validate] FAILED")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    print("[validate] OK")
    print(f"  year={args.year}")
    print(f"  feature_cols={len(FEAT_COLS)}")
    print(f"  gold_cols={len(GOLD_MODEL_INPUT_COLS)}")
    print(f"  gold={gold_path}")
    if args.year >= 2026:
        print(f"  combined_training={combined_path}")
    print(f"  daily_player_state={state_path}")
    print(f"  game_player_store={player_path}")


def validate_gold_table(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return None, [f"missing gold file: {path}"]

    df = pd.read_csv(path)
    cols = list(df.columns)
    if cols != GOLD_MODEL_INPUT_COLS:
        missing = [c for c in GOLD_MODEL_INPUT_COLS if c not in cols]
        extra = [c for c in cols if c not in GOLD_MODEL_INPUT_COLS]
        errors.append(
            f"{path.name} schema mismatch: expected exact ordered {len(GOLD_MODEL_INPUT_COLS)} cols, "
            f"got {len(cols)}; missing={missing[:10]} extra={extra[:10]}"
        )

    stale_cols = [
        c for c in cols
        if re.search(r"_(p(?:8|9|10|11|12))_", c)
        or c.endswith("_player_id")
        or c.endswith("_player_name")
        or c.endswith("_strength_pre")
        or c.endswith("_origin_city_pre")
        or c.endswith("_current_city_pre")
    ]
    if stale_cols:
        errors.append(f"{path.name} stale/debug columns leaked into gold: {stale_cols[:10]}")

    if "game_id" in df.columns and df["game_id"].duplicated().any():
        errors.append(f"{path.name} has duplicate game_id rows")
    errors.extend(validate_binary_label(df, path.name))

    meta_required = [
        "game_id", "game_ts", "game_date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_elo_pre", "away_elo_pre",
        "p_elo", "base_margin",
    ]
    missing_meta = [c for c in meta_required if c not in df.columns]
    if missing_meta:
        errors.append(f"{path.name} missing gold metadata columns: {missing_meta}")
    else:
        null_meta = df[meta_required].isna().sum()
        bad_meta = null_meta[null_meta > 0].to_dict()
        if bad_meta:
            errors.append(f"{path.name} metadata has nulls: {bad_meta}")

    missing_features = [c for c in FEAT_COLS if c not in df.columns]
    if missing_features:
        errors.append(f"{path.name} missing model feature columns: {missing_features[:10]}")
    else:
        null_features = df[FEAT_COLS].isna().sum()
        bad_features = null_features[null_features > 0].to_dict()
        if bad_features:
            errors.append(f"{path.name} model features have nulls: {bad_features}")

    if "p_elo" in df.columns:
        p_elo = pd.to_numeric(df["p_elo"], errors="coerce")
        if p_elo.isna().any() or not ((p_elo > 0) & (p_elo < 1)).all():
            errors.append(f"{path.name} p_elo must be non-null and strictly between 0 and 1")
    if "base_margin" in df.columns:
        base_margin = pd.to_numeric(df["base_margin"], errors="coerce")
        if base_margin.isna().any() or not np.isfinite(base_margin).all():
            errors.append(f"{path.name} base_margin must be finite for every row")

    return df, errors


def validate_binary_label(df: pd.DataFrame, name: str) -> list[str]:
    if "home_win" not in df.columns:
        return [f"{name} missing home_win label column"]
    labels = pd.to_numeric(df["home_win"], errors="coerce")
    if labels.isna().any():
        return [f"{name} has missing/non-numeric home_win labels"]
    bad = sorted(set(labels.astype(float)) - {0.0, 1.0})
    if bad:
        return [f"{name} home_win labels must be binary 0/1; saw {bad[:10]}"]
    return []


def validate_bronze_run_manifests() -> list[str]:
    run_root = REPO_ROOT / "data" / "bronze_runs"
    if not run_root.exists():
        return []

    errors: list[str] = []
    required_if_accepted = ("endpoint", "request_url", "canonical_path", "sha256")
    for manifest_path in sorted(run_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{manifest_path.relative_to(REPO_ROOT)} unreadable manifest: {exc}")
            continue

        if manifest.get("accepted") is True:
            missing = [key for key in required_if_accepted if not manifest.get(key)]
            if missing:
                errors.append(
                    f"{manifest_path.relative_to(REPO_ROOT)} accepted bronze manifest missing fields: {missing}"
                )

    return errors


def validate_franchise_continuity(year: int, gold: pd.DataFrame | None) -> list[str]:
    if gold is None or gold.empty:
        return []

    errors: list[str] = []
    map_path = REPO_ROOT / "data" / "config" / "franchise_map.csv"
    if not map_path.exists():
        return [f"missing franchise map: {map_path}"]

    fmap = pd.read_csv(map_path, dtype={"team_id": str, "franchise_id": str})
    fmap["start_year"] = pd.to_numeric(fmap["start_year"], errors="coerce").astype("Int64")
    fmap["end_year"] = pd.to_numeric(fmap["end_year"], errors="coerce").astype("Int64")

    current_team_ids = _team_ids_from_gold(gold)
    for team_id in sorted(current_team_ids):
        rows = fmap[fmap["team_id"].astype(str) == team_id]
        if rows.empty:
            continue
        active = rows[(rows["start_year"] <= year) & (rows["end_year"] >= year)]
        if active.empty:
            errors.append(
                f"franchise_map has current team_id={team_id} but no row covers year={year}; "
                "this would fall back to raw team_id and lose franchise priors"
            )

    prev_gold_path = REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{year-1}_REGPST.csv"
    style_path = REPO_ROOT / "data" / "silver_plus" / f"game_franchise_style_profile_{year}_REGPST.csv"
    if prev_gold_path.exists() and style_path.exists():
        prev_gold = pd.read_csv(prev_gold_path, usecols=["home_team_id", "away_team_id"])
        prior_team_ids = _team_ids_from_gold(prev_gold)
        style = pd.read_csv(style_path)
        needed = {"team_id", "franchise_id", "games_played_before_game", "prior_source"}
        if needed.issubset(style.columns):
            games_before = pd.to_numeric(style["games_played_before_game"], errors="coerce").fillna(-1)
            bad = style[
                style["team_id"].astype(str).isin(prior_team_ids)
                & (games_before == 0)
                & (style["prior_source"].astype(str) == "league_init")
            ]
            if len(bad):
                sample = bad[["team_id", "franchise_id", "game_id", "prior_source"]].head(5).to_dict("records")
                errors.append(
                    f"{style_path.name} has existing prior-season teams initialized from league_init: {sample}"
                )

    return errors


def _team_ids_from_gold(df: pd.DataFrame) -> set[str]:
    team_ids: set[str] = set()
    for col in ("home_team_id", "away_team_id"):
        if col in df.columns:
            team_ids.update(str(v) for v in df[col].dropna().astype(str))
    return team_ids


if __name__ == "__main__":
    main()
