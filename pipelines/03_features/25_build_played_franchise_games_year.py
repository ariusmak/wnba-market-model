import argparse
from pathlib import Path
import pandas as pd


ALLSTAR_REGEX = r"all[\s-]?star"


def main(year: int):
    in_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}")

    df = pd.read_csv(in_path)

    if "title" not in df.columns:
        raise ValueError(
            f"{in_path} is missing required column 'title'. "
            "This script filters All-Star games by title."
        )

    title = df["title"].fillna("").astype(str)
    is_allstar = title.str.contains(ALLSTAR_REGEX, case=False, regex=True)

    removed = df[is_allstar].copy()
    kept = df[~is_allstar].copy()

    out_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    print(f"\n{year}:")
    print(f"  input games: {len(df)}")
    print(f"  removed all-star games: {len(removed)}")
    print(f"  output games: {len(kept)}")
    print(f"  wrote: {out_path}")

    if len(removed) > 0:
        # Print removed games for sanity
        cols_to_show = [c for c in [
            "game_id", "scheduled", "season_type", "title",
            "home_id", "away_id", "home_name", "away_name",
            "home_market", "away_market"
        ] if c in removed.columns]

        print("  removed games:")
        print(removed[cols_to_show].to_string(index=False))

        # Expectation: typically 1 all-star game per season (sometimes 0)
        if len(removed) > 2:
            print("  WARNING: More than 2 games matched the All-Star filter. Double-check titles.")

    # Extra sanity: no all-star titles remain
    if kept["title"].fillna("").astype(str).str.contains(ALLSTAR_REGEX, case=False, regex=True).any():
        raise RuntimeError("Sanity check failed: All-Star title still present in franchise manifest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)