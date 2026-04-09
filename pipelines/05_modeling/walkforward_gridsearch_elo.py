import argparse
from pathlib import Path

import pandas as pd

from gridsearch_elo import parse_int_list, parse_float_list, run_gridsearch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-train-years", required=True, help="Comma-separated years, e.g. 2015,2016,2017,2018,2019")
    ap.add_argument("--all-years", required=True, help="Comma-separated years in order, e.g. 2015,2016,...,2025")

    ap.add_argument("--H", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)

    ap.add_argument("--outdir", default="data/silver/walkforward_elo")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    init_train = parse_int_list(args.initial_train_years)
    all_years = parse_int_list(args.all_years)

    H_list = parse_float_list(args.H)
    K_list = parse_float_list(args.K)
    a_list = parse_float_list(args.a)
    b_list = parse_float_list(args.b)

    # sanity: ensure init_train is a prefix subset of all_years
    all_set = set(all_years)
    missing = [y for y in init_train if y not in all_set]
    if missing:
        raise ValueError(f"initial_train_years contains years not in all_years: {missing}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build walk-forward steps:
    # train grows by 1 year each step; test is next year after last train year
    steps = []
    train = sorted(init_train)

    while True:
        last_train = max(train)
        # find next year in the provided all_years sequence
        idx = all_years.index(last_train)
        if idx + 1 >= len(all_years):
            break
        test_year = all_years[idx + 1]
        steps.append((train.copy(), [test_year]))
        train.append(test_year)

    summary_rows = []

    for train_years, test_years in steps:
        print(f"\n=== WALKFORWARD: train={train_years} test={test_years} ===")
        res = run_gridsearch(
            train_years=train_years,
            test_years=test_years,
            H_list=H_list,
            K_list=K_list,
            a_list=a_list,
            b_list=b_list,
            verbose=args.verbose,
        )

        out_step = outdir / f"grid_train{train_years[0]}_{train_years[-1]}_test{test_years[0]}.csv"
        res.to_csv(out_step, index=False)
        best = res.iloc[0].to_dict()

        summary_rows.append({
            "train_start": train_years[0],
            "train_end": train_years[-1],
            "test_year": test_years[0],
            "best_H": best["H"],
            "best_K": best["K"],
            "best_a": best["a"],
            "best_b": best["b"],
            "train_logloss": best["train_logloss"],
            "test_logloss": best["test_logloss"],
            "train_brier": best["train_brier"],
            "test_brier": best["test_brier"],
            "step_file": out_step.name,
        })

        print("best:", {k: summary_rows[-1][k] for k in ["best_H","best_K","best_a","best_b","train_logloss","test_logloss"]})
        print("wrote:", out_step.resolve())

    summary = pd.DataFrame(summary_rows)
    out_summary = outdir / "walkforward_summary.csv"
    summary.to_csv(out_summary, index=False)
    print("\nWROTE SUMMARY:", out_summary.resolve())


if __name__ == "__main__":
    main()