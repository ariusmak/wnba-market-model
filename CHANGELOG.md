# Organized Repository — Inclusion/Exclusion Decisions

This document logs all decisions made when building the `organized/` folder from the development workspace.

---

## Included — Main Pipeline

### Pipeline Scripts (`pipelines/`)
All 34+ numbered pipeline scripts were copied and organized into stages:
- `01_ingestion/` — Sportradar API backfill scripts (schedules, game summaries, injuries)
- `02_parsing/` — Bronze JSON → silver CSV parsing and normalization
- `03_features/` — Feature engineering (Elo, player state, recent form, style, schedule context)
- `04_gold/` — Final XGBoost input assembly
- `05_modeling/` — XGBoost CV, ensemble creation, calibration, Elo grid search
- `06_markets/` — Kalshi and Polymarket data ingestion pipelines

**No code changes** to pipeline scripts (they reference `../data/` paths which remain valid from `organized/pipelines/`).

### Source Code (`src/srwnba/`, `utils/`)
Copied verbatim. Includes:
- `src/srwnba/` — API client, config, Elo engine, franchise mapping, caching
- `utils/` — Kalshi and Polymarket API clients, ingestion, normalization

### Result Notebooks (`notebooks/analysis/`, `notebooks/trading/`)
| Notebook | Location | Path Changes |
|----------|----------|--------------|
| `prelim.ipynb` | `notebooks/analysis/` | `REPO = Path('.').resolve().parent` → `.parent.parent` (2 levels up) |
| `forecasting_results.ipynb` | `notebooks/analysis/` | `Path("..")` → `Path("../..)")` |
| `model_comparison.py` | `notebooks/analysis/` | None needed |
| `final_comparisons.ipynb` | `notebooks/analysis/` | `Path('..')` → `Path('../..')` |
| `logit.ipynb` | `notebooks/analysis/` | `Path('..')` → `Path('../..')` |
| `trading_results2.ipynb` | `notebooks/trading/` | `Path('..')` → `Path('../..')` |

### Tuning Notebooks (`notebooks/elo_tuning/`, `notebooks/xgb_tuning/`)
| Notebook | Location | Path Changes |
|----------|----------|--------------|
| `elo_tuning.ipynb` | `notebooks/elo_tuning/` | Already present from prior copy |
| `elo_tuning - Copy.ipynb` | `notebooks/elo_tuning/` | Already present |
| `XGB_tuning3.ipynb` | `notebooks/xgb_tuning/` | `Path('..')` → `Path('../..')` |
| `XGB_tuning2.ipynb` | `notebooks/xgb_tuning/` | `Path('..')` → `Path('../..')` |
| `complexity_curve.ipynb` | `notebooks/xgb_tuning/` | Newly added |

### Documentation
| File | Notes |
|------|-------|
| `README.md` | Rewritten with actual results, architecture, exploration summary, pipeline docs |
| `CLAUDE.md` | Kept as-is (detailed methodology specification) |
| `docs/tuning_methodology.md` | New: full tuning strategy with Stage 3 top-10 config table |
| `data/xgb_stage3_top10.csv` | New: extracted top 10 XGB configs from 4,864 + 1,296 evaluated |
| `data/spec_sheets/` | Kept: table specifications for all intermediate data tables |

---

## Included — Scratchwork (`notebooks/scratchwork/`)

These notebooks document exploration that informed design decisions but are not part of the final pipeline:

| Notebook | Why included |
|----------|-------------|
| `ensemble_comparison.ipynb` | Documents ensemble exploration (row + time-block bootstrap) and why it was dropped |
| `trading_results.ipynb` | Original trading notebook with Kalshi + Polymarket, convergence, confidence gate exploration |
| `kalshi_trading.ipynb` | Early Kalshi market exploration |
| `poly_trading.ipynb` | Polymarket exploration — documents why it was dropped (thin liquidity) |
| `NN_test.ipynb` | Neural network alternative — documents why XGBoost was preferred |
| `XGBpure.ipynb` | XGBoost without Elo base_margin — confirms architecture decision |
| `logit.ipynb` | Logistic regression benchmark (also in analysis/ for the main pipeline) |
---

## Excluded

| Item | Reason |
|------|--------|
| `notebooks/XGB_tuning.ipynb` | Early XGBoost tuning (Stage 1) — used a flawed CV methodology. Findings (promising hyperparameter region) carried forward into the corrected Stage 3 grid search (`XGB_tuning3.ipynb`) |
| `notebooks/XGB_tuning2.ipynb` | Intermediate XGBoost tuning (Stage 2) — same CV methodology issue as Stage 1. Superseded by Stage 3 |
| `notebooks/trading_comparisons.ipynb` | Superseded by `trading_results2.ipynb` |
| `notebooks/z_scratch/` | Raw scratch/debug notebooks with no research value |
| `notebooks/_*.py` temp scripts | Build/update scripts used during notebook construction (e.g., `_add_section9.py`, `_build_nb2.py`, `_run_trading2.py`) |
| `notebooks/calibration_run.py` | Already in `pipelines/05_modeling/` |
| `notebooks/gridsearch_elo.py` | Already in `pipelines/05_modeling/` |
| `notebooks/walkforward_gridsearch_elo.py` | Already in `pipelines/05_modeling/` |
| `notebooks/model_comparison.py` (original) | Copied to `notebooks/analysis/` |
| `notebooks/kalshi/` subdirectory | Market pipeline scripts already in `pipelines/06_markets/kalshi/` |
| `notebooks/polymarket/` subdirectory | Market pipeline scripts already in `pipelines/06_markets/polymarket/` |
| `NBA/` directory | Cross-sport validation spinoff, separate project scope |
| `data/` (raw data files) | Data directories exist as empty placeholders with README. Raw data too large for version control and requires API keys to reproduce. |
| `notebooks/2020_omit_test.ipynb` | Deleted in working tree (2020 season omission test, results incorporated into main analysis) |
| `notebooks/calibration.ipynb` | Deleted in working tree |
| `notebooks/lean_features_test.ipynb` | Deleted in working tree |
| `notebooks/trading_comparisons.ipynb` | Deleted in working tree (superseded by trading_results2) |
| `organized/_build_xgb_top10.py` | Temp build script, output saved as `data/xgb_stage3_top10.csv` |
| `organized/_fix_paths.py` | Temp path-fixing script |
| `organized/_find_chosen_config.py` | Temp analysis script |

---

## Path Changes Summary

All notebooks in `organized/notebooks/{subdir}/` are 2 directory levels below `organized/`. The original notebooks were 1 level below the project root (in `notebooks/`). Path references were updated:

- `Path("..").resolve()` → `Path("../..").resolve()` (for `PROJECT_ROOT`)
- `Path('.').resolve().parent` → `Path('.').resolve().parent.parent` (for `REPO`)

No changes were made to pipeline scripts, source code, or utility modules.
