# Scratchwork / Exploration Notebooks

These notebooks document exploratory analysis that informed final design decisions but are **not** part of the main results pipeline. They are included for completeness and reproducibility of the research process.

| Notebook | Purpose | Outcome |
|----------|---------|---------|
| `ensemble_comparison.ipynb` | Bootstrap ensemble (row & time-block) vs single model | Ensemble did not meaningfully improve over single model; dropped from final pipeline |
| `trading_results.ipynb` | Initial trading backtest (Kalshi + Polymarket, convergence + settlement, confidence gate) | Superseded by `trading_results2.ipynb` which narrows to Kalshi-only, settlement-only |
| `kalshi_trading.ipynb` | Early Kalshi market exploration and trade logic prototyping | Superseded by final trading notebook |
| `poly_trading.ipynb` | Polymarket CLOB exploration and trade logic | Polymarket dropped: thin WNBA liquidity, wide spreads, unreliable fill assumptions |
| `NN_test.ipynb` | Neural network (MLP) alternative to XGBoost | Did not outperform XGBoost; higher variance across folds |
| `XGBpure.ipynb` | XGBoost without Elo base_margin (features only) | Confirmed Elo-as-base-margin architecture is superior to features-only |

**Note:** Earlier XGBoost tuning iterations (Stages 1–2) are not included here. Those searches explored the hyperparameter space and identified the promising region (depth 4–8, mcw 2–3, moderate regularization) that informed the final Stage 3 grid in `xgb_tuning/XGB_tuning3.ipynb`. The final grid search was conducted with corrected walk-forward CV methodology.
