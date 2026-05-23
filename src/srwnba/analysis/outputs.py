"""Standardized output paths + save helpers for paper figures and tables."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Project root resolves to .../organized/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str, *, dpi: int = 200) -> Path:
    """Save a matplotlib figure to outputs/<name>.png at consistent DPI."""
    path = OUTPUTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved: {path.relative_to(PROJECT_ROOT)}")
    return path


def save_table(df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
    path = OUTPUTS_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  saved: {path.relative_to(PROJECT_ROOT)}")
    return path
