"""Opt-in segmentation by QuantRiskLab rate and VIX states.

Nothing in the running site imports this module. A caller chooses the model
and an as_of. The state is the filtered probability from the latest estimate
whose cutoff is on or before that date. Gaps longer than 14 days stay empty.
Smoothed probabilities are not used.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(r"F:\projects\QuantRiskLab\data\regime_states.csv")


def regime_path() -> Path:
    override = os.environ.get("QUANTRISKLAB_REGIMES", "").strip()
    return Path(override) if override else DEFAULT_CSV


def load_states(path: Path | None = None) -> pd.DataFrame:
    source = path or regime_path()
    if not source.exists():
        raise FileNotFoundError(f"还没有离线状态文件：{source}")
    return pd.read_csv(source)


def align_states(panel: pd.DataFrame, dates, model_id: str, as_of: date) -> pd.DataFrame:
    dates_frame = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    chosen = panel[(panel["model_id"] == model_id) & (pd.to_datetime(panel["estimation_end"]) <= pd.Timestamp(as_of))]
    if chosen.empty:
        dates_frame["filtered_high"] = np.nan
        dates_frame["high_state"] = np.nan
        dates_frame["estimation_end"] = pd.NaT
        return dates_frame
    end = pd.to_datetime(chosen["estimation_end"]).max()
    sample = chosen[pd.to_datetime(chosen["estimation_end"]) == end].copy()
    sample["observation_date"] = pd.to_datetime(sample["observation_date"])
    sample = sample[sample["observation_date"] <= pd.Timestamp(as_of)].sort_values("observation_date")
    merged = pd.merge_asof(
        dates_frame,
        sample[["observation_date", "filtered_high"]],
        left_on="date",
        right_on="observation_date",
        direction="backward",
        tolerance=pd.Timedelta(days=14),
    )
    merged["estimation_end"] = end
    merged.loc[merged["filtered_high"].isna(), "estimation_end"] = pd.NaT
    merged["high_state"] = np.where(merged["filtered_high"].isna(), np.nan, (merged["filtered_high"] >= 0.5).astype(float))
    return merged