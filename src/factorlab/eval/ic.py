"""Rank IC、IC 衰减与 IC_IR。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from scipy.stats import rankdata

from factorlab.eval.metrics import summarize_series


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"Spearman 需要两条等长一维序列，当前 {x.shape} 与 {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def rank_ic_series(
    panel: pl.DataFrame,
    factor: str,
    *,
    ret_col: str = "fwd_ret",
    date_col: str = "date",
    min_names: int = 15,
) -> pl.DataFrame:
    ordered = panel.sort(date_col)
    dates = ordered[date_col].to_numpy()
    f = ordered[factor].to_numpy()
    r = ordered[ret_col].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    rows = []
    for start, end in zip(starts, ends, strict=True):
        ic = spearman(f[start:end], r[start:end])
        n_valid = int(np.sum(np.isfinite(f[start:end]) & np.isfinite(r[start:end])))
        if n_valid < min_names:
            ic = float("nan")
        rows.append({date_col: _as_date(dates[start]), "ic": ic, "n": n_valid})
    frame = pl.DataFrame(rows)
    if frame.height and frame[date_col].dtype != pl.Date:
        frame = frame.with_columns(pl.col(date_col).cast(pl.Date))
    return frame


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def ic_decay(
    panel: pl.DataFrame,
    factor: str,
    *,
    horizons: int = 6,
    date_col: str = "date",
    min_names: int = 15,
    direction: int = 1,
) -> pl.DataFrame:
    """未来 h 个月累计收益上的策略 Rank IC。direction 把因子调成「高者做多」。"""

    rows = []
    signed = panel.with_columns((pl.col(factor) * direction).alias("_signed"))
    for h in range(1, horizons + 1):
        col = f"fwd_ret_{h}"
        if col not in panel.columns:
            raise KeyError(f"面板缺少 {col}，请先调用 add_horizon_returns")
        series = rank_ic_series(signed, "_signed", ret_col=col, date_col=date_col, min_names=min_names)
        stats = summarize_series(series["ic"].to_numpy(), nw_lags=max(h - 1, 1))
        rows.append({"horizon": h, **stats})
    return pl.DataFrame(rows)
