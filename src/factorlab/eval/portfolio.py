"""复合得分、分组多头与市值基准权重。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorlab.eval.backtest import assign_quantile_groups
from factorlab.schema import FACTORS


def add_composite_score(
    panel: pl.DataFrame,
    *,
    date_col: str = "date",
    min_coverage: float = 0.5,
) -> pl.DataFrame:
    """方向调整后的中性化因子等权 z-score。高分对应研究假设下的多头。"""

    ordered = panel.sort(date_col)
    n = ordered.height
    score = np.zeros(n)
    count = np.zeros(n)
    dates = ordered[date_col].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], n]
    specs = [spec for spec in FACTORS if spec.neu_name in ordered.columns]
    if not specs:
        raise KeyError("复合得分需要先生成 *_neu 列")
    columns = {spec.neu_name: ordered[spec.neu_name].to_numpy() for spec in specs}
    directions = {spec.neu_name: spec.direction for spec in specs}
    for start, end in zip(starts, ends, strict=True):
        for name, values in columns.items():
            raw = values[start:end].astype(float)
            mask = np.isfinite(raw)
            if int(mask.sum()) < 5:
                continue
            mu = float(raw[mask].mean())
            sd = float(raw[mask].std(ddof=0))
            if sd < 1e-12:
                continue
            z = np.full(end - start, np.nan)
            z[mask] = (raw[mask] - mu) / sd
            good = np.isfinite(z)
            block_score = score[start:end]
            block_count = count[start:end]
            block_score[good] = block_score[good] + directions[name] * z[good]
            block_count[good] = block_count[good] + 1.0
    need = max(1, int(np.ceil(len(specs) * min_coverage)))
    out = np.divide(score, count, out=np.full(n, np.nan), where=count >= need)
    return ordered.with_columns(pl.Series("composite", out))


def book_weights(
    panel: pl.DataFrame,
    *,
    n_groups: int = 5,
    date_col: str = "date",
    id_col: str = "ticker",
    score_col: str = "composite",
    ret_col: str = "fwd_ret",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """多头为复合得分最高一组的等权组合；基准为当期有收益股票的市值加权。"""

    ordered = panel.sort([date_col, id_col])
    dates = ordered[date_col].to_numpy()
    ids = ordered[id_col].to_list()
    score = ordered[score_col].to_numpy()
    ret = ordered[ret_col].to_numpy()
    cap = ordered["mktcap"].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    port_rows: list[dict] = []
    bench_rows: list[dict] = []
    long_group = n_groups
    for start, end in zip(starts, ends, strict=True):
        date = dates[start]
        tradable = np.isfinite(ret[start:end]) & np.isfinite(cap[start:end]) & (cap[start:end] > 0)
        if int(tradable.sum()) == 0:
            continue
        caps = cap[start:end][tradable]
        bench_names = [ids[start + i] for i in np.flatnonzero(tradable)]
        bench_w = caps / caps.sum()
        for name, w in zip(bench_names, bench_w, strict=True):
            bench_rows.append({date_col: _as_date(date), id_col: name, "weight": float(w)})
        scored = tradable & np.isfinite(score[start:end])
        groups = assign_quantile_groups(
            np.where(scored, score[start:end], np.nan),
            n_groups,
        )
        members = [ids[start + i] for i in np.flatnonzero(groups == long_group)]
        if len(members) < 2:
            continue
        weight = 1.0 / len(members)
        for name in members:
            port_rows.append({date_col: _as_date(date), id_col: name, "weight": weight})
    portfolio = _frame(port_rows, date_col, id_col)
    benchmark = _frame(bench_rows, date_col, id_col)
    return portfolio, benchmark


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _frame(rows: list[dict], date_col: str, id_col: str) -> pl.DataFrame:
    if not rows:
        return _empty(date_col, id_col)
    return pl.DataFrame(rows).with_columns(
        pl.col(date_col).cast(pl.Date),
        pl.col(id_col).cast(pl.Utf8),
        pl.col("weight").cast(pl.Float64),
    )


def _empty(date_col: str, id_col: str) -> pl.DataFrame:
    return pl.DataFrame(schema={date_col: pl.Date, id_col: pl.Utf8, "weight": pl.Float64})
