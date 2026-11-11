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


def neutral_book_weights(
    panel: pl.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "ticker",
    score_col: str = "composite",
    ret_col: str = "fwd_ret",
    industry_col: str = "industry",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """行业权重和大小市值桶都贴着基准。每个桶里只留复合得分最高的一只。"""

    if industry_col not in panel.columns:
        raise KeyError("中性组合需要行业列")
    ordered = panel.sort([date_col, id_col])
    dates = ordered[date_col].to_numpy()
    ids = ordered[id_col].to_list()
    score = ordered[score_col].to_numpy().astype(float)
    ret = ordered[ret_col].to_numpy().astype(float)
    cap = ordered["mktcap"].to_numpy().astype(float)
    industries = [_industry_label(item) for item in ordered[industry_col].to_list()]
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    port_rows: list[dict] = []
    bench_rows: list[dict] = []
    for start, end in zip(starts, ends, strict=True):
        date = dates[start]
        tradable = np.isfinite(ret[start:end]) & np.isfinite(cap[start:end]) & (cap[start:end] > 0)
        if int(tradable.sum()) < 2:
            continue
        index = np.flatnonzero(tradable)
        caps = cap[start:end][tradable]
        names = [ids[start + i] for i in index]
        scores = score[start:end][tradable]
        labels = [industries[start + i] for i in index]
        total = float(caps.sum())
        bench_w = caps / total
        for name, weight in zip(names, bench_w, strict=True):
            bench_rows.append({date_col: _as_date(date), id_col: name, "weight": float(weight)})
        held: dict[str, float] = {}
        for label in dict.fromkeys(labels):
            members = np.flatnonzero(np.array(labels) == label)
            _assign_bucket(held, names, caps, scores, members, total)
        for name, weight in held.items():
            if weight > 0:
                port_rows.append({date_col: _as_date(date), id_col: name, "weight": float(weight)})
    return _frame(port_rows, date_col, id_col), _frame(bench_rows, date_col, id_col)


def _industry_label(value: object) -> str:
    if value is None:
        return "未分类"
    text = str(value).strip()
    return text or "未分类"


def _assign_bucket(
    held: dict[str, float],
    names: list[str],
    caps: np.ndarray,
    scores: np.ndarray,
    members: np.ndarray,
    total: float,
) -> None:
    member_caps = caps[members]
    if members.size == 1 or float(np.nanmax(member_caps) - np.nanmin(member_caps)) <= 1e-12:
        _place(held, names, caps, scores, members, float(member_caps.sum()) / total)
        return
    median = float(np.median(member_caps))
    small = members[member_caps <= median]
    large = members[member_caps > median]
    for bucket in (small, large):
        if bucket.size == 0:
            continue
        _place(held, names, caps, scores, bucket, float(caps[bucket].sum()) / total)


def _place(
    held: dict[str, float],
    names: list[str],
    caps: np.ndarray,
    scores: np.ndarray,
    bucket: np.ndarray,
    share: float,
) -> None:
    finite = bucket[np.isfinite(scores[bucket])]
    pick_from = finite if finite.size else bucket
    if finite.size:
        pick = int(pick_from[int(np.argmax(scores[pick_from]))])
    else:
        pick = int(pick_from[int(np.argmax(caps[pick_from]))])
    held[names[pick]] = held.get(names[pick], 0.0) + share


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
