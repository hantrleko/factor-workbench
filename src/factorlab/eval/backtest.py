"""单因子分组回测：等权分组净值、多空收益、最大回撤与换手。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from factorlab.eval.metrics import one_way_turnover, performance_stats


def assign_quantile_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    """把有限值分成 1..n_groups，1 为最低。无效值记为 0。组间人数最多差 1。"""

    if n_groups < 2:
        raise ValueError("分组数至少为 2")
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"因子值必须是一维，当前 shape={x.shape}")
    out = np.zeros(x.shape[0], dtype=int)
    mask = np.isfinite(x)
    n = int(mask.sum())
    if n < n_groups:
        return out
    order = np.argsort(x[mask], kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    groups = np.minimum(ranks * n_groups // n, n_groups - 1) + 1
    out[np.flatnonzero(mask)] = groups
    return out


@dataclass
class QuantileBacktest:
    n_groups: int
    long_group: int
    short_group: int
    group_nav: pl.DataFrame
    long_short: pl.DataFrame
    summary: dict[str, float]
    group_summary: pl.DataFrame
    weights: pl.DataFrame


def quantile_backtest(
    panel: pl.DataFrame,
    factor: str,
    *,
    direction: int,
    n_groups: int = 5,
    ret_col: str = "fwd_ret",
    date_col: str = "date",
    id_col: str = "ticker",
    min_names: int | None = None,
) -> QuantileBacktest:
    if direction not in (1, -1):
        raise ValueError("direction 只能是 +1 或 -1")
    floor = n_groups * 3
    min_obs = min_names or max(20, floor)
    ordered = panel.sort([date_col, id_col])
    dates = ordered[date_col].to_numpy()
    ids = ordered[id_col].to_list()
    values = ordered[factor].to_numpy()
    rets = ordered[ret_col].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]

    long_group = n_groups if direction > 0 else 1
    short_group = 1 if direction > 0 else n_groups
    nav_rows: list[dict] = []
    ls_rows: list[dict] = []
    weight_rows: list[dict] = []
    prev_group: dict[int, dict[str, float]] = {}
    prev_ls: dict[str, float] | None = None
    wealth = {g: 1.0 for g in range(1, n_groups + 1)}
    ls_wealth = 1.0

    for start, end in zip(starts, ends, strict=True):
        date = _as_date(dates[start])
        val = values[start:end]
        ret = rets[start:end]
        names = ids[start:end]
        usable = np.isfinite(val) & np.isfinite(ret)
        if int(usable.sum()) < min_obs:
            continue
        groups = assign_quantile_groups(np.where(usable, val, np.nan), n_groups)
        group_ret: dict[int, float] = {}
        group_w: dict[int, dict[str, float]] = {}
        for g in range(1, n_groups + 1):
            take = groups == g
            members = [names[i] for i in np.flatnonzero(take)]
            if not members:
                continue
            group_ret[g] = float(np.mean(ret[take]))
            weight = 1.0 / len(members)
            group_w[g] = {name: weight for name in members}
            wealth[g] *= 1.0 + group_ret[g]
            turn = (
                one_way_turnover(prev_group[g], group_w[g]) if g in prev_group else float("nan")
            )
            nav_rows.append(
                {
                    date_col: date,
                    "group": g,
                    "ret": group_ret[g],
                    "nav": wealth[g],
                    "n": len(members),
                    "turnover": turn,
                }
            )
        prev_group.update(group_w)
        if long_group in group_ret and short_group in group_ret:
            long_w = group_w[long_group]
            short_w = group_w[short_group]
            ls_w = {name: w for name, w in long_w.items()}
            for name, w in short_w.items():
                ls_w[name] = ls_w.get(name, 0.0) - w
            r_ls = group_ret[long_group] - group_ret[short_group]
            ls_wealth *= 1.0 + r_ls
            turn_ls = one_way_turnover(prev_ls, ls_w) if prev_ls is not None else float("nan")
            prev_ls = ls_w
            for name, held in long_w.items():
                weight_rows.append(
                    {date_col: date, id_col: name, "side": "long", "group": long_group, "weight": held}
                )
            for name, held in short_w.items():
                weight_rows.append(
                    {date_col: date, id_col: name, "side": "short", "group": short_group, "weight": held}
                )
            ls_rows.append(
                {
                    date_col: date,
                    "ret": r_ls,
                    "nav": ls_wealth,
                    "turnover": turn_ls,
                    "long_ret": group_ret[long_group],
                    "short_ret": group_ret[short_group],
                }
            )

    group_nav = _dated(nav_rows, date_col, _empty_group_nav(date_col))
    long_short = _dated(ls_rows, date_col, _empty_ls(date_col))
    weights = _dated(weight_rows, date_col, _empty_weights(date_col, id_col))
    group_summary_rows = []
    for g in range(1, n_groups + 1):
        sub = group_nav.filter(pl.col("group") == g) if group_nav.height else group_nav
        stats = performance_stats(sub["ret"].to_numpy() if sub.height else np.array([]))
        turns = sub["turnover"].to_numpy() if sub.height else np.array([])
        stats["group"] = g
        stats["avg_turnover"] = _finite_mean(turns)
        group_summary_rows.append(stats)
    ls_stats = performance_stats(long_short["ret"].to_numpy() if long_short.height else np.array([]))
    ls_turns = long_short["turnover"].to_numpy() if long_short.height else np.array([])
    ls_stats["avg_turnover"] = _finite_mean(ls_turns)
    ls_stats["long_group"] = long_group
    ls_stats["short_group"] = short_group
    return QuantileBacktest(
        n_groups=n_groups,
        long_group=long_group,
        short_group=short_group,
        group_nav=group_nav,
        long_short=long_short,
        summary=ls_stats,
        group_summary=pl.DataFrame(group_summary_rows),
        weights=weights,
    )


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _dated(rows: list[dict], date_col: str, empty: pl.DataFrame) -> pl.DataFrame:
    if not rows:
        return empty
    frame = pl.DataFrame(rows)
    if frame[date_col].dtype != pl.Date:
        frame = frame.with_columns(
            pl.Series(date_col, [_as_date(v) for v in frame[date_col].to_list()], dtype=pl.Date)
        )
    return frame


def _empty_group_nav(date_col: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            date_col: pl.Date,
            "group": pl.Int64,
            "ret": pl.Float64,
            "nav": pl.Float64,
            "n": pl.Int64,
            "turnover": pl.Float64,
        }
    )


def _empty_weights(date_col: str, id_col: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            date_col: pl.Date,
            id_col: pl.Utf8,
            "side": pl.Utf8,
            "group": pl.Int64,
            "weight": pl.Float64,
        }
    )


def _empty_ls(date_col: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            date_col: pl.Date,
            "ret": pl.Float64,
            "nav": pl.Float64,
            "turnover": pl.Float64,
            "long_ret": pl.Float64,
            "short_ret": pl.Float64,
        }
    )
