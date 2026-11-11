"""从缓存面板现算分组、中性化开关和多头超额。不改官方结论。"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from factorlab.eval.backtest import quantile_backtest
from factorlab.eval.ic import rank_ic_series
from factorlab.eval.metrics import performance_stats, summarize_series
from factorlab.research.costs import impacts_for_dates, net_returns, participation_by_date
from factorlab.schema import FACTORS
from factorlab.web.explain import _scale_bars


def view_settings(args) -> tuple[int, bool]:
    groups = 10 if str(args.get("groups", "5")) == "10" else 5
    neu = str(args.get("neu", "1")) != "0"
    return groups, neu


def without_borrow(preset: dict) -> dict:
    copied = dict(preset)
    copied["borrow_monthly"] = 0.0
    return copied


def slice_panel(panel: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    sample = panel
    if start:
        sample = sample.filter(pl.col("date") >= dt.date.fromisoformat(start))
    if end:
        sample = sample.filter(pl.col("date") <= dt.date.fromisoformat(end))
    return sample


def run_book(panel, spec, *, n_groups: int, neu: bool, min_names: int, respect_tradable: bool):
    column = spec.neu_name if neu and spec.neu_name in panel.columns else spec.name
    if column not in panel.columns:
        return None
    respect = bool(respect_tradable and "tradable" in panel.columns)
    backtest = quantile_backtest(
        panel,
        column,
        direction=spec.direction,
        n_groups=n_groups,
        min_names=min_names,
        respect_tradable=respect,
    )
    return backtest, column


def _mean_finite(values: list) -> float | None:
    kept = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            kept.append(number)
    if not kept:
        return None
    return sum(kept) / len(kept)


def benchmarks(panel: pl.DataFrame) -> dict[str, tuple[float, float]]:
    if "fwd_ret" not in panel.columns or "mktcap" not in panel.columns:
        return {}
    valid = panel.filter(
        pl.col("fwd_ret").is_not_null()
        & pl.col("fwd_ret").is_finite()
        & pl.col("mktcap").is_not_null()
        & pl.col("mktcap").is_finite()
        & (pl.col("mktcap") > 0)
    )
    if valid.height == 0:
        return {}
    frame = valid.group_by("date").agg(
        pl.col("fwd_ret").mean().alias("ew"),
        (pl.col("fwd_ret") * pl.col("mktcap")).sum().alias("cw_num"),
        pl.col("mktcap").sum().alias("cw_den"),
    )
    out = {}
    for row in frame.to_dicts():
        day = str(row["date"])[:10]
        den = float(row["cw_den"])
        cw = float(row["cw_num"]) / den if den else float("nan")
        out[day] = (float(row["ew"]), cw)
    return out


def long_leg(backtest, panel: pl.DataFrame, preset: dict) -> dict:
    frame = backtest.long_short
    if frame.height == 0:
        return {}
    dates = [str(item)[:10] for item in frame["date"].to_list()]
    long_ret = [float(item) for item in frame["long_ret"].to_list()]
    turn_rows = backtest.group_nav.filter(pl.col("group") == backtest.long_group)
    turns_map = {
        str(row["date"])[:10]: row["turnover"]
        for row in turn_rows.to_dicts()
    }
    bench = benchmarks(panel)
    kept_dates = []
    kept_long = []
    kept_turns = []
    ew = []
    cw = []
    for day, ret in zip(dates, long_ret, strict=True):
        pair = bench.get(day)
        if pair is None or not math.isfinite(pair[0]) or not math.isfinite(pair[1]):
            continue
        kept_dates.append(day)
        kept_long.append(ret)
        kept_turns.append(turns_map.get(day))
        ew.append(pair[0])
        cw.append(pair[1])
    if not kept_long:
        return {}
    cost = without_borrow(preset)
    long_net, _impacts = apply_preset(
        kept_long,
        kept_turns,
        kept_dates,
        cost,
        backtest.weights,
        panel,
        side="long",
    )
    excess = [left - right for left, right in zip(kept_long, ew, strict=True)]
    excess_cw = [left - right for left, right in zip(kept_long, cw, strict=True)]
    excess_net = [left - right for left, right in zip(long_net, ew, strict=True)]
    return {
        "dates": kept_dates,
        "long": kept_long,
        "ew": ew,
        "cw": cw,
        "excess": excess,
        "excess_cw": excess_cw,
        "long_net": long_net,
        "excess_net": excess_net,
        "long_ann": float(performance_stats(kept_long)["ann_return"]),
        "ew_ann": float(performance_stats(ew)["ann_return"]),
        "cw_ann": float(performance_stats(cw)["ann_return"]),
        "excess_ann": float(performance_stats(excess)["ann_return"]),
        "excess_cw_ann": float(performance_stats(excess_cw)["ann_return"]),
        "long_net_ann": float(performance_stats(long_net)["ann_return"]),
        "excess_net_ann": float(performance_stats(excess_net)["ann_return"]),
        "turnover": _mean_finite(kept_turns),
    }


def group_rows(backtest) -> list[dict]:
    rows = []
    for item in backtest.group_summary.sort("group").to_dicts():
        group_id = int(item["group"])
        ann = float(item["ann_return"])
        if not math.isfinite(ann):
            ann = 0.0
        rows.append(
            {
                "group": group_id,
                "ann_return": ann,
                "is_long": group_id == backtest.long_group,
                "is_short": group_id == backtest.short_group,
            }
        )
    _scale_bars(rows)
    return rows


def apply_preset(gross, turns, dates, preset, weights, panel, *, side: str | None = None):
    """完整成本在有月成交额时用参与率冲击，其余预设仍用固定费率。"""

    impacts = None
    base = float(preset.get("impact_bps") or 0.0)
    if base > 0 and weights is not None and panel is not None and "dollar_volume" in panel.columns:
        by_date = participation_by_date(weights, panel, base_bps=base, side=side)
        impacts = impacts_for_dates(by_date, dates)
    return net_returns(gross, turns, preset, impacts), impacts


def average_rank_corr(panel: pl.DataFrame, names: list[str]) -> list[dict]:
    """各月截面 Spearman 的平均。列是去掉市值和行业之后的因子。"""

    specs = [spec for spec in FACTORS if spec.name in names and spec.neu_name in panel.columns]
    if len(specs) < 2 or panel.height == 0:
        return []
    columns = [spec.neu_name for spec in specs]
    totals = [[0.0 for _ in specs] for _ in specs]
    counts = [[0 for _ in specs] for _ in specs]
    for day in panel["date"].unique().to_list():
        block = panel.filter(pl.col("date") == day).select(columns)
        values = [block[name].to_numpy().astype(float) for name in columns]
        for left in range(len(specs)):
            for right in range(left + 1, len(specs)):
                rho = _spearman(values[left], values[right])
                if rho is None:
                    continue
                totals[left][right] += rho
                totals[right][left] += rho
                counts[left][right] += 1
                counts[right][left] += 1
    rows = []
    for left, spec in enumerate(specs):
        cells = []
        for right in range(len(specs)):
            if left == right:
                cells.append(1.0)
            elif counts[left][right] == 0:
                cells.append(None)
            else:
                cells.append(totals[left][right] / counts[left][right])
        rows.append({"label": spec.label, "cells": cells})
    return rows


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 20:
        return None
    a = left[mask]
    b = right[mask]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if float(ra.std()) <= 1e-12 or float(rb.std()) <= 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def strategy_ic(panel, spec, column: str, min_names: int) -> dict:
    if column not in panel.columns or panel.height == 0:
        return {}
    signed = panel.with_columns((pl.col(column) * spec.direction).alias("_signed"))
    ic = rank_ic_series(signed, "_signed", min_names=min_names)
    stats = summarize_series(ic["ic"].to_numpy(), nw_lags=1)
    return {"mean": float(stats["mean"]), "nw_t": float(stats["nw_t"]), "n": int(stats["n"])}
