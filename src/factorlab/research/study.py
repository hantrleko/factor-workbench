"""在预先划定的开发样本、滚动测试和保留样本上评估因子。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from factorlab.eval.backtest import assign_quantile_groups, quantile_backtest
from factorlab.eval.ic import rank_ic_series
from factorlab.eval.metrics import performance_stats, summarize_series
from factorlab.research.costs import FULL_COST, annualized_net, net_returns
from factorlab.research.oos import holm_adjust, one_sided_p_from_t, rolling_windows, split_dates
from factorlab.research.status import research_status
from factorlab.schema import FACTORS


def _subset(panel: pl.DataFrame, dates: list, date_col: str = "date") -> pl.DataFrame:
    allowed = set(dates)
    return panel.filter(pl.col(date_col).is_in(list(allowed)))


def _group_means(result) -> list[dict]:
    rows = []
    for item in result.group_summary.sort("group").to_dicts():
        group_id = int(item["group"])
        rows.append(
            {
                "group": group_id,
                "ann_return": float(item["ann_return"]),
                "is_long": group_id == result.long_group,
                "is_short": group_id == result.short_group,
            }
        )
    return rows


def _monotone(groups: list[dict], direction: int) -> bool:
    if len(groups) < 3:
        return False
    score = [g["group"] if direction > 0 else -g["group"] for g in groups]
    returns = [g["ann_return"] for g in groups]
    if len(set(np.round(returns, 8))) < 2:
        return False
    from scipy.stats import spearmanr

    rho, _ = spearmanr(score, returns)
    return bool(math.isfinite(rho) and rho > 0)


def _book_arrays(result):
    frame = result.long_short
    return frame["date"].to_list(), frame["ret"].to_list(), frame["turnover"].to_list()


def _metrics_for_dates(panel, spec, dates: list) -> dict | None:
    sample = _subset(panel, dates)
    if sample.height == 0:
        return None
    signed = sample.with_columns((pl.col(spec.neu_name) * spec.direction).alias("_signed"))
    ic = rank_ic_series(signed, "_signed", min_names=12)
    ic_stats = summarize_series(ic["ic"].to_numpy(), nw_lags=1)
    backtest = quantile_backtest(sample, spec.neu_name, direction=spec.direction, n_groups=5, min_names=12)
    dates_ls, gross, turns = _book_arrays(backtest)
    groups = _group_means(backtest)
    net = net_returns(gross, turns, FULL_COST)
    net_stats = performance_stats(net)
    net_summary = summarize_series(np.asarray(net, dtype=float), nw_lags=1)
    exposure = long_short_exposure(sample, spec.neu_name, spec.direction)
    return {
        "ic": ic_stats,
        "gross": backtest.summary,
        "groups": groups,
        "monotone": _monotone(groups, spec.direction),
        "net_ann": float(net_stats["ann_return"]),
        "net_mdd": float(net_stats["max_drawdown"]),
        "net_nw_t": float(net_summary["nw_t"]),
        "dates": [str(d) for d in dates_ls],
        "gross_returns": [float(v) for v in gross],
        "net_returns": [float(v) for v in net],
        "turnover": [None if v is None or not math.isfinite(float(v)) else float(v) for v in turns],
        "exposure": exposure,
    }


def long_short_exposure(
    panel: pl.DataFrame,
    factor: str,
    direction: int,
    *,
    min_names: int = 12,
    n_groups: int = 5,
) -> dict:
    """多头等权相对空头等权的行业权重和对数市值差。正的规模差表示多头公司更大。"""

    empty = {"size_spread": None, "industries": [], "n_months": 0}
    if factor not in panel.columns or "mktcap" not in panel.columns or "industry" not in panel.columns:
        return empty
    if direction not in (1, -1) or panel.height == 0:
        return empty
    ordered = panel.sort(["date", "ticker"])
    dates = ordered["date"].to_numpy()
    values = ordered[factor].to_numpy()
    caps = ordered["mktcap"].to_numpy()
    industries = ordered["industry"].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    long_group = n_groups if direction > 0 else 1
    short_group = 1 if direction > 0 else n_groups
    size_gaps: list[float] = []
    month_weights: list[tuple[dict[str, float], dict[str, float]]] = []
    for start, end in zip(starts, ends, strict=True):
        val = np.asarray(values[start:end], dtype=float)
        cap = np.asarray(caps[start:end], dtype=float)
        labels = industries[start:end]
        usable = np.isfinite(val) & np.isfinite(cap) & (cap > 0)
        if int(usable.sum()) < min_names:
            continue
        groups = assign_quantile_groups(np.where(usable, val, np.nan), n_groups)

        def _leg(group_id: int) -> tuple[np.ndarray, np.ndarray] | None:
            take = groups == group_id
            if not np.any(take):
                return None
            return cap[take], labels[take]

        long_leg = _leg(long_group)
        short_leg = _leg(short_group)
        if long_leg is None or short_leg is None:
            continue
        size_gaps.append(float(np.log(long_leg[0]).mean() - np.log(short_leg[0]).mean()))
        month_weights.append((_weights(long_leg[1]), _weights(short_leg[1])))
    if not month_weights:
        return empty
    names: set[str] = set()
    for long_w, short_w in month_weights:
        names.update(long_w)
        names.update(short_w)
    rows = []
    n_months = len(month_weights)
    for name in names:
        long_weight = sum(long_w.get(name, 0.0) for long_w, _short_w in month_weights) / n_months
        short_weight = sum(short_w.get(name, 0.0) for _long_w, short_w in month_weights) / n_months
        rows.append(
            {
                "name": name,
                "long_weight": float(long_weight),
                "short_weight": float(short_weight),
                "active": float(long_weight - short_weight),
            }
        )
    rows.sort(key=lambda item: abs(item["active"]), reverse=True)
    return {
        "size_spread": float(np.mean(size_gaps)),
        "industries": rows,
        "n_months": n_months,
    }


def _weights(labels: np.ndarray) -> dict[str, float]:
    names = [("" if item is None else str(item)) for item in labels]
    if not names:
        return {}
    share = 1.0 / len(names)
    out: dict[str, float] = {}
    for name in names:
        out[name] = out.get(name, 0.0) + share
    return out


def evaluate_universe(panel: pl.DataFrame) -> list[dict]:
    dates = panel["date"].unique().sort().to_list()
    development, holdout = split_dates(dates)
    windows = rolling_windows(development)
    test_dates = [day for window in windows for day in window["test"]]
    reports = []
    oos_p = []
    kept = []
    for spec in FACTORS:
        if spec.neu_name not in panel.columns:
            continue
        kept.append(spec)
        dev = _metrics_for_dates(panel, spec, development)
        oos = _metrics_for_dates(panel, spec, test_dates) if test_dates else None
        held = _metrics_for_dates(panel, spec, holdout)
        candidate = bool(
            dev
            and dev["ic"]["mean"] > 0
            and math.isfinite(dev["ic"]["nw_t"])
            and dev["ic"]["nw_t"] >= 1.96
            and dev["gross"].get("ann_return", 0) > 0
        )
        oos_t = float(oos["net_nw_t"]) if oos else float("nan")
        oos_p.append(one_sided_p_from_t(oos_t, int(oos["ic"]["n"]) if oos else 0))
        reports.append(
            {
                "spec": spec,
                "candidate": candidate,
                "development": dev,
                "oos": oos,
                "holdout": held,
                "full": _metrics_for_dates(panel, spec, dates),
                "windows": len(windows),
            }
        )
    adjusted = holm_adjust([1.0 if not math.isfinite(p) else p for p in oos_p])
    for report, p_adj in zip(reports, adjusted, strict=True):
        oos = report["oos"]
        held = report["holdout"]
        label, tone = research_status(
            in_sample_candidate=report["candidate"],
            oos_nw_t=float(oos["net_nw_t"]) if oos else float("nan"),
            oos_net_positive=bool(oos and oos["net_ann"] > 0),
            holdout_net_positive=bool(held and held["net_ann"] > 0),
            holm_p=float(p_adj),
            monotone=bool(oos and oos["monotone"]),
        )
        report["holm_p"] = float(p_adj)
        report["status"] = label
        report["tone"] = tone
        report["holdout_dates"] = [str(d) for d in holdout]
        report["development_start"] = str(development[0])
        report["development_end"] = str(development[-1])
    return reports
