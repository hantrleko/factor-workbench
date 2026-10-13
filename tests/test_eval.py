"""Rank IC、分组与换手。"""

from datetime import date

import numpy as np
import polars as pl

from factorlab.eval.backtest import assign_quantile_groups, quantile_backtest
from factorlab.eval.ic import spearman
from factorlab.eval.metrics import max_drawdown, one_way_turnover, performance_stats


def test_spearman_and_groups():
    x = np.arange(30, dtype=float)
    assert spearman(x, x) == 1.0
    assert abs(spearman(x, -x) + 1.0) < 1e-12
    groups = assign_quantile_groups(np.arange(50, dtype=float), 10)
    counts = np.bincount(groups)[1:]
    assert counts.max() - counts.min() <= 1
    assert set(groups.tolist()) == set(range(1, 11))
    assert assign_quantile_groups(np.array([1.0, np.nan, 2.0]), 5).tolist() == [0, 0, 0]


def test_long_short_follows_direction_and_drawdown():
    n = 40
    factor = np.linspace(-1.0, 1.0, n)
    fwd = factor * 0.02
    panel = pl.DataFrame(
        {
            "date": [date(2021, 1, 29)] * n,
            "ticker": [f"S{i:03d}" for i in range(n)],
            "factor": factor,
            "fwd_ret": fwd,
        }
    )
    high = quantile_backtest(panel, "factor", direction=1, n_groups=5, min_names=20)
    low = quantile_backtest(panel, "factor", direction=-1, n_groups=5, min_names=20)
    assert high.summary["ann_return"] > 0
    assert low.summary["ann_return"] < 0
    assert high.long_group == 5 and high.short_group == 1
    assert low.long_group == 1 and low.short_group == 5
    assert max_drawdown(np.array([-0.5, 0.1])) == -0.5
    stats = performance_stats(np.array([0.1, 0.1]))
    assert abs(stats["total_return"] - (1.1**2 - 1)) < 1e-12


def test_backtest_turnover_ignores_the_first_rebalance():
    n = 30
    rows = []
    for month, sign in ((date(2021, 1, 29), 1.0), (date(2021, 2, 26), -1.0)):
        factor = sign * np.linspace(-1.0, 1.0, n)
        for i, value in enumerate(factor):
            rows.append(
                {
                    "date": month,
                    "ticker": f"S{i:03d}",
                    "factor": float(value),
                    "fwd_ret": float(value * 0.01),
                }
            )
    result = quantile_backtest(
        pl.DataFrame(rows), "factor", direction=1, n_groups=5, min_names=20
    )
    assert np.isfinite(result.summary["avg_turnover"])
    assert result.summary["avg_turnover"] > 0


def test_turnover_identity():
    assert one_way_turnover({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    # 完全换仓：卖掉 a、b，买入 c、d
    assert one_way_turnover({"a": 0.5, "b": 0.5}, {"c": 0.5, "d": 0.5}) == 1.0
