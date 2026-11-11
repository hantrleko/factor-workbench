"""停牌和近似涨跌停不能新开仓，卖不出的旧仓按原权重延续。"""

import datetime as dt

import polars as pl

from factorlab.eval.backtest import quantile_backtest


def test_suspended_holding_keeps_its_weight():
    day1 = dt.date(2022, 1, 31)
    day2 = dt.date(2022, 2, 28)
    names = ["A", "B", "C", "D"]
    rows = []
    for name, factor, ret in zip(names, (1.0, 2.0, 3.0, 4.0), (0.01, 0.02, 0.03, 0.04), strict=True):
        rows.append({"date": day1, "ticker": name, "factor": factor, "fwd_ret": ret, "tradable": True})
    for name, factor, ret, tradable in zip(
        names,
        (1.0, 2.0, 3.0, 4.0),
        (0.1, 0.1, 0.2, 0.5),
        (True, True, True, False),
        strict=True,
    ):
        rows.append({"date": day2, "ticker": name, "factor": factor, "fwd_ret": ret, "tradable": tradable})
    panel = pl.DataFrame(rows)
    result = quantile_backtest(
        panel,
        "factor",
        direction=1,
        n_groups=2,
        min_names=3,
        respect_tradable=True,
    )
    second = result.weights.filter(pl.col("date") == day2).filter(pl.col("side") == "long")
    held = {row["ticker"]: row["weight"] for row in second.to_dicts()}
    assert held["D"] == 0.5
    assert abs(held["C"] - 0.5) < 1e-12
    long_ret = result.long_short.filter(pl.col("date") == day2)["long_ret"][0]
    assert abs(float(long_ret) - 0.35) < 1e-12

    ignored = quantile_backtest(panel, "factor", direction=1, n_groups=2, min_names=3)
    fresh = ignored.weights.filter(pl.col("date") == day2).filter(pl.col("ticker") == "D")
    assert fresh.height == 1
