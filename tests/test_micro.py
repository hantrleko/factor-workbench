"""微观结构因子与远期收益对齐。"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from factorlab.factors.microstructure import (
    amihud_monthly,
    capm_residual_vol,
    microstructure_panel,
    prepare_prices,
)


def test_capm_ivol_matches_ols():
    rng = np.random.default_rng(1)
    y = rng.normal(size=20)
    x = rng.normal(size=20)
    vol = capm_residual_vol(y, x, min_obs=15)
    design = np.column_stack([np.ones(20), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    assert abs(vol - np.sqrt(np.sum(resid**2) / 18)) < 1e-12


def test_amihud_and_reversal_and_forward_window():
    days = [date(2021, 1, 4) + timedelta(days=i) for i in range(20)]
    days = [d for d in days if d.weekday() < 5][:12]
    feb = [date(2021, 2, 1) + timedelta(days=i) for i in range(20)]
    feb = [d for d in feb if d.weekday() < 5][:12]
    rows = []
    for d in days:
        rows.append(
            {
                "date": d,
                "ticker": "A",
                "close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
                "ret": 0.01,
                "mktcap": 1000.0,
                "industry": "电子",
            }
        )
    for d in feb:
        rows.append(
            {
                "date": d,
                "ticker": "A",
                "close": 11.0,
                "volume": 100.0,
                "amount": 2000.0,
                "ret": -0.02,
                "mktcap": 1100.0,
                "industry": "电子",
            }
        )
    panel = microstructure_panel(pl.DataFrame(rows)).sort("date")
    jan = panel.row(0, named=True)
    expected_amihud = abs(0.01) / 1000.0
    assert abs(jan["amihud"] - expected_amihud) < 1e-12
    expected_month = (1.01 ** 12) - 1
    assert abs(jan["month_ret"] - expected_month) < 1e-12
    assert abs(jan["reversal"] + expected_month) < 1e-12
    assert abs(jan["fwd_ret"] - ((0.98 ** 12) - 1)) < 1e-12
    assert panel["fwd_ret"].to_list()[1] is None


def test_ivol_on_two_stocks():
    rng = np.random.default_rng(2)
    days = [date(2021, 3, 1) + timedelta(days=i) for i in range(40)]
    days = [d for d in days if d.weekday() < 5][:16]
    rows = []
    rets = {}
    for ticker, scale in (("A", 1.0), ("B", 1.4)):
        series = rng.normal(0.001, 0.02 * scale, size=len(days))
        rets[ticker] = series
        for d, ret in zip(days, series, strict=True):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": 10.0,
                    "volume": 1_000.0,
                    "amount": 10_000.0,
                    "ret": float(ret),
                    "mktcap": 1_000_000.0 if ticker == "A" else 3_000_000.0,
                    "industry": "医药",
                }
            )
    px = prepare_prices(pl.DataFrame(rows))
    out = amihud_monthly(px)
    assert out.height == 2
    from factorlab.factors.microstructure import ivol_monthly, market_return

    mkt = market_return(px).sort("date")
    # 首日没有滞后市值，市场收益从第二天开始
    mkt_map = dict(zip(mkt["date"].to_list(), mkt["mkt_ret"].to_list(), strict=True))
    ivol = ivol_monthly(px)
    for ticker in ("A", "B"):
        sub = px.filter(pl.col("ticker") == ticker).sort("date")
        paired_ret = []
        paired_mkt = []
        for d, ret in zip(sub["date"].to_list(), sub["ret"].to_list(), strict=True):
            if d in mkt_map:
                paired_ret.append(ret)
                paired_mkt.append(mkt_map[d])
        expected = capm_residual_vol(np.array(paired_ret), np.array(paired_mkt), min_obs=15)
        got = ivol.filter(pl.col("ticker") == ticker)["ivol"][0]
        assert abs(got - expected) < 1e-10
