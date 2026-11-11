"""因子个数多于月份时，波动分解仍给出数字，并说明只用了自身方差。"""

import numpy as np
import polars as pl

from factorlab.risk.barra import _risk_decomposition


def test_risk_decomposition_falls_back_when_factors_exceed_months():
    dates = [f"{2020 + index // 12:04d}-{(index % 12) + 1:02d}-28" for index in range(14)]
    returns: dict = {"date": dates, "country": [0.01] * 14}
    for index in range(13):
        returns[f"ind::{index}"] = [0.001 * (index + 1)] * 14
    exposures: dict = {"date": [dates[-1]], "ticker": ["A"], "country": [1.0]}
    for index in range(13):
        exposures[f"ind::{index}"] = [0.05]
    specific = pl.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["A"] * 14 + ["B"] * 14,
            "specific_return": list(np.linspace(-0.02, 0.02, 28)),
        }
    )
    weights = pl.DataFrame({"date": [dates[-1]], "ticker": ["A"], "weight": [1.0]})
    risk = _risk_decomposition(
        pl.DataFrame(returns),
        specific,
        pl.DataFrame(exposures),
        weights,
        weights,
        date_col="date",
        id_col="ticker",
        asof=dates[-1],
    )
    assert risk["total_vol_ann"] == risk["total_vol_ann"]
    assert risk["factor_vol_ann"] == 0.0
    assert "自身方差" in risk["note"]
