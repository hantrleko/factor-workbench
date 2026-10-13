"""中性化的维度与正交性。"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from factorlab.neutralize.residual import (
    ols_neutralize_one,
    orthogonality_diagnostics,
    wls_residual,
)


def test_wls_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="X"):
        wls_residual(np.ones(5), np.ones((4, 2)), np.ones(5))
    with pytest.raises(ValueError, match="w"):
        wls_residual(np.ones(5), np.ones((5, 2)), np.ones(4))


def test_residual_is_orthogonal_and_wipes_size_and_industry():
    rng = np.random.default_rng(0)
    n = 90
    cap = rng.lognormal(mean=10.0, sigma=0.6, size=n)
    industry = np.array([f"I{i % 4}" for i in range(n)])
    y = rng.normal(size=n) + 0.4 * np.log(cap) + (industry == "I2").astype(float)
    frame = pl.DataFrame(
        {
            "date": [date(2021, 1, 29)] * n,
            "mktcap": cap,
            "industry": industry,
            "factor": y,
        }
    )
    from factorlab.neutralize.residual import neutralize_factor_frame

    out = neutralize_factor_frame(frame, "factor", winsor=None)
    diag = orthogonality_diagnostics(out, "factor_neu")
    assert diag["max_abs_size_corr"] < 1e-8
    assert diag["max_abs_industry_mean"] < 1e-8

    size_resid = ols_neutralize_one(np.log(cap), cap, industry, winsor=None)
    assert np.nanmax(np.abs(size_resid)) < 1e-8
    dummy = (industry == "I1").astype(float)
    ind_resid = ols_neutralize_one(dummy, cap, industry, winsor=None)
    assert np.nanmax(np.abs(ind_resid)) < 1e-8


def test_too_few_names_stay_missing():
    cap = np.array([10.0, 20.0, 30.0])
    industry = np.array(["A", "B", "A"])
    y = np.array([1.0, 2.0, 3.0])
    resid = ols_neutralize_one(y, cap, industry)
    assert np.isnan(resid).all()
