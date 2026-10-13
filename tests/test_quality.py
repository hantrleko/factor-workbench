"""财务因子的手算与时点。"""

from datetime import date

import polars as pl

from factorlab.factors.quality import fundamental_factors


def _statement(**kwargs):
    base = {
        "ticker": "A",
        "fiscal_year": 2020,
        "fiscal_quarter": 1,
        "available_date": date(2020, 4, 30),
        "total_assets": 200.0,
        "current_assets": 100.0,
        "cash": 20.0,
        "current_liabilities": 40.0,
        "short_term_debt": 10.0,
        "tax_payable": 5.0,
        "depreciation": 0.0,
        "net_income": 0.0,
        "revenue": 0.0,
        "capex": 0.0,
        "equity": 100.0,
    }
    base.update(kwargs)
    return base


def test_sloan_dupont_and_publication_lag():
    rows = [
        _statement(),
        _statement(
            fiscal_quarter=2,
            available_date=date(2020, 7, 31),
            depreciation=4,
            net_income=2,
            revenue=25,
            capex=5,
        ),
        _statement(
            fiscal_quarter=3,
            available_date=date(2020, 10, 31),
            depreciation=4,
            net_income=3,
            revenue=25,
            capex=5,
        ),
        _statement(
            fiscal_quarter=4,
            available_date=date(2021, 1, 31),
            depreciation=4,
            net_income=2,
            revenue=25,
            capex=5,
        ),
        _statement(
            fiscal_year=2021,
            fiscal_quarter=1,
            available_date=date(2021, 4, 30),
            total_assets=220,
            current_assets=130,
            cash=25,
            current_liabilities=50,
            short_term_debt=12,
            tax_payable=6,
            depreciation=3,
            net_income=3,
            revenue=25,
            capex=5,
            equity=110,
        ),
    ]
    funds = pl.DataFrame(rows)
    spine = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": [date(2021, 3, 31), date(2021, 4, 30)],
        }
    )
    out = fundamental_factors(spine, funds).sort("date")
    early = out.row(0, named=True)
    late = out.row(1, named=True)
    assert early["available_date"] == date(2021, 1, 31)
    assert early["sloan_accrual"] is None
    assert late["available_date"] == date(2021, 4, 30)
    assert abs(late["sloan_accrual"] - 3 / 210) < 1e-12
    assert abs(late["capex_intensity"] - 20 / 210) < 1e-12
    assert abs(late["dupont_pm"] - 0.1) < 1e-12
    assert abs(late["dupont_at"] - 100 / 210) < 1e-12
    assert abs(late["dupont_em"] - 2.0) < 1e-12
    assert abs(late["dupont_roe"] - late["dupont_pm"] * late["dupont_at"] * late["dupont_em"]) < 1e-12
