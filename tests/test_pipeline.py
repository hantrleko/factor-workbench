"""时点不被未来行情改写，并且整条链路能写出报告。"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from factorlab.data.loader import load_prices
from factorlab.data.synthetic import make_synthetic_universe
from factorlab.factors.panel import build_factor_panel
from factorlab.pipeline import run_analysis


def test_chinese_aliases(tmp_path: Path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "日期,股票代码,收盘价,成交量,市值,行业\n2021-01-04,S0001,10,100,1000,电子\n",
        encoding="utf-8",
    )
    loaded = load_prices(path)
    assert loaded.columns[:6] == ["date", "ticker", "close", "volume", "mktcap", "industry"]


def test_past_factors_do_not_change_when_future_prices_are_removed():
    prices, funds = make_synthetic_universe(
        n_stocks=24,
        start="2022-01-04",
        end="2023-09-29",
        seed=3,
    )
    full = build_factor_panel(prices, funds, horizons=2)
    cutoff = date(2023, 3, 31)
    cut = build_factor_panel(prices.filter(pl.col("date") <= cutoff), funds, horizons=2)
    cols = [
        "sloan_accrual",
        "capex_intensity",
        "dupont_pm",
        "dupont_at",
        "dupont_em",
        "amihud",
        "ivol",
        "reversal",
    ]
    a = full.filter(pl.col("date") <= cutoff).select(["ticker", "date", *cols]).sort(["ticker", "date"])
    b = cut.filter(pl.col("date") <= cutoff).select(["ticker", "date", *cols]).sort(["ticker", "date"])
    assert a.select(["ticker", "date"]).equals(b.select(["ticker", "date"]))
    for col in cols:
        left = a[col].to_numpy()
        right = b[col].to_numpy()
        both = np.isfinite(left) & np.isfinite(right)
        assert both.any()
        assert np.max(np.abs(left[both] - right[both])) < 1e-8


def test_report_files(tmp_path: Path):
    prices, funds = make_synthetic_universe(
        n_stocks=36,
        start="2022-01-04",
        end="2023-06-30",
        seed=11,
    )
    result = run_analysis(
        prices,
        funds,
        n_groups=(5,),
        out_dir=tmp_path,
        meta={"note": "单元测试用的合成样本。"},
        progress=None,
    )
    html = Path(result.artifacts["html"])
    pdf = Path(result.artifacts["pdf"])
    assert html.exists() and "策略 Rank IC" in html.read_text(encoding="utf-8")
    assert pdf.exists() and pdf.stat().st_size > 20_000
    assert result.artifacts["pdf_engine"] in {"weasyprint", "fpdf2"}
    active = result.brinson_totals["active_return"]
    parts = (
        result.brinson_totals["allocation"]
        + result.brinson_totals["selection"]
        + result.brinson_totals["interaction"]
    )
    assert abs(active - parts) < 1e-8
