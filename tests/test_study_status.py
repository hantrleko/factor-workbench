"""手造的月度面板也能跑出研究状态，而且不会写成通过。"""

from datetime import date

import polars as pl

from factorlab.research.study import evaluate_universe


def test_pre_specified_panel_is_not_labeled_as_passed():
    rows = []
    for month in range(48):
        year = 2020 + (month // 12)
        month_number = month % 12 + 1
        day = date(year, month_number, 28)
        for name in range(20):
            rows.append(
                {
                    "date": day,
                    "ticker": f"S{name:02d}",
                    "sloan_accrual_neu": float(name),
                    "fwd_ret": 0.001 * name,
                    "mktcap": 1e9,
                    "industry": "测试",
                }
            )
    reports = evaluate_universe(pl.DataFrame(rows))
    assert len(reports) == 1
    assert "通过" not in reports[0]["status"]
    assert reports[0]["status"] in {"研究中", "样本内候选，尚未验证", "样本外仍成立"}
    exposure = reports[0]["full"]["exposure"]
    assert exposure["n_months"] > 0
    assert abs(exposure["size_spread"]) < 1e-8
    assert exposure["industries"][0]["name"] == "测试"
