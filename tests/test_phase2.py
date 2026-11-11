"""二阶段：中性组合、参与率冲击、IC 与多空背离。"""

from datetime import date

import polars as pl

from factorlab.eval.portfolio import neutral_book_weights
from factorlab.research.costs import (
    FULL_COST,
    impact_bps_for_participation,
    net_returns,
    participation_by_date,
)
from factorlab.web.browse import average_rank_corr
from factorlab.web.explain import ic_return_conflict


def test_participation_anchor_and_square_root():
    assert abs(impact_bps_for_participation(0.10, 10) - 10) < 1e-12
    assert abs(impact_bps_for_participation(0.40, 10) - 20) < 1e-12
    assert impact_bps_for_participation(0.0, 10) == 0.0


def test_replaced_impact_keeps_commission_spread_and_borrow():
    gross = [0.02]
    turns = [0.5]
    impacts = [0.0004]
    net = net_returns(gross, turns, FULL_COST, impacts)
    commission_spread = 2.0 * 0.5 * (3.0 + 5.0) / 10000.0
    expected = 0.02 - commission_spread - 0.0004 - FULL_COST["borrow_monthly"]
    assert abs(net[0] - expected) < 1e-12


def test_participation_matches_flat_bps_at_the_anchor():
    days = [date(2020, 1, 31), date(2020, 2, 29)]
    weights = pl.DataFrame(
        {
            "date": [days[0], days[0], days[1]],
            "ticker": ["a", "b", "a"],
            "side": ["long", "long", "long"],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    panel = pl.DataFrame(
        {
            "date": [days[0], days[0], days[1], days[1]],
            "ticker": ["a", "b", "a", "b"],
            "dollar_volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    # 第二个月 |Δw| 各为 0.5。n=2，中位数 100，规模 = 0.1×100×2。
    # 参与率 = 0.5×20/100 = 0.10，冲击 10bp，成本 = 0.5×2×10/10000。
    costs = participation_by_date(weights, panel, base_bps=10)
    assert abs(costs["2020-01-31"]) < 1e-12
    assert abs(costs["2020-02-29"] - 0.001) < 1e-12


def test_neutral_book_matches_industry_and_picks_the_high_score():
    day = date(2020, 1, 31)
    panel = pl.DataFrame(
        {
            "date": [day] * 5,
            "ticker": ["a", "b", "e", "c", "d"],
            "composite": [1.0, 3.0, 5.0, 0.0, 9.0],
            "fwd_ret": [0.01] * 5,
            "mktcap": [100.0, 300.0, 80.0, 100.0, 300.0],
            "industry": ["银行", "银行", "银行", "证券", "证券"],
        }
    )
    port, bench = neutral_book_weights(panel)
    weights = dict(zip(port["ticker"].to_list(), port["weight"].to_list(), strict=True))
    total = 880.0
    assert abs(weights["e"] - 180.0 / total) < 1e-9
    assert abs(weights["b"] - 300.0 / total) < 1e-9
    assert "a" not in weights
    assert abs(weights["c"] - 100.0 / total) < 1e-9
    assert abs(weights["d"] - 300.0 / total) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    bench_w = dict(zip(bench["ticker"].to_list(), bench["weight"].to_list(), strict=True))
    bank_book = weights["e"] + weights["b"]
    bank_bench = bench_w["a"] + bench_w["b"] + bench_w["e"]
    assert abs(bank_book - bank_bench) < 1e-9


def test_ic_positive_and_long_short_negative_is_explained():
    groups = [
        {"group": 1, "ann_return": 0.01, "is_short": True},
        {"group": 5, "ann_return": -0.02, "is_long": True},
    ]
    text = ic_return_conflict(0.04, -0.021, [-0.2] + [0.01] * 9, groups)
    assert "策略 IC 是 0.040" in text
    assert "多空年化" in text
    assert "最差的 1 个月" in text
    assert ic_return_conflict(0.04, 0.02, [0.01] * 10, groups) == ""
    assert ic_return_conflict(-0.01, -0.02, [-0.01] * 10, groups) == ""


def test_rank_corr_of_opposite_factors_is_minus_one():
    day = date(2020, 1, 31)
    raw = list(range(25))
    panel = pl.DataFrame(
        {
            "date": [day] * 25,
            "sloan_accrual_neu": [float(item) for item in raw],
            "capex_intensity_neu": [float(-item) for item in raw],
        }
    )
    rows = average_rank_corr(panel, ["sloan_accrual", "capex_intensity"])
    assert len(rows) == 2
    assert abs(rows[0]["cells"][1] + 1.0) < 1e-9
