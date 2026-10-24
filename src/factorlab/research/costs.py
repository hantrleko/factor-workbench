"""交易成本：佣金、点差、冲击和借券，全部按明示假设计算。"""

from __future__ import annotations

import math

from factorlab.eval.metrics import performance_stats

# 单边费率，按成交金额计一次。冲击成本是固定假设，不是逐笔估计。
FULL_COST = {
    "name": "完整",
    "commission_bps": 3.0,
    "half_spread_bps": 5.0,
    "impact_bps": 10.0,
    "borrow_monthly": 0.003,
}
PRESETS = {
    "zero": {"name": "不扣费", "commission_bps": 0.0, "half_spread_bps": 0.0, "impact_bps": 0.0, "borrow_monthly": 0.0},
    "commission": {"name": "仅佣金", "commission_bps": 3.0, "half_spread_bps": 0.0, "impact_bps": 0.0, "borrow_monthly": 0.0},
    "full": FULL_COST,
}


def side_cost(preset: dict) -> float:
    bps = preset["commission_bps"] + preset["half_spread_bps"] + preset["impact_bps"]
    return bps / 10000.0


def flat_side(bps: float) -> dict:
    """单一单边费率。买卖各付一次，不含点差、冲击和借券。"""

    return {
        "name": f"单边 {bps:g}bp",
        "commission_bps": float(bps),
        "half_spread_bps": 0.0,
        "impact_bps": 0.0,
        "borrow_monthly": 0.0,
    }


def month_cost_parts(turnover, preset: dict) -> dict[str, float]:
    """把一个月的成本拆成佣金、半点差、冲击和借券。

    单位都是组合名义收益。多空空头名义为 1。没有换手时交易成本为 0，借券费仍计。
    """

    traded = 0.0
    if turnover is not None and math.isfinite(float(turnover)):
        traded = float(turnover)
    commission = 2.0 * traded * float(preset["commission_bps"]) / 10000.0
    half_spread = 2.0 * traded * float(preset["half_spread_bps"]) / 10000.0
    impact = 2.0 * traded * float(preset["impact_bps"]) / 10000.0
    borrow = float(preset["borrow_monthly"])
    return {
        "commission": commission,
        "half_spread": half_spread,
        "impact": impact,
        "borrow": borrow,
        "total": commission + half_spread + impact + borrow,
    }


def net_returns(gross_returns, turnovers, preset: dict) -> list[float]:
    """月净收益 = 毛收益 − 佣金 − 半点差 − 冲击 − 借券。"""

    return [
        float(gross) - month_cost_parts(turn, preset)["total"]
        for gross, turn in zip(gross_returns, turnovers, strict=True)
    ]


def annualized_net(gross_returns, turnovers, preset: dict) -> float:
    nets = net_returns(gross_returns, turnovers, preset)
    if not nets:
        return float("nan")
    return float(performance_stats(nets)["ann_return"])
