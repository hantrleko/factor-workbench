"""交易成本：佣金、点差、冲击和借券，全部按明示假设计算。"""

from __future__ import annotations

import math

import polars as pl

from factorlab.eval.metrics import performance_stats

# 单边费率，按成交金额计一次。
# 冲击的 10bp 是参与率锚点：单只成交占该股当月成交额 10% 时记 10bp。
# 没有月成交额时，完整成本仍按固定 10bp。
REF_PARTICIPATION = 0.10
PARTICIPATION_COST_TEXT = (
    "完整成本 = 单边佣金 3bp + 半点差 5bp，买卖各付一次，另加空头每月 30bp 借券费。"
    "冲击按成交额占比：假设组合规模等于当月成交额中位数的 10% 乘以当月股票数，"
    "单只成交占该股月成交额 10% 时冲击为 10bp，其余按参与率的平方根缩放，参与率最高按 100% 计。"
    "没有月成交额时，冲击退回固定 10bp。"
)
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


def net_returns(gross_returns, turnovers, preset: dict, impacts=None) -> list[float]:
    """月净收益 = 毛收益 − 佣金 − 半点差 − 冲击 − 借券。

    impacts 给出时，用它替换固定冲击，佣金、点差和借券仍按 preset。
    """

    if impacts is not None and len(impacts) != len(list(gross_returns)):
        raise ValueError("冲击序列和收益序列长度不一致")
    out = []
    for index, (gross, turn) in enumerate(zip(gross_returns, turnovers, strict=True)):
        parts = month_cost_parts(turn, preset)
        drag = parts["total"]
        if impacts is not None:
            extra = impacts[index]
            extra = 0.0 if extra is None or not math.isfinite(float(extra)) else float(extra)
            drag = parts["total"] - parts["impact"] + extra
        out.append(float(gross) - drag)
    return out


def impact_bps_for_participation(participation: float, base_bps: float) -> float:
    """参与率等于锚点时回到 base_bps，其余按平方根缩放。"""

    if base_bps <= 0 or not math.isfinite(participation) or participation <= 0:
        return 0.0
    return float(base_bps) * math.sqrt(min(float(participation), 1.0) / REF_PARTICIPATION)


def participation_by_date(weights, panel, *, base_bps: float, side: str | None = None) -> dict[str, float] | None:
    """每个形成日的冲击，单位是收益。side 为空时用多头权重减空头权重。"""

    if weights is None or getattr(weights, "height", 0) == 0:
        return None
    if panel is None or "dollar_volume" not in panel.columns or base_bps <= 0:
        return None
    id_col = "ticker" if "ticker" in weights.columns else "id"
    books: dict[str, dict[str, float]] = {}
    named = weights.select(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("date"),
        pl.col(id_col).cast(pl.Utf8).alias(id_col),
        pl.col("side"),
        pl.col("weight"),
    )
    for row in named.iter_rows(named=True):
        if side and row["side"] != side:
            continue
        weight = float(row["weight"])
        if not math.isfinite(weight):
            continue
        signed = weight if side or row["side"] == "long" else -weight
        book = books.setdefault(row["date"], {})
        book[row[id_col]] = book.get(row[id_col], 0.0) + signed
    if not books:
        return None
    volume = panel.select(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("date"),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("dollar_volume").cast(pl.Float64),
    )
    lookup: dict[tuple[str, str], float] = {}
    medians: dict[str, float] = {}
    counts: dict[str, int] = {}
    buckets: dict[str, list[float]] = {}
    for row in volume.iter_rows(named=True):
        value = row["dollar_volume"]
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            continue
        lookup[(row["date"], row["ticker"])] = float(value)
        buckets.setdefault(row["date"], []).append(float(value))
    for day, values in buckets.items():
        medians[day] = _median(values)
        counts[day] = len(values)
    out: dict[str, float] = {}
    prev: dict[str, float] | None = None
    for day in sorted(books):
        current = books[day]
        if prev is None:
            out[day] = 0.0
            prev = current
            continue
        names = set(prev) | set(current)
        changes = {name: abs(current.get(name, 0.0) - prev.get(name, 0.0)) for name in names}
        traded = sum(changes.values())
        median = medians.get(day) or 0.0
        count = counts.get(day) or 0
        if traded <= 0:
            out[day] = 0.0
        elif median <= 0 or count <= 0:
            out[day] = traded * float(base_bps) / 10000.0
        else:
            aum = REF_PARTICIPATION * median * count
            cost = 0.0
            for name, change in changes.items():
                if change <= 0:
                    continue
                own = lookup.get((day, name))
                participation = 1.0 if own is None or own <= 0 else min(change * aum / own, 1.0)
                cost += change * impact_bps_for_participation(participation, base_bps) / 10000.0
            out[day] = cost
        prev = current
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def impacts_for_dates(by_date: dict[str, float] | None, dates) -> list[float] | None:
    if by_date is None:
        return None
    return [float(by_date.get(str(day)[:10], 0.0)) for day in dates]


def annualized_net(gross_returns, turnovers, preset: dict) -> float:
    nets = net_returns(gross_returns, turnovers, preset)
    if not nets:
        return float("nan")
    return float(performance_stats(nets)["ann_return"])
