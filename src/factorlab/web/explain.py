"""把分组收益翻成一句人话，并按统一口径给出样本内结论和扣费敏感性。"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from factorlab.eval.backtest import quantile_backtest
from factorlab.eval.metrics import performance_stats
from factorlab.report.formatters import pct
from factorlab.schema import FACTORS

_BY_NAME = {spec.name: spec for spec in FACTORS}

# 单边费率：佣金加价差，按成交金额计一次。买和卖各付一次。
COST_PER_SIDE = (0.0005, 0.0010, 0.0020)
DEMO_NOTICE = "生成过程包含预设因子溢价，不代表真实投资表现。"
CN_NOTICE = (
    "行情来自证券宝，财务来自东方财富公告日。收益用前复权价格，已包含分红和拆股。"
    "上市腿不是完整的历史指数成分，退市股票只在退市日前可交易。这是研究记录，不是可交易 Alpha。"
)
SUPPORT_RULE = (
    "“样本内候选，尚未验证”指开发样本上 Newey–West t ≥ 1.96，且扣费前多空为正。"
    "这不是可交易 Alpha。样本外和完整成本没有同时成立时，状态是“研究中”。"
)


def attach_group_reads(out: Path, rows: list[dict]) -> None:
    path = out / "factor_panel.parquet"
    if not path.is_file():
        for row in rows:
            row["because"] = ""
            row["groups"] = []
            row["net"] = []
            row["net_10"] = None
        return
    panel = pl.read_parquet(path)
    for row in rows:
        spec = _BY_NAME.get(row.get("name", ""))
        if spec is None or spec.neu_name not in panel.columns:
            row["because"] = ""
            row["groups"] = []
            row["net"] = []
            row["net_10"] = None
            continue
        result = quantile_backtest(
            panel,
            spec.neu_name,
            direction=spec.direction,
            n_groups=5,
        )
        groups = []
        for item in result.group_summary.sort("group").to_dicts():
            group_id = int(item["group"])
            groups.append(
                {
                    "group": group_id,
                    "ann_return": float(item["ann_return"]),
                    "is_long": group_id == result.long_group,
                    "is_short": group_id == result.short_group,
                }
            )
        _scale_bars(groups)
        row["groups"] = groups
        row["long_group"] = result.long_group
        row["because"] = _because(result.long_group, groups)
        row["net"] = _net_cases(result)
        row["net_10"] = next((item["ann_return"] for item in row["net"] if item["bps"] == 10), None)


def _scale_bars(groups: list[dict]) -> None:
    span = max((abs(item["ann_return"]) for item in groups), default=0.0)
    if span <= 0:
        span = 1.0
    for item in groups:
        item["width"] = max(4.0, abs(item["ann_return"]) / span * 100.0)
        item["tone"] = "pos" if item["ann_return"] >= 0 else "neg"


def _because(long_group: int, groups: list[dict]) -> str:
    if not groups:
        return ""
    long = next(item for item in groups if item["group"] == long_group)
    short = next((item for item in groups if item.get("is_short")), None)
    best = max(groups, key=lambda item: item["ann_return"])
    long_text = pct(long["ann_return"])
    if short is not None and long["ann_return"] <= short["ann_return"]:
        sentence = (
            f"该买的第{long_group}组年化 {long_text}，"
            f"低于该空的第{short['group']}组（{pct(short['ann_return'])}）。"
        )
    elif short is not None:
        sentence = (
            f"该买的第{long_group}组年化 {long_text}，"
            f"该空的第{short['group']}组年化 {pct(short['ann_return'])}。"
        )
    else:
        sentence = f"该买的第{long_group}组年化 {long_text}。"
    gap = best["ann_return"] - long["ann_return"]
    if best["group"] != long_group and gap >= 0.03:
        sentence += f"五组里最高的是第{best['group']}组（{pct(best['ann_return'])}），不是这一端。"
    return sentence


def industry_tilt(groups: list[dict]) -> str:
    usable = [item for item in groups if "wp" in item and "wb" in item]
    if len(usable) < 2:
        return ""
    gap = lambda item: float(item["wp"]) - float(item["wb"])
    over = [item for item in sorted(usable, key=gap, reverse=True) if gap(item) > 0][:2]
    under = [item for item in sorted(usable, key=gap) if gap(item) < 0][:2]
    if not over or not under:
        return ""
    over_names = "、".join(item["group"] for item in over)
    under_names = "、".join(item["group"] for item in under)
    return f"按期间平均权重，比市值基准多拿的是{over_names}，少拿的是{under_names}。"


def month_net_returns(gross_returns, turnovers, cost_per_side: float) -> list[float]:
    """月净收益 = 毛收益 − 单边换手 × 2 × 单边费率。首月没有换手，成本记 0。"""

    nets = []
    for gross, turn in zip(gross_returns, turnovers, strict=True):
        traded = 0.0 if turn is None or not math.isfinite(float(turn)) else float(turn)
        nets.append(float(gross) - 2.0 * traded * cost_per_side)
    return nets


def _net_cases(result) -> list[dict]:
    frame = result.long_short
    if frame.height == 0:
        return []
    gross = frame["ret"].to_list()
    turns = frame["turnover"].to_list()
    cases = []
    for cost in COST_PER_SIDE:
        nets = month_net_returns(gross, turns, cost)
        cases.append(
            {
                "bps": int(round(cost * 10000)),
                "ann_return": float(performance_stats(nets)["ann_return"]),
            }
        )
    return cases


def verdict(row: dict) -> tuple[str, str]:
    """与详情页同一条规则：Newey–West t，外加扣费前多空收益的符号。"""

    try:
        ic = float(row["ic"])
        t_stat = round(float(row["nw_t"]), 2)
        long_short = float(row["ls_ann"])
    except (KeyError, TypeError, ValueError):
        return "样本不足", "muted"
    if not math.isfinite(t_stat):
        return "样本不足", "muted"
    if ic > 0 and t_stat >= 1.96 and long_short > 0:
        return "样本内候选，尚未验证", "muted"
    if ic > 0 and long_short > 0:
        return "毛收益为正，NW t 不足", "muted"
    if ic > 0 and long_short <= 0:
        return "排序略对，多空没赚到", "neg"
    return "样本内不支持", "neg"


def scoreboard(rows: list[dict]) -> list[dict]:
    order = (
        ("样本外仍成立", "pos"),
        ("样本内候选，尚未验证", "muted"),
        ("研究中", "muted"),
        ("毛收益为正，NW t 不足", "muted"),
        ("排序略对，多空没赚到", "neg"),
        ("样本内不支持", "neg"),
        ("样本不足", "muted"),
    )
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("verdict") or ""
        counts[label] = counts.get(label, 0) + 1
    return [
        {"label": label, "n": counts[label], "tone": tone}
        for label, tone in order
        if counts.get(label, 0) > 0
    ]
