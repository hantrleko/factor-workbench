"""把分组收益翻成一句人话，并标出该买的那一组赚了多少。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from factorlab.eval.backtest import quantile_backtest
from factorlab.report.formatters import pct
from factorlab.schema import FACTORS

_BY_NAME = {spec.name: spec for spec in FACTORS}


def attach_group_reads(out: Path, rows: list[dict]) -> None:
    path = out / "factor_panel.parquet"
    if not path.is_file():
        for row in rows:
            row.setdefault("because", "")
            row.setdefault("groups", [])
        return
    panel = pl.read_parquet(path)
    for row in rows:
        spec = _BY_NAME.get(row.get("name", ""))
        if spec is None or spec.neu_name not in panel.columns:
            row["because"] = ""
            row["groups"] = []
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
    return f"比市值基准多拿的是{over_names}，少拿的是{under_names}。"


def scoreboard(rows: list[dict]) -> list[dict]:
    order = (
        ("支持该假设", "pos"),
        ("多空为正，但不够稳", "muted"),
        ("排序略对，多空没赚到", "neg"),
        ("不支持该假设", "neg"),
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
