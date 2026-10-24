"""把点时行情和公告日财务跑成可导出的研究结果。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import polars as pl

from factorlab.data.cn_pit import RELEASE, fetch_and_store, refresh_fundamentals_from_cache
from factorlab.factors.panel import build_factor_panel
from factorlab.neutralize.residual import neutralize_factor_frame
from factorlab.research.ledger import write_trace
from factorlab.research.study import evaluate_universe
from factorlab.schema import FACTORS


def _jsonable(report: dict) -> dict:
    spec = report["spec"]
    full = report["full"] or {}
    gross = full.get("gross") or {}
    ic = full.get("ic") or {}
    dev = report["development"] or {}
    dev_ic = dev.get("ic") or {}
    dev_gross = dev.get("gross") or {}
    oos = report["oos"] or {}
    held = report["holdout"] or {}
    return {
        "name": spec.name,
        "label": spec.label,
        "family": spec.family,
        "question": spec.question,
        "formula": spec.formula,
        "definition": spec.definition,
        "direction_text": spec.direction_text,
        "verdict": report["status"],
        "verdict_tone": report["tone"],
        "because": _because(report),
        "ic": ic.get("mean"),
        "ir": ic.get("ir"),
        "t": ic.get("t"),
        "nw_t": ic.get("nw_t"),
        "dev_nw_t": dev_ic.get("nw_t"),
        "dev_ic": dev_ic.get("mean"),
        "dev_ls_ann": dev_gross.get("ann_return"),
        "hit": ic.get("hit"),
        "ls_ann": gross.get("ann_return"),
        "ls_mdd": gross.get("max_drawdown"),
        "turnover": gross.get("avg_turnover"),
        "net_full": full.get("net_ann"),
        "net_mdd": full.get("net_mdd"),
        "groups": full.get("groups") or [],
        "monotone": full.get("monotone"),
        "oos_nw_t": oos.get("net_nw_t"),
        "oos_net_ann": oos.get("net_ann"),
        "oos_monotone": oos.get("monotone"),
        "holdout_net_ann": held.get("net_ann"),
        "holdout_nw_t": held.get("net_nw_t"),
        "holm_p": report["holm_p"],
        "dates": full.get("dates") or [],
        "gross_returns": full.get("gross_returns") or [],
        "turnover_series": full.get("turnover") or [],
        "holdout_start": report["holdout_dates"][0] if report["holdout_dates"] else "",
        "development_start": report["development_start"],
        "development_end": report["development_end"],
        "exposure": full.get("exposure") or {"size_spread": None, "industries": [], "n_months": 0},
        "oos_exposure": (oos.get("exposure") if oos else None)
        or {"size_spread": None, "industries": [], "n_months": 0},
        "net_returns": full.get("net_returns") or [],
    }


def _because(report: dict) -> str:
    oos = report["oos"] or {}
    held = report["holdout"] or {}
    return (
        f"开发样本 {report['development_start']} 至 {report['development_end']}，"
        f"Rank IC 的 Newey–West t 为 {_num((report['development'] or {}).get('ic', {}).get('nw_t'))}，"
        f"毛多空年化为 {_pct(((report['development'] or {}).get('gross') or {}).get('ann_return'))}。"
        f"滚动测试完整成本后的 Newey–West t 为 {_num(oos.get('net_nw_t'))}，"
        f"保留样本完整成本后的年化为 {_pct(held.get('net_ann'))}。"
        f"Holm p 为 {_num(report['holm_p'])}。"
    )


def _num(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            number = value.item()
            if isinstance(number, float) and not math.isfinite(number):
                return None
            return number
    except ImportError:
        pass
    return value


def _thick_dates(panel: pl.DataFrame, min_names: int = 12) -> list:
    counts = panel.group_by("date").len().filter(pl.col("len") >= min_names).sort("date")
    return counts["date"].to_list()


def build(refresh: bool = False) -> Path:
    if refresh or not (RELEASE / "prices.parquet").is_file():
        fetch_and_store()
    elif (RELEASE / "provenance.json").is_file():
        refresh_fundamentals_from_cache()
    prices = pl.read_parquet(RELEASE / "prices.parquet")
    funds = pl.read_parquet(RELEASE / "fundamentals.parquet")
    panel = build_factor_panel(prices, funds)
    kept_dates = _thick_dates(panel, 12)
    panel = panel.filter(pl.col("date").is_in(kept_dates))
    for spec in FACTORS:
        panel = neutralize_factor_frame(panel, spec.name, out_col=spec.neu_name)
    reports = evaluate_universe(panel)
    provenance = json.loads((RELEASE / "provenance.json").read_text(encoding="utf-8"))
    payload = {
        "mode": "cn_pit",
        "ready": True,
        "generated_at": provenance["fetched_at"],
        "universe": {
            "start": str(panel["date"].min()),
            "end": str(panel["date"].max()),
            "n_stocks": panel["ticker"].n_unique(),
            "n_industries": panel["industry"].n_unique(),
            "n_dates": panel["date"].n_unique(),
            "note": (
                provenance["universe_rule"]
                + " 回测只保留至少有 12 只股票的月末，因此退市潮之后的薄截面没有进入滚动测试和保留样本。"
            ),
            "min_names": 12,
        },
        "provenance": provenance,
        "methods": {
            "grouping": "每个月末按中性化因子序位均分 5 组，组 1 最低，组内等权。并列用稳定排序。",
            "neutralization": "市值平方根加权最小二乘：截距、对数市值，以及当月至少有 2 只股票的行业哑元（丢掉一个基准行业）。只有一只股票的行业无法估计行业效应，这些股票只去掉市值。回归前按截面 1% 与 99% 缩尾。截面样本数少于参数个数加 5 时，该月残差为空。",
            "missing": "公告日缺失的报告整期丢弃，不用报告期期末代替。折旧在一季报或三季报缺失时，应计因子沿用最近一份已经公告且能够计算的报表。因子算不出来的股票当月不参与排序。未上市或已退市的股票不在当月行情中。",
            "oos": "最后 12 个月是保留样本。之前的月份上用 24 个月训练、随后 6 个月测试，测试月不与训练月重叠。滚动只用于报告样本外表现，不用来挑选因子。",
            "multiple_testing": "8 个因子的滚动测试 Newey–West t 做 Holm 校正。",
            "costs": "完整成本 = 单边佣金 3bp + 半点差 5bp + 冲击 10bp，买卖各付一次，另加空头每月 30bp 借券费。",
        },
        "summary": [_jsonable(report) for report in reports],
        "brinson": {},
        "barra": {"factors": [], "r2": None},
    }
    record = write_trace(
        RELEASE,
        panel,
        meta={"seed": None, "embed_premium": None},
        generated_at=datetime.now().isoformat(timespec="seconds"),
        mode="cn_pit",
        ic_min_names=12,
        book_min_names=12,
        estimate_brinson=False,
    )
    payload["run"] = record
    panel.write_parquet(RELEASE / "factor_panel.parquet")
    (RELEASE / "study.json").write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return RELEASE / "study.json"


if __name__ == "__main__":
    print(build())
