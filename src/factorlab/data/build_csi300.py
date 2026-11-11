"""在沪深 300 点时成分上重跑现有 8 个因子。"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import polars as pl

from factorlab.attribution.book import attribute_portfolio
from factorlab.data.build_study import _json_ready, _jsonable, _thick_dates
from factorlab.data.csi300_membership import load_history, restrict_to_members
from factorlab.data.csi300_pit import (
    LIMITATIONS,
    RELEASE,
    UNIVERSE_RULE,
    active_history,
    active_release,
    assemble_release,
)
from factorlab.eval.portfolio import add_composite_score, book_weights, neutral_book_weights
from factorlab.research.costs import PARTICIPATION_COST_TEXT
from factorlab.factors.panel import build_factor_panel
from factorlab.neutralize.residual import neutralize_factor_frame
from factorlab.report.render import _pretty
from factorlab.research.ledger import write_trace
from factorlab.research.study import evaluate_universe
from factorlab.risk.barra import estimate_barra
from factorlab.schema import FACTORS

MIN_NAMES = 80


def build() -> None:
    global RELEASE
    RELEASE = active_release()
    dataset = os.environ.get("FACTORLAB_DATASET", "csi300_pit_v1")
    if dataset.startswith("csi500"):
        mode = "csi500"
        pool = "中证500"
    elif dataset.startswith("csi800"):
        mode = "csi800"
        pool = "中证800"
    else:
        mode = "csi300"
        pool = "沪深300"
    rule = os.environ.get("FACTORLAB_UNIVERSE_RULE", UNIVERSE_RULE)
    if not (RELEASE / "prices.parquet").is_file() or not (RELEASE / "fundamentals.parquet").is_file():
        assemble_release()
    prices = pl.read_parquet(RELEASE / "prices.parquet")
    funds = pl.read_parquet(RELEASE / "fundamentals.parquet")
    n_prices = prices["ticker"].n_unique()
    n_funds = funds["ticker"].n_unique()
    if n_prices < 450 or n_funds < 400:
        raise RuntimeError(
            f"行情 {n_prices} 只、财务 {n_funds} 只，覆盖不够，先不要把残缺样本写成{pool}研究结果"
        )
    if prices["date"].dtype != pl.Date:
        prices = prices.with_columns(pl.col("date").cast(pl.Utf8).str.to_date())
    panel = build_factor_panel(prices, funds)
    panel = restrict_to_members(panel, load_history(active_history()))
    kept = _thick_dates(panel, MIN_NAMES)
    if len(kept) <= 36:
        raise RuntimeError(f"至少有 {MIN_NAMES} 只股票的月末只有 {len(kept)} 个，先补齐行情和财务再出结果")
    panel = panel.filter(pl.col("date").is_in(kept))
    for spec in FACTORS:
        panel = neutralize_factor_frame(panel, spec.name, out_col=spec.neu_name)
    reports = evaluate_universe(panel, min_names=MIN_NAMES, respect_tradable=True)
    brinson, barra, brinson_periods, brinson_neutral = _portfolio_blocks(panel)
    provenance = {}
    path = RELEASE / "provenance.json"
    if path.is_file():
        import json
        provenance = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "mode": mode,
        "ready": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe": {
            "start": str(panel["date"].min()),
            "end": str(panel["date"].max()),
            "n_stocks": panel["ticker"].n_unique(),
            "n_industries": panel["industry"].n_unique(),
            "n_dates": panel["date"].n_unique(),
            "note": (
                rule
                + f" 回测只保留至少 {MIN_NAMES} 只股票的月末。"
                + " 月末成交量为 0，或当日涨跌幅达到 9.5% 的股票不能新开仓；已经持有且卖不出的仓位按原权重延续。"
            ),
            "min_names": MIN_NAMES,
        },
        "provenance": provenance or {"limitations": list(LIMITATIONS), "universe_rule": rule},
        "methods": {
            "grouping": "每个月末按中性化因子序位均分 5 组，组 1 最低，组内等权。并列用稳定排序。月末成交量为 0 或涨跌幅达到 9.5% 的股票不能新开仓，卖不出的旧仓按原权重延续。",
            "neutralization": "市值平方根加权最小二乘：截距、对数市值，以及当月至少有 2 只股票的行业哑元。只有一只股票的行业只去掉市值。回归前按截面 1% 与 99% 缩尾。",
            "missing": "公告日晚于报告期才使用。折旧缺失时，应计因子沿用最近一份已经公告且能够计算的报表。因子缺失的股票当月不参与该因子排序。",
            "oos": "最后 12 个月是保留样本。之前用 24 个月训练、随后 6 个月测试，测试月不与训练月重叠。滚动只报告样本外表现，不用来挑选因子。",
            "multiple_testing": "8 个因子的滚动测试 Newey–West t 做 Holm 校正。",
            "costs": PARTICIPATION_COST_TEXT,
        },
        "summary": [_jsonable(report) for report in reports],
        "brinson": brinson,
        "brinson_neutral": brinson_neutral,
        "barra": barra,
    }
    record = write_trace(
        RELEASE,
        panel,
        brinson_periods=brinson_periods,
        meta={"seed": None, "embed_premium": None},
        generated_at=payload["generated_at"],
        mode=mode,
        ic_min_names=MIN_NAMES,
        book_min_names=MIN_NAMES,
        estimate_brinson=False,
    )
    payload["run"] = record
    panel.write_parquet(RELEASE / "factor_panel.parquet")
    (RELEASE / "study.json").write_text(
        __import__("json").dumps(_json_ready(payload), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(RELEASE / "study.json", "months", panel["date"].n_unique(), "names", panel["ticker"].n_unique())


def _portfolio_blocks(panel: pl.DataFrame) -> tuple[dict, dict, pl.DataFrame, dict]:
    scored = add_composite_score(panel).with_columns(pl.col("mktcap").log().alias("log_mcap"))
    portfolio, benchmark = book_weights(scored)
    periods, groups, totals = attribute_portfolio(scored, portfolio, benchmark)
    neutral_block: dict = {}
    neutral_port, _neutral_bench = neutral_book_weights(scored)
    if not neutral_port.is_empty():
        _periods_n, groups_n, totals_n = attribute_portfolio(scored, neutral_port, benchmark)
        neutral_block = {**totals_n, "groups": groups_n.to_dicts()}
    periods.write_parquet(RELEASE / "brinson_periods.parquet")
    barra = estimate_barra(
        scored,
        ["log_mcap", *[spec.neu_name for spec in FACTORS]],
        portfolio_weights=portfolio,
        benchmark_weights=benchmark,
    )
    rows = []
    for column in barra.factor_returns.columns:
        if column == "date":
            continue
        series = barra.factor_returns[column].drop_nulls().to_numpy()
        series = series[np.isfinite(series)]
        if series.size < 3:
            continue
        std = float(series.std(ddof=1))
        rows.append(
            {
                "name": _pretty(column),
                "ann_ret": float(series.mean() * 12.0),
                "ann_vol": float(std * np.sqrt(12.0)) if std > 0 else None,
                "t": float(series.mean() / (std / np.sqrt(series.size))) if std > 0 else None,
            }
        )
    r2 = barra.r_squared["r2"].drop_nulls()
    risk = barra.risk or {}
    brinson = {
        **totals,
        "groups": groups.to_dicts(),
    }
    summary = {
        "r2": float(r2.mean()) if r2.len() else None,
        "factors": rows,
        "total_vol": risk.get("total_vol_ann"),
        "factor_vol": risk.get("factor_vol_ann"),
        "specific_vol": risk.get("specific_vol_ann"),
        "risk_note": risk.get("note") or "",
    }
    return brinson, summary, periods, neutral_block


if __name__ == "__main__":
    build()
