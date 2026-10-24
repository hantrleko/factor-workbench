"""把一次检验拆成可以逐期核对的 CSV，并写下运行配置。"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import polars as pl

from factorlab.eval.backtest import quantile_backtest
from factorlab.eval.ic import rank_ic_series
from factorlab.research.costs import FULL_COST, flat_side, month_cost_parts
from factorlab.research.lineage import manifest, run_config
from factorlab.schema import FACTORS


def write_trace(
    out_dir: str | Path,
    panel: pl.DataFrame,
    *,
    brinson_periods: pl.DataFrame | None = None,
    meta: dict | None = None,
    generated_at: str | None = None,
    mode: str = "synthetic",
    ic_min_names: int = 15,
    book_min_names: int = 20,
    estimate_brinson: bool = True,
) -> dict:
    """写出 run.json 和四份逐期表。Brinson 只在调用方已经算过，或明确要求现算时写入。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(meta or {})
    stamp = generated_at or datetime.now().isoformat(timespec="seconds")
    periods = brinson_periods
    brinson_state = "provided" if periods is not None and periods.height else "absent"
    if periods is None and estimate_brinson and "composite" in panel.columns:
        from factorlab.attribution.book import attribute_portfolio
        from factorlab.eval.portfolio import book_weights

        portfolio, benchmark = book_weights(panel)
        periods, _groups, _totals = attribute_portfolio(panel, portfolio, benchmark)
        brinson_state = "recomputed"
    dates = panel["date"].unique().sort()
    config = run_config(
        mode=mode,
        seed=info.get("seed"),
        embed_premium=info.get("embed_premium"),
        start=str(dates[0])[:10],
        end=str(dates[-1])[:10],
        n_stocks=panel["ticker"].n_unique(),
        ic_min_names=ic_min_names,
        book_min_names=book_min_names,
    )
    record = manifest(config, generated_at=stamp, brinson=brinson_state)
    books, weights = _book_and_weights(panel, book_min_names)
    _write_csv(out / "periods_ic.csv", _ic_rows(panel, ic_min_names), _IC_FIELDS)
    _write_csv(out / "periods_long_short.csv", books, _BOOK_FIELDS)
    _write_csv(out / "periods_weights.csv", weights, _WEIGHT_FIELDS)
    _write_csv(out / "periods_brinson.csv", _brinson_rows(periods), _BRINSON_FIELDS)
    (out / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


_IC_FIELDS = ["factor", "date", "ic", "n"]
_BOOK_FIELDS = [
    "factor",
    "date",
    "gross_return",
    "turnover",
    "commission",
    "half_spread",
    "impact",
    "borrow",
    "net_full",
    "net_5bp",
    "net_10bp",
    "net_20bp",
]
_WEIGHT_FIELDS = ["factor", "date", "ticker", "side", "group", "weight"]
_BRINSON_FIELDS = [
    "date",
    "portfolio_return",
    "benchmark_return",
    "active_return",
    "allocation",
    "selection",
    "interaction",
    "allocation_linked",
    "selection_linked",
    "interaction_linked",
]


def _ic_rows(panel: pl.DataFrame, min_names: int) -> list[dict]:
    rows = []
    for spec in FACTORS:
        if spec.neu_name not in panel.columns:
            continue
        signed = panel.with_columns((pl.col(spec.neu_name) * spec.direction).alias("_signed"))
        series = rank_ic_series(signed, "_signed", min_names=min_names)
        for item in series.to_dicts():
            rows.append(
                {
                    "factor": spec.name,
                    "date": _day(item["date"]),
                    "ic": _num(item["ic"]),
                    "n": item["n"],
                }
            )
    return rows


def _book_and_weights(panel: pl.DataFrame, min_names: int) -> tuple[list[dict], list[dict]]:
    books = []
    weights = []
    flat = {bps: flat_side(bps) for bps in (5, 10, 20)}
    for spec in FACTORS:
        if spec.neu_name not in panel.columns:
            continue
        backtest = quantile_backtest(
            panel,
            spec.neu_name,
            direction=spec.direction,
            n_groups=5,
            min_names=min_names,
        )
        for item in backtest.long_short.to_dicts():
            parts = month_cost_parts(item["turnover"], FULL_COST)
            gross = float(item["ret"])
            row = {
                "factor": spec.name,
                "date": _day(item["date"]),
                "gross_return": _num(gross),
                "turnover": _num(item["turnover"]),
                "commission": _num(parts["commission"]),
                "half_spread": _num(parts["half_spread"]),
                "impact": _num(parts["impact"]),
                "borrow": _num(parts["borrow"]),
                "net_full": _num(gross - parts["total"]),
            }
            for bps, preset in flat.items():
                row[f"net_{bps}bp"] = _num(gross - month_cost_parts(item["turnover"], preset)["total"])
            books.append(row)
        for item in backtest.weights.to_dicts():
            weights.append(
                {
                    "factor": spec.name,
                    "date": _day(item["date"]),
                    "ticker": item["ticker"],
                    "side": item["side"],
                    "group": item["group"],
                    "weight": _num(item["weight"]),
                }
            )
    return books, weights


def _brinson_rows(periods: pl.DataFrame | None) -> list[dict]:
    if periods is None or periods.height == 0:
        return []
    rows = []
    for item in periods.to_dicts():
        portfolio = float(item["portfolio_return"])
        benchmark = float(item["benchmark_return"])
        rows.append(
            {
                "date": _day(item["date"]),
                "portfolio_return": _num(portfolio),
                "benchmark_return": _num(benchmark),
                "active_return": _num(item.get("active_return", portfolio - benchmark)),
                "allocation": _num(item["allocation"]),
                "selection": _num(item["selection"]),
                "interaction": _num(item["interaction"]),
                "allocation_linked": _num(item.get("allocation_linked")),
                "selection_linked": _num(item.get("selection_linked")),
                "interaction_linked": _num(item.get("interaction_linked")),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _day(value) -> str:
    return str(value)[:10]


def _num(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"
