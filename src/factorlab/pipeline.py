"""从行情和财务到因子报告的一条链路。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from factorlab.attribution.book import attribute_portfolio
from factorlab.eval.backtest import QuantileBacktest, quantile_backtest
from factorlab.eval.ic import ic_decay, rank_ic_series
from factorlab.eval.metrics import summarize_series
from factorlab.eval.portfolio import add_composite_score, book_weights
from factorlab.factors.panel import build_factor_panel
from factorlab.neutralize.residual import neutralize_factor_frame, orthogonality_diagnostics
from factorlab.risk.barra import BarraResult, estimate_barra
from factorlab.schema import FACTORS, WEIGHT_COLUMNS, FactorSpec, apply_aliases, require_columns


@dataclass
class AnalysisResult:
    panel: pl.DataFrame
    factor_reports: list[dict]
    brinson_periods: pl.DataFrame
    brinson_groups: pl.DataFrame
    brinson_totals: dict[str, float]
    barra: BarraResult
    portfolio: pl.DataFrame
    benchmark: pl.DataFrame
    meta: dict
    artifacts: dict[str, str] = field(default_factory=dict)


def run_analysis(
    prices: pl.DataFrame,
    fundamentals: pl.DataFrame,
    *,
    portfolio_weights: pl.DataFrame | None = None,
    benchmark_weights: pl.DataFrame | None = None,
    n_groups: tuple[int, ...] = (5, 10),
    out_dir: str | Path | None = None,
    make_report: bool = True,
    meta: dict | None = None,
    progress=print,
) -> AnalysisResult:
    """跑通因子构造、中性化、IC、分组回测、Brinson 与 Barra。"""

    info = dict(meta or {})
    _say(progress, "构造月末因子面板")
    panel = build_factor_panel(prices, fundamentals)
    _say(progress, "行业与市值中性化")
    factor_reports: list[dict] = []
    ortho_by_name: dict[str, dict[str, float]] = {}
    for spec in FACTORS:
        panel = neutralize_factor_frame(panel, spec.name, out_col=spec.neu_name)
        ortho = orthogonality_diagnostics(panel, spec.neu_name)
        _guard_orthogonality(spec, ortho)
        ortho_by_name[spec.name] = ortho
    panel = add_composite_score(panel)
    panel = panel.with_columns(pl.col("mktcap").log().alias("log_mcap"))

    for spec in FACTORS:
        _say(progress, f"评估 {spec.label}")
        signed = panel.with_columns((pl.col(spec.neu_name) * spec.direction).alias("_signed"))
        raw = panel.with_columns((pl.col(spec.name) * spec.direction).alias("_raw_signed"))
        ic = rank_ic_series(signed, "_signed")
        ic_raw = rank_ic_series(raw, "_raw_signed")
        backtests: dict[int, QuantileBacktest] = {}
        for groups in n_groups:
            backtests[groups] = quantile_backtest(
                panel,
                spec.neu_name,
                direction=spec.direction,
                n_groups=groups,
            )
        factor_reports.append(
            {
                "spec": spec,
                "ic": ic,
                "ic_stats": summarize_series(ic["ic"].to_numpy(), nw_lags=1),
                "ic_raw_stats": summarize_series(ic_raw["ic"].to_numpy(), nw_lags=1),
                "decay": ic_decay(panel, spec.neu_name, direction=spec.direction),
                "ortho": ortho_by_name[spec.name],
                "backtests": backtests,
            }
        )

    _say(progress, "构造组合并做 Brinson 归因")
    built_port, built_bench = book_weights(panel)
    portfolio = _as_weights(portfolio_weights) if portfolio_weights is not None else built_port
    benchmark = _as_weights(benchmark_weights) if benchmark_weights is not None else built_bench
    if portfolio_weights is None:
        info.setdefault(
            "portfolio_rule",
            "组合为复合得分最高的五分位等权。复合得分是各中性化因子截面 z-score 按研究方向等权平均，"
            "至少覆盖一半因子。基准是当月有下期收益的股票按市值加权。分组键为行业。",
        )
    else:
        info.setdefault("portfolio_rule", "组合权重由调用方提供；未提供基准时使用市值加权基准。")
    periods, groups, totals = attribute_portfolio(panel, portfolio, benchmark)

    _say(progress, "估计 Barra 截面因子收益")
    style_cols = ["log_mcap", *[spec.neu_name for spec in FACTORS]]
    barra = estimate_barra(
        panel,
        style_cols,
        portfolio_weights=portfolio,
        benchmark_weights=benchmark,
    )
    result = AnalysisResult(
        panel=panel,
        factor_reports=factor_reports,
        brinson_periods=periods,
        brinson_groups=groups,
        brinson_totals=totals,
        barra=barra,
        portfolio=portfolio,
        benchmark=benchmark,
        meta=info,
    )
    if out_dir is not None:
        result.artifacts.update(_write_tables(result, Path(out_dir)))
        if make_report:
            _say(progress, "生成 HTML / PDF 报告")
            from factorlab.report.render import render_report

            result.artifacts.update(render_report(result, out_dir))
    return result


def _say(progress, message: str) -> None:
    if progress is not None:
        progress(message)


def _guard_orthogonality(spec: FactorSpec, ortho: dict[str, float]) -> None:
    size = ortho["max_abs_size_corr"]
    industry = ortho["max_abs_industry_mean"]
    if np.isfinite(size) and size > 1e-5:
        raise RuntimeError(f"{spec.label} 中性化后仍与市值相关，最大加权相关 {size:.3e}")
    if np.isfinite(industry) and abs(industry) > 1e-5:
        raise RuntimeError(
            f"{spec.label} 中性化后行业加权均值未消去，最大绝对值 {industry:.3e}"
        )


def _as_weights(df: pl.DataFrame) -> pl.DataFrame:
    out = apply_aliases(df)
    require_columns(out, WEIGHT_COLUMNS, "权重")
    if out["date"].dtype == pl.Datetime:
        out = out.with_columns(pl.col("date").dt.date().alias("date"))
    else:
        out = out.with_columns(pl.col("date").cast(pl.Date))
    return out.with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("weight").cast(pl.Float64),
    )


def _write_tables(result: AnalysisResult, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": out_dir / "factor_panel.parquet",
        "brinson_periods": out_dir / "brinson_periods.parquet",
        "brinson_groups": out_dir / "brinson_groups.parquet",
        "barra_returns": out_dir / "barra_factor_returns.parquet",
        "portfolio": out_dir / "portfolio_weights.parquet",
        "benchmark": out_dir / "benchmark_weights.parquet",
    }
    result.panel.write_parquet(paths["panel"])
    result.brinson_periods.write_parquet(paths["brinson_periods"])
    result.brinson_groups.write_parquet(paths["brinson_groups"])
    result.barra.factor_returns.write_parquet(paths["barra_returns"])
    result.portfolio.write_parquet(paths["portfolio"])
    result.benchmark.write_parquet(paths["benchmark"])
    return {key: str(path) for key, path in paths.items()}
