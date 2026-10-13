"""把分析结果画成图并写出 HTML / PDF。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from factorlab.report.charts import (
    plot_decay,
    plot_effects,
    plot_exposures,
    plot_group_nav,
    plot_ic,
    plot_ls,
    plot_waterfall,
)
from factorlab.report.formatters import num, pct
from factorlab.report.pdf_report import write_pdf
from factorlab.schema import FACTORS

_LABELS = {spec.neu_name: spec.label for spec in FACTORS}
_LABELS.update({spec.name: spec.label for spec in FACTORS})
_LABELS["log_mcap"] = "对数市值"
_LABELS["country"] = "国家"


def render_report(result, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(result, fig_dir)
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["pct"] = pct
    env.filters["num"] = num
    html = env.get_template("template.html.j2").render(**context)
    html_path = out / "factor_report.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path = out / "factor_report.pdf"
    engine = write_pdf(context, pdf_path, html_path)
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(_public_summary(context), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "html": str(html_path),
        "pdf": str(pdf_path),
        "pdf_engine": engine,
        "summary": str(summary_path),
    }


def build_context(result, fig_dir: Path) -> dict:
    panel = result.panel
    summary_rows = []
    ortho_rows = []
    factors = []
    for item in result.factor_reports:
        spec = item["spec"]
        ic_stats = item["ic_stats"]
        bt5 = item["backtests"].get(5) or next(iter(item["backtests"].values()))
        summary_rows.append(
            {
                "label": spec.label,
                "family": spec.family,
                "ic": ic_stats["mean"],
                "ir": ic_stats["ir"],
                "t": ic_stats["t"],
                "nw_t": ic_stats["nw_t"],
                "hit": ic_stats["hit"],
                "ls_ann": bt5.summary.get("ann_return"),
                "ls_mdd": bt5.summary.get("max_drawdown"),
                "turnover": bt5.summary.get("avg_turnover"),
            }
        )
        ortho = item["ortho"]
        ortho_rows.append(
            {
                "label": spec.label,
                "size_corr": ortho["max_abs_size_corr"],
                "industry_mean": ortho["max_abs_industry_mean"],
                "n": ortho["n_dates"],
            }
        )
        factors.append(_factor_block(item, fig_dir))

    brinson = _brinson_block(result, fig_dir)
    barra = _barra_block(result, fig_dir)
    dates = panel["date"].unique().sort()
    return {
        "title": "单因子检验与业绩归因诊断报告",
        "subtitle": "财务质量与微观结构因子 · Brinson 归因 · Barra 截面风险",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "universe": {
            "start": str(dates[0])[:10],
            "end": str(dates[-1])[:10],
            "n_stocks": panel["ticker"].n_unique(),
            "n_industries": panel["industry"].n_unique(),
            "n_dates": dates.len(),
            "note": result.meta.get("note", ""),
        },
        "summary_rows": summary_rows,
        "ortho_rows": ortho_rows,
        "factors": factors,
        "brinson": brinson,
        "barra": barra,
    }


def _factor_block(item: dict, fig_dir: Path) -> dict:
    spec = item["spec"]
    stem = spec.name
    ic = item["ic"]
    decay = item["decay"]
    charts = {
        "ic": plot_ic(
            ic["date"].to_list(),
            ic["ic"].to_numpy(),
            fig_dir / f"{stem}_ic.png",
            f"{spec.label} · 中性化后策略 Rank IC",
        ),
        "ic_path": str(fig_dir / f"{stem}_ic.png"),
        "decay": plot_decay(
            decay["horizon"].to_list(),
            decay["mean"].to_list(),
            fig_dir / f"{stem}_decay.png",
            f"{spec.label} · IC 衰减",
        ),
        "decay_path": str(fig_dir / f"{stem}_decay.png"),
    }
    backtests = []
    for n_groups, bt in item["backtests"].items():
        nav_uri = ""
        ls_uri = ""
        nav_path = fig_dir / f"{stem}_nav_{n_groups}.png"
        ls_path = fig_dir / f"{stem}_ls_{n_groups}.png"
        if bt.group_nav.height:
            dates, navs = _nav_by_group(bt.group_nav)
            nav_uri = plot_group_nav(
                dates,
                navs,
                nav_path,
                f"{spec.label} · {n_groups} 分组净值",
                bt.long_group,
            )
        if bt.long_short.height:
            ls_uri = plot_ls(
                bt.long_short["date"].to_list(),
                bt.long_short["nav"].to_list(),
                ls_path,
                f"{spec.label} · 多空（第{bt.long_group}组 − 第{bt.short_group}组）",
            )
        backtests.append(
            {
                "n_groups": n_groups,
                "long_group": bt.long_group,
                "short_group": bt.short_group,
                "ann_return": bt.summary.get("ann_return"),
                "ann_vol": bt.summary.get("ann_vol"),
                "sharpe": bt.summary.get("sharpe"),
                "max_drawdown": bt.summary.get("max_drawdown"),
                "turnover": bt.summary.get("avg_turnover"),
                "groups": bt.group_summary.to_dicts(),
                "nav": nav_uri,
                "ls": ls_uri,
                "nav_path": str(nav_path) if nav_uri else "",
                "ls_path": str(ls_path) if ls_uri else "",
            }
        )
    return {
        "label": spec.label,
        "family": spec.family,
        "definition": spec.definition,
        "formula": spec.formula,
        "direction_text": spec.direction_text,
        "ic": item["ic_stats"]["mean"],
        "ir": item["ic_stats"]["ir"],
        "n": item["ic_stats"]["n"],
        "ic_raw": item["ic_raw_stats"]["mean"],
        "decay": decay.to_dicts(),
        "charts": charts,
        "backtests": backtests,
    }


def _nav_by_group(frame):
    dates: dict[int, list] = {}
    navs: dict[int, list] = {}
    for group in sorted(frame["group"].unique().to_list()):
        sub = frame.filter(pl.col("group") == group).sort("date")
        dates[int(group)] = sub["date"].to_list()
        navs[int(group)] = sub["nav"].to_list()
    return dates, navs


def _brinson_block(result, fig_dir: Path) -> dict:
    totals = result.brinson_totals
    periods = result.brinson_periods
    waterfall = plot_waterfall(
        ["配置", "选股", "交互", "超额合计"],
        [
            totals["allocation"],
            totals["selection"],
            totals["interaction"],
            totals["active_return"],
        ],
        fig_dir / "brinson_waterfall.png",
        "Brinson–Fachler 累计主动收益",
    )
    timeline = plot_effects(
        periods["date"].to_list(),
        periods["allocation"].to_list(),
        periods["selection"].to_list(),
        periods["interaction"].to_list(),
        fig_dir / "brinson_timeline.png",
        "单期配置、选股与交互",
    )
    return {
        "rule": result.meta.get("portfolio_rule", ""),
        **totals,
        "groups": result.brinson_groups.to_dicts(),
        "waterfall": waterfall,
        "timeline": timeline,
        "waterfall_path": str(fig_dir / "brinson_waterfall.png"),
        "timeline_path": str(fig_dir / "brinson_timeline.png"),
        "periods": periods.select(
            ["date", "portfolio_return", "benchmark_return", "allocation", "selection", "interaction"]
        ).to_dicts(),
    }


def _pretty(name: str) -> str:
    if name.startswith("ind::"):
        return name.split("::", 1)[1]
    return _LABELS.get(name, name)


def _barra_block(result, fig_dir: Path) -> dict:
    barra = result.barra
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
                "ann_vol": float(std * np.sqrt(12.0)) if std > 0 else float("nan"),
                "t": float(series.mean() / (std / np.sqrt(series.size))) if std > 0 else float("nan"),
            }
        )
    risk = barra.risk or {}
    exposure_uri = ""
    exposure_path = ""
    names = list(risk.get("exposure_names") or [])
    values = list(risk.get("exposure_values") or [])
    if names and values and len(names) == len(values):
        exposure_path = str(fig_dir / "barra_exposure.png")
        exposure_uri = plot_exposures(
            [_pretty(name) for name in names],
            [float(v) for v in values],
            fig_dir / "barra_exposure.png",
            "最后一期主动暴露",
        )
    r2 = barra.r_squared["r2"].drop_nulls()
    return {
        "r2": float(r2.mean()) if r2.len() else float("nan"),
        "note": str(risk.get("note") or "因子收益是多元回归系数，已经同时控制行业和其他风格。"),
        "factors": rows,
        "exposure": exposure_uri,
        "exposure_path": exposure_path,
        "total_vol": risk.get("total_vol_ann"),
        "factor_vol": risk.get("factor_vol_ann"),
        "specific_vol": risk.get("specific_vol_ann"),
        "n_periods": risk.get("n_periods"),
        "n_factors": risk.get("n_factors"),
    }


def _public_summary(context: dict) -> dict:
    def scrub(value):
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key not in {"charts", "nav", "ls", "waterfall", "timeline", "exposure"}
                and not str(key).endswith("_path")
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    payload = {
        "generated_at": context["generated_at"],
        "universe": context["universe"],
        "summary": context["summary_rows"],
        "orthogonality": context["ortho_rows"],
        "brinson": {
            key: context["brinson"][key]
            for key in (
                "portfolio_return",
                "benchmark_return",
                "active_return",
                "allocation",
                "selection",
                "interaction",
                "n_periods",
                "groups",
            )
            if key in context["brinson"]
        },
        "barra": {
            "r2": context["barra"]["r2"],
            "factors": context["barra"]["factors"],
            "total_vol": context["barra"]["total_vol"],
            "factor_vol": context["barra"]["factor_vol"],
            "specific_vol": context["barra"]["specific_vol"],
        },
    }
    return scrub(payload)
