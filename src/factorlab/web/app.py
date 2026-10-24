"""读取一次分析结果，用网页浏览因子、归因和风险。"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
from pathlib import Path

import polars as pl
from flask import Flask, Response, abort, render_template, request, send_from_directory

from factorlab.eval.backtest import quantile_backtest
from factorlab.eval.metrics import performance_stats
from factorlab.report.formatters import num, pct
from factorlab.research.costs import PRESETS, annualized_net, flat_side, month_cost_parts, net_returns
from factorlab.research.ledger import write_trace
from factorlab.research.study import long_short_exposure
from factorlab.web.svg import level_paths, nav_paths
from factorlab.schema import FACTORS
from factorlab.web.explain import (
    DEMO_NOTICE,
    CN_NOTICE,
    SUPPORT_RULE,
    _scale_bars,
    attach_group_reads,
    industry_tilt,
    scoreboard,
    verdict,
)

_BY_LABEL = {spec.label: spec for spec in FACTORS}
_BY_NAME = {spec.name: spec for spec in FACTORS}


def create_app(out_dir: str | Path) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
    )
    out = Path(out_dir).resolve()
    app.config["OUT"] = out

    @app.context_processor
    def inject_notices():
        notice = DEMO_NOTICE
        title = "合成数据演示。"
        study = out / "study.json"
        if study.is_file():
            try:
                mode = json.loads(study.read_text(encoding="utf-8")).get("mode")
            except (OSError, json.JSONDecodeError):
                mode = None
            if mode == "cn_pit":
                notice = CN_NOTICE
                title = "真实数据研究台。"
        brand = "合成数据上的计算流程演示"
        if title.startswith("真实"):
            brand = "真实点时数据上的研究流程"
        return {
            "demo_notice": notice,
            "banner_title": title,
            "support_rule": SUPPORT_RULE,
            "brand_note": brand,
        }

    @app.route("/")
    def home():
        payload = _load(out)
        return render_template("index.html", page="home", payload=payload, pct=pct, num=num)

    @app.route("/factor/<name>")
    def factor(name: str):
        spec = _BY_NAME.get(name)
        if spec is None:
            abort(404)
        payload = _load(out)
        row = next((item for item in payload["summary"] if item["label"] == spec.label), None)
        ortho = next(
            (item for item in (payload.get("orthogonality") or []) if item["label"] == spec.label),
            None,
        )
        charts = {
            "ic": _chart(out, f"{spec.name}_ic.png"),
            "decay": _chart(out, f"{spec.name}_decay.png"),
            "nav5": _chart(out, f"{spec.name}_nav_5.png"),
            "ls5": _chart(out, f"{spec.name}_ls_5.png"),
            "nav10": _chart(out, f"{spec.name}_nav_10.png"),
            "ls10": _chart(out, f"{spec.name}_ls_10.png"),
        }
        start = request.args.get("start", "")
        end = request.args.get("end", "")
        cost_name = request.args.get("cost", "full")
        bps_raw = request.args.get("bps", "").strip()
        if cost_name not in PRESETS:
            cost_name = "full"
        date_error = _date_error(start, end)
        if date_error:
            start, end = "", ""
        bps_error, preset = _cost_preset(cost_name, bps_raw)
        if row is not None and not row.get("gross_returns"):
            _fill_series(out, row, spec, _book_min_names(payload))
        curve = _browse_curve(row, start, end, preset, out, spec.neu_name, spec.direction)
        sensitivity = _sensitivity(row)
        cost_drag = _cost_drag(curve.get("turnovers") or [], preset) if curve else {}
        return render_template(
            "factor.html",
            page="factor",
            payload=payload,
            spec=spec,
            row=row,
            ortho=ortho,
            charts=charts,
            curve=curve,
            cost_name=cost_name,
            start=start,
            end=end,
            bps=bps_raw,
            bps_error=bps_error,
            date_error=date_error,
            sensitivity=sensitivity,
            cost_drag=cost_drag,
            presets=PRESETS,
            pct=pct,
            num=num,
        )

    @app.route("/brinson")
    def brinson():
        payload = _load(out)
        return render_template(
            "brinson.html",
            page="brinson",
            payload=payload,
            waterfall=_chart(out, "brinson_waterfall.png"),
            timeline=_chart(out, "brinson_timeline.png"),
            pct=pct,
            num=num,
        )

    @app.route("/barra")
    def barra():
        payload = _load(out)
        return render_template(
            "barra.html",
            page="barra",
            payload=payload,
            exposure=_chart(out, "barra_exposure.png"),
            pct=pct,
            num=num,
        )

    @app.route("/figures/<path:filename>")
    def figures(filename: str):
        if Path(filename).name != filename or not filename.endswith(".png"):
            abort(404)
        directory = out / "figures"
        if not (directory / filename).is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/download/report.html")
    def report_html():
        return _send_file(out, "factor_report.html")

    @app.route("/download/report.pdf")
    def report_pdf():
        return _send_file(out, "factor_report.pdf")

    @app.route("/download/study.json")
    def study_json():
        if not (out / "study.json").is_file():
            abort(404)
        return send_from_directory(out, "study.json", as_attachment=True, download_name="study.json")

    @app.route("/download/run.json")
    def run_json():
        _load(out)
        return _send_download(out, "run.json")

    @app.route("/download/periods_ic.csv")
    @app.route("/download/periods_long_short.csv")
    @app.route("/download/periods_weights.csv")
    @app.route("/download/periods_brinson.csv")
    def period_file():
        filename = request.path.rsplit("/", 1)[-1]
        _load(out)
        return _send_download(out, filename)

    @app.route("/download/factors.csv")
    def factors_csv():
        payload = _load(out)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        real = payload.get("mode") == "cn_pit"
        if real:
            writer.writerow(
                [
                    "因子",
                    "检验问题",
                    "结论",
                    "策略IC",
                    "普通t",
                    "Newey-West t",
                    "毛多空几何年化",
                    "完整成本后几何年化",
                    "完整成本后最大回撤",
                    "月均单边换手",
                    "五分组单调",
                    "滚动测试完整成本Newey-West t",
                    "滚动测试完整成本年化",
                    "滚动测试五分组单调",
                    "保留样本完整成本年化",
                    "保留样本Newey-West t",
                    "Holm p",
                    "规模暴露",
                ]
            )
        else:
            writer.writerow(
                [
                    "因子",
                    "检验问题",
                    "结论",
                    "策略IC",
                    "普通t",
                    "Newey-West t",
                    "毛多空几何年化",
                    "扣10bp后几何年化",
                    "月均单边换手",
                ]
            )
        for row in payload["summary"]:
            if real:
                writer.writerow(
                    [
                        row.get("label", ""),
                        row.get("question", ""),
                        row.get("verdict", ""),
                        row.get("ic", ""),
                        row.get("t", ""),
                        row.get("nw_t", ""),
                        row.get("ls_ann", ""),
                        row.get("net_full", ""),
                        row.get("net_mdd", ""),
                        row.get("turnover", ""),
                        row.get("monotone", ""),
                        row.get("oos_nw_t", ""),
                        row.get("oos_net_ann", ""),
                        row.get("oos_monotone", ""),
                        row.get("holdout_net_ann", ""),
                        row.get("holdout_nw_t", ""),
                        row.get("holm_p", ""),
                        (row.get("exposure") or {}).get("size_spread", ""),
                    ]
                )
            else:
                writer.writerow(
                    [
                        row.get("label", ""),
                        row.get("question", ""),
                        row.get("verdict", ""),
                        row.get("ic", ""),
                        row.get("t", ""),
                        row.get("nw_t", ""),
                        row.get("ls_ann", ""),
                        row.get("net_10", ""),
                        row.get("turnover", ""),
                    ]
                )
        body = "\ufeff" + buffer.getvalue()
        return Response(
            body,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=factors.csv"},
        )

    return app


_CACHE: dict[tuple, dict] = {}


def _load(out: Path) -> dict:
    path = out / "summary.json"
    panel = out / "factor_panel.parquet"
    study_path = out / "study.json"
    cache_key = (
        str(out),
        study_path.stat().st_mtime if study_path.is_file() else 0,
        path.stat().st_mtime if path.is_file() else 0,
        panel.stat().st_mtime if panel.is_file() else 0,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    if study_path.is_file():
        data = json.loads(study_path.read_text(encoding="utf-8"))
        data["ready"] = True
        data.setdefault("summary", [])
        data.setdefault("orthogonality", [])
        data.setdefault("brinson", {})
        data.setdefault("barra", {"factors": []})
        data.setdefault("universe", {})
        for row in data["summary"]:
            _scale_bars(row.get("groups") or [])
        data["scoreboard"] = scoreboard(data["summary"])
        data["brinson_read"] = _brinson_read(data.get("brinson") or {})
        data["industry_tilt"] = industry_tilt((data.get("brinson") or {}).get("groups") or [])
        data["has_pdf"] = (out / "factor_report.pdf").is_file()
        data["has_html"] = (out / "factor_report.html").is_file()
        _attach_trace(out, data)
        _CACHE[cache_key] = data
        return data
    if not path.is_file():
        empty = {
            "ready": False,
            "summary": [],
            "orthogonality": [],
            "brinson": {},
            "barra": {"factors": []},
            "universe": {},
            "generated_at": "",
            "factors": list(FACTORS),
        }
        _CACHE[cache_key] = empty
        return empty
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ready"] = True
    data.setdefault("summary", [])
    data.setdefault("orthogonality", [])
    data.setdefault("brinson", {})
    data.setdefault("barra", {"factors": []})
    data.setdefault("universe", {})
    data["factors"] = list(FACTORS)
    for row in data["summary"]:
        spec = _BY_LABEL.get(row.get("label", ""))
        row["name"] = spec.name if spec else ""
        row["question"] = spec.question if spec else row.get("label", "")
        label, tone = verdict(row)
        row["verdict"] = label
        row["verdict_tone"] = tone
        row.setdefault("because", "")
        row.setdefault("groups", [])
        row.setdefault("net", [])
        row.setdefault("net_10", None)
        row.setdefault("ls_mdd", None)
        row.setdefault("ls_ann", None)
        row.setdefault("turnover", None)
    attach_group_reads(out, data["summary"])
    data["scoreboard"] = scoreboard(data["summary"])
    data["brinson_read"] = _brinson_read(data.get("brinson") or {})
    data["industry_tilt"] = industry_tilt((data.get("brinson") or {}).get("groups") or [])
    data["has_pdf"] = (out / "factor_report.pdf").is_file()
    data["has_html"] = (out / "factor_report.html").is_file()
    _attach_trace(out, data)
    _CACHE.clear()
    _CACHE[cache_key] = data
    return data


def _brinson_read(brinson: dict) -> str:
    try:
        allocation = float(brinson["allocation"])
        selection = float(brinson["selection"])
        interaction = float(brinson["interaction"])
    except (KeyError, TypeError, ValueError):
        return ""
    pieces = {"行业配置": allocation, "选股": selection, "交互": interaction}
    main = max(pieces, key=lambda name: abs(pieces[name]))
    detail = "，".join(f"{name} {pct(value)}" for name, value in pieces.items())
    return f"累计贡献：{detail}。超额主要来自{main}。这些数字未年化，也未扣费。"


def _browse_curve(
    row: dict | None,
    start: str,
    end: str,
    preset: dict,
    out: Path,
    factor: str,
    direction: int,
) -> dict:
    if not row or not row.get("gross_returns"):
        return {}
    dates = [str(item)[:10] for item in row["dates"]]
    gross = []
    turns = []
    kept_dates = []
    for day, value, turn in zip(dates, row["gross_returns"], row["turnover_series"], strict=True):
        if start and day < start:
            continue
        if end and day > end:
            continue
        kept_dates.append(day)
        gross.append(value)
        turns.append(turn)
    net = net_returns(gross, turns, preset) if gross else []
    gross_path, net_path = nav_paths([gross, net]) if gross else ("", "")
    gross_dd = _drawdown(gross)
    net_dd = _drawdown(net)
    dd_gross, dd_net = level_paths([gross_dd, net_dd]) if gross else ("", "")
    gross_stats = performance_stats(gross) if gross else {}
    net_stats = performance_stats(net) if net else {}
    return {
        "dates": kept_dates,
        "gross": gross_path,
        "net": net_path,
        "drawdown_gross": dd_gross,
        "drawdown_net": dd_net,
        "gross_mdd": gross_stats.get("max_drawdown"),
        "net_mdd": net_stats.get("max_drawdown"),
        "turnovers": turns,
        "turnover": _mean_finite(turns),
        "exposure": _window_exposure(out, factor, direction, start, end) or row.get("exposure") or {},
        "official": not start and not end and preset["name"] == "完整",
        "n_months": len(kept_dates),
    }


def _drawdown(returns: list[float]) -> list[float]:
    level = 1.0
    peak = 1.0
    path = []
    for value in returns:
        level *= 1.0 + float(value)
        peak = max(peak, level)
        path.append(level / peak - 1.0)
    return path


def _mean_finite(values: list) -> float | None:
    kept = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            kept.append(number)
    if not kept:
        return None
    return sum(kept) / len(kept)


def _window_exposure(out: Path, factor: str, direction: int, start: str, end: str) -> dict | None:
    panel = _read_panel(out)
    if panel is None or factor not in panel.columns:
        return None
    sample = panel
    try:
        if start:
            sample = sample.filter(pl.col("date") >= dt.date.fromisoformat(start))
        if end:
            sample = sample.filter(pl.col("date") <= dt.date.fromisoformat(end))
    except ValueError:
        return None
    return long_short_exposure(sample, factor, direction)


_PANEL: dict[tuple, pl.DataFrame] = {}


def _read_panel(out: Path) -> pl.DataFrame | None:
    path = out / "factor_panel.parquet"
    if not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime)
    cached = _PANEL.get(key)
    if cached is not None:
        return cached
    frame = pl.read_parquet(path)
    _PANEL.clear()
    _PANEL[key] = frame
    return frame


def _chart(out: Path, filename: str) -> str | None:
    if (out / "figures" / filename).is_file():
        return f"/figures/{filename}"
    return None


def _send_file(out: Path, filename: str):
    if not (out / filename).is_file():
        abort(404)
    return send_from_directory(out, filename, as_attachment=False)


def _send_download(out: Path, filename: str):
    if not (out / filename).is_file():
        abort(404)
    return send_from_directory(out, filename, as_attachment=True, download_name=filename)


def _date_error(start: str, end: str) -> str:
    try:
        if start:
            dt.date.fromisoformat(start)
        if end:
            dt.date.fromisoformat(end)
    except ValueError:
        return "日期要写成 YYYY-MM-DD。"
    if start and end and start > end:
        return "起始日期不能晚于结束日期。"
    return ""


def _cost_preset(cost_name: str, bps_raw: str) -> tuple[str, dict]:
    if not bps_raw:
        return "", PRESETS[cost_name]
    try:
        value = float(bps_raw)
    except ValueError:
        return "单边成本要写成 0 到 200 之间的数字，单位是 bp。", PRESETS[cost_name]
    if not math.isfinite(value) or value < 0 or value > 200:
        return "单边成本要写成 0 到 200 之间的数字，单位是 bp。", PRESETS[cost_name]
    return "", flat_side(value)


def _book_min_names(payload: dict) -> int:
    config = (payload.get("run") or {}).get("config") or {}
    if config.get("book_min_names"):
        return int(config["book_min_names"])
    return 12 if payload.get("mode") == "cn_pit" else 20


def _fill_series(out: Path, row: dict, spec, min_names: int) -> None:
    panel = _read_panel(out)
    if panel is None or spec.neu_name not in panel.columns:
        return
    backtest = quantile_backtest(
        panel,
        spec.neu_name,
        direction=spec.direction,
        n_groups=5,
        min_names=min_names,
    )
    frame = backtest.long_short
    row["dates"] = [str(item)[:10] for item in frame["date"].to_list()]
    row["gross_returns"] = [float(item) for item in frame["ret"].to_list()]
    row["turnover_series"] = [
        None if item is None or not math.isfinite(float(item)) else float(item)
        for item in frame["turnover"].to_list()
    ]


def _sensitivity(row: dict | None) -> list[dict]:
    if not row:
        return []
    if not row.get("gross_returns"):
        return list(row.get("net") or [])
    rows = []
    for bps in (5, 10, 20):
        rows.append(
            {
                "bps": bps,
                "ann_return": annualized_net(row["gross_returns"], row["turnover_series"], flat_side(bps)),
            }
        )
    return rows


def _cost_drag(turnovers: list, preset: dict) -> dict:
    if not turnovers:
        return {}
    keys = ("commission", "half_spread", "impact", "borrow", "total")
    totals = {key: 0.0 for key in keys}
    for turn in turnovers:
        parts = month_cost_parts(turn, preset)
        for key in keys:
            totals[key] += parts[key]
    count = len(turnovers)
    return {key: totals[key] / count for key in keys}


def _missing_styles(barra: dict) -> list[str]:
    factors = barra.get("factors") or []
    if not factors:
        return []
    present = {str(row.get("name")) for row in factors}
    missing = []
    if "对数市值" not in present:
        missing.append("对数市值")
    for spec in FACTORS:
        if spec.label not in present:
            missing.append(spec.label)
    return missing


def _attach_trace(out: Path, data: dict) -> None:
    run_path = out / "run.json"
    if not run_path.is_file() and (out / "factor_panel.parquet").is_file():
        panel = pl.read_parquet(out / "factor_panel.parquet")
        mode = data.get("mode") or _infer_mode(data)
        periods = None
        estimate = mode != "cn_pit" and "composite" in panel.columns
        brinson_path = out / "brinson_periods.parquet"
        if brinson_path.is_file():
            periods = pl.read_parquet(brinson_path)
            estimate = False
        note = str((data.get("universe") or {}).get("note") or "")
        seed = _seed_from_note(note)
        ic_min = 12 if mode == "cn_pit" else 15
        book_min = 12 if mode == "cn_pit" else 20
        write_trace(
            out,
            panel,
            brinson_periods=periods,
            meta={
                "seed": seed,
                "embed_premium": True if "溢价" in note else None,
            },
            generated_at=str(data.get("generated_at") or ""),
            mode=mode,
            ic_min_names=ic_min,
            book_min_names=book_min,
            estimate_brinson=estimate,
        )
    if run_path.is_file():
        data["run"] = json.loads(run_path.read_text(encoding="utf-8"))
    data["barra_excluded"] = _missing_styles(data.get("barra") or {})


def _infer_mode(data: dict) -> str:
    note = str((data.get("universe") or {}).get("note") or "")
    if "合成" in note:
        return "synthetic"
    return "external"


def _seed_from_note(note: str) -> int | None:
    import re

    found = re.search(r"随机种子\s*(\d+)", note)
    if not found:
        return None
    return int(found.group(1))
