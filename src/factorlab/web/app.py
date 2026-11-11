"""读取一次分析结果，用网页浏览因子、归因和风险。"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
import math
from pathlib import Path

import polars as pl
from flask import Flask, Response, abort, g, render_template, request, send_from_directory

from factorlab.eval.backtest import quantile_backtest
from factorlab.eval.metrics import performance_stats
from factorlab.report.formatters import num, pct
from factorlab.research.costs import PRESETS, annualized_net, flat_side, month_cost_parts, net_returns
from factorlab.research.ledger import write_trace
from factorlab.research.study import long_short_exposure
from factorlab.schema import FACTORS
from factorlab.web.browse import (
    apply_preset,
    average_rank_corr,
    group_rows,
    long_leg,
    run_book,
    slice_panel,
    strategy_ic,
    view_settings,
)
from factorlab.web.runs import diff_runs, list_runs
from factorlab.web.svg import level_overlay, lined_chart, nav_overlay
from factorlab.web.explain import (
    DEMO_NOTICE,
    CSI_NOTICE,
    CN_NOTICE,
    SUPPORT_RULE,
    _scale_bars,
    attach_group_reads,
    ic_return_conflict,
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
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.before_request
    def _bind_universe():
        items = _universe_catalog()
        g.universes = items
        g.universe_note = ""
        g.universe_id = ""
        g.universe_q = ""
        g.out = out
        if request.path in {"/health", "/robots.txt"} or request.path.startswith("/static"):
            return
        choice = request.args.get("universe", "").strip()
        known = {item["id"] for item in items}
        if choice and choice not in known:
            abort(404)
        chosen = next((item for item in items if item["id"] == choice and item["ready"]), None)
        if choice and chosen is None:
            g.universe_note = "所选样本还没有研究结果，页面仍停在当前结果。"
        if chosen is not None:
            g.out = chosen["path"].resolve()
            g.universe_id = chosen["id"]
        else:
            matched = next(
                (item for item in items if item["ready"] and item["path"].resolve() == out.resolve()),
                None,
            )
            g.universe_id = matched["id"] if matched else ""
        g.universe_q = f"?universe={g.universe_id}" if g.universe_id else ""

    @app.context_processor
    def inject_notices():
        active = getattr(g, "out", out)
        notice = DEMO_NOTICE
        title = "合成数据演示。"
        study = active / "study.json"
        mode = None
        if study.is_file():
            try:
                mode = json.loads(study.read_text(encoding="utf-8")).get("mode")
            except (OSError, json.JSONDecodeError):
                mode = None
            if mode == "csi300":
                notice = CSI_NOTICE
                title = "沪深300点时研究台。"
            elif mode == "cn_pit":
                notice = CN_NOTICE
                title = "真实数据研究台。"
        brand = "合成数据上的计算流程演示"
        if mode == "csi300":
            brand = "沪深300点时成分上的研究流程"
        elif title.startswith("真实"):
            brand = "真实点时数据上的研究流程"
        return {
            "demo_notice": notice,
            "banner_title": title,
            "support_rule": SUPPORT_RULE,
            "brand_note": brand,
            "universes": getattr(g, "universes", []),
            "universe_id": getattr(g, "universe_id", ""),
            "universe_q": getattr(g, "universe_q", ""),
            "universe_note": getattr(g, "universe_note", ""),
        }

    @app.route("/health")
    def health():
        return {"ok": True}

    @app.route("/robots.txt")
    def robots():
        return Response("User-agent: *\nAllow: /\n", mimetype="text/plain; charset=utf-8")

    @app.after_request
    def _cache_headers(response):
        path = request.path
        if path.startswith("/static") or path.startswith("/figures"):
            response.headers["Cache-Control"] = "public, max-age=86400"
        elif path == "/robots.txt":
            response.headers["Cache-Control"] = "public, max-age=86400"
        elif path.startswith("/download") or path.endswith(".csv"):
            response.headers.setdefault("Cache-Control", "private, max-age=3600")
            response.headers.setdefault("Vary", "Accept-Encoding")
        else:
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.errorhandler(404)
    def not_found(_err):
        choice = request.args.get("universe", "").strip()[:40]
        message = f"没有名为 {choice} 的样本。" if choice else "这个地址不存在。"
        return render_template("error.html", page="error", title="没有这一页", message=message), 404

    @app.errorhandler(500)
    def server_error(_err):
        return (
            render_template(
                "error.html",
                page="error",
                title="这一页没有算出来",
                message="参数或数据出了问题。结论没有被改写。",
            ),
            500,
        )

    @app.route("/")
    def home():
        payload = _load(g.out)
        return render_template("index.html", page="home", payload=payload, pct=pct, num=num)

    @app.route("/factor/<name>")
    def factor(name: str):
        spec = _BY_NAME.get(name)
        if spec is None:
            abort(404)
        payload = _load(g.out)
        row = next((item for item in payload["summary"] if item["label"] == spec.label), None)
        ortho = next(
            (item for item in (payload.get("orthogonality") or []) if item["label"] == spec.label),
            None,
        )
        charts = {
            "ic": _chart(g.out, f"{spec.name}_ic.png"),
            "decay": _chart(g.out, f"{spec.name}_decay.png"),
            "nav5": _chart(g.out, f"{spec.name}_nav_5.png"),
            "ls5": _chart(g.out, f"{spec.name}_ls_5.png"),
            "nav10": _chart(g.out, f"{spec.name}_nav_10.png"),
            "ls10": _chart(g.out, f"{spec.name}_ls_10.png"),
        }
        view = _factor_view(g.out, spec, payload, request.args)
        sensitivity = _sensitivity(row)
        return render_template(
            "factor.html",
            page="factor",
            payload=payload,
            spec=spec,
            row=row,
            ortho=ortho,
            charts=charts,
            sensitivity=sensitivity,
            presets=PRESETS,
            pct=pct,
            num=num,
            **view,
        )

    @app.route("/brinson")
    def brinson():
        payload = _load(g.out)
        return render_template(
            "brinson.html",
            page="brinson",
            payload=payload,
            waterfall=_chart(g.out, "brinson_waterfall.png"),
            timeline=_chart(g.out, "brinson_timeline.png"),
            pct=pct,
            num=num,
        )

    @app.route("/barra")
    def barra():
        payload = _load(g.out)
        return render_template(
            "barra.html",
            page="barra",
            payload=payload,
            exposure=_chart(g.out, "barra_exposure.png"),
            pct=pct,
            num=num,
        )

    @app.route("/figures/<path:filename>")
    def figures(filename: str):
        if Path(filename).name != filename or not filename.endswith(".png"):
            abort(404)
        directory = g.out / "figures"
        if not (directory / filename).is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/download/report.html")
    def report_html():
        return _send_file(g.out, "factor_report.html")

    @app.route("/download/report.pdf")
    def report_pdf():
        return _send_file(g.out, "factor_report.pdf")

    @app.route("/download/study.json")
    def study_json():
        return _send_download(g.out, "study.json")

    @app.route("/download/run.json")
    def run_json():
        _load(g.out)
        return _send_download(g.out, "run.json")

    @app.route("/download/periods_ic.csv")
    @app.route("/download/periods_long_short.csv")
    @app.route("/download/periods_weights.csv")
    @app.route("/download/periods_brinson.csv")
    def period_file():
        filename = request.path.rsplit("/", 1)[-1]
        _load(g.out)
        return _send_download(g.out, filename)

    @app.route("/download/factors.csv")
    def factors_csv():
        payload = _load(g.out)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        real = bool(payload.get("real"))
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
        return _download_response(body, "factors.csv", "text/csv; charset=utf-8")

    @app.route("/factor/<name>/series.csv")
    def factor_series(name: str):
        spec = _BY_NAME.get(name)
        if spec is None:
            abort(404)
        payload = _load(g.out)
        view = _factor_view(g.out, spec, payload, request.args)
        curve = view["curve"] or {}
        long_book = curve.get("long_book") or {}
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["date", "long_short_gross", "long_short_net", "long_gross", "equal_weight", "cap_weight", "long_net"]
        )
        gross = curve.get("gross_values") or []
        net = curve.get("net_values") or []
        dates = curve.get("dates") or []
        long_map = {day: (long_book.get("long") or [])[index] for index, day in enumerate(long_book.get("dates") or [])}
        ew_map = {day: (long_book.get("ew") or [])[index] for index, day in enumerate(long_book.get("dates") or [])}
        cw_map = {day: (long_book.get("cw") or [])[index] for index, day in enumerate(long_book.get("dates") or [])}
        long_net_map = {
            day: (long_book.get("long_net") or [])[index] for index, day in enumerate(long_book.get("dates") or [])
        }
        for index, day in enumerate(dates):
            writer.writerow(
                [
                    day,
                    gross[index] if index < len(gross) else "",
                    net[index] if index < len(net) else "",
                    long_map.get(day, ""),
                    ew_map.get(day, ""),
                    cw_map.get(day, ""),
                    long_net_map.get(day, ""),
                ]
            )
        body = "\ufeff" + buffer.getvalue()
        return _download_response(body, f"{spec.name}_series.csv", "text/csv; charset=utf-8")

    @app.route("/compare")
    def compare():
        payload = _load(g.out)
        start = request.args.get("start", "")
        end = request.args.get("end", "")
        cost_name = request.args.get("cost", "full")
        bps_raw = request.args.get("bps", "").strip()
        if cost_name not in PRESETS:
            cost_name = "full"
        first, last = _sample_bounds(payload)
        date_error = _date_error(start, end, first, last)
        bps_error, preset = _cost_preset(cost_name, bps_raw)
        groups, neu = view_settings(request.args)
        picked = [name for name in request.args.getlist("factors") if name in _BY_NAME]
        trimmed = len(picked) > 3
        picked = picked[:3]
        rows = []
        chart = None
        panel = None if date_error else _read_panel(g.out)
        if panel is not None and picked:
            respect = "tradable" in panel.columns
            sample = slice_panel(panel, start, end)
            min_names = _book_min_names(payload)
            packed = []
            for name in picked:
                spec = _BY_NAME[name]
                ran = run_book(
                    sample,
                    spec,
                    n_groups=groups,
                    neu=neu,
                    min_names=min_names,
                    respect_tradable=respect,
                )
                if ran is None:
                    continue
                backtest, column = ran
                frame = backtest.long_short
                gross = [float(item) for item in frame["ret"].to_list()]
                turns = [
                    None if item is None or not math.isfinite(float(item)) else float(item)
                    for item in frame["turnover"].to_list()
                ]
                dates = [str(item)[:10] for item in frame["date"].to_list()]
                nets, _impacts = apply_preset(
                    gross,
                    turns,
                    dates,
                    preset,
                    backtest.weights,
                    sample,
                ) if gross else ([], None)
                ic = strategy_ic(sample, spec, column, min_names)
                leg = long_leg(backtest, sample, preset)
                packed.append(
                    {
                        "spec": spec,
                        "ic": ic.get("mean"),
                        "nw_t": ic.get("nw_t"),
                        "ls_ann": performance_stats(gross)["ann_return"] if gross else None,
                        "net_ann": performance_stats(nets)["ann_return"] if nets else None,
                        "excess_net_ann": leg.get("excess_net_ann"),
                        "dates": dates,
                        "net": nets,
                    }
                )
            if packed:
                common = set(packed[0]["dates"])
                for item in packed[1:]:
                    common &= set(item["dates"])
                common_dates = [day for day in packed[0]["dates"] if day in common]
                series = []
                for item in packed:
                    mapping = dict(zip(item["dates"], item["net"], strict=True))
                    series.append([mapping[day] for day in common_dates])
                if common_dates:
                    chart = lined_chart(
                        series,
                        common_dates,
                        [item["spec"].label for item in packed],
                        kind="nav",
                    )
            rows = packed
        corr_panel = None
        if panel is not None:
            corr_panel = sample if picked else slice_panel(panel, start, end)
        corr_names = picked if len(picked) >= 2 else [spec.name for spec in FACTORS]
        corr = average_rank_corr(corr_panel, corr_names) if corr_panel is not None else []
        return render_template(
            "compare.html",
            page="compare",
            payload=payload,
            factors=list(FACTORS),
            selected=picked,
            trimmed=trimmed,
            rows=rows,
            chart=chart,
            groups=groups,
            neu=neu,
            cost_name=cost_name,
            start=start,
            end=end,
            bps=bps_raw,
            bps_error=bps_error,
            date_error=date_error,
            sample_first=first,
            sample_last=last,
            window_label="全样本" if not date_error and not start and not end and groups == 5 and neu else "当前窗口",
            corr=corr,
            presets=PRESETS,
            pct=pct,
            num=num,
        )

    @app.route("/runs")
    def runs_page():
        project = Path(__file__).resolve().parents[3]
        cards = list_runs(project)
        by_key = {item["key"]: item for item in cards}
        left_id = request.args.get("left", "")
        right_id = request.args.get("right", "")
        if left_id not in by_key:
            left_id = cards[0]["key"] if cards else ""
        if right_id not in by_key:
            right_id = cards[1]["key"] if len(cards) > 1 else ""
        left = by_key.get(left_id)
        right = by_key.get(right_id)
        diff: list[dict] = []
        same = False
        if left and right and left_id != right_id:
            diff = diff_runs(left, right)
            same = not diff
        return render_template(
            "runs.html",
            page="runs",
            runs=cards,
            left=left,
            right=right,
            left_id=left_id,
            right_id=right_id,
            diff=diff,
            same=same,
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
        _explain_missing_risk(data)
        for row in data["summary"]:
            _scale_bars(row.get("groups") or [])
        data["scoreboard"] = scoreboard(data["summary"])
        data["brinson_read"] = _brinson_read(data.get("brinson") or {})
        data.setdefault("brinson_neutral", {})
        data["brinson_neutral_read"] = _brinson_read(data.get("brinson_neutral") or {})
        data["industry_tilt"] = industry_tilt((data.get("brinson") or {}).get("groups") or [])
        data["has_pdf"] = (out / "factor_report.pdf").is_file()
        data["has_html"] = (out / "factor_report.html").is_file()
        data["real"] = data.get("mode") in {"cn_pit", "csi300", "csi500", "csi800"}
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
    data["real"] = data.get("mode") in {"cn_pit", "csi300", "csi500", "csi800"}
    _explain_missing_risk(data)
    _attach_trace(out, data)
    _CACHE.clear()
    _CACHE[cache_key] = data
    return data


def _explain_missing_risk(data: dict) -> None:
    barra = data.get("barra") or {}
    if not barra or barra.get("total_vol") is not None or barra.get("risk_note"):
        return
    if not barra.get("factors"):
        return
    n_factors = len(barra["factors"])
    n_dates = (data.get("universe") or {}).get("n_dates") or "未知"
    barra["risk_note"] = (
        f"因子收益有 {n_factors} 列，样本约 {n_dates} 个月。"
        "不少行业因子在部分月份是空的，把含空值的月份整行删掉之后，剩下的月份不够估计协方差，"
        "所以主动波动、因子部分和特质部分没有数字。"
    )
    data["barra"] = barra


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
    *,
    min_names: int = 12,
    n_groups: int = 5,
    respect_tradable: bool = False,
) -> dict:
    if not row or not row.get("gross_returns"):
        return {}
    dates = [str(item)[:10] for item in row["dates"]]
    net_all = list(row.get("net_returns") or [])
    impact_all = list(row.get("impact_series") or [])
    use_stored = (
        preset.get("name") == "完整"
        and len(net_all) == len(dates)
        and len(impact_all) == len(dates)
    )
    gross = []
    turns = []
    kept_dates = []
    kept_net = []
    kept_impact = []
    for index, (day, value, turn) in enumerate(zip(dates, row["gross_returns"], row["turnover_series"], strict=True)):
        if start and day < start:
            continue
        if end and day > end:
            continue
        kept_dates.append(day)
        gross.append(value)
        turns.append(turn)
        if use_stored:
            kept_net.append(float(net_all[index]))
            kept_impact.append(float(impact_all[index]))
    return _pack_curve(
        kept_dates,
        gross,
        turns,
        preset,
        _window_exposure(
            out,
            factor,
            direction,
            start,
            end,
            min_names=min_names,
            n_groups=n_groups,
            respect_tradable=respect_tradable,
        )
        or row.get("exposure")
        or {},
        official=not start and not end and preset["name"] == "完整",
        impacts=kept_impact if use_stored else None,
        net_override=kept_net if use_stored else None,
    )


def _pack_curve(dates, gross, turns, preset, exposure, *, official: bool, impacts=None, net_override=None) -> dict:
    if net_override is not None:
        net = [float(value) for value in net_override]
    else:
        net = net_returns(gross, turns, preset, impacts) if gross else []
    if gross:
        (gross_path, net_path), (gross_points, net_points) = nav_overlay([gross, net], dates)
        gross_dd = _drawdown(gross)
        net_dd = _drawdown(net)
        (dd_gross, dd_net), (dd_gross_points, dd_net_points) = level_overlay([gross_dd, net_dd], dates)
    else:
        gross_path = net_path = dd_gross = dd_net = ""
        gross_points = net_points = dd_gross_points = dd_net_points = []
        gross_dd = net_dd = []
    gross_stats = performance_stats(gross) if gross else {}
    net_stats = performance_stats(net) if net else {}
    nav_chart = lined_chart([gross, net], dates, ["毛净值", "净净值"], kind="nav") if gross else None
    dd_chart = lined_chart([gross_dd, net_dd], dates, ["毛回撤", "净回撤"], kind="level") if gross else None
    return {
        "dates": dates,
        "gross": gross_path,
        "net": net_path,
        "gross_points": gross_points,
        "net_points": net_points,
        "drawdown_gross": dd_gross,
        "drawdown_net": dd_net,
        "drawdown_gross_points": dd_gross_points,
        "drawdown_net_points": dd_net_points,
        "nav_chart": nav_chart,
        "dd_chart": dd_chart,
        "gross_values": gross,
        "net_values": net,
        "gross_mdd": gross_stats.get("max_drawdown"),
        "net_mdd": net_stats.get("max_drawdown"),
        "turnovers": turns,
        "turnover": _mean_finite(turns),
        "exposure": exposure,
        "official": official,
        "n_months": len(dates),
        "impacts": impacts,
        "participation": impacts is not None,
    }


def _factor_view(out: Path, spec, payload: dict, args) -> dict:
    start = args.get("start", "")
    end = args.get("end", "")
    cost_name = args.get("cost", "full")
    bps_raw = args.get("bps", "").strip()
    if cost_name not in PRESETS:
        cost_name = "full"
    first, last = _sample_bounds(payload)
    date_error = _date_error(start, end, first, last)
    bps_error, preset = _cost_preset(cost_name, bps_raw)
    groups, neu = view_settings(args)
    row = next((item for item in payload["summary"] if item["label"] == spec.label), None)
    min_names = _book_min_names(payload)
    if row is not None and not row.get("gross_returns"):
        _fill_series(out, row, spec, min_names)
    panel = None if date_error else _read_panel(out)
    respect = panel is not None and "tradable" in panel.columns
    official_settings = groups == 5 and neu
    live = None
    column = spec.neu_name if neu and spec.neu_name else spec.name
    if panel is not None:
        sample = slice_panel(panel, start, end)
        ran = run_book(
            sample,
            spec,
            n_groups=groups,
            neu=neu,
            min_names=min_names,
            respect_tradable=respect,
        )
        if ran is not None:
            backtest, column = ran
            live = {"backtest": backtest, "sample": sample, "column": column}
    if official_settings and row is not None and row.get("gross_returns"):
        curve = _browse_curve(
            row,
            start,
            end,
            preset,
            out,
            spec.neu_name,
            spec.direction,
            min_names=min_names,
            n_groups=5,
            respect_tradable=respect,
        )
    elif live is not None:
        frame = live["backtest"].long_short
        dates = [str(item)[:10] for item in frame["date"].to_list()]
        gross = [float(item) for item in frame["ret"].to_list()]
        turns = [
            None if item is None or not math.isfinite(float(item)) else float(item)
            for item in frame["turnover"].to_list()
        ]
        exposure = long_short_exposure(
            live["sample"],
            column,
            spec.direction,
            min_names=min_names,
            n_groups=groups,
            respect_tradable=respect,
        )
        _nets, impacts = apply_preset(
            gross,
            turns,
            dates,
            preset,
            live["backtest"].weights,
            live["sample"],
        )
        curve = _pack_curve(dates, gross, turns, preset, exposure, official=False, impacts=impacts)
        curve["groups"] = group_rows(live["backtest"])
    else:
        curve = {}
    if live is not None and curve:
        curve["long_book"] = long_leg(live["backtest"], live["sample"], preset)
        long_dates = curve["long_book"].get("dates") or []
        if long_dates:
            long_paths, long_points = nav_overlay(
                [curve["long_book"]["long"], curve["long_book"]["ew"], curve["long_book"]["cw"]],
                long_dates,
            )
            curve["long_book"]["paths"] = long_paths
            curve["long_book"]["points"] = long_points
            curve["long_book"]["chart"] = lined_chart(
                [curve["long_book"]["long"], curve["long_book"]["ew"], curve["long_book"]["cw"]],
                long_dates,
                ["多头", "等权基准", "市值基准"],
                colors=["#12343b", "#0f5f63", "#a24b32"],
                kind="nav",
            )
    if curve:
        curve["official"] = bool(curve.get("official") and official_settings)
        curve["custom"] = not official_settings
    return {
        "curve": curve,
        "cost_name": cost_name,
        "start": start,
        "end": end,
        "bps": bps_raw,
        "bps_error": bps_error,
        "date_error": date_error,
        "cost_drag": _cost_drag(curve.get("turnovers") or [], preset, curve.get("impacts")) if curve else {},
        "conflict": ic_return_conflict(
            row.get("ic") if row else None,
            row.get("ls_ann") if row else None,
            row.get("gross_returns") if row else None,
            row.get("groups") if row else None,
        ) if row is not None and payload.get("real") else "",
        "groups": groups,
        "neu": neu,
        "sample_first": first,
        "sample_last": last,
        "window_label": "全样本" if not date_error and not start and not end and groups == 5 and neu else "当前窗口",
    }


def _universe_catalog() -> list[dict]:
    project = Path(__file__).resolve().parents[3]
    rows = [
        ("csi300", "沪深300点时", project / "data" / "releases" / "csi300_pit_v1", "study.json"),
        ("csi500", "中证500点时", project / "data" / "releases" / "csi500_pit_v1", "study.json"),
        ("csi800", "中证800点时", project / "data" / "releases" / "csi800_pit_v1", "study.json"),
        ("cn_pit", "16只点时", project / "data" / "releases" / "cn_pit_v1", "study.json"),
        ("demo", "合成演示", project / "demo", "summary.json"),
    ]
    catalog = []
    for ident, label, path, marker in rows:
        catalog.append({"id": ident, "label": label, "path": path, "ready": (path / marker).is_file()})
    return catalog


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


def _window_exposure(
    out: Path,
    factor: str,
    direction: int,
    start: str,
    end: str,
    *,
    min_names: int = 12,
    n_groups: int = 5,
    respect_tradable: bool = False,
) -> dict | None:
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
    return long_short_exposure(
        sample,
        factor,
        direction,
        min_names=min_names,
        n_groups=n_groups,
        respect_tradable=respect_tradable,
    )


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


_MIME = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".pdf": "application/pdf",
}


def _download_response(body: bytes | str, filename: str, mimetype: str, *, attachment: bool = True) -> Response:
    raw = body.encode("utf-8") if isinstance(body, str) else body
    headers = {
        "Cache-Control": "private, max-age=3600",
        "Vary": "Accept-Encoding",
    }
    if attachment:
        headers["Content-Disposition"] = f"attachment; filename={filename}"
    accept = request.headers.get("Accept-Encoding", "")
    compressible = filename.endswith((".csv", ".json", ".html"))
    if compressible and "gzip" in accept.lower() and len(raw) >= 2048:
        raw = gzip.compress(raw, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
    return Response(raw, mimetype=mimetype, headers=headers)


def _send_file(out: Path, filename: str):
    return _send_named(out, filename, attachment=False)


def _send_download(out: Path, filename: str):
    return _send_named(out, filename, attachment=True)


def _send_named(out: Path, filename: str, *, attachment: bool):
    path = out / filename
    if not path.is_file():
        abort(404)
    if path.suffix.lower() == ".pdf":
        response = send_from_directory(
            out,
            filename,
            as_attachment=attachment,
            download_name=filename if attachment else None,
        )
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return _download_response(path.read_bytes(), filename, mime, attachment=attachment)


def _sample_bounds(payload: dict) -> tuple[str, str]:
    universe = payload.get("universe") or {}
    return str(universe.get("start") or "")[:10], str(universe.get("end") or "")[:10]


def _date_error(start: str, end: str, first: str = "", last: str = "") -> str:
    try:
        if start:
            dt.date.fromisoformat(start)
        if end:
            dt.date.fromisoformat(end)
    except ValueError:
        return "日期要写成 YYYY-MM-DD。"
    if start and end and start > end:
        return "起始日期不能晚于结束日期。"
    if last and start and start > last:
        return f"起始日期晚于样本最后一天 {last}。"
    if first and end and end < first:
        return f"结束日期早于样本第一天 {first}。"
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
    if payload.get("mode") == "csi300":
        return int(config.get("book_min_names") or 80)
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


def _cost_drag(turnovers: list, preset: dict, impacts=None) -> dict:
    if not turnovers:
        return {}
    keys = ("commission", "half_spread", "impact", "borrow", "total")
    totals = {key: 0.0 for key in keys}
    for index, turn in enumerate(turnovers):
        parts = month_cost_parts(turn, preset)
        if impacts is not None and index < len(impacts):
            extra = impacts[index]
            extra = 0.0 if extra is None or not math.isfinite(float(extra)) else float(extra)
            parts["total"] = parts["total"] - parts["impact"] + extra
            parts["impact"] = extra
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
        if mode == "csi300":
            ic_min, book_min = 80, 80
        elif mode == "cn_pit":
            ic_min, book_min = 12, 12
        else:
            ic_min, book_min = 15, 20
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
