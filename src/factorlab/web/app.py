"""读取一次分析结果，用网页浏览因子、归因和风险。"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from flask import Flask, Response, abort, render_template, send_from_directory

from factorlab.report.formatters import num, pct
from factorlab.schema import FACTORS
from factorlab.web.explain import (
    DEMO_NOTICE,
    SUPPORT_RULE,
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
        return {"demo_notice": DEMO_NOTICE, "support_rule": SUPPORT_RULE}

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
        ortho = next((item for item in payload["orthogonality"] if item["label"] == spec.label), None)
        charts = {
            "ic": _chart(out, f"{spec.name}_ic.png"),
            "decay": _chart(out, f"{spec.name}_decay.png"),
            "nav5": _chart(out, f"{spec.name}_nav_5.png"),
            "ls5": _chart(out, f"{spec.name}_ls_5.png"),
            "nav10": _chart(out, f"{spec.name}_nav_10.png"),
            "ls10": _chart(out, f"{spec.name}_ls_10.png"),
        }
        return render_template(
            "factor.html",
            page="factor",
            payload=payload,
            spec=spec,
            row=row,
            ortho=ortho,
            charts=charts,
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

    @app.route("/download/factors.csv")
    def factors_csv():
        payload = _load(out)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
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
    cache_key = (
        str(path),
        path.stat().st_mtime if path.is_file() else 0,
        panel.stat().st_mtime if panel.is_file() else 0,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
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
    attach_group_reads(out, data["summary"])
    data["scoreboard"] = scoreboard(data["summary"])
    data["brinson_read"] = _brinson_read(data.get("brinson") or {})
    data["industry_tilt"] = industry_tilt((data.get("brinson") or {}).get("groups") or [])
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


def _chart(out: Path, filename: str) -> str | None:
    if (out / "figures" / filename).is_file():
        return f"/figures/{filename}"
    return None


def _send_file(out: Path, filename: str):
    if not (out / filename).is_file():
        abort(404)
    return send_from_directory(out, filename, as_attachment=False)
