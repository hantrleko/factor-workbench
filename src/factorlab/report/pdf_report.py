"""PDF 报告。优先用 WeasyPrint 渲染 HTML；缺少 Pango 时改用 fpdf2。"""

from __future__ import annotations

import logging
from pathlib import Path

from factorlab.report.formatters import num, pct

logger = logging.getLogger("factorlab")

_REGULAR = (
    Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
)
_BOLD = (
    Path(r"C:\Windows\Fonts\Noto Sans SC Bold (TrueType).otf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
)


def write_pdf(context: dict, pdf_path: Path, html_path: Path) -> str:
    if _try_weasyprint(html_path, pdf_path):
        return "weasyprint"
    _write_fpdf(context, pdf_path)
    return "fpdf2"


def _try_weasyprint(html_path: Path, pdf_path: Path) -> bool:
    try:
        from weasyprint import HTML
    except Exception:
        return False
    try:
        HTML(filename=str(html_path), encoding="utf-8").write_pdf(str(pdf_path))
    except Exception as exc:
        logger.warning("WeasyPrint 未能生成 PDF：%s", exc)
        return False
    return pdf_path.exists() and pdf_path.stat().st_size > 0


def _first_font(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("没有找到可用的中文字体，无法生成 PDF")


def _write_fpdf(context: dict, pdf_path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    regular = _first_font(_REGULAR)
    bold = _first_font(_BOLD)
    pdf.add_font("Body", "", str(regular))
    pdf.add_font("Body", "B", str(bold))
    pdf.add_page()
    pdf.set_fill_color(18, 52, 59)
    pdf.rect(0, 0, 210, 42, style="F")
    pdf.set_text_color(246, 243, 237)
    pdf.set_font("Body", "B", 16)
    pdf.set_xy(12, 12)
    pdf.cell(186, 8, context["title"])
    pdf.set_font("Body", "", 10)
    pdf.set_xy(12, 22)
    pdf.cell(186, 6, context["subtitle"])
    uni = context["universe"]
    pdf.set_xy(12, 32)
    pdf.cell(
        186,
        6,
        f"{uni['start']} 至 {uni['end']}    {uni['n_stocks']} 只股票    {uni['n_industries']} 个行业",
    )
    pdf.set_text_color(28, 36, 48)
    pdf.set_xy(12, 48)
    pdf.set_font("Body", "", 9)
    pdf.multi_cell(186, 5, uni["note"])
    pdf.ln(2)
    _heading(pdf, "汇总")
    _table(
        pdf,
        ["因子", "策略IC", "IC_IR", "t", "多空年化", "最大回撤", "换手"],
        [
            [
                row["label"],
                num(row["ic"], 3),
                num(row["ir"], 2),
                num(row["t"], 2),
                pct(row["ls_ann"]),
                pct(row["ls_mdd"]),
                pct(row["turnover"]),
            ]
            for row in context["summary_rows"]
        ],
        [36, 22, 20, 18, 28, 28, 24],
    )
    _heading(pdf, "Brinson 归因")
    br = context["brinson"]
    _table(
        pdf,
        ["组合", "基准", "主动", "配置", "选股", "交互"],
        [[
            pct(br["portfolio_return"]),
            pct(br["benchmark_return"]),
            pct(br["active_return"]),
            pct(br["allocation"]),
            pct(br["selection"]),
            pct(br["interaction"]),
        ]],
        [31, 31, 31, 31, 31, 31],
    )
    _image(pdf, br.get("waterfall_path"))
    _image(pdf, br.get("timeline_path"))
    _heading(pdf, "行业贡献（Carino 链接）")
    _table(
        pdf,
        ["行业", "组合权重", "基准权重", "配置", "选股", "交互", "合计"],
        [
            [
                row["group"],
                pct(row["wp"]),
                pct(row["wb"]),
                pct(row["allocation"]),
                pct(row["selection"]),
                pct(row["interaction"]),
                pct(row["total"]),
            ]
            for row in br["groups"]
        ],
        [32, 26, 26, 24, 24, 24, 24],
    )

    barra = context["barra"]
    _heading(pdf, "Barra 因子收益")
    pdf.set_font("Body", "", 9)
    pdf.multi_cell(186, 5, f"平均截面 R² {num(barra['r2'], 3)}。{barra['note']}")
    _table(
        pdf,
        ["因子", "年化收益", "年化波动", "t"],
        [
            [row["name"], pct(row["ann_ret"]), pct(row["ann_vol"]), num(row["t"], 2)]
            for row in barra["factors"]
        ],
        [70, 38, 38, 30],
    )
    if barra.get("exposure_path"):
        _image(pdf, barra["exposure_path"])
        pdf.set_font("Body", "", 9)
        pdf.multi_cell(
            186,
            5,
            "主动年化波动 "
            f"{pct(barra['total_vol'])}，因子 {pct(barra['factor_vol'])}，"
            f"特质 {pct(barra['specific_vol'])}。",
        )

    for factor in context["factors"]:
        pdf.add_page()
        _heading(pdf, factor["label"])
        pdf.set_font("Body", "", 9)
        pdf.multi_cell(186, 5, factor["definition"])
        pdf.multi_cell(186, 5, f"方向：{factor['direction_text']}")
        pdf.multi_cell(
            186,
            5,
            f"中性化策略 IC {num(factor['ic'], 3)}，IC_IR {num(factor['ir'], 2)}，"
            f"原始策略 IC {num(factor['ic_raw'], 3)}。",
        )
        _image(pdf, factor["charts"].get("ic_path"))
        _image(pdf, factor["charts"].get("decay_path"))
        if factor["backtests"]:
            bt = factor["backtests"][0]
            pdf.set_font("Body", "", 9)
            pdf.multi_cell(
                186,
                5,
                f"{bt['n_groups']} 分组多空年化 {pct(bt['ann_return'])}，"
                f"最大回撤 {pct(bt['max_drawdown'])}，换手 {pct(bt['turnover'])}。",
            )
            _image(pdf, bt.get("nav_path"))
            _image(pdf, bt.get("ls_path"))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def _heading(pdf, text: str) -> None:
    pdf.ln(2)
    pdf.set_text_color(15, 95, 99)
    pdf.set_font("Body", "B", 13)
    pdf.cell(186, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(28, 36, 48)


def _table(pdf, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    pdf.set_font("Body", "B", 8)
    pdf.set_fill_color(243, 239, 231)
    for header, width in zip(headers, widths, strict=True):
        pdf.cell(width, 6, header, border=0, fill=True)
    pdf.ln()
    pdf.set_font("Body", "", 8)
    for row in rows:
        if pdf.get_y() > 275:
            pdf.add_page()
        for value, width in zip(row, widths, strict=True):
            pdf.cell(width, 6, str(value), border=0)
        pdf.ln()


def _image(pdf, path: str | None) -> None:
    if not path or not Path(path).exists():
        return
    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.ln(1)
    pdf.image(path, w=186)
    pdf.ln(2)
