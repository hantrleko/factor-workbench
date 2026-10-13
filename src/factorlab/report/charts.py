"""诊断报告用的静态图。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_FONT_READY = False


def _setup_font() -> None:
    global _FONT_READY
    if not _FONT_READY:
        for candidate in (
            Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ):
            if candidate.exists():
                font_manager.fontManager.addfont(str(candidate))
        _FONT_READY = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c2b6")
    ax.spines["bottom"].set_color("#c9c2b6")
    ax.tick_params(colors="#3d4450", labelsize=8)
    ax.grid(axis="y", color="#e4dfd6", linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_ic(dates: list, values: np.ndarray, path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.1), facecolor="#fbfaf7")
    _style(ax)
    x = np.asarray(values, dtype=float)
    ax.axhline(0.0, color="#8a8175", linewidth=0.8)
    ax.plot(dates, x, color="#0f5f63", linewidth=1.3)
    mean = np.nanmean(x)
    if np.isfinite(mean):
        ax.axhline(mean, color="#a24b32", linewidth=1.0, linestyle="--", label=f"均值 {mean:.3f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_ylabel("Rank IC")
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_decay(horizons: list[int], values: list[float], path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.1), facecolor="#fbfaf7")
    _style(ax)
    colors = ["#0f5f63" if (np.isfinite(v) and v >= 0) else "#a24b32" for v in values]
    ax.bar([str(h) for h in horizons], values, color=colors, width=0.72)
    ax.axhline(0.0, color="#8a8175", linewidth=0.8)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_xlabel("远期月数")
    ax.set_ylabel("策略 Rank IC")
    return _finish(fig, path)


def plot_group_nav(
    frame_dates: dict[int, list],
    frame_nav: dict[int, list],
    path: Path,
    title: str,
    long_group: int,
) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.3), facecolor="#fbfaf7")
    _style(ax)
    groups = sorted(frame_nav)
    colors = plt.cm.GnBu(np.linspace(0.3, 0.95, len(groups)))
    for color, group in zip(colors, groups, strict=True):
        width = 2.0 if group == long_group else 1.1
        label = f"第{group}组" + (" · 多头" if group == long_group else "")
        ax.plot(frame_dates[group], frame_nav[group], color=color, linewidth=width, label=label)
    ax.axhline(1.0, color="#8a8175", linewidth=0.7)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_ylabel("净值")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_ls(dates: list, nav: list, path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.1), facecolor="#fbfaf7")
    _style(ax)
    ax.plot(dates, nav, color="#12343b", linewidth=1.6)
    ax.axhline(1.0, color="#8a8175", linewidth=0.7)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_ylabel("多空净值")
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_waterfall(labels: list[str], values: list[float], path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.3), facecolor="#fbfaf7")
    _style(ax)
    running = 0.0
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        if i == len(values) - 1:
            bottom = 0.0
            height = value
            color = "#12343b"
        else:
            bottom = running
            height = value
            color = "#2f7d6d" if value >= 0 else "#a24b32"
            running += value
        ax.bar(label, height, bottom=bottom, color=color, width=0.62)
    ax.axhline(0.0, color="#8a8175", linewidth=0.8)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_ylabel("累计主动收益")
    return _finish(fig, path)


def plot_effects(dates: list, alloc: list, sel: list, inter: list, path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 3.3), facecolor="#fbfaf7")
    _style(ax)
    ax.plot(dates, alloc, color="#0f5f63", linewidth=1.2, label="配置")
    ax.plot(dates, sel, color="#c47b2b", linewidth=1.2, label="选股")
    ax.plot(dates, inter, color="#6d645b", linewidth=1.0, label="交互")
    ax.axhline(0.0, color="#8a8175", linewidth=0.7)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_ylabel("单期效应")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_exposures(labels: list[str], values: list[float], path: Path, title: str) -> str:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.28 * len(labels) + 1.2)), facecolor="#fbfaf7")
    _style(ax)
    order = np.argsort(np.abs(values))
    labels_s = [labels[i] for i in order]
    values_s = [values[i] for i in order]
    colors = ["#2f7d6d" if v >= 0 else "#a24b32" for v in values_s]
    ax.barh(labels_s, values_s, color=colors, height=0.7)
    ax.axvline(0.0, color="#8a8175", linewidth=0.8)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
    ax.set_xlabel("主动暴露")
    return _finish(fig, path)
