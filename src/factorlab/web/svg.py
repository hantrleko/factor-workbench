"""把月度收益画成不依赖图片文件的净值曲线。"""

from __future__ import annotations


def nav_paths(series: list[list[float]], width: int = 640, height: int = 180, pad: int = 16) -> list[str]:
    paths, _markers = nav_overlay(series, width=width, height=height, pad=pad)
    return paths


def nav_overlay(
    series: list[list[float]],
    labels: list[str] | None = None,
    width: int = 640,
    height: int = 180,
    pad: int = 16,
) -> tuple[list[str], list[list[dict]]]:
    wealths = []
    for values in series:
        level = 1.0
        wealth = []
        for value in values:
            level *= 1.0 + float(value)
            wealth.append(level)
        wealths.append(wealth)
    if not any(wealths):
        return ["" for _ in series], [[] for _ in series]
    low = min(min(item) for item in wealths if item)
    high = max(max(item) for item in wealths if item)
    low = min(low, 1.0)
    high = max(high, 1.0)
    span = high - low or 1.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    paths = []
    markers = []
    for wealth in wealths:
        coords = []
        points = []
        for index, point in enumerate(wealth):
            x = pad if len(wealth) == 1 else pad + usable_w * index / (len(wealth) - 1)
            y = pad + usable_h * (1 - (point - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
            label = ""
            if labels and index < len(labels):
                label = str(labels[index])[:10]
            title = f"{label} 净值 {point:.3f}".strip()
            points.append({"x": f"{x:.1f}", "y": f"{y:.1f}", "title": title})
        paths.append(" ".join(coords))
        markers.append(points)
    return paths, markers


def level_paths(
    series: list[list[float]],
    *,
    width: int = 640,
    height: int = 140,
    pad: int = 16,
    include_zero: bool = True,
) -> list[str]:
    """把已经算好的水平值（例如回撤）画成折线。"""

    paths, _markers = level_overlay(
        series,
        width=width,
        height=height,
        pad=pad,
        include_zero=include_zero,
    )
    return paths


_PALETTE = ["#12343b", "#a24b32", "#0f5f63"]


def lined_chart(
    series: list[list[float]],
    labels: list[str] | None,
    names: list[str],
    *,
    kind: str = "nav",
    colors: list[str] | None = None,
    width: int = 720,
    height: int = 248,
) -> dict:
    """带纵轴、日期刻度、图例和悬停数值的折线。kind=nav 把收益复利成净值。"""

    palette = colors or _PALETTE
    pad_l, pad_r, pad_t, pad_b = 64, 18, 36, 36
    x0, x1 = pad_l, width - pad_r
    y0, y1 = pad_t, height - pad_b
    empty = {
        "width": width,
        "height": height,
        "series": [],
        "x_ticks": [],
        "y_ticks": [],
        "legend": [],
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }
    plotted: list[list[float]] = []
    for values in series:
        if kind == "nav":
            level = 1.0
            wealth: list[float] = []
            for value in values:
                level *= 1.0 + float(value)
                wealth.append(level)
            plotted.append(wealth)
        else:
            plotted.append([float(value) for value in values])
    usable = [item for item in plotted if item]
    if not usable:
        return empty
    low = min(min(item) for item in usable)
    high = max(max(item) for item in usable)
    if kind == "nav":
        low, high = min(low, 1.0), max(high, 1.0)
    else:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low or 1.0
    usable_w = x1 - x0
    usable_h = y1 - y0

    def at(index: int, point: float, count: int) -> tuple[float, float]:
        x = x0 if count <= 1 else x0 + usable_w * index / (count - 1)
        y = y0 + usable_h * (1.0 - (point - low) / span)
        return x, y

    out_series = []
    legend = []
    legend_x = float(x0)
    for index, values in enumerate(plotted):
        color = palette[index % len(palette)]
        name = names[index] if index < len(names) else f"序列 {index + 1}"
        count = len(values)
        coords = []
        points = []
        for step, point in enumerate(values):
            x, y = at(step, point, count)
            coords.append(f"{x:.1f},{y:.1f}")
            label = str(labels[step])[:10] if labels and step < len(labels) else ""
            if kind == "nav":
                title = f"{name} {label} 净值 {point:.3f}".strip()
            else:
                title = f"{name} {label} {point:.2%}".strip()
            points.append({"x": f"{x:.1f}", "y": f"{y:.1f}", "title": title})
        out_series.append({"name": name, "color": color, "path": " ".join(coords), "points": points})
        legend.append({"x": int(round(legend_x)), "name": name, "color": color})
        legend_x += 22 + 13 * len(name)

    y_ticks = []
    for step in range(4):
        value = low + span * step / 3.0
        _x, y = at(0, value, 2)
        label = f"{value:.2f}" if kind == "nav" else f"{value:.0%}"
        y_ticks.append({"y": f"{y:.1f}", "label": label})

    n_ref = max(len(item) for item in usable)
    indexes = [0] if n_ref <= 1 else sorted({0, n_ref // 3, (2 * n_ref) // 3, n_ref - 1})
    x_ticks = []
    seen: set[str] = set()
    for position, index in enumerate(indexes):
        if not labels or index >= len(labels):
            continue
        label = str(labels[index])[:7]
        if not label or label in seen:
            continue
        seen.add(label)
        x, _y = at(index, low, n_ref)
        if position == 0:
            anchor = "start"
        elif position == len(indexes) - 1:
            anchor = "end"
        else:
            anchor = "middle"
        x_ticks.append({"x": f"{x:.1f}", "label": label, "anchor": anchor})
    return {
        "width": width,
        "height": height,
        "series": out_series,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "legend": legend,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def level_overlay(
    series: list[list[float]],
    labels: list[str] | None = None,
    *,
    width: int = 640,
    height: int = 140,
    pad: int = 16,
    include_zero: bool = True,
) -> tuple[list[str], list[list[dict]]]:
    usable = [item for item in series if item]
    if not usable:
        return ["" for _ in series], [[] for _ in series]
    low = min(min(item) for item in usable)
    high = max(max(item) for item in usable)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low or 1.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    paths = []
    markers = []
    for values in series:
        coords = []
        points = []
        for index, point in enumerate(values):
            x = pad if len(values) <= 1 else pad + usable_w * index / (len(values) - 1)
            y = pad + usable_h * (1 - (float(point) - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
            label = ""
            if labels and index < len(labels):
                label = str(labels[index])[:10]
            points.append({"x": f"{x:.1f}", "y": f"{y:.1f}", "title": f"{label} {float(point):.2%}".strip()})
        paths.append(" ".join(coords))
        markers.append(points)
    return paths, markers
