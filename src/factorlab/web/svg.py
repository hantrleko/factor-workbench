"""把月度收益画成不依赖图片文件的净值曲线。"""

from __future__ import annotations


def nav_paths(series: list[list[float]], width: int = 640, height: int = 180, pad: int = 16) -> list[str]:
    wealths = []
    for values in series:
        level = 1.0
        wealth = []
        for value in values:
            level *= 1.0 + float(value)
            wealth.append(level)
        wealths.append(wealth)
    if not any(wealths):
        return ["" for _ in series]
    low = min(min(item) for item in wealths if item)
    high = max(max(item) for item in wealths if item)
    low = min(low, 1.0)
    high = max(high, 1.0)
    span = high - low or 1.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    paths = []
    for wealth in wealths:
        coords = []
        for index, point in enumerate(wealth):
            x = pad if len(wealth) == 1 else pad + usable_w * index / (len(wealth) - 1)
            y = pad + usable_h * (1 - (point - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
        paths.append(" ".join(coords))
    return paths


def level_paths(
    series: list[list[float]],
    *,
    width: int = 640,
    height: int = 140,
    pad: int = 16,
    include_zero: bool = True,
) -> list[str]:
    """把已经算好的水平值（例如回撤）画成折线。"""

    usable = [item for item in series if item]
    if not usable:
        return ["" for _ in series]
    low = min(min(item) for item in usable)
    high = max(max(item) for item in usable)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low or 1.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    paths = []
    for values in series:
        coords = []
        for index, point in enumerate(values):
            x = pad if len(values) <= 1 else pad + usable_w * index / (len(values) - 1)
            y = pad + usable_h * (1 - (float(point) - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
        paths.append(" ".join(coords))
    return paths
