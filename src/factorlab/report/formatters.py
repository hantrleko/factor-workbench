"""报告里的数字格式。"""

from __future__ import annotations

import math


def _finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def pct(value: object, digits: int = 2) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: object, digits: int = 3) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def plain_date(value: object) -> str:
    if value is None:
        return "—"
    return str(value)[:10]
