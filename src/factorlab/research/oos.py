"""时间切分、滚动样本外和 Holm 校正。阈值不能用保留样本来挑。"""

from __future__ import annotations

import math

HOLDOUT_MONTHS = 12
TRAIN_MONTHS = 24
TEST_MONTHS = 6


def split_dates(dates: list, holdout: int = HOLDOUT_MONTHS) -> tuple[list, list]:
    ordered = list(dates)
    if len(ordered) <= holdout + TRAIN_MONTHS:
        raise ValueError(f"月末数量 {len(ordered)} 不够同时留下训练窗和 {holdout} 个月保留样本")
    return ordered[:-holdout], ordered[-holdout:]


def rolling_windows(
    dates: list,
    train: int = TRAIN_MONTHS,
    test: int = TEST_MONTHS,
) -> list[dict]:
    """在已经去掉保留样本的日期上滚动。测试月不与训练月重叠。"""

    windows = []
    start = train
    while start + test <= len(dates):
        windows.append(
            {
                "train": list(dates[start - train : start]),
                "test": list(dates[start : start + test]),
            }
        )
        start += test
    return windows


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm 逐步校正。输入顺序与输出顺序一致。"""

    indexed = [(i, float(p)) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    adjusted = [1.0] * total
    running = 0.0
    for rank, (origin, p_value) in enumerate(indexed):
        step = min(1.0, (total - rank) * p_value)
        running = max(running, step)
        adjusted[origin] = running
    return adjusted


def one_sided_p_from_t(t_stat: float, observations: int) -> float:
    """预注册方向是收益为正，所以用右尾。"""

    if observations < 3 or not math.isfinite(t_stat):
        return float("nan")
    from scipy.stats import t as student_t

    return float(1.0 - student_t.cdf(t_stat, df=observations - 1))
