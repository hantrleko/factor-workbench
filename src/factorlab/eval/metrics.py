"""收益序列上的绩效统计与 Newey–West 标准误。"""

from __future__ import annotations

import numpy as np


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def newey_west_mean_se(values: np.ndarray, lags: int) -> float:
    """样本均值的 Newey–West 标准误（Bartlett 核）。"""

    x = finite(values)
    t_obs = x.size
    if t_obs < 3:
        return float("nan")
    centered = x - x.mean()
    gamma0 = float(np.dot(centered, centered) / t_obs)
    variance = gamma0
    max_lag = min(int(lags), t_obs - 2)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / t_obs)
        variance += 2.0 * weight * gamma
    variance = max(variance, 0.0)
    return float(np.sqrt(variance / t_obs))


def summarize_series(values: np.ndarray, nw_lags: int = 1) -> dict[str, float]:
    x = finite(values)
    t_obs = int(x.size)
    if t_obs == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "t": float("nan"),
            "nw_t": float("nan"),
            "hit": float("nan"),
        }
    mean = float(x.mean())
    if t_obs < 2:
        return {
            "n": t_obs,
            "mean": mean,
            "std": float("nan"),
            "ir": float("nan"),
            "t": float("nan"),
            "nw_t": float("nan"),
            "hit": float(np.mean(x > 0)),
        }
    std = float(x.std(ddof=1))
    ir = mean / std if std > 0 else float("nan")
    t_stat = mean / (std / np.sqrt(t_obs)) if std > 0 else float("nan")
    nw_se = newey_west_mean_se(x, nw_lags)
    nw_t = mean / nw_se if np.isfinite(nw_se) and nw_se > 0 else float("nan")
    return {
        "n": t_obs,
        "mean": mean,
        "std": std,
        "ir": float(ir),
        "t": float(t_stat),
        "nw_t": float(nw_t),
        "hit": float(np.mean(x > 0)),
    }


def performance_stats(returns: np.ndarray, periods_per_year: int = 12) -> dict[str, float]:
    """由简单收益计算净值、年化收益、波动、夏普与最大回撤。

    最大回撤从期初净值为 1 开始计算。夏普比率假设无风险利率为 0。
    """

    rets = finite(returns)
    n_obs = int(rets.size)
    empty = {
        "n": n_obs,
        "ann_return": float("nan"),
        "ann_vol": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "hit": float("nan"),
        "total_return": float("nan"),
    }
    if n_obs == 0:
        return empty
    wealth_steps = np.cumprod(1.0 + rets)
    total = float(wealth_steps[-1] - 1.0)
    if n_obs < 2 or np.any(wealth_steps <= 0):
        empty.update({"total_return": total, "hit": float(np.mean(rets > 0))})
        if n_obs >= 1 and wealth_steps[-1] > 0:
            empty["ann_return"] = float(wealth_steps[-1] ** (periods_per_year / n_obs) - 1.0)
        return empty
    ann_return = float(wealth_steps[-1] ** (periods_per_year / n_obs) - 1.0)
    std = float(rets.std(ddof=1))
    vol = float(std * np.sqrt(periods_per_year))
    sharpe = float(rets.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = np.concatenate(([1.0], wealth_steps))
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return {
        "n": n_obs,
        "ann_return": ann_return,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit": float(np.mean(rets > 0)),
        "total_return": total,
    }


def max_drawdown(returns: np.ndarray) -> float:
    return float(performance_stats(returns)["max_drawdown"])


def one_way_turnover(prev: dict[str, float], curr: dict[str, float]) -> float:
    """目标权重的单边换手：0.5 * Σ|w_t − w_{t−1}|。不计收益漂移。"""

    keys = set(prev) | set(curr)
    gross = 0.0
    for key in keys:
        gross += abs(curr.get(key, 0.0) - prev.get(key, 0.0))
    return 0.5 * gross
