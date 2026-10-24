"""截面加权最小二乘残差：去掉市值与行业暴露。

每个调仓日单独回归，避免把不同日期的股票叠进同一设计矩阵：

    y = α · 1 + β · z(log 市值) + Σ_{k≠base} γ_k · 1_{行业=k} + ε

权重默认是市值的平方根（Barra 常用的 WLS 权重）。残差 ε 满足
X' W ε = 0：加权均值为 0，与市值正交，并且每个行业的加权残差和为 0。
被省略的基准行业由截距吸收，因此它的加权残差和同样为 0。
"""

from __future__ import annotations

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits


def wls_residual(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """加权最小二乘残差。

    Parameters
    ----------
    y : (n,)
    x : (n, k)
    w : (n,) 非负权重。最小化 ||sqrt(w) ⊙ (y − Xβ)||。

    Returns
    -------
    resid : (n,)
    beta : (k,)
    """

    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float).reshape(-1)
    if y_arr.ndim != 1:
        raise ValueError(f"y 必须是一维，当前 shape={y_arr.shape}")
    if x_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X 的行数必须等于 y 的长度，当前 X={x_arr.shape}，y={y_arr.shape}"
        )
    if w_arr.shape != y_arr.shape:
        raise ValueError(f"w 的形状必须与 y 一致，当前 w={w_arr.shape}，y={y_arr.shape}")
    if not (np.isfinite(y_arr).all() and np.isfinite(x_arr).all() and np.isfinite(w_arr).all()):
        raise ValueError("y、X、w 含有非有限值，调用前需要先丢掉缺失样本")
    if np.any(w_arr < 0):
        raise ValueError("权重不能为负")
    if x_arr.shape[1] == 0:
        raise ValueError("X 至少需要一列")

    sw = np.sqrt(w_arr)
    x_weighted = x_arr * sw[:, None]
    y_weighted = y_arr * sw
    # 小截面、行业哑元接近共线时，多线程 BLAS 会让残差在两次运行之间漂移。
    with threadpool_limits(limits=1, user_api="blas"):
        beta = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)[0]
    beta = np.ravel(beta)
    if beta.shape != (x_arr.shape[1],):
        raise RuntimeError(
            f"系数维度异常：beta={beta.shape}，X 列数={x_arr.shape[1]}"
        )
    resid = y_arr - x_arr @ beta
    if resid.shape != y_arr.shape:
        raise RuntimeError(f"残差维度异常：resid={resid.shape}，y={y_arr.shape}")
    return resid, beta


def _weights(mktcap: np.ndarray, weight: str) -> np.ndarray:
    cap = np.asarray(mktcap, dtype=float)
    if weight == "sqrt_cap":
        return np.sqrt(np.clip(cap, 0.0, None))
    if weight == "cap":
        return np.clip(cap, 0.0, None)
    if weight == "equal":
        return np.ones(cap.shape[0], dtype=float)
    raise ValueError("weight 只能是 sqrt_cap、cap 或 equal")


def _winsorize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if values.size < 5 or not (0.0 <= lower < upper <= 1.0):
        return values.copy()
    lo, hi = np.quantile(values, [lower, upper])
    return np.clip(values, lo, hi)


def ols_neutralize_one(
    y: np.ndarray,
    mktcap: np.ndarray,
    industry: np.ndarray,
    *,
    weight: str = "sqrt_cap",
    winsor: tuple[float, float] | None = (0.01, 0.99),
    min_extra: int = 5,
    min_industry_count: int = 2,
) -> np.ndarray:
    """对单个截面做市值、行业中性化。无法估计的位置返回 NaN。"""

    y_arr = np.asarray(y, dtype=float)
    cap = np.asarray(mktcap, dtype=float)
    if y_arr.shape != cap.shape or y_arr.ndim != 1:
        raise ValueError(
            f"y 与市值长度必须一致，当前 y={y_arr.shape}，mktcap={cap.shape}"
        )
    ind = np.asarray(industry)
    if ind.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"行业标签长度必须等于 y，当前 industry={ind.shape}，y={y_arr.shape}"
        )

    out = np.full(y_arr.shape[0], np.nan, dtype=float)
    size = np.log(cap)
    w = _weights(cap, weight)
    ind_text = np.array(["" if v is None else str(v) for v in ind], dtype=object)
    valid_ind = np.array([text not in {"", "None", "nan", "NaN"} for text in ind_text])
    mask = (
        np.isfinite(y_arr)
        & np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(size)
        & np.isfinite(w)
        & (w > 0)
        & valid_ind
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out

    yy = y_arr[idx].astype(float, copy=True)
    if winsor is not None:
        yy = _winsorize(yy, winsor[0], winsor[1])
    ss = size[idx]
    ww = w[idx]
    labels = ind_text[idx].astype(str)
    share = ww / ww.sum()
    mu = float(np.sum(share * ss))
    sd = float(np.sqrt(np.sum(share * (ss - mu) ** 2)))

    blocks: list[np.ndarray] = [np.ones((idx.size, 1), dtype=float)]
    if sd > 1e-12:
        blocks.append(((ss - mu) / sd).reshape(-1, 1))
    codes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    multi = [code for code, count in zip(codes, counts, strict=True) if count >= min_industry_count]
    if len(multi) >= 2:
        dummy_codes = multi[1:]
    elif len(multi) == 1 and int(counts.sum()) > int(counts[codes == multi[0]][0]):
        dummy_codes = multi
    else:
        dummy_codes = []
    if dummy_codes:
        dummies = np.zeros((idx.size, len(dummy_codes)), dtype=float)
        for column, code in enumerate(dummy_codes):
            dummies[labels == code, column] = 1.0
        blocks.append(dummies)
    design = np.hstack(blocks)
    if design.shape[0] != idx.size:
        raise RuntimeError(
            f"设计矩阵行数异常：X={design.shape}，样本数={idx.size}"
        )
    if idx.size < design.shape[1] + min_extra:
        return out

    resid, _beta = wls_residual(yy, design, ww)
    out[idx] = resid
    return out


def neutralize_factor_frame(
    df: pl.DataFrame,
    factor: str,
    *,
    date_col: str = "date",
    size_col: str = "mktcap",
    industry_col: str = "industry",
    weight: str = "sqrt_cap",
    out_col: str | None = None,
    winsor: tuple[float, float] | None = (0.01, 0.99),
    min_extra: int = 5,
    min_industry_count: int = 2,
) -> pl.DataFrame:
    """按日期对整张面板做中性化，新增一列残差因子。"""

    if factor not in df.columns:
        raise KeyError(factor)
    target = out_col or f"{factor}_neu"
    ordered = df.sort(date_col)
    dates = ordered[date_col].to_numpy()
    y = ordered[factor].to_numpy()
    cap = ordered[size_col].to_numpy()
    industry = ordered[industry_col].to_numpy()
    resid = np.full(ordered.height, np.nan, dtype=float)
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    for start, end in zip(starts, ends, strict=True):
        resid[start:end] = ols_neutralize_one(
            y[start:end],
            cap[start:end],
            industry[start:end],
            weight=weight,
            winsor=winsor,
            min_extra=min_extra,
            min_industry_count=min_industry_count,
        )
    return ordered.with_columns(pl.Series(target, resid))


def orthogonality_diagnostics(
    df: pl.DataFrame,
    resid_col: str,
    *,
    date_col: str = "date",
    size_col: str = "mktcap",
    industry_col: str = "industry",
    weight: str = "sqrt_cap",
) -> dict[str, float]:
    """检查残差相对市值和行业的加权相关性。数值应接近 0。"""

    ordered = df.sort(date_col)
    dates = ordered[date_col].to_numpy()
    resid = ordered[resid_col].to_numpy()
    cap = ordered[size_col].to_numpy()
    industry = ordered[industry_col].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], ordered.height]
    size_corrs: list[float] = []
    industry_means: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        corr, means = _date_orthogonality(
            resid[start:end], cap[start:end], industry[start:end], weight
        )
        if np.isfinite(corr):
            size_corrs.append(abs(corr))
        industry_means.extend(abs(v) for v in means if np.isfinite(v))
    return {
        "max_abs_size_corr": float(max(size_corrs) if size_corrs else np.nan),
        "max_abs_industry_mean": float(max(industry_means) if industry_means else np.nan),
        "n_dates": float(len(size_corrs)),
    }


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    share = w / w.sum()
    x_c = x - np.sum(share * x)
    y_c = y - np.sum(share * y)
    denom = np.sqrt(np.sum(share * x_c**2) * np.sum(share * y_c**2))
    if denom <= 1e-18:
        return float("nan")
    return float(np.sum(share * x_c * y_c) / denom)


def _date_orthogonality(
    resid: np.ndarray,
    mktcap: np.ndarray,
    industry: np.ndarray,
    weight: str,
) -> tuple[float, list[float]]:
    r = np.asarray(resid, dtype=float)
    cap = np.asarray(mktcap, dtype=float)
    w = _weights(cap, weight)
    labels = np.array(["" if v is None else str(v) for v in industry], dtype=object)
    mask = np.isfinite(r) & np.isfinite(cap) & (cap > 0) & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < 5:
        return float("nan"), []
    rr = r[mask]
    ww = w[mask]
    size = np.log(cap[mask])
    labs = labels[mask].astype(str)
    corr = _weighted_corr(rr, size, ww)
    means: list[float] = []
    for code in np.unique(labs):
        take = labs == code
        if not np.any(take):
            continue
        means.append(float(np.sum(ww[take] * rr[take]) / np.sum(ww[take])))
    return corr, means
