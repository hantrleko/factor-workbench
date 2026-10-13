"""行业约束下的截面 WLS 因子收益，以及组合风险拆分。

每个月末用下一期收益对当期暴露回归：

    r = f_country · 1 + Σ_j f_j · 1_{行业=j} + Σ_s f_s · z_s + ε

行业因子收益满足市值加权和为零：c' f_ind = 0。因此国家因子吸收市场，
行业因子是相对市场的偏离。风格因子先做市值加权去均值和标准化，
系数可读成「每 1 个标准差暴露对应的收益」。

回归权重默认是市值平方根；行业约束和风格去均值使用市值权重。
这是 Barra 结构的一个可核对子集，不是商业版 USE4 的复制。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from factorlab.neutralize.residual import wls_residual


@dataclass
class BarraResult:
    factor_returns: pl.DataFrame
    r_squared: pl.DataFrame
    specific: pl.DataFrame
    industries: list[str]
    styles: list[str]
    exposures: pl.DataFrame | None
    risk: dict[str, float | list[str]]


def _zscore_cap(values: np.ndarray, cap_weight: np.ndarray) -> np.ndarray | None:
    mu = float(np.sum(cap_weight * values))
    var = float(np.sum(cap_weight * (values - mu) ** 2))
    if var <= 1e-18:
        return None
    return (values - mu) / np.sqrt(var)


def _estimate_cross_section(
    ret: np.ndarray,
    mktcap: np.ndarray,
    industry: np.ndarray,
    styles: np.ndarray,
    style_names: list[str],
    *,
    weight: str,
    min_extra: int,
) -> dict | None:
    n = ret.shape[0]
    if mktcap.shape != (n,) or industry.shape[0] != n:
        raise ValueError(
            f"收益、市值、行业长度不一致：ret={ret.shape}，mktcap={mktcap.shape}，"
            f"industry={industry.shape}"
        )
    if styles.ndim != 2 or styles.shape[0] != n:
        raise ValueError(f"风格矩阵必须是 (n, s)，当前 {styles.shape}，n={n}")
    if styles.shape[1] != len(style_names):
        raise ValueError(
            f"风格列名数量 {len(style_names)} 与矩阵列数 {styles.shape[1]} 不一致"
        )

    cap = np.asarray(mktcap, dtype=float)
    if weight == "sqrt_cap":
        w_reg = np.sqrt(np.clip(cap, 0.0, None))
    elif weight == "cap":
        w_reg = np.clip(cap, 0.0, None)
    elif weight == "equal":
        w_reg = np.ones(n, dtype=float)
    else:
        raise ValueError("weight 只能是 sqrt_cap、cap 或 equal")

    labels = np.array(["" if v is None else str(v) for v in industry], dtype=object)
    kept: list[int] = []
    kept_names: list[str] = []
    finite_style = np.isfinite(styles) if styles.shape[1] else np.ones((n, 0), dtype=bool)
    for j, name in enumerate(style_names):
        if float(np.mean(finite_style[:, j])) >= 0.6:
            kept.append(j)
            kept_names.append(name)

    mask = np.isfinite(ret) & np.isfinite(cap) & (cap > 0) & np.isfinite(w_reg) & (w_reg > 0)
    mask &= np.array([text not in {"", "None", "nan", "NaN"} for text in labels])
    for j in kept:
        mask &= finite_style[:, j]
    idx = np.flatnonzero(mask)
    if idx.size < 8:
        return None

    r = ret[idx]
    wc = cap[idx]
    wc = wc / wc.sum()
    wr = w_reg[idx]
    ind = labels[idx].astype(str)
    codes, inverse = np.unique(ind, return_inverse=True)
    k = int(codes.size)
    if k < 2:
        return None
    dummies = np.eye(k, dtype=float)[inverse]
    cap_share = np.array([float(wc[inverse == j].sum()) for j in range(k)])
    order = np.argsort(cap_share)
    if cap_share[order[-1]] <= 1e-10:
        return None

    style_blocks: list[np.ndarray] = []
    used_styles: list[str] = []
    for col, name in zip((styles[idx][:, j] for j in kept), kept_names, strict=True):
        scored = _zscore_cap(col, wc)
        if scored is None:
            continue
        style_blocks.append(scored.reshape(-1, 1))
        used_styles.append(name)
    style_z = np.hstack(style_blocks) if style_blocks else np.zeros((idx.size, 0))

    d_ord = dummies[:, order]
    c_ord = cap_share[order]
    scale = c_ord[:-1] / c_ord[-1]
    basis = d_ord[:, :-1] - np.outer(d_ord[:, -1], scale)
    design = np.hstack([np.ones((idx.size, 1)), basis, style_z]) if style_z.shape[1] else np.hstack(
        [np.ones((idx.size, 1)), basis]
    )
    if design.shape != (idx.size, 1 + (k - 1) + style_z.shape[1]):
        raise RuntimeError(
            f"Barra 设计矩阵维度异常：X={design.shape}，"
            f"期望列数={1 + (k - 1) + style_z.shape[1]}"
        )
    if idx.size < design.shape[1] + min_extra:
        return None

    resid, beta = wls_residual(r, design, wr)
    f_country = float(beta[0])
    free = beta[1:k]
    if free.shape != (k - 1,):
        raise RuntimeError(f"行业自由系数维度异常：{free.shape}，行业数={k}")
    f_style = beta[k:]
    if f_style.shape != (style_z.shape[1],):
        raise RuntimeError(
            f"风格系数维度异常：beta_style={f_style.shape}，风格数={style_z.shape[1]}"
        )
    f_ord = np.zeros(k, dtype=float)
    f_ord[:-1] = free
    f_ord[-1] = float(-np.dot(c_ord[:-1], free) / c_ord[-1])
    inv_order = np.empty(k, dtype=int)
    inv_order[order] = np.arange(k)
    f_ind = f_ord[inv_order]
    if abs(float(np.dot(cap_share, f_ind))) > 1e-6:
        raise RuntimeError(
            f"行业因子收益不满足市值加权和为零：{float(np.dot(cap_share, f_ind)):.3e}"
        )

    fitted = f_country + dummies @ f_ind
    if style_z.shape[1]:
        fitted = fitted + style_z @ f_style
    resid_check = r - fitted
    if not np.allclose(resid_check, resid, atol=1e-8):
        raise RuntimeError("因子收益重构的残差与 WLS 残差不一致")

    wrn = wr / wr.sum()
    center = float(np.sum(wrn * r))
    sst = float(np.sum(wrn * (r - center) ** 2))
    sse = float(np.sum(wrn * resid**2))
    r2 = float(1.0 - sse / sst) if sst > 1e-18 else float("nan")

    blocks = [np.ones((idx.size, 1)), dummies]
    if style_z.shape[1]:
        blocks.append(style_z)
    exposure = np.hstack(blocks)
    if exposure.shape != (idx.size, 1 + k + style_z.shape[1]):
        raise RuntimeError(f"暴露矩阵维度异常：{exposure.shape}")

    return {
        "codes": codes.tolist(),
        "styles": used_styles,
        "country": f_country,
        "industry": {code: float(val) for code, val in zip(codes, f_ind, strict=True)},
        "style": {name: float(val) for name, val in zip(used_styles, f_style, strict=True)},
        "r2": r2,
        "idx": idx,
        "resid": resid,
        "exposure": exposure,
        "cap_share_dot": float(np.dot(cap_share, f_ind)),
    }


def estimate_barra(
    panel: pl.DataFrame,
    style_cols: list[str],
    *,
    date_col: str = "date",
    id_col: str = "ticker",
    ret_col: str = "fwd_ret",
    size_col: str = "mktcap",
    industry_col: str = "industry",
    weight: str = "sqrt_cap",
    portfolio_weights: pl.DataFrame | None = None,
    benchmark_weights: pl.DataFrame | None = None,
) -> BarraResult:
    """估计月度因子收益。若给出组合与基准权重，则对最后一期做主动风险拆分。"""

    needed = [date_col, id_col, ret_col, size_col, industry_col, *style_cols]
    missing = [col for col in needed if col not in panel.columns]
    if missing:
        raise KeyError(f"Barra 输入缺少字段：{', '.join(missing)}")

    ordered = panel.sort([date_col, id_col])
    industries = sorted(
        {
            str(v)
            for v in ordered[industry_col].drop_nulls().unique().to_list()
            if str(v) not in {"", "None", "nan"}
        }
    )
    dates = ordered[date_col].unique().sort().to_list()
    ret_rows: list[dict] = []
    r2_rows: list[dict] = []
    spec_chunks: list[pl.DataFrame] = []
    last_pack: dict | None = None
    last_date = None
    last_ids: list | None = None

    for dt in dates:
        block = ordered.filter(pl.col(date_col) == dt)
        styles = np.column_stack(
            [block[col].to_numpy().astype(float) for col in style_cols]
        ) if style_cols else np.zeros((block.height, 0))
        pack = _estimate_cross_section(
            block[ret_col].to_numpy().astype(float),
            block[size_col].to_numpy().astype(float),
            block[industry_col].to_numpy(),
            styles,
            style_cols,
            weight=weight,
            min_extra=5,
        )
        if pack is None:
            continue
        row: dict[str, object] = {date_col: dt, "country": pack["country"]}
        for code in industries:
            row[f"ind::{code}"] = pack["industry"].get(code, float("nan"))
        for name in style_cols:
            row[name] = pack["style"].get(name, float("nan"))
        ret_rows.append(row)
        r2_rows.append({date_col: dt, "r2": pack["r2"]})
        spec_chunks.append(
            pl.DataFrame(
                {
                    date_col: [dt] * pack["idx"].size,
                    id_col: block[id_col].to_numpy()[pack["idx"]],
                    "specific_return": pack["resid"],
                }
            )
        )
        last_pack = pack
        last_date = dt
        last_ids = block[id_col].to_list()

    if not ret_rows:
        raise ValueError("没有足够的截面能够估计 Barra 因子收益")

    factor_returns = pl.DataFrame(ret_rows).sort(date_col)
    r_squared = pl.DataFrame(r2_rows).sort(date_col)
    specific = pl.concat(spec_chunks, how="vertical") if spec_chunks else pl.DataFrame()
    exposures = None
    risk: dict[str, float | list[str]] = {}
    if last_pack is not None and last_ids is not None and last_date is not None:
        columns = ["country", *[f"ind::{c}" for c in last_pack["codes"]], *last_pack["styles"]]
        expo = last_pack["exposure"]
        if expo.shape[1] != len(columns):
            raise RuntimeError(
                f"暴露列名数量 {len(columns)} 与矩阵列数 {expo.shape[1]} 不一致"
            )
        exposures = pl.DataFrame(
            {
                date_col: [last_date] * expo.shape[0],
                id_col: [last_ids[i] for i in last_pack["idx"]],
                **{col: expo[:, j] for j, col in enumerate(columns)},
            }
        )
        if portfolio_weights is not None and benchmark_weights is not None:
            risk = _risk_decomposition(
                factor_returns,
                specific,
                exposures,
                portfolio_weights,
                benchmark_weights,
                date_col=date_col,
                id_col=id_col,
                asof=last_date,
            )
    return BarraResult(
        factor_returns=factor_returns,
        r_squared=r_squared,
        specific=specific,
        industries=industries,
        styles=style_cols,
        exposures=exposures,
        risk=risk,
    )


def _risk_decomposition(
    factor_returns: pl.DataFrame,
    specific: pl.DataFrame,
    exposures: pl.DataFrame,
    portfolio_weights: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    *,
    date_col: str,
    id_col: str,
    asof,
) -> dict[str, float | list[str]]:
    factor_cols = [c for c in factor_returns.columns if c != date_col]
    expo_cols = [c for c in exposures.columns if c not in {date_col, id_col}]
    common = [c for c in factor_cols if c in expo_cols]
    if len(common) < 2:
        return {}
    hist = factor_returns.select(common).drop_nulls()
    if hist.height < len(common) + 2:
        return {
            "note": "因子收益样本短于因子个数，未计算协方差。",
            "n_periods": float(hist.height),
            "n_factors": float(len(common)),
        }
    cov = np.cov(hist.to_numpy(), rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    pw = portfolio_weights.filter(pl.col(date_col) == asof)
    bw = benchmark_weights.filter(pl.col(date_col) == asof)
    merged = (
        exposures.select([id_col, *common])
        .join(pw.select(id_col, pl.col("weight").alias("wp")), on=id_col, how="left")
        .join(bw.select(id_col, pl.col("weight").alias("wb")), on=id_col, how="left")
        .with_columns(
            pl.col("wp").fill_null(0.0),
            pl.col("wb").fill_null(0.0),
        )
    )
    wp = merged["wp"].to_numpy()
    wb = merged["wb"].to_numpy()
    if wp.sum() > 0:
        wp = wp / wp.sum()
    if wb.sum() > 0:
        wb = wb / wb.sum()
    active = wp - wb
    x = merged.select(common).to_numpy()
    if x.shape != (active.shape[0], len(common)):
        raise RuntimeError(f"主动暴露矩阵维度异常：X={x.shape}，h={active.shape}")
    exposure = x.T @ active
    factor_var = float(exposure.T @ cov @ exposure)

    spec = (
        specific.group_by(id_col)
        .agg(
            pl.col("specific_return").std(ddof=1).alias("spec_vol"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("spec_vol") ** 2).alias("spec_var"))
    )
    vars_ = spec.filter(pl.col("n") >= 8)["spec_var"].drop_nulls().to_numpy()
    median = float(np.median(vars_)) if vars_.size else 0.0
    spec = spec.with_columns(
        pl.when(pl.col("n") >= 8)
        .then(0.5 * pl.col("spec_var") + 0.5 * median)
        .otherwise(median)
        .alias("var_shrunk")
    )
    var_map = dict(zip(spec[id_col].to_list(), spec["var_shrunk"].to_list(), strict=True))
    spec_var = 0.0
    for ticker, h in zip(merged[id_col].to_list(), active, strict=True):
        spec_var += float(h) ** 2 * float(var_map.get(ticker, median))
    return {
        "asof": str(asof),
        "factor_var_m": factor_var,
        "specific_var_m": float(spec_var),
        "total_var_m": float(factor_var + spec_var),
        "factor_vol_ann": float(np.sqrt(max(factor_var, 0.0) * 12.0)),
        "specific_vol_ann": float(np.sqrt(max(spec_var, 0.0) * 12.0)),
        "total_vol_ann": float(np.sqrt(max(factor_var + spec_var, 0.0) * 12.0)),
        "n_periods": float(hist.height),
        "n_factors": float(len(common)),
        "exposure_names": common,
        "exposure_values": [float(v) for v in exposure],
    }
