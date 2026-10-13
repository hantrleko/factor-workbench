"""Brinson–Fachler 配置 / 选股 / 交互分解，以及 Carino 多期链接。

单期（Brinson and Fachler, 1985），R_b 为基准总收益：

    配置_j     = (w_p,j − w_b,j) · (r_b,j − R_b)
    选股_j     = w_b,j · (r_p,j − r_b,j)
    交互_j     = (w_p,j − w_b,j) · (r_p,j − r_b,j)

三项之和等于组合收益减基准收益。BHB（Brinson, Hood, Beebower, 1986）
把配置项改成 (w_p − w_b) · r_b，不加基准总收益扣减。

基准在该组没有持仓时，把组基准收益记为 R_b，新增持仓的主动收益进入交互项。
组合在该组没有持仓时，组组合收益记为组基准收益，主动收益只留在配置项。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BrinsonPeriod:
    portfolio_return: float
    benchmark_return: float
    allocation: float
    selection: float
    interaction: float
    by_group: pl.DataFrame

    @property
    def active_return(self) -> float:
        return self.portfolio_return - self.benchmark_return


def _as_1d(name: str, values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} 必须是长度为 {n} 的一维数组，当前 shape={arr.shape}")
    return arr


def brinson_fachler(
    weight_p: np.ndarray,
    weight_b: np.ndarray,
    asset_return: np.ndarray,
    group: np.ndarray,
    *,
    model: str = "bf",
) -> BrinsonPeriod:
    """对单一持有期做 Brinson 分解。权重会在有效样本内重新归一。"""

    if model not in {"bf", "bhb"}:
        raise ValueError("model 只能是 bf 或 bhb")
    groups = np.asarray(group)
    n = groups.shape[0]
    if groups.ndim != 1:
        raise ValueError(f"group 必须是一维，当前 shape={groups.shape}")
    wp = _as_1d("weight_p", weight_p, n)
    wb = _as_1d("weight_b", weight_b, n)
    ret = _as_1d("asset_return", asset_return, n)
    labels = np.array(["" if v is None else str(v) for v in groups], dtype=object)
    mask = np.isfinite(wp) & np.isfinite(wb) & np.isfinite(ret) & np.array(
        [text not in {"", "None", "nan"} for text in labels]
    )
    if int(mask.sum()) == 0:
        raise ValueError("Brinson 没有有效的资产观测")
    wp = wp[mask].copy()
    wb = wb[mask].copy()
    ret = ret[mask]
    labels = labels[mask].astype(str)
    if wp.sum() <= 0 or wb.sum() <= 0:
        raise ValueError("组合或基准权重之和必须为正")
    wp = wp / wp.sum()
    wb = wb / wb.sum()
    port_ret = float(np.dot(wp, ret))
    bench_ret = float(np.dot(wb, ret))

    rows: list[dict[str, float | str]] = []
    for name in np.unique(labels):
        take = labels == name
        wpj = float(wp[take].sum())
        wbj = float(wb[take].sum())
        rpj = float(np.dot(wp[take], ret[take]) / wpj) if wpj > 1e-15 else float("nan")
        rbj = float(np.dot(wb[take], ret[take]) / wbj) if wbj > 1e-15 else float("nan")
        if wpj <= 1e-15 and wbj <= 1e-15:
            continue
        if wbj <= 1e-15:
            rb_use = bench_ret
            rp_use = rpj
        elif wpj <= 1e-15:
            rb_use = rbj
            rp_use = rbj
        else:
            rb_use = rbj
            rp_use = rpj
        excess_group = rb_use - bench_ret
        alloc = (wpj - wbj) * (excess_group if model == "bf" else rb_use)
        sel = wbj * (rp_use - rb_use)
        inter = (wpj - wbj) * (rp_use - rb_use)
        rows.append(
            {
                "group": name,
                "wp": wpj,
                "wb": wbj,
                "rp": rp_use,
                "rb": rb_use,
                "allocation": float(alloc),
                "selection": float(sel),
                "interaction": float(inter),
            }
        )
    table = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "group": pl.Utf8,
            "wp": pl.Float64,
            "wb": pl.Float64,
            "rp": pl.Float64,
            "rb": pl.Float64,
            "allocation": pl.Float64,
            "selection": pl.Float64,
            "interaction": pl.Float64,
        }
    )
    allocation = float(table["allocation"].sum()) if table.height else 0.0
    selection = float(table["selection"].sum()) if table.height else 0.0
    interaction = float(table["interaction"].sum()) if table.height else 0.0
    active = port_ret - bench_ret
    gap = abs(allocation + selection + interaction - active)
    if gap > 1e-8:
        raise RuntimeError(
            f"Brinson 三项之和与主动收益不一致：差额 {gap:.3e}，"
            f"主动收益 {active:.6f}，配置 {allocation:.6f}，"
            f"选股 {selection:.6f}，交互 {interaction:.6f}"
        )
    return BrinsonPeriod(
        portfolio_return=port_ret,
        benchmark_return=bench_ret,
        allocation=allocation,
        selection=selection,
        interaction=interaction,
        by_group=table,
    )


def _carino_k(port: float, bench: float) -> float:
    gap = port - bench
    if abs(gap) < 1e-12:
        return 1.0 / (1.0 + port)
    if port <= -1.0 or bench <= -1.0:
        raise ValueError("Carino 链接要求单期收益大于 -100%")
    return float((np.log1p(port) - np.log1p(bench)) / gap)


def carino_link(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    allocation: np.ndarray,
    selection: np.ndarray,
    interaction: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """把各期算术归因缩放到累计主动收益 R_p − R_b。

    k_t = ln((1+r_p,t)/(1+r_b,t)) / (r_p,t − r_b,t)
    K   = ln((1+R_p)/(1+R_b)) / (R_p − R_b)
    调整后的效应 = (k_t / K) × 当期效应。
    """

    rp = np.asarray(portfolio_returns, dtype=float).reshape(-1)
    rb = np.asarray(benchmark_returns, dtype=float).reshape(-1)
    alloc = np.asarray(allocation, dtype=float).reshape(-1)
    sel = np.asarray(selection, dtype=float).reshape(-1)
    inter = np.asarray(interaction, dtype=float).reshape(-1)
    n = rp.shape[0]
    for name, arr in (
        ("benchmark_returns", rb),
        ("allocation", alloc),
        ("selection", sel),
        ("interaction", inter),
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} 长度必须为 {n}，当前 shape={arr.shape}")
    if n == 0:
        raise ValueError("Carino 链接至少需要一期")
    if np.any(rp <= -1) or np.any(rb <= -1):
        raise ValueError("收益序列含有小于等于 -100% 的观测")

    growth_p = float(np.prod(1.0 + rp) - 1.0)
    growth_b = float(np.prod(1.0 + rb) - 1.0)
    k_t = np.array([_carino_k(a, b) for a, b in zip(rp, rb, strict=True)])
    k_total = _carino_k(growth_p, growth_b)
    if abs(k_total) < 1e-18:
        scale = np.ones(n)
    else:
        scale = k_t / k_total
    adj_alloc = scale * alloc
    adj_sel = scale * sel
    adj_inter = scale * inter
    return {
        "scale": scale,
        "allocation": adj_alloc,
        "selection": adj_sel,
        "interaction": adj_inter,
        "portfolio_return": growth_p,
        "benchmark_return": growth_b,
        "active_return": growth_p - growth_b,
        "allocation_total": float(adj_alloc.sum()),
        "selection_total": float(adj_sel.sum()),
        "interaction_total": float(adj_inter.sum()),
    }
