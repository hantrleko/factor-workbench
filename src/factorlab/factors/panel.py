"""把质量因子和微观结构因子对齐到月末面板，并生成多期远期收益。"""

from __future__ import annotations

import polars as pl

from factorlab.factors.microstructure import microstructure_panel
from factorlab.factors.quality import fundamental_factors
from factorlab.schema import FACTORS


def add_horizon_returns(panel: pl.DataFrame, horizons: int = 6) -> pl.DataFrame:
    """fwd_ret_h 是从形成月末起未来 h 个自然月的累计简单收益。"""

    if horizons < 1:
        raise ValueError("horizons 至少为 1")
    out = panel
    for h in range(1, horizons + 1):
        lead = panel.select(
            "ticker",
            pl.col("month").dt.offset_by(f"-{h}mo").alias("month"),
            pl.col("month_ret").alias(f"_rh{h}"),
        )
        out = out.join(lead, on=["ticker", "month"], how="left")
    expressions = []
    for h in range(1, horizons + 1):
        gross = pl.lit(1.0)
        for j in range(1, h + 1):
            gross = gross * (1.0 + pl.col(f"_rh{j}"))
        expressions.append((gross - 1.0).alias(f"fwd_ret_{h}"))
    out = out.with_columns(expressions)
    drop_cols = [f"_rh{h}" for h in range(1, horizons + 1)]
    return out.drop(drop_cols)


def build_factor_panel(
    prices: pl.DataFrame,
    fundamentals: pl.DataFrame,
    *,
    horizons: int = 6,
) -> pl.DataFrame:
    micro = microstructure_panel(prices)
    quality = fundamental_factors(micro.select(["ticker", "date"]), fundamentals)
    panel = micro.join(quality, on=["ticker", "date"], how="left")
    panel = add_horizon_returns(panel, horizons=horizons)
    _assert_unique(panel)
    _assert_forward_alignment(panel)
    factor_names = [spec.name for spec in FACTORS]
    return panel.select(
        [
            "date",
            "ticker",
            "month",
            "industry",
            "mktcap",
            "month_ret",
            "fwd_ret",
            *[f"fwd_ret_{h}" for h in range(1, horizons + 1)],
            *factor_names,
            "dupont_roe",
            "available_date",
        ]
    )


def _assert_unique(panel: pl.DataFrame) -> None:
    n_unique = panel.select(["ticker", "date"]).unique().height
    if n_unique != panel.height:
        raise RuntimeError("因子面板存在重复的 (ticker, date)，拒绝继续以免收益被重复计算")


def _assert_forward_alignment(panel: pl.DataFrame) -> None:
    if "fwd_ret_1" not in panel.columns:
        return
    both = panel.select(["fwd_ret", "fwd_ret_1"]).drop_nulls()
    if both.height == 0:
        return
    gap = (both["fwd_ret"] - both["fwd_ret_1"]).abs().max()
    if gap is not None and float(gap) > 1e-8:
        raise RuntimeError(f"1 个月远期收益与 fwd_ret 不一致，最大偏差 {float(gap):.3e}")
