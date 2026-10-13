"""把月末组合权重和行业收益送进 Brinson，并做 Carino 链接。"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlab.attribution.brinson import brinson_fachler, carino_link


def attribute_portfolio(
    panel: pl.DataFrame,
    portfolio: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "ticker",
    group_col: str = "industry",
    ret_col: str = "fwd_ret",
    model: str = "bf",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    if portfolio.is_empty() or benchmark.is_empty():
        raise ValueError("组合或基准权重为空，无法归因")
    port_dates = set(portfolio[date_col].unique().to_list())
    bench_dates = set(benchmark[date_col].unique().to_list())
    dates = sorted(port_dates & bench_dates)
    period_rows: list[dict] = []
    group_frames: list[pl.DataFrame] = []
    for dt in dates:
        day = panel.filter(pl.col(date_col) == dt).select([id_col, group_col, ret_col])
        if day.is_empty():
            continue
        wp = portfolio.filter(pl.col(date_col) == dt).select(id_col, pl.col("weight").alias("wp"))
        wb = benchmark.filter(pl.col(date_col) == dt).select(id_col, pl.col("weight").alias("wb"))
        merged = (
            day.join(wp, on=id_col, how="left")
            .join(wb, on=id_col, how="left")
            .with_columns(pl.col("wp").fill_null(0.0), pl.col("wb").fill_null(0.0))
            .filter(pl.col(ret_col).is_finite() & pl.col(group_col).is_not_null())
        )
        if merged.height == 0 or merged["wp"].sum() <= 0 or merged["wb"].sum() <= 0:
            continue
        result = brinson_fachler(
            merged["wp"].to_numpy(),
            merged["wb"].to_numpy(),
            merged[ret_col].to_numpy(),
            merged[group_col].to_numpy(),
            model=model,
        )
        period_rows.append(
            {
                date_col: dt,
                "portfolio_return": result.portfolio_return,
                "benchmark_return": result.benchmark_return,
                "active_return": result.active_return,
                "allocation": result.allocation,
                "selection": result.selection,
                "interaction": result.interaction,
            }
        )
        group_frames.append(
            result.by_group.with_columns(pl.lit(dt).cast(pl.Date).alias(date_col))
        )
    if not period_rows:
        raise ValueError("没有同时具备组合权重、基准权重和远期收益的月份")

    periods = (
        pl.DataFrame(period_rows)
        .with_columns(pl.col(date_col).cast(pl.Date))
        .sort(date_col)
    )
    linked = carino_link(
        periods["portfolio_return"].to_numpy(),
        periods["benchmark_return"].to_numpy(),
        periods["allocation"].to_numpy(),
        periods["selection"].to_numpy(),
        periods["interaction"].to_numpy(),
    )
    scale = np.asarray(linked["scale"], dtype=float)
    periods = periods.with_columns(
        pl.Series("scale", scale),
        pl.Series("allocation_linked", np.asarray(linked["allocation"], dtype=float)),
        pl.Series("selection_linked", np.asarray(linked["selection"], dtype=float)),
        pl.Series("interaction_linked", np.asarray(linked["interaction"], dtype=float)),
    )
    groups = pl.concat(group_frames, how="vertical").join(
        periods.select([date_col, "scale"]), on=date_col, how="left"
    )
    groups = groups.with_columns(
        (pl.col("allocation") * pl.col("scale")).alias("allocation_linked"),
        (pl.col("selection") * pl.col("scale")).alias("selection_linked"),
        (pl.col("interaction") * pl.col("scale")).alias("interaction_linked"),
    )
    summary = (
        groups.group_by("group")
        .agg(
            pl.col("allocation_linked").sum().alias("allocation"),
            pl.col("selection_linked").sum().alias("selection"),
            pl.col("interaction_linked").sum().alias("interaction"),
            pl.col("wp").mean().alias("wp"),
            pl.col("wb").mean().alias("wb"),
        )
        .with_columns(
            (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias("total")
        )
        .sort("total", descending=True)
    )
    totals = {
        "portfolio_return": float(linked["portfolio_return"]),
        "benchmark_return": float(linked["benchmark_return"]),
        "active_return": float(linked["active_return"]),
        "allocation": float(linked["allocation_total"]),
        "selection": float(linked["selection_total"]),
        "interaction": float(linked["interaction_total"]),
        "n_periods": float(periods.height),
    }
    residual = abs(
        totals["allocation"] + totals["selection"] + totals["interaction"] - totals["active_return"]
    )
    if residual > 1e-6:
        raise RuntimeError(f"Carino 链接后的归因项与累计主动收益不一致：{residual:.3e}")
    for column in ("allocation", "selection", "interaction"):
        gap = abs(float(summary[column].sum()) - totals[column])
        if gap > 1e-6:
            raise RuntimeError(f"行业{column}之和与总计不一致：{gap:.3e}")
    return periods, summary, totals
