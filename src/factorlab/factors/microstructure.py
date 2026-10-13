"""市场微观结构因子：Amihud、特质波动率、短期反转，以及远期收益。

形成月是已经走完的自然月。因子在该月最后一个交易日可知。
fwd_ret 是下一个自然月的收益，窗口严格晚于形成日，不包含形成月本身。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlab.schema import PRICE_COLUMNS, apply_aliases, require_columns


def prepare_prices(prices: pl.DataFrame) -> pl.DataFrame:
    df = apply_aliases(prices)
    require_columns(df, PRICE_COLUMNS, "行情数据")
    if df["date"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("date").dt.date().alias("date"))
    else:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("industry").cast(pl.Utf8),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("mktcap").cast(pl.Float64),
    )
    df = df.sort(["ticker", "date"])
    if "ret" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret")
        )
    else:
        df = df.with_columns(pl.col("ret").cast(pl.Float64))
    if "amount" in df.columns:
        dollar = pl.col("amount").cast(pl.Float64)
    else:
        dollar = pl.col("close") * pl.col("volume")
    df = df.with_columns(
        dollar.alias("dollar_volume"),
        pl.col("mktcap").shift(1).over("ticker").alias("mktcap_lag"),
        pl.col("date").dt.truncate("1mo").alias("month"),
    )
    df = df.filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())
    return df


def capm_residual_vol(ret: np.ndarray, market: np.ndarray, min_obs: int = 15) -> float:
    """带截距的 CAPM 残差标准差，分母为 N − 2。"""

    y = np.asarray(ret, dtype=float)
    x = np.asarray(market, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError(f"收益与市场收益形状必须一致，当前 ret={y.shape}，market={x.shape}")
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = y.size
    if n < min_obs:
        return float("nan")
    x_c = x - x.mean()
    y_c = y - y.mean()
    sxx = float(np.dot(x_c, x_c))
    if sxx <= 1e-18:
        return float("nan")
    syy = float(np.dot(y_c, y_c))
    sxy = float(np.dot(x_c, y_c))
    sse = syy - (sxy ** 2) / sxx
    sse = max(sse, 0.0)
    return float(np.sqrt(sse / (n - 2)))


def market_return(prices: pl.DataFrame) -> pl.DataFrame:
    """用前一日市值加权的市场收益，避免用当日收盘市值给当日收益加权。"""

    usable = prices.filter(pl.col("mktcap_lag").is_not_null() & (pl.col("mktcap_lag") > 0))
    return (
        usable.group_by("date")
        .agg(
            (
                (pl.col("mktcap_lag") * pl.col("ret")).sum() / pl.col("mktcap_lag").sum()
            ).alias("mkt_ret")
        )
        .sort("date")
    )


def monthly_returns(prices: pl.DataFrame, min_days: int = 10) -> pl.DataFrame:
    gross = prices.with_columns((1.0 + pl.col("ret")).alias("gross"))
    out = gross.group_by(["ticker", "month"]).agg(
        pl.col("gross").product().alias("gross_m"),
        pl.len().alias("n_days"),
        pl.col("date").max().alias("date"),
        pl.col("mktcap").sort_by("date").last().alias("mktcap"),
        pl.col("industry").sort_by("date").last().alias("industry"),
    )
    out = out.with_columns((pl.col("gross_m") - 1.0).alias("month_ret"))
    return out.filter(pl.col("n_days") >= min_days)


def add_forward_return(monthly: pl.DataFrame) -> pl.DataFrame:
    """fwd_ret(t) = 下一个自然月的 month_ret，而不是下一行观测。"""

    cal = (
        monthly.select("month")
        .unique()
        .sort("month")
        .with_columns(pl.col("month").dt.offset_by("1mo").alias("next_month"))
    )
    nxt = monthly.select(
        "ticker",
        pl.col("month").alias("next_month"),
        pl.col("month_ret").alias("fwd_ret"),
    )
    return (
        monthly.join(cal, on="month", how="left")
        .join(nxt, on=["ticker", "next_month"], how="left")
        .drop("next_month")
    )


def amihud_monthly(prices: pl.DataFrame, min_days: int = 10) -> pl.DataFrame:
    daily = prices.with_columns(
        pl.when(pl.col("dollar_volume") > 0)
        .then(pl.col("ret").abs() / pl.col("dollar_volume"))
        .otherwise(None)
        .alias("illiq_d")
    )
    return (
        daily.group_by(["ticker", "month"])
        .agg(
            pl.col("illiq_d").drop_nulls().mean().alias("amihud"),
            pl.col("illiq_d").drop_nulls().len().alias("n_illiq"),
        )
        .with_columns(
            pl.when(pl.col("n_illiq") >= min_days).then(pl.col("amihud")).otherwise(None).alias("amihud")
        )
        .select(["ticker", "month", "amihud"])
    )


def ivol_monthly(prices: pl.DataFrame, min_days: int = 15) -> pl.DataFrame:
    mkt = market_return(prices)
    df = prices.join(mkt, on="date", how="inner")
    df = df.with_columns(
        (pl.col("ret") - pl.col("ret").mean().over(["ticker", "month"])).alias("dy"),
        (pl.col("mkt_ret") - pl.col("mkt_ret").mean().over(["ticker", "month"])).alias("dx"),
    )
    stats = df.group_by(["ticker", "month"]).agg(
        pl.len().alias("n"),
        (pl.col("dy") ** 2).sum().alias("syy"),
        (pl.col("dx") ** 2).sum().alias("sxx"),
        (pl.col("dy") * pl.col("dx")).sum().alias("sxy"),
    )
    sse = pl.col("syy") - (pl.col("sxy") ** 2) / pl.col("sxx")
    return stats.with_columns(
        pl.when((pl.col("n") >= min_days) & (pl.col("sxx") > 1e-18))
        .then((sse.clip(lower_bound=0.0) / (pl.col("n") - 2)).sqrt())
        .otherwise(None)
        .alias("ivol")
    ).select(["ticker", "month", "ivol"])


def microstructure_panel(prices: pl.DataFrame) -> pl.DataFrame:
    px = prepare_prices(prices)
    monthly = add_forward_return(monthly_returns(px))
    monthly = monthly.join(amihud_monthly(px), on=["ticker", "month"], how="left")
    monthly = monthly.join(ivol_monthly(px), on=["ticker", "month"], how="left")
    monthly = monthly.with_columns((-pl.col("month_ret")).alias("reversal"))
    return monthly.select(
        [
            "date",
            "ticker",
            "month",
            "industry",
            "mktcap",
            "month_ret",
            "fwd_ret",
            "amihud",
            "ivol",
            "reversal",
        ]
    )
