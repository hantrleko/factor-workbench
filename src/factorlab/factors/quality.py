"""财务质量因子：Sloan 应计、资本开支强度、杜邦三项。

时点规则：调仓日 t 只能使用 available_date <= t 的报表。
同比变化和 TTM 都挂在「已经可用的最新报告」上，向后取同一公司更早的财季，
因此滞后项的披露日必然不晚于最新报告。
"""

from __future__ import annotations

import polars as pl

from factorlab.schema import FUNDAMENTAL_COLUMNS, apply_aliases, require_columns


def prepare_fundamentals(fundamentals: pl.DataFrame) -> pl.DataFrame:
    df = apply_aliases(fundamentals)
    require_columns(df, FUNDAMENTAL_COLUMNS, "财务数据")
    df = df.with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("fiscal_year").cast(pl.Int32),
        pl.col("fiscal_quarter").cast(pl.Int32),
        pl.col("available_date").cast(pl.Date),
    )
    numeric = [
        "total_assets",
        "current_assets",
        "cash",
        "current_liabilities",
        "short_term_debt",
        "tax_payable",
        "depreciation",
        "net_income",
        "revenue",
        "capex",
        "equity",
    ]
    df = df.with_columns([pl.col(col).cast(pl.Float64) for col in numeric])
    bad_q = df.filter((pl.col("fiscal_quarter") < 1) | (pl.col("fiscal_quarter") > 4))
    if bad_q.height:
        raise ValueError("fiscal_quarter 必须是 1 到 4")
    df = df.with_columns(
        (pl.col("fiscal_year") * 4 + pl.col("fiscal_quarter")).alias("fq")
    )
    df = df.sort(["ticker", "fq", "available_date"]).unique(
        subset=["ticker", "fq"],
        keep="last",
        maintain_order=True,
    )
    return _attach_lags(df)


def _attach_lags(funds: pl.DataFrame) -> pl.DataFrame:
    base = funds.sort(["ticker", "fq"])
    lag = base.select(
        "ticker",
        (pl.col("fq") + 4).alias("fq"),
        pl.col("total_assets").alias("total_assets_lag4"),
        pl.col("current_assets").alias("current_assets_lag4"),
        pl.col("cash").alias("cash_lag4"),
        pl.col("current_liabilities").alias("current_liabilities_lag4"),
        pl.col("short_term_debt").alias("short_term_debt_lag4"),
        pl.col("tax_payable").alias("tax_payable_lag4"),
        pl.col("equity").alias("equity_lag4"),
    )
    out = base.join(lag, on=["ticker", "fq"], how="left")
    for lag_q in range(4):
        shifted = base.select(
            "ticker",
            (pl.col("fq") + lag_q).alias("fq"),
            pl.col("depreciation").alias(f"dep_l{lag_q}"),
            pl.col("capex").alias(f"capex_l{lag_q}"),
            pl.col("net_income").alias(f"ni_l{lag_q}"),
            pl.col("revenue").alias(f"rev_l{lag_q}"),
        )
        out = out.join(shifted, on=["ticker", "fq"], how="left")
    out = out.with_columns(
        (pl.col("dep_l0") + pl.col("dep_l1") + pl.col("dep_l2") + pl.col("dep_l3")).alias("dep_ttm"),
        (pl.col("capex_l0") + pl.col("capex_l1") + pl.col("capex_l2") + pl.col("capex_l3")).alias(
            "capex_ttm"
        ),
        (pl.col("ni_l0") + pl.col("ni_l1") + pl.col("ni_l2") + pl.col("ni_l3")).alias("ni_ttm"),
        (pl.col("rev_l0") + pl.col("rev_l1") + pl.col("rev_l2") + pl.col("rev_l3")).alias("rev_ttm"),
    )
    avg_assets = (pl.col("total_assets") + pl.col("total_assets_lag4")) / 2.0
    avg_equity = (pl.col("equity") + pl.col("equity_lag4")) / 2.0
    numer = (
        (pl.col("current_assets") - pl.col("current_assets_lag4"))
        - (pl.col("cash") - pl.col("cash_lag4"))
        - (
            (pl.col("current_liabilities") - pl.col("current_liabilities_lag4"))
            - (pl.col("short_term_debt") - pl.col("short_term_debt_lag4"))
            - (pl.col("tax_payable") - pl.col("tax_payable_lag4"))
        )
        - pl.col("dep_ttm")
    )
    out = out.with_columns(
        pl.when(avg_assets > 0).then(numer / avg_assets).otherwise(None).alias("sloan_accrual"),
        pl.when(avg_assets > 0).then(pl.col("capex_ttm") / avg_assets).otherwise(None).alias(
            "capex_intensity"
        ),
        pl.when(pl.col("rev_ttm") > 0).then(pl.col("ni_ttm") / pl.col("rev_ttm")).otherwise(None).alias(
            "dupont_pm"
        ),
        pl.when(avg_assets > 0).then(pl.col("rev_ttm") / avg_assets).otherwise(None).alias("dupont_at"),
        pl.when(avg_equity > 0).then(avg_assets / avg_equity).otherwise(None).alias("dupont_em"),
    )
    out = out.with_columns(
        pl.when(
            pl.col("dupont_pm").is_not_null()
            & pl.col("dupont_at").is_not_null()
            & pl.col("dupont_em").is_not_null()
        )
        .then(pl.col("dupont_pm") * pl.col("dupont_at") * pl.col("dupont_em"))
        .otherwise(None)
        .alias("dupont_roe")
    )
    return out


def fundamental_factors(spine: pl.DataFrame, fundamentals: pl.DataFrame) -> pl.DataFrame:
    """把最新可用报表上的质量因子贴到 (ticker, date) 骨架上。"""

    if not {"ticker", "date"}.issubset(set(spine.columns)):
        raise KeyError("骨架需要 ticker 和 date")
    funds = prepare_fundamentals(fundamentals)
    left = spine.select(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("date").cast(pl.Date),
    ).unique(maintain_order=True).sort(["ticker", "date"])
    right = funds.sort(["ticker", "available_date"])
    joined = left.join_asof(
        right,
        left_on="date",
        right_on="available_date",
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    )
    keep = [
        "ticker",
        "date",
        "sloan_accrual",
        "capex_intensity",
        "dupont_pm",
        "dupont_at",
        "dupont_em",
        "dupont_roe",
        "available_date",
        "fiscal_year",
        "fiscal_quarter",
    ]
    return joined.select([col for col in keep if col in joined.columns])
