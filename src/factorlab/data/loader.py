"""读取 parquet 或 csv，并套用中英文字段别名。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from factorlab.schema import (
    FUNDAMENTAL_COLUMNS,
    PRICE_COLUMNS,
    WEIGHT_COLUMNS,
    apply_aliases,
    require_columns,
)


def _read(path: str | Path) -> pl.DataFrame:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() == ".parquet":
        return pl.read_parquet(src)
    if src.suffix.lower() == ".csv":
        return pl.read_csv(src, try_parse_dates=True)
    raise ValueError(f"只支持 parquet 和 csv：{src}")


def load_prices(path: str | Path) -> pl.DataFrame:
    df = apply_aliases(_read(path))
    require_columns(df, PRICE_COLUMNS, "行情数据")
    return df


def load_fundamentals(path: str | Path) -> pl.DataFrame:
    df = apply_aliases(_read(path))
    require_columns(df, FUNDAMENTAL_COLUMNS, "财务数据")
    return df


def load_weights(path: str | Path) -> pl.DataFrame:
    df = apply_aliases(_read(path))
    require_columns(df, WEIGHT_COLUMNS, "组合权重")
    return df
