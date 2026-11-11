"""沪深 300 历史成分。成员关系用调入日和调出日，不用抓取日的成分名单。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "reference" / "csi300_history.csv"
SOURCE_URL = "https://raw.githubusercontent.com/unliftedq/index-constitution/main/history/csi300.csv"


def load_history(path: Path = HISTORY_PATH) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=2000)
    frame = frame.rename({"opt-in": "opt_in", "opt-out": "opt_out"})
    frame = frame.with_columns(
        pl.col("symbol").cast(pl.Utf8).str.slice(0, 2).str.to_lowercase().alias("exchange"),
        pl.col("symbol").cast(pl.Utf8).str.slice(2, 6).alias("code"),
        pl.col("name").cast(pl.Utf8),
        pl.col("opt_in").str.to_date(strict=False),
        pl.when(pl.col("opt_out").is_null() | (pl.col("opt_out") == ""))
        .then(None)
        .otherwise(pl.col("opt_out").str.to_date(strict=False))
        .alias("opt_out"),
    )
    missing = frame.filter(pl.col("opt_in").is_null())
    if missing.height:
        raise ValueError(f"成分表有 {missing.height} 行缺少调入日")
    return frame.select(["code", "exchange", "name", "opt_in", "opt_out"]).sort(["code", "opt_in"])


def _as_date(day: date | datetime | str) -> date:
    if isinstance(day, datetime):
        return day.date()
    if isinstance(day, date):
        return day
    return date.fromisoformat(str(day)[:10])


def members_on(history: pl.DataFrame, day: date | datetime | str) -> list[str]:
    """调入日当天进入，调出日当天退出。同一只股票多段任期只要有一段覆盖就算在内。"""

    point = _as_date(day)
    covered = history.filter(
        (pl.col("opt_in") <= point) & (pl.col("opt_out").is_null() | (pl.col("opt_out") > point))
    )
    return covered["code"].unique().sort().to_list()


def symbols_overlapping(
    history: pl.DataFrame,
    start: date | str,
    end: date | str,
) -> pl.DataFrame:
    """窗口内至少有一天是成分股的代码。一只股票保留最早的名称和交易所。"""

    start_day = _as_date(start)
    end_day = _as_date(end)
    overlap = history.filter(
        (pl.col("opt_in") <= end_day) & (pl.col("opt_out").is_null() | (pl.col("opt_out") > start_day))
    )
    return (
        overlap.sort(["code", "opt_in"])
        .unique(subset=["code"], keep="last")
        .select(["code", "exchange", "name"])
        .sort("code")
    )


def restrict_to_members(panel: pl.DataFrame, history: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """只留下形成日当天仍在指数里的股票。"""

    intervals = history.select(
        pl.col("code").alias("ticker"),
        "opt_in",
        "opt_out",
    )
    joined = panel.join(intervals, on="ticker", how="inner")
    joined = joined.filter(
        (pl.col(date_col) >= pl.col("opt_in"))
        & (pl.col("opt_out").is_null() | (pl.col(date_col) < pl.col("opt_out")))
    )
    return joined.drop(["opt_in", "opt_out"]).unique(subset=["ticker", date_col], keep="first", maintain_order=True)
