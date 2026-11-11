"""行业空档：公告前沿用最早行业，始终缺失的用缺口表。"""

import polars as pl

from factorlab.data.csi300_pit import _fill_industry_gaps


def test_carry_earliest_industry_and_gap_table(monkeypatch):
    monkeypatch.setattr(
        "factorlab.data.csi300_pit._read_industry_gap",
        lambda: pl.DataFrame({"ticker": ["000002"], "gap_industry": ["证券Ⅱ"]}),
    )
    frame = pl.DataFrame(
        {
            "_i": [0, 1, 2, 3],
            "date": ["2018-01-31", "2018-02-28", "2018-01-31", "2018-02-28"],
            "ticker": ["000001", "000001", "000002", "000002"],
            "industry": ["未分类", "银行Ⅱ", "未分类", "未分类"],
        }
    ).with_columns(pl.col("date").str.to_date())
    filled = _fill_industry_gaps(frame)
    assert filled["industry"].to_list() == ["银行Ⅱ", "银行Ⅱ", "证券Ⅱ", "证券Ⅱ"]
