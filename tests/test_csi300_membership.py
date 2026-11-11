"""沪深 300 成分用调入调出日，不用抓取日名单。"""

from datetime import date
from pathlib import Path

import polars as pl

from factorlab.data.csi300_membership import load_history, members_on, restrict_to_members


def _history(tmp_path: Path) -> pl.DataFrame:
    path = tmp_path / "csi300.csv"
    path.write_text(
        "symbol,name,opt-in,opt-out\n"
        "SZ000008,神州高铁,2016-12-12,2018-06-11\n"
        "SZ000027,深圳能源,2013-07-01,2013-12-16\n"
        "SZ000027,深圳能源,2014-06-16,2017-06-12\n"
        "SH600519,贵州茅台,2005-04-08,\n",
        encoding="utf-8",
    )
    return load_history(path)


def test_membership_uses_entry_and_exit_dates(tmp_path: Path):
    history = _history(tmp_path)
    assert members_on(history, date(2016, 12, 11)) == ["000027", "600519"]
    assert "000008" in members_on(history, date(2016, 12, 12))
    assert "000008" in members_on(history, date(2018, 6, 10))
    assert "000008" not in members_on(history, date(2018, 6, 11))
    assert "000027" not in members_on(history, date(2014, 1, 1))
    assert "000027" in members_on(history, date(2015, 1, 1))
    assert "600519" in members_on(history, date(2024, 12, 31))


def test_cross_section_drops_names_outside_their_spell(tmp_path: Path):
    history = _history(tmp_path)
    panel = pl.DataFrame(
        {
            "date": [date(2018, 6, 10), date(2018, 6, 11), date(2020, 1, 31)],
            "ticker": ["000008", "000008", "600519"],
            "fwd_ret": [0.01, 0.02, 0.03],
        }
    )
    kept = restrict_to_members(panel, history).sort(["date", "ticker"])
    assert kept["ticker"].to_list() == ["000008", "600519"]
    assert kept["date"].to_list()[0] == date(2018, 6, 10)
