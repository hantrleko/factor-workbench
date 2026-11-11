"""日期越界、未知样本和空截面不再变成 500。"""

from pathlib import Path

import polars as pl

from factorlab.eval.backtest import quantile_backtest
from factorlab.web.app import _date_error, create_app


def test_out_of_range_date_is_a_message():
    message = _date_error("2030-01-01", "", "2018-01-31", "2024-12-31")
    assert "晚于" in message
    assert _date_error("2010-01-01", "2010-06-01", "2018-01-31", "2024-12-31")
    assert _date_error("not-a-date", "") == "日期要写成 YYYY-MM-DD。"
    assert _date_error("2019-01-31", "2020-01-31", "2018-01-31", "2024-12-31") == ""


def test_empty_panel_backtest_does_not_crash():
    panel = pl.DataFrame(
        {
            "date": [],
            "ticker": [],
            "fwd_ret": [],
            "value": [],
        }
    ).cast({"date": pl.Date, "ticker": pl.Utf8, "fwd_ret": pl.Float64, "value": pl.Float64})
    result = quantile_backtest(panel, "value", direction=1, n_groups=5, min_names=3)
    assert result.long_short.height == 0


def test_unknown_universe_is_404_and_health_is_ok():
    root = Path(__file__).resolve().parents[1]
    app = create_app(root / "data" / "releases" / "cn_pit_v1")
    client = app.test_client()
    missing = client.get("/?universe=zzz")
    assert missing.status_code == 404
    assert "没有名为 zzz 的样本" in missing.get_data(as_text=True)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["ok"] is True


def test_factor_page_survives_a_date_past_the_sample():
    root = Path(__file__).resolve().parents[1]
    app = create_app(root / "data" / "releases" / "cn_pit_v1")
    client = app.test_client()
    page = client.get("/factor/amihud?start=2030-01-01")
    assert page.status_code == 200
    assert "晚于" in page.get_data(as_text=True)
