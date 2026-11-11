"""第三阶段：坐标轴、因子配置、运行对照、下载压缩。"""

import gzip
import json
from datetime import date
from pathlib import Path

from factorlab.data.schedule import is_due, main
from factorlab.schema import FACTORS, load_factor_specs
from factorlab.web.app import create_app
from factorlab.web.runs import diff_runs, list_runs
from factorlab.web.svg import lined_chart


_DIRECTIONS = (
    ("sloan_accrual", -1),
    ("capex_intensity", -1),
    ("dupont_pm", 1),
    ("dupont_at", 1),
    ("dupont_em", -1),
    ("amihud", 1),
    ("ivol", -1),
    ("reversal", 1),
)


def test_factor_config_keeps_the_eight_directions():
    assert [(spec.name, spec.direction) for spec in FACTORS] == list(_DIRECTIONS)
    assert all(spec.label and spec.formula for spec in FACTORS)


def test_config_rejects_a_factor_without_a_calculation(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "config" / "factors.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows.append(
        {
            "name": "momentum",
            "label": "动量",
            "family": "价格",
            "direction": 1,
            "question": "过去涨得多的股票，下个月是否还涨？",
            "definition": "过去十二个月收益。",
            "formula": "过去十二个月收益",
        }
    )
    path = tmp_path / "factors.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    try:
        load_factor_specs(path)
    except ValueError as exc:
        assert "还没有计算函数" in str(exc)
    else:
        raise AssertionError("多出来的因子应该被拒绝")


def test_chart_has_date_ticks_and_hover_value():
    chart = lined_chart(
        [[0.01, -0.02, 0.03, 0.01]],
        ["2018-01-31", "2018-02-28", "2018-03-30", "2018-04-30"],
        ["净值"],
    )
    assert chart["y_ticks"]
    assert any(tick["label"].startswith("2018-") for tick in chart["x_ticks"])
    title = chart["series"][0]["points"][0]["title"]
    assert "净值" in title
    assert "2018-01-31" in title
    assert chart["legend"][0]["name"] == "净值"
    assert isinstance(chart["legend"][0]["x"], int)


def test_run_diff_shows_a_changed_stock_count(tmp_path: Path):
    _write_run(tmp_path / "data" / "releases" / "left", "aaa111", 100)
    _write_run(tmp_path / "data" / "releases" / "right", "bbb222", 400)
    cards = list_runs(tmp_path)
    assert {card["run_id"] for card in cards} == {"aaa111", "bbb222"}
    left, right = cards
    diff = diff_runs(left, right)
    assert any(row["field"] == "股票数" for row in diff)


def test_schedule_is_saturday_and_does_not_fetch_by_itself():
    assert is_due(date(2026, 11, 14), {"weekdays": ["saturday"]})
    assert not is_due(date(2026, 11, 11), {"weekdays": ["saturday"]})
    assert main([]) == 0


def test_weights_download_is_gzipped_and_static_is_cached(tmp_path: Path):
    payload = ("a,b\n" + "1,2\n" * 700).encode("utf-8")
    assert len(payload) >= 2048
    (tmp_path / "periods_weights.csv").write_bytes(payload)
    app = create_app(tmp_path)
    client = app.test_client()
    downloaded = client.get("/download/periods_weights.csv", headers={"Accept-Encoding": "gzip"})
    assert downloaded.status_code == 200
    assert downloaded.headers["Content-Encoding"] == "gzip"
    assert "max-age=3600" in downloaded.headers["Cache-Control"]
    assert gzip.decompress(downloaded.data).startswith(b"a,b")
    static = client.get("/static/site.css")
    assert static.status_code == 200
    assert "max-age=86400" in static.headers["Cache-Control"]
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "User-agent" in robots.get_data(as_text=True)


def test_runs_page_renders(tmp_path: Path):
    app = create_app(tmp_path)
    page = app.test_client().get("/runs")
    assert page.status_code == 200
    assert "运行记录" in page.get_data(as_text=True)


def _write_run(directory: Path, run_id: str, n_stocks: int) -> None:
    directory.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "generated_at": "2026-01-01T00:00:00",
        "source_commit": "abc",
        "config": {
            "mode": "csi300",
            "start": "2018-01-31",
            "end": "2024-12-31",
            "n_stocks": n_stocks,
            "factors": [["amihud", 1], ["ivol", -1]],
            "n_groups": 5,
            "ic_min_names": 80,
            "book_min_names": 80,
            "full_cost_bps": {"commission": 3, "half_spread": 5, "impact": 10},
        },
    }
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
