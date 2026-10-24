"""逐期文件要能按公开容差重建首页上的多空和 Brinson。"""

import csv
import json
from pathlib import Path

from factorlab.data.synthetic import make_synthetic_universe
from factorlab.eval.metrics import performance_stats
from factorlab.pipeline import run_analysis
from factorlab.research.costs import flat_side, month_cost_parts, net_returns
from factorlab.research.lineage import run_id


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_same_config_keeps_the_run_id_and_rebuilds_within_tolerance(tmp_path: Path):
    prices, funds = make_synthetic_universe(
        n_stocks=28,
        start="2022-01-04",
        end="2023-06-30",
        seed=11,
        embed_premium=True,
    )
    result = run_analysis(
        prices,
        funds,
        n_groups=(5,),
        out_dir=tmp_path,
        make_report=False,
        meta={"note": "复算测试", "seed": 11, "embed_premium": True},
        progress=None,
    )
    record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert record["run_id"] == run_id(record["config"])
    assert record["config"]["seed"] == 11
    assert record["tolerance_abs"] == 1e-8
    assert (tmp_path / "periods_ic.csv").stat().st_size > 100
    assert (tmp_path / "periods_weights.csv").stat().st_size > 100

    books = _rows(tmp_path / "periods_long_short.csv")
    sloan = [row for row in books if row["factor"] == "sloan_accrual"]
    assert len(sloan) > 8
    gross = [float(row["gross_return"]) for row in sloan]
    report = result.factor_reports[0]
    published = report["backtests"][5].summary["ann_return"]
    assert report["spec"].name == "sloan_accrual"
    assert abs(performance_stats(gross)["ann_return"] - published) < 1e-8

    direct = net_returns(gross, [float(row["turnover"]) if row["turnover"] else None for row in sloan], flat_side(10))
    from_file = [float(row["net_10bp"]) for row in sloan]
    assert max(abs(a - b) for a, b in zip(direct, from_file, strict=True)) < 1e-10
    parts = month_cost_parts(sloan[1]["turnover"], flat_side(10))
    assert abs(float(sloan[1]["gross_return"]) - parts["total"] - float(sloan[1]["net_10bp"])) < 1e-10

    brinson = _rows(tmp_path / "periods_brinson.csv")
    assert brinson
    linked = sum(float(row["allocation_linked"]) for row in brinson)
    assert abs(linked - result.brinson_totals["allocation"]) < 1e-8
    linked_sum = (
        sum(float(row["allocation_linked"]) for row in brinson)
        + sum(float(row["selection_linked"]) for row in brinson)
        + sum(float(row["interaction_linked"]) for row in brinson)
    )
    assert abs(linked_sum - result.brinson_totals["active_return"]) < 1e-8

    summary = {
        "generated_at": record["generated_at"],
        "universe": {
            "start": record["config"]["start"],
            "end": record["config"]["end"],
            "n_stocks": record["config"]["n_stocks"],
            "n_dates": 12,
            "note": "样本为合成数据。随机种子 11。",
        },
        "summary": [
            {
                "label": report["spec"].label,
                "ic": report["ic_stats"]["mean"],
                "t": report["ic_stats"]["t"],
                "nw_t": report["ic_stats"]["nw_t"],
                "ls_ann": published,
                "turnover": report["backtests"][5].summary.get("avg_turnover"),
            }
        ],
        "brinson": result.brinson_totals,
        "barra": {"factors": [{"name": "对数市值"}, {"name": "应计利润（Sloan）"}]},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    from factorlab.web.app import create_app

    client = create_app(tmp_path).test_client()
    home = client.get("/").get_data(as_text=True)
    assert record["run_id"] in home
    assert "逐月多空与成本" in home
    page = client.get("/factor/sloan_accrual?bps=10").get_data(as_text=True)
    assert "10 bp" in page
    assert "运行" not in page or record["run_id"] in page
    bad = client.get("/factor/sloan_accrual?bps=abc").get_data(as_text=True)
    assert "0 到 200" in bad
    bad_date = client.get("/factor/sloan_accrual?start=2022/1/1").get_data(as_text=True)
    assert "YYYY-MM-DD" in bad_date
    downloaded = client.get("/download/periods_long_short.csv")
    assert downloaded.status_code == 200
    assert b"sloan_accrual" in downloaded.data
