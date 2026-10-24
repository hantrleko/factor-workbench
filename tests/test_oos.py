"""滚动切分、多重检验校正和研究状态。"""

from datetime import date

from factorlab.research.costs import FULL_COST, net_returns
from factorlab.research.oos import holm_adjust, rolling_windows, split_dates
from factorlab.research.status import research_status


def test_holdout_is_the_last_twelve_months():
    dates = [date(2020, 1, 31)] * 0
    dates = [date(2020 + i // 12, i % 12 + 1, 1) for i in range(48)]
    development, holdout = split_dates(dates)
    assert len(holdout) == 12
    assert development[-1] < holdout[0]
    assert holdout[-1] == dates[-1]


def test_rolling_tests_do_not_overlap_training():
    dates = list(range(48))
    development, _ = split_dates(dates)
    windows = rolling_windows(development)
    assert windows
    for window in windows:
        assert set(window["train"]).isdisjoint(window["test"])
        assert max(window["train"]) < min(window["test"])


def test_holm_is_at_least_as_large_as_the_raw_p():
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert adjusted[0] >= 0.01
    assert max(adjusted) <= 1


def test_full_cost_includes_borrow_even_without_turnover():
    nets = net_returns([0.01], [None], FULL_COST)
    assert abs(nets[0] - (0.01 - FULL_COST["borrow_monthly"])) < 1e-12


def test_failed_oos_is_not_called_a_pass():
    label, _tone = research_status(
        in_sample_candidate=True,
        oos_nw_t=0.4,
        oos_net_positive=False,
        holdout_net_positive=False,
        holm_p=0.4,
        monotone=False,
    )
    assert label == "样本内候选，尚未验证"
    assert "通过" not in label
