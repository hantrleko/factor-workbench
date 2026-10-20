"""分组结论要指出该买的一组，以及实际上谁最高。"""

from factorlab.web.explain import _because, industry_tilt, scoreboard


def test_because_points_at_the_group_that_won():
    groups = [
        {"group": 1, "ann_return": 0.046, "is_short": True},
        {"group": 4, "ann_return": 0.121},
        {"group": 5, "ann_return": 0.012, "is_long": True},
    ]
    text = _because(5, groups)
    assert "第5组" in text
    assert "第1组" in text
    assert "第4组" in text
    assert "12.10%" in text


def test_because_when_the_prescribed_group_beats_the_short_side():
    groups = [
        {"group": 1, "ann_return": 0.080, "is_long": True},
        {"group": 2, "ann_return": 0.090},
        {"group": 5, "ann_return": -0.012, "is_short": True},
    ]
    text = _because(1, groups)
    assert "该买的第1组年化 8.00%" in text
    assert "该空的第5组年化 -1.20%" in text
    assert "不是这一端" not in text


def test_industry_tilt_names_both_sides():
    text = industry_tilt(
        [
            {"group": "医药", "wp": 0.16, "wb": 0.09},
            {"group": "银行", "wp": 0.10, "wb": 0.14},
            {"group": "食品饮料", "wp": 0.11, "wb": 0.15},
        ]
    )
    assert text.startswith("按期间平均权重，比市值基准多拿的是医药")
    assert "银行" in text and "食品饮料" in text


def test_scoreboard_keeps_a_stable_order():
    rows = [
        {"verdict": "样本内通过"},
        {"verdict": "排序略对，多空没赚到"},
        {"verdict": "样本内通过"},
    ]
    labels = [item["label"] for item in scoreboard(rows)]
    assert labels == ["样本内通过", "排序略对，多空没赚到"]
    assert scoreboard(rows)[0]["n"] == 2


def test_verdict_uses_newey_west_not_the_ordinary_t():
    from factorlab.web.explain import verdict

    passed = verdict({"ic": 0.05, "t": 3.0, "nw_t": 2.2, "ls_ann": 0.04})
    assert passed[0] == "样本内通过"
    weak = verdict({"ic": 0.05, "t": 3.0, "nw_t": 1.2, "ls_ann": 0.04})
    assert weak[0] == "毛收益为正，NW t 不足"


def test_cost_charges_both_sides_of_one_way_turnover():
    from factorlab.web.explain import month_net_returns

    nets = month_net_returns([0.01, 0.02], [None, 1.0], 0.001)
    assert nets[0] == 0.01
    assert abs(nets[1] - (0.02 - 0.002)) < 1e-12
