"""Brinson 恒等式、Carino 链接和 Barra 约束。"""

import numpy as np

from factorlab.attribution.brinson import brinson_fachler, carino_link
from factorlab.risk.barra import _estimate_cross_section


def test_brinson_fachler_textbook_numbers():
    weight_p = np.array([0.4, 0.0, 0.6, 0.0])
    weight_b = np.array([0.0, 0.5, 0.0, 0.5])
    asset_return = np.array([0.12, 0.10, 0.06, 0.04])
    group = np.array(["A", "A", "B", "B"])
    result = brinson_fachler(weight_p, weight_b, asset_return, group, model="bf")
    assert abs(result.portfolio_return - 0.084) < 1e-12
    assert abs(result.benchmark_return - 0.07) < 1e-12
    assert abs(result.allocation - (-0.006)) < 1e-12
    assert abs(result.selection - 0.02) < 1e-12
    assert abs(result.interaction) < 1e-12
    assert abs(result.active_return - 0.014) < 1e-12


def test_new_industry_still_adds_up():
    weight_p = np.array([0.0, 1.0])
    weight_b = np.array([1.0, 0.0])
    asset_return = np.array([0.05, 0.20])
    group = np.array(["基准行业", "新行业"])
    result = brinson_fachler(weight_p, weight_b, asset_return, group)
    assert abs(result.allocation + result.selection + result.interaction - result.active_return) < 1e-12


def test_carino_links_to_cumulative_active_return():
    rp = np.array([0.10, -0.02, 0.05])
    rb = np.array([0.04, 0.01, 0.03])
    active = rp - rb
    linked = carino_link(rp, rb, active, np.zeros(3), np.zeros(3))
    growth_p = np.prod(1 + rp) - 1
    growth_b = np.prod(1 + rb) - 1
    assert abs(linked["active_return"] - (growth_p - growth_b)) < 1e-12
    assert abs(linked["allocation_total"] - linked["active_return"]) < 1e-10
    assert abs(linked["selection_total"]) < 1e-12
    assert abs(linked["interaction_total"]) < 1e-12


def test_barra_industry_constraint_and_shapes():
    rng = np.random.default_rng(4)
    n = 60
    ret = rng.normal(0.01, 0.04, size=n)
    cap = rng.lognormal(10.0, 0.5, size=n)
    industry = np.array([f"I{i % 3}" for i in range(n)])
    styles = rng.normal(size=(n, 2))
    pack = _estimate_cross_section(
        ret, cap, industry, styles, ["价值", "动量"], weight="sqrt_cap", min_extra=5
    )
    assert pack is not None
    assert abs(pack["cap_share_dot"]) < 1e-8
    assert pack["exposure"].shape == (pack["idx"].size, 1 + 3 + 2)
    assert pack["exposure"].shape[0] == pack["resid"].shape[0]
    beta_len = 1 + 2 + 2  # 国家 + 两个自由行业系数 + 两个风格
    assert pack["exposure"].shape[1] == 6
    del beta_len
