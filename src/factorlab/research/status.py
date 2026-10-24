"""研究状态。未通过样本外和完整成本检验时，不得写成通过。"""

from __future__ import annotations


def research_status(
    *,
    in_sample_candidate: bool,
    oos_nw_t: float,
    oos_net_positive: bool,
    holdout_net_positive: bool,
    holm_p: float,
    monotone: bool,
) -> tuple[str, str]:
    passed = (
        in_sample_candidate
        and oos_nw_t >= 1.96
        and oos_net_positive
        and holdout_net_positive
        and holm_p < 0.05
        and monotone
    )
    if passed:
        return "样本外仍成立", "pos"
    if in_sample_candidate:
        return "样本内候选，尚未验证", "muted"
    return "研究中", "muted"
