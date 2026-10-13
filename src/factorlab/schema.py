"""因子定义、字段契约与列名别名。"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class FactorSpec:
    """单因子说明书。

    direction = +1：因子值越高，预期下期收益越高，多头取最高分组。
    direction = -1：因子值越低，预期下期收益越高，多头取最低分组。
    原始因子列始终保持经济含义，不在存储时翻号。
    """

    name: str
    label: str
    family: str
    direction: int
    question: str
    definition: str
    formula: str

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name} 的 direction 只能是 +1 或 -1")

    @property
    def direction_text(self) -> str:
        if self.direction > 0:
            return "做多高因子值（最高分组）"
        return "做多低因子值（最低分组）"

    @property
    def neu_name(self) -> str:
        return f"{self.name}_neu"


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="sloan_accrual",
        label="应计利润（Sloan）",
        family="财务质量",
        direction=-1,
        question="应计利润更低的股票，下个月收益是否更高？",
        definition=(
            "资产负债表法应计利润，用平均总资产缩放。"
            "高应计往往对应更低的后续收益（Sloan, 1996）。"
        ),
        formula=(
            "[(Δ流动资产 − Δ现金) − (Δ流动负债 − Δ短期借款 − Δ应交税费) − 折旧摊销TTM]"
            " / 平均总资产"
        ),
    ),
    FactorSpec(
        name="capex_intensity",
        label="资本开支强度",
        family="财务质量",
        direction=-1,
        question="资本开支强度更低的股票，下个月收益是否更高？",
        definition=(
            "资本开支 TTM / 平均总资产。高资本开支公司后续收益往往更低"
            "（Titman, Wei, Xie, 2004），因此多头取低资本开支。"
        ),
        formula="资本开支TTM / 平均总资产",
    ),
    FactorSpec(
        name="dupont_pm",
        label="杜邦销售净利率",
        family="财务质量",
        direction=1,
        question="销售净利率更高的股票，下个月收益是否更高？",
        definition="杜邦分解的盈利能力驱动项。净利润 TTM / 营业收入 TTM。",
        formula="净利润TTM / 营业收入TTM",
    ),
    FactorSpec(
        name="dupont_at",
        label="杜邦资产周转率",
        family="财务质量",
        direction=1,
        question="资产周转更快的股票，下个月收益是否更高？",
        definition="杜邦分解的经营效率驱动项。营业收入 TTM / 平均总资产。",
        formula="营业收入TTM / 平均总资产",
    ),
    FactorSpec(
        name="dupont_em",
        label="杜邦权益乘数",
        family="财务质量",
        direction=-1,
        question="杠杆更低的股票，下个月收益是否更高？",
        definition=(
            "杜邦分解的杠杆驱动项。平均总资产 / 平均股东权益。"
            "质量组合默认规避高杠杆，因此方向为做多低权益乘数。"
            "原始 Rank IC 仍按未翻号的乘数报告。"
        ),
        formula="平均总资产 / 平均股东权益",
    ),
    FactorSpec(
        name="amihud",
        label="Amihud 非流动性",
        family="市场微观结构",
        direction=1,
        question="成交更困难的股票，下个月收益是否更高？",
        definition=(
            "月内日频 |收益率| / 成交额 的均值（Amihud, 2002）。"
            "数值越大，非流动性溢价越高。"
        ),
        formula="(1/D) Σ |r_d| / 成交额_d",
    ),
    FactorSpec(
        name="ivol",
        label="特质波动率",
        family="市场微观结构",
        direction=-1,
        question="特质波动更低的股票，下个月收益是否更高？",
        definition=(
            "月内日收益对市值加权市场收益做带截距回归后，残差标准差"
            "（Ang, Hodrick, Xing, Zhang, 2006 的 CAPM 版本）。"
            "高特质波动后续收益往往更低。"
        ),
        formula="sqrt( SSE / (N − 2) )，N 为月内有效交易日",
    ),
    FactorSpec(
        name="reversal",
        label="短期反转",
        family="市场微观结构",
        direction=1,
        question="上个月跌得多的股票，这个月是否涨回来？",
        definition=(
            "形成月已实现收益的相反数（Jegadeesh, 1990）。"
            "因子值越高，表示过去一个月跌得越多，做多近期输家。"
        ),
        formula="− 形成月收益率",
    ),
)


PRICE_COLUMNS = ("date", "ticker", "close", "volume", "mktcap", "industry")
PRICE_OPTIONAL = ("ret", "amount")
FUNDAMENTAL_COLUMNS = (
    "ticker",
    "fiscal_year",
    "fiscal_quarter",
    "available_date",
    "total_assets",
    "current_assets",
    "cash",
    "current_liabilities",
    "short_term_debt",
    "tax_payable",
    "depreciation",
    "net_income",
    "revenue",
    "capex",
    "equity",
)
WEIGHT_COLUMNS = ("date", "ticker", "weight")

ALIASES: dict[str, str] = {
    "日期": "date",
    "股票代码": "ticker",
    "证券代码": "ticker",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "市值": "mktcap",
    "行业": "industry",
    "收益率": "ret",
    "报告期年": "fiscal_year",
    "报告期季": "fiscal_quarter",
    "披露日": "available_date",
    "可用日": "available_date",
    "总资产": "total_assets",
    "流动资产": "current_assets",
    "货币资金": "cash",
    "流动负债": "current_liabilities",
    "短期借款": "short_term_debt",
    "应交税费": "tax_payable",
    "折旧摊销": "depreciation",
    "净利润": "net_income",
    "营业收入": "revenue",
    "资本开支": "capex",
    "股东权益": "equity",
    "权重": "weight",
}


def apply_aliases(df: pl.DataFrame) -> pl.DataFrame:
    rename = {
        src: dst
        for src, dst in ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    return df.rename(rename) if rename else df


def require_columns(df: pl.DataFrame, columns: tuple[str, ...], kind: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{kind}缺少字段：{', '.join(missing)}")


def factor_by_name(name: str) -> FactorSpec:
    for spec in FACTORS:
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in FACTORS)
    raise KeyError(f"未知因子 {name}。已知因子：{known}")
