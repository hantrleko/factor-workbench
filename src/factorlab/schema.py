"""因子定义、字段契约与列名别名。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


# 这 8 个名字在 factors 包里有计算函数。配置不能凭空多加一个。
COMPUTED_FACTORS = frozenset(
    {
        "sloan_accrual",
        "capex_intensity",
        "dupont_pm",
        "dupont_at",
        "dupont_em",
        "amihud",
        "ivol",
        "reversal",
    }
)


def load_factor_specs(path: Path | None = None) -> tuple[FactorSpec, ...]:
    """名称、方向和文字读配置文件。没有计算函数的名字会被拒绝。"""

    source = path or Path(__file__).resolve().parents[2] / "config" / "factors.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    specs: list[FactorSpec] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("enabled", True) is False:
            continue
        name = str(row["name"])
        if name in seen:
            raise ValueError(f"因子配置里 {name} 重复")
        if name not in COMPUTED_FACTORS:
            raise ValueError(f"配置里的 {name} 还没有计算函数。不会把它当成新的研究因子。")
        seen.add(name)
        specs.append(
            FactorSpec(
                name=name,
                label=str(row["label"]),
                family=str(row["family"]),
                direction=int(row["direction"]),
                question=str(row["question"]),
                definition=str(row["definition"]),
                formula=str(row["formula"]),
            )
        )
    missing = COMPUTED_FACTORS - seen
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"因子配置缺少已有计算：{joined}")
    return tuple(specs)


FACTORS: tuple[FactorSpec, ...] = load_factor_specs()


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
