"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from factorlab.data.loader import load_fundamentals, load_prices, load_weights
from factorlab.data.synthetic import make_synthetic_universe
from factorlab.pipeline import run_analysis
from factorlab.report.formatters import num, pct


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="轻量级多因子检验与 Brinson/Barra 归因工作台")
    parser.add_argument("--demo", action="store_true", help="使用可复现的合成样本")
    parser.add_argument("--prices", help="行情 parquet 或 csv")
    parser.add_argument("--fundamentals", help="财务 parquet 或 csv")
    parser.add_argument("--weights", help="可选组合权重，列：date, ticker, weight")
    parser.add_argument("--benchmark", help="可选基准权重")
    parser.add_argument("--out", default="output/demo", help="输出目录")
    parser.add_argument("--groups", default="5,10", help="分组数，逗号分隔")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stocks", type=int, default=48)
    parser.add_argument("--start", default="2021-01-04")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--no-premium", action="store_true", help="合成数据不嵌入因子溢价")
    parser.add_argument("--no-pdf", action="store_true", help="只写表格，不生成报告")
    args = parser.parse_args(argv)

    groups = tuple(int(part) for part in args.groups.split(",") if part.strip())
    if args.demo or (not args.prices and not args.fundamentals):
        prices, fundamentals = make_synthetic_universe(
            n_stocks=args.stocks,
            start=args.start,
            end=args.end,
            seed=args.seed,
            embed_premium=not args.no_premium,
        )
        note = (
            "样本为合成数据，用来核对计算链路。"
            + (
                "数据生成过程嵌入了温和的因子溢价，报告里的 IC 和多空收益不是市场结论。"
                if not args.no_premium
                else "本次未嵌入因子溢价。"
            )
            + f" 随机种子 {args.seed}。"
        )
        meta = {"note": note, "seed": args.seed, "embed_premium": not args.no_premium}
        data_dir = Path(args.out) / "synthetic"
        data_dir.mkdir(parents=True, exist_ok=True)
        prices.write_parquet(data_dir / "prices.parquet")
        fundamentals.write_parquet(data_dir / "fundamentals.parquet")
    else:
        if not args.prices or not args.fundamentals:
            parser.error("外部数据需要同时提供 --prices 和 --fundamentals")
        prices = load_prices(args.prices)
        fundamentals = load_fundamentals(args.fundamentals)
        meta = {"note": "样本来自调用方提供的行情与财务数据。因子只使用披露日不晚于月末的报表。"}

    result = run_analysis(
        prices,
        fundamentals,
        portfolio_weights=load_weights(args.weights) if args.weights else None,
        benchmark_weights=load_weights(args.benchmark) if args.benchmark else None,
        n_groups=groups,
        out_dir=args.out,
        make_report=not args.no_pdf,
        meta=meta,
    )
    print()
    print(f"{'因子':<16}{'策略IC':>10}{'IC_IR':>8}{'t':>8}{'多空年化':>10}{'回撤':>10}")
    for item in result.factor_reports:
        stats = item["ic_stats"]
        bt = item["backtests"][groups[0]]
        label = item["spec"].label
        print(
            f"{label:<16}"
            f"{num(stats['mean'], 3):>10}"
            f"{num(stats['ir'], 2):>8}"
            f"{num(stats['t'], 2):>8}"
            f"{pct(bt.summary.get('ann_return')):>10}"
            f"{pct(bt.summary.get('max_drawdown')):>10}"
        )
    totals = result.brinson_totals
    print()
    print(
        "Brinson 累计主动收益 "
        f"{pct(totals['active_return'])} = 配置 {pct(totals['allocation'])}"
        f" + 选股 {pct(totals['selection'])} + 交互 {pct(totals['interaction'])}"
    )
    if result.artifacts.get("html"):
        engine = result.artifacts.get("pdf_engine", "")
        print(f"HTML  {result.artifacts['html']}")
        print(f"PDF   {result.artifacts.get('pdf', '')}  ({engine})")
