# 轻量级多因子检验与业绩归因工作台

面向买方投研的单因子检验和组合归因。因子、中性化、Rank IC、分组回测、Brinson 和 Barra 分成独立模块，每月截面单独回归，避免把不同日期或不同长度的数组广播到一起。

默认样本是可复现的合成数据，用来核对计算链路。嵌入的因子溢价写在数据生成过程里，不是市场结论。换成自己的行情和财务表之后，同一条命令会出报告。

## 运行

```powershell
cd F:\factor-workbench
$env:PYTHONPATH = "src"
python -m factorlab --demo --out output/demo
```

报告在 `output/demo/factor_report.html` 和 `factor_report.pdf`。本机没有 WeasyPrint 所依赖的 Pango 时，PDF 由 fpdf2 生成，页面内容与 HTML 对应；装好 WeasyPrint 后会自动改用 HTML 渲染。

打开工作台网站：

```powershell
python -m factorlab.web --out output/demo --port 5057
```

浏览器访问 http://127.0.0.1:5057 。页面从该目录的 `summary.json` 和 `figures` 读取，可在总览、单因子、归因和风险之间切换。仓库里的 `demo/` 是同一份结果，线上部署没有本机 `output/demo` 时用它。

外部数据：

```powershell
python -m factorlab --prices data/prices.parquet --fundamentals data/fundamentals.parquet --out output/live
```

可选 `--weights` 和 `--benchmark`，列是 `date, ticker, weight`。不提供时，组合是复合得分最高的五分位等权，基准是市值加权。

## 因子

| 因子 | 定义 | 做多方向 |
| --- | --- | --- |
| 应计利润 | Sloan 资产负债表应计 / 平均总资产 | 低应计 |
| 资本开支强度 | 资本开支 TTM / 平均总资产 | 低资本开支 |
| 销售净利率、资产周转率 | 杜邦经营驱动项 | 高 |
| 权益乘数 | 平均总资产 / 平均股东权益 | 低杠杆 |
| Amihud 非流动性 | 月内日频 \|收益\| / 成交额 | 高非流动性 |
| 特质波动率 | 日收益对市值加权市场的残差波动 | 低波动 |
| 短期反转 | 形成月收益的相反数 | 高（近期下跌） |

财务因子只使用 `available_date <= 月末` 的报表。同比和 TTM 都从这份已披露报告往前取，不会用到尚未披露的财季。远期收益是形成月末之后的自然月，不包含形成月本身。

中性化在每个月末做市值平方根加权最小二乘：截距、对数市值、行业哑元（去掉一个基准行业）。残差与市值正交，每个行业的加权残差均值应为 0。报告里有这项检查。

## 行情字段

`date, ticker, close, volume, mktcap, industry`。可选 `ret, amount`。没有收益时用收盘价计算，没有成交额时用收盘价乘成交量。中文列名（日期、股票代码、收盘价、成交量、市值、行业、成交额、收益率）会自动识别。

## 财务字段

`ticker, fiscal_year, fiscal_quarter, available_date, total_assets, current_assets, cash, current_liabilities, short_term_debt, tax_payable, depreciation, net_income, revenue, capex, equity`。

折旧摊销、资本开支、净利润、营业收入按报告期填写，程序自己滚四个季度。应交税费或短期借款如果没有，请把该列填 0，不要删列。

## 归因与风险

Brinson–Fachler 把主动收益拆成配置、选股和交互。多期用 Carino 系数链接，三项之和等于累计主动收益。基准没有持仓的行业，基准收益用基准总收益代替。

Barra 部分是可核对的子集：国家因子、市值加权和为零的行业因子、对数市值以及中性化后的研究因子。回归权重是市值平方根。它不是商业风险模型的复制。

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```
