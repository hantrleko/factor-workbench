"""从东财公告日报表和证券宝行情组装一个可复现的沪市研究池。

上市腿是写死的 8 只 2016 年前已上市的非金融股，不是历史指数成分重建。
退市腿来自上交所退市名单，只在上市日至退市日之间进入截面。
收益使用前复权价，从而计入分红和拆股。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import akshare as ak
import baostock as bs
import pandas as pd
import polars as pl

RELEASE = Path(__file__).resolve().parents[3] / "data" / "releases" / "cn_pit_v1"
CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "cn_pit_v1"

LISTED_CODES = (
    "600519",
    "600276",
    "600887",
    "600309",
    "600585",
    "600031",
    "600436",
    "600690",
)
LISTED_NAMES = {
    "600519": "贵州茅台",
    "600276": "恒瑞医药",
    "600887": "伊利股份",
    "600309": "万华化学",
    "600585": "海螺水泥",
    "600031": "三一重工",
    "600436": "片仔癀",
    "600690": "海尔智家",
}
UNIVERSE_RULE = (
    "退市腿为上交所 2016 年前上市、2022–2024 年退市的 A 股，按退市日从新到旧取最多 8 只。"
    "上市腿为写死的 8 只 2016 年前已上市的非金融股，不是历史沪深300成分重建。"
    "股票只在上市日至退市日之间进入截面。"
)
LIMITATIONS = (
    "上市腿使用抓取日仍在交易的公司，这一腿有幸存者偏差",
    "行业是抓取日的东财行业，不是历史时点行业",
    "市值是换手率反推的流通市值，不是总市值",
    "退市日使用上交所名单中的暂停上市日期字段",
    "行情缓存为 2019-01-01 至 2024-12-31 的证券宝日线",
    "现金流量表补充资料里的折旧在一季报和三季报经常缺失；缺失时不倒推，应计因子沿用最近一份已经公告且能够计算的报表",
)


def _cache_path(kind: str, key: str) -> Path:
    path = CACHE / kind
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.parquet"


def _read_cache(kind: str, key: str) -> pd.DataFrame | None:
    path = _cache_path(kind, key)
    if path.is_file():
        return pd.read_parquet(path)
    return None


def _write_cache(kind: str, key: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(_cache_path(kind, key), index=False)


def _call(kind: str, key: str, func):
    cached = _read_cache(kind, key)
    if cached is not None and len(cached):
        return cached
    last_error = None
    for attempt in range(5):
        try:
            frame = func()
            if frame is not None and len(frame):
                _write_cache(kind, key, frame)
                return frame
        except Exception as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    if last_error:
        raise last_error
    return pd.DataFrame()


def select_delisted(limit: int = 8) -> list[dict]:
    frame = _call("meta", "sh_delist", lambda: ak.stock_info_sh_delist(symbol="全部"))
    frame = frame.rename(columns={frame.columns[0]: "code", frame.columns[1]: "name", frame.columns[2]: "listed", frame.columns[3]: "delist"})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].str.startswith("60")]
    frame["listed"] = pd.to_datetime(frame["listed"])
    frame["delist"] = pd.to_datetime(frame["delist"])
    frame = frame[(frame["listed"] <= "2016-01-01") & (frame["delist"].between("2022-01-01", "2024-12-31"))]
    frame = frame.sort_values("delist", ascending=False).head(limit)
    return [
        {
            "code": row.code,
            "name": row.name,
            "listed": row.listed.date().isoformat(),
            "delist": row.delist.date().isoformat(),
            "leg": "delisted",
        }
        for row in frame.itertuples(index=False)
    ]


KNOWN_INDUSTRY = {
    "600519": "白酒",
    "600276": "化学制药",
    "600887": "食品",
    "600309": "化学制品",
    "600585": "水泥",
    "600031": "工程机械",
    "600436": "中药",
    "600690": "白色家电",
    "600104": "汽车",
    "600048": "房地产开发",
    "600009": "机场",
    "601012": "光伏设备",
    "601888": "旅游零售",
    "600406": "电网设备",
    "600760": "航空装备",
}


def _industry(code: str) -> str:
    if code in KNOWN_INDUSTRY:
        return KNOWN_INDUSTRY[code]
    try:
        info = _call("meta", f"info_{code}", lambda: ak.stock_individual_info_em(symbol=code))
    except Exception:
        return "未分类"
    column_item = info.columns[0]
    column_value = info.columns[1]
    mapping = dict(zip(info[column_item].astype(str), info[column_value].astype(str)))
    return mapping.get("行业", "未分类")


def _prices(code: str) -> pd.DataFrame:
    """未复权价算市值，前复权价算收益，从而把分红和拆股算进收益率。"""

    cached = _read_cache("prices", f"{code}_panel")
    if cached is not None and len(cached):
        cached["日期"] = pd.to_datetime(cached["日期"])
        return cached
    try:
        frame = _prices_baostock(code)
    except Exception:
        frame = _prices_sina(code)
    _write_cache("prices", f"{code}_panel", frame.assign(日期=frame["日期"].astype(str)))
    frame["日期"] = pd.to_datetime(frame["日期"])
    return frame


def _prices_baostock(code: str) -> pd.DataFrame:
    fields = "date,close,volume,amount,turn"

    def grab(flag: str, key: str) -> pd.DataFrame:
        def run():
            _ensure_bs_login()
            result = bs.query_history_k_data_plus(
                f"sh.{code}",
                fields,
                start_date="2019-01-01",
                end_date="2024-12-31",
                frequency="d",
                adjustflag=flag,
            )
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise RuntimeError(result.error_msg)
            frame = pd.DataFrame(rows, columns=fields.split(","))
            if frame.empty:
                raise RuntimeError("空行情")
            return frame

        return _call("prices", f"{code}_{key}", run)

    raw = grab("3", "raw")
    adjusted = grab("2", "qfq")
    for frame in (raw, adjusted):
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ["close", "volume", "amount", "turn"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    merged = raw.merge(adjusted[["date", "close"]], on="date", suffixes=("", "_adj")).sort_values("date")
    shares = merged["volume"] / (merged["turn"] / 100.0).replace(0, pd.NA)
    return pd.DataFrame(
        {
            "日期": merged["date"],
            "收盘": merged["close"],
            "成交量": merged["volume"],
            "成交额": merged["amount"],
            "mktcap": merged["close"] * shares,
            "ret": merged["close_adj"].pct_change(),
        }
    )


def _prices_sina(code: str) -> pd.DataFrame:
    def run(adjust: str) -> pd.DataFrame:
        frame = ak.stock_zh_a_daily(
            symbol=f"sh{code}",
            start_date="20190101",
            end_date="20241231",
            adjust=adjust,
        )
        if frame is None or len(frame) == 0:
            raise RuntimeError("新浪无行情")
        return frame

    raw = _call("prices", f"{code}_sina_raw", lambda: run(""))
    adjusted = _call("prices", f"{code}_sina_qfq", lambda: run("qfq"))
    raw = raw.copy()
    adjusted = adjusted.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    adjusted["date"] = pd.to_datetime(adjusted["date"])
    merged = raw.merge(adjusted[["date", "close"]], on="date", suffixes=("", "_adj")).sort_values("date")
    for column in ["close", "volume", "amount", "outstanding_share", "close_adj"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return pd.DataFrame(
        {
            "日期": merged["date"],
            "收盘": merged["close"],
            "成交量": merged["volume"],
            "成交额": merged["amount"],
            "mktcap": merged["close"] * merged["outstanding_share"],
            "ret": merged["close_adj"].pct_change(),
        }
    )


_BS_READY = False


def _ensure_bs_login() -> None:
    global _BS_READY
    if _BS_READY:
        return
    last_error = "登录失败"
    for _attempt in range(2):
        login = bs.login()
        if login.error_code == "0":
            _BS_READY = True
            return
        last_error = login.error_msg
        time.sleep(1)
    raise RuntimeError(last_error)


def _statement(code: str, kind: str):
    symbol = f"SH{code}"
    for key in (f"{code}_{kind}", f"{code}_{kind}_delisted"):
        cached = _read_cache("statements", key)
        if cached is not None and len(cached):
            return cached
    functions = {
        "bs": (ak.stock_balance_sheet_by_report_em, ak.stock_balance_sheet_by_report_delisted_em),
        "is": (ak.stock_profit_sheet_by_report_em, ak.stock_profit_sheet_by_report_delisted_em),
        "cf": (ak.stock_cash_flow_sheet_by_report_em, ak.stock_cash_flow_sheet_by_report_delisted_em),
    }
    listed_fn, delisted_fn = functions[kind]

    def pull(fn):
        return fn(symbol=symbol)

    try:
        return _call("statements", f"{code}_{kind}", lambda: pull(listed_fn))
    except Exception:
        return _call("statements", f"{code}_{kind}_delisted", lambda: pull(delisted_fn))


def _quarter_from_type(report_type: str) -> int | None:
    mapping = {"一季报": 1, "中报": 2, "三季报": 3, "年报": 4}
    return mapping.get(str(report_type))


def _ytd_to_quarter(values: pd.Series, quarters: pd.Series) -> pd.Series:
    """中报、三季报和年报是年初至今。只有紧邻的上一季存在时才相减，否则该季流量记为空。"""

    out = []
    previous = None
    previous_quarter = None
    for value, quarter in zip(values, quarters, strict=True):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number) or quarter is None or (isinstance(quarter, float) and pd.isna(quarter)):
            out.append(pd.NA)
            continue
        quarter = int(quarter)
        if quarter == 1:
            flow = number
        elif previous is not None and previous_quarter == quarter - 1:
            flow = number - previous
        else:
            flow = pd.NA
        out.append(flow)
        if pd.notna(number):
            previous = number
            previous_quarter = quarter
    return pd.Series(out, index=values.index)


def _numeric(frame: pd.DataFrame, name: str, fill: float | None = None) -> pd.Series:
    if name not in frame.columns:
        series = pd.Series([fill] * len(frame), index=frame.index, dtype="float64")
        return series
    values = pd.to_numeric(frame[name], errors="coerce")
    if fill is not None:
        values = values.fillna(fill)
    return values


def _fundamentals_one(code: str) -> pd.DataFrame:
    balance = _statement(code, "bs")
    income = _statement(code, "is")
    cash = _statement(code, "cf")
    keys = ["REPORT_DATE", "REPORT_TYPE", "NOTICE_DATE"]
    merged = balance[keys + [c for c in ["TOTAL_ASSETS", "TOTAL_LIAB_EQUITY", "TOTAL_CURRENT_ASSETS", "MONETARYFUNDS", "TOTAL_CURRENT_LIAB", "SHORT_LOAN", "TAX_PAYABLE", "TOTAL_EQUITY"] if c in balance.columns]]
    merged = merged.merge(income[keys + [c for c in ["OPERATE_INCOME", "NETPROFIT"] if c in income.columns]], on=keys, how="inner")
    cash_cols = [c for c in ["CONSTRUCT_LONG_ASSET", "FA_IR_DEPR"] if c in cash.columns]
    merged = merged.merge(cash[keys + cash_cols], on=keys, how="left")
    merged["NOTICE_DATE"] = pd.to_datetime(merged["NOTICE_DATE"])
    merged["REPORT_DATE"] = pd.to_datetime(merged["REPORT_DATE"])
    merged = merged.dropna(subset=["NOTICE_DATE"])
    merged["quarter"] = merged["REPORT_TYPE"].map(_quarter_from_type)
    merged = merged.dropna(subset=["quarter"]).sort_values("REPORT_DATE")
    merged = merged[merged["NOTICE_DATE"] >= merged["REPORT_DATE"]].copy()
    ytd_columns = {
        "revenue_ytd": "OPERATE_INCOME",
        "net_income_ytd": "NETPROFIT",
        "capex_ytd": "CONSTRUCT_LONG_ASSET",
        "depreciation_ytd": "FA_IR_DEPR",
    }
    ytd_values = {name: _numeric(merged, source) for name, source in ytd_columns.items()}
    for column in ["OPERATE_INCOME", "NETPROFIT", "CONSTRUCT_LONG_ASSET", "FA_IR_DEPR"]:
        if column in merged.columns:
            merged[column] = _ytd_to_quarter(merged[column], merged["quarter"])
    assets = _numeric(merged, "TOTAL_ASSETS")
    if assets.isna().all():
        assets = _numeric(merged, "TOTAL_LIAB_EQUITY")
    return pd.DataFrame(
        {
            "ticker": code,
            "fiscal_year": merged["REPORT_DATE"].dt.year.astype(int),
            "fiscal_quarter": merged["quarter"].astype(int),
            "available_date": merged["NOTICE_DATE"].dt.date,
            "total_assets": assets,
            "current_assets": _numeric(merged, "TOTAL_CURRENT_ASSETS"),
            "cash": _numeric(merged, "MONETARYFUNDS"),
            "current_liabilities": _numeric(merged, "TOTAL_CURRENT_LIAB"),
            "short_term_debt": _numeric(merged, "SHORT_LOAN", fill=0.0),
            "tax_payable": _numeric(merged, "TAX_PAYABLE", fill=0.0),
            "depreciation": _numeric(merged, "FA_IR_DEPR"),
            "net_income": _numeric(merged, "NETPROFIT"),
            "revenue": _numeric(merged, "OPERATE_INCOME"),
            "capex": _numeric(merged, "CONSTRUCT_LONG_ASSET"),
            "equity": _numeric(merged, "TOTAL_EQUITY"),
            "revenue_ytd": ytd_values["revenue_ytd"],
            "net_income_ytd": ytd_values["net_income_ytd"],
            "capex_ytd": ytd_values["capex_ytd"],
            "depreciation_ytd": ytd_values["depreciation_ytd"],
        }
    )


    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        return _build_tables_logged_in()
    finally:
        bs.logout()


def _build_tables_logged_in() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    delisted = select_delisted(8)
    members = list(delisted)
    for code in LISTED_CODES:
        members.append(
            {
                "code": code,
                "name": LISTED_NAMES.get(code, code),
                "listed": "2016-01-01",
                "delist": None,
                "leg": "listed_snapshot",
            }
        )
    price_frames = []
    fundamental_frames = []
    industries = {}
    failures = []
    for member in members:
        code = member["code"]
        print(f"fetch {code} {member['leg']}", flush=True)
        try:
            industries[code] = KNOWN_INDUSTRY.get(code, "已退市" if member["leg"] == "delisted" else "未分类")
            prices = _prices(code)
            listed = pd.Timestamp(member["listed"])
            delist = pd.Timestamp(member["delist"]) if member["delist"] else None
            keep = prices["日期"] >= listed
            if delist is not None:
                keep &= prices["日期"] <= delist
            prices = prices.loc[keep].copy()
            if prices.empty:
                failures.append({"code": code, "error": "退市日之前没有行情"})
                continue
            price_frames.append(
                pd.DataFrame(
                    {
                        "date": prices["日期"].dt.date,
                        "ticker": code,
                        "close": pd.to_numeric(prices["收盘"], errors="coerce"),
                        "volume": pd.to_numeric(prices["成交量"], errors="coerce"),
                        "amount": pd.to_numeric(prices["成交额"], errors="coerce"),
                        "ret": prices["ret"],
                        "mktcap": prices["mktcap"],
                        "industry": industries[code],
                    }
                )
            )
            fundamental_frames.append(_fundamentals_one(code))
        except Exception as exc:
            failures.append({"code": code, "error": str(exc)[:300]})
        time.sleep(0.4)
    if not price_frames:
        raise RuntimeError("没有抓到任何股票的行情")
    prices_out = pl.from_pandas(pd.concat(price_frames, ignore_index=True)).drop_nulls(["date", "close", "ret"])
    funds_out = pl.from_pandas(pd.concat(fundamental_frames, ignore_index=True))
    provenance = {
        "dataset_id": "cn_pit_v1",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "akshare_version": getattr(ak, "__version__", "unknown"),
        "price_source": "证券宝日行情。收益用前复权收盘价，计入分红和拆股；市值用未复权收盘价乘以成交量除以换手率",
        "market_cap_source": "未复权收盘价 × 成交量 / 换手率，近似自由流通市值",
        "fundamental_source": "东方财富按报告期资产负债表、利润表、现金流量表",
        "announcement_rule": "available_date = NOTICE_DATE，且必须不早于 REPORT_DATE；公告日缺失则丢弃该期",
        "universe_rule": UNIVERSE_RULE,
        "members": members,
        "industries_asof_fetch": industries,
        "failures": failures,
        "limitations": list(LIMITATIONS),
    }
    return prices_out, funds_out, provenance


def build_tables() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    try:
        return _build_tables_logged_in()
    finally:
        if _BS_READY:
            bs.logout()


def refresh_fundamentals_from_cache(release: Path = RELEASE) -> dict:
    """用已经落地的报表缓存重算财务表，不再访问网络。"""

    provenance_path = release / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    frames = []
    for member in provenance["members"]:
        code = member["code"]
        if member.get("leg") == "listed_snapshot":
            member["name"] = LISTED_NAMES.get(code, member.get("name") or code)
        frames.append(_fundamentals_one(code))
    funds = pl.from_pandas(pd.concat(frames, ignore_index=True))
    skip = {"ticker", "fiscal_year", "fiscal_quarter", "available_date"}
    funds = funds.with_columns(
        [pl.col(col).cast(pl.Float64, strict=False).fill_nan(None) for col in funds.columns if col not in skip]
    )
    if funds["available_date"].dtype != pl.Date:
        funds = funds.with_columns(pl.col("available_date").cast(pl.Date))
    release.mkdir(parents=True, exist_ok=True)
    funds.write_parquet(release / "fundamentals.parquet")
    provenance["universe_rule"] = UNIVERSE_RULE
    provenance["limitations"] = list(LIMITATIONS)
    provenance["fundamentals_built_at"] = datetime.now().isoformat(timespec="seconds")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def fetch_and_store(release: Path = RELEASE) -> dict:
    prices, funds, provenance = build_tables()
    release.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(release / "prices.parquet")
    funds.write_parquet(release / "fundamentals.parquet")
    (release / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance
