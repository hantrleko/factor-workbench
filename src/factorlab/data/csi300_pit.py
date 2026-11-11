"""按沪深 300 历史成分拉取行情和公告日财务。失败的股票记下来，下次从缓存继续。"""

from __future__ import annotations

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from factorlab.data.csi300_membership import SOURCE_URL, load_history, symbols_overlapping

RELEASE = Path(__file__).resolve().parents[3] / "data" / "releases" / "csi300_pit_v1"
CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "csi300_pit_v1"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path


def active_cache() -> Path:
    return _env_path("FACTORLAB_CACHE", CACHE)


def active_release() -> Path:
    return _env_path("FACTORLAB_RELEASE", RELEASE)


def active_history() -> Path:
    from factorlab.data.csi300_membership import HISTORY_PATH

    return _env_path("FACTORLAB_MEMBERSHIP", HISTORY_PATH)
PRICE_START = "2018-01-01"
PRICE_END = "2024-12-31"
SAMPLE_START = "2019-01-01"
SAMPLE_END = "2024-12-31"

UNIVERSE_RULE = (
    "股票池是沪深300历史成分，成员关系来自中证指数公司公告整理的调入日和调出日。"
    "形成日当天必须已经调入且尚未调出。退市股只在它仍是成分股的日期出现，不再单独加权。"
    f"成分表来源 {SOURCE_URL} 。"
)
LIMITATIONS = (
    "行业用东方财富报表上的行业名称，按公告日向后对齐；该字段有可能被数据商回填成较新的分类。"
    "某只股票第一次公告之前的交易日，沿用它最早一份报表上的行业名称。"
    "报表里始终没有行业的股票，用 data/reference/csi300_industry_gap.csv 里的当前行业补齐，这不是历史变更序列",
    "市值是未复权收盘价乘以成交量除以换手率，近似自由流通市值",
    "收益用前复权收盘价，计入分红和拆股",
    "特质波动率里的市场收益是样本价格的市值加权，不是沪深300官方指数收益",
    "银行等缺少流动资产或折旧的公司，在应计因子截面里自然空缺",
    "折旧在一季报和三季报经常缺失；缺失时不倒推，应计因子沿用最近一份已经公告且能够计算的报表",
)


def quarter_ends(start_year: int = 2017, end_year: int = 2024) -> list[str]:
    days = []
    for year in range(start_year, end_year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            days.append(date(year, month, day).isoformat())
    return days


def _cache(kind: str, key: str) -> Path:
    path = active_cache() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.parquet"


def _read(kind: str, key: str) -> pd.DataFrame | None:
    roots = [active_cache()]
    if active_cache().resolve() != CACHE.resolve():
        roots.append(CACHE)
    for root in roots:
        path = root / kind / f"{key}.parquet"
        if path.is_file() and path.stat().st_size > 0:
            try:
                return pd.read_parquet(path)
            except Exception:
                continue
    return None


def _write(kind: str, key: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(_cache(kind, key), index=False)


def target_symbols() -> pd.DataFrame:
    history = load_history(active_history())
    return symbols_overlapping(history, SAMPLE_START, SAMPLE_END).to_pandas()


def _bs_symbol(code: str, exchange: str) -> str:
    return f"{exchange.lower()}.{code}"


def _em_symbol(code: str, exchange: str) -> str:
    return f"{exchange.upper()}{code}"


def fetch_industries() -> pd.DataFrame:
    frames = []
    for day in quarter_ends(2018, 2024):
        cached = _read("industry", day)
        if cached is None:
            print(f"industry {day}", flush=True)
            cached = _industry_one(day)
            _write("industry", day, cached)
        frames.append(cached)
    tape = pd.concat(frames, ignore_index=True)
    tape.to_parquet(active_cache() / "industry_tape.parquet", index=False)
    return tape


def _get_json(url: str, params: dict, attempts: int = 5) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=40)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("空响应")


def _industry_one(day: str) -> pd.DataFrame:
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_DMSK_FN_INCOME",
        "columns": "SECURITY_CODE,NOTICE_DATE,INDUSTRY_NAME",
        "filter": f"(REPORT_DATE='{day}')",
        "pageSize": "500",
        "pageNumber": "1",
        "sortColumns": "NOTICE_DATE",
        "sortTypes": "-1",
    }
    payload = _get_json(url, params)["result"]
    pages = int(payload["pages"])
    rows = list(payload["data"] or [])
    for page in range(2, pages + 1):
        params["pageNumber"] = str(page)
        rows.extend(_get_json(url, params)["result"]["data"] or [])
        time.sleep(0.2)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "available_date", "industry"])
    frame = frame.rename(columns={"SECURITY_CODE": "ticker", "NOTICE_DATE": "available_date", "INDUSTRY_NAME": "industry"})
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce").dt.date.astype(str)
    return frame.loc[frame["industry"].notna(), ["ticker", "available_date", "industry"]]


def fetch_prices(symbols: pd.DataFrame | None = None) -> dict:
    import baostock as bs

    symbols = target_symbols() if symbols is None else symbols
    failures = []
    def relogin() -> None:
        bs.logout()
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)

    socket.setdefaulttimeout(20)
    relogin()
    try:
        fetched = 0
        for index, row in enumerate(symbols.itertuples(index=False), start=1):
            key = f"{row.code}_panel"
            if _read("prices", key) is not None:
                continue
            if fetched and fetched % 40 == 0:
                relogin()
            print(f"price {index}/{len(symbols)} {row.code}", flush=True)
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    frame = _price_one(row.code, row.exchange)
                    _write("prices", key, frame)
                    fetched += 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"price retry {row.code} {attempt + 1} {exc}", flush=True)
                    time.sleep(1.5 * (attempt + 1))
                    try:
                        relogin()
                    except Exception as login_exc:
                        print(f"relogin {login_exc}", flush=True)
                        time.sleep(3)
            if last_error is not None:
                failures.append({"code": row.code, "error": str(last_error)[:240]})
                print(f"price fail {row.code} {last_error}", flush=True)
            time.sleep(0.05)
    finally:
        bs.logout()
    (active_cache() / "price_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"requested": int(len(symbols)), "failures": failures}


def _price_one(code: str, exchange: str) -> pd.DataFrame:
    import baostock as bs

    fields = "date,close,volume,amount,turn"

    def grab(flag: str) -> pd.DataFrame:
        result = bs.query_history_k_data_plus(
            _bs_symbol(code, exchange),
            fields,
            start_date=PRICE_START,
            end_date=PRICE_END,
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
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ["close", "volume", "amount", "turn"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    raw = grab("3")
    adjusted = grab("2")
    merged = raw.merge(adjusted[["date", "close"]], on="date", suffixes=("", "_adj")).sort_values("date")
    shares = merged["volume"] / (merged["turn"] / 100.0).replace(0, pd.NA)
    return pd.DataFrame(
        {
            "date": merged["date"].dt.date.astype(str),
            "ticker": code,
            "close": merged["close"],
            "volume": merged["volume"],
            "amount": merged["amount"],
            "mktcap": merged["close"] * shares,
            "ret": merged["close_adj"].pct_change(),
        }
    )


def fetch_fundamentals(symbols: pd.DataFrame | None = None, workers: int = 4) -> dict:
    symbols = target_symbols() if symbols is None else symbols
    dates = quarter_ends()
    failures = []
    pending = []
    for row in symbols.itertuples(index=False):
        if _read("fundamentals", row.code) is None:
            pending.append((row.code, row.exchange))
    print(f"fundamentals pending {len(pending)}/{len(symbols)}", flush=True)

    def one(item: tuple[str, str]) -> tuple[str, str | None]:
        code, exchange = item
        try:
            frame = _fundamental_one(code, exchange, dates)
            _write("fundamentals", code, frame)
            return code, None
        except Exception as exc:
            return code, str(exc)[:240]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            code, error = future.result()
            if error:
                failures.append({"code": code, "error": error})
                print(f"fund fail {index}/{len(pending)} {code} {error}", flush=True)
            elif index % 20 == 0:
                print(f"fund {index}/{len(pending)}", flush=True)
    (active_cache() / "fundamental_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"requested": int(len(symbols)), "pending": len(pending), "failures": failures}


def _company_type(symbol: str) -> str:
    roots = [active_cache()]
    if active_cache().resolve() != CACHE.resolve():
        roots.append(CACHE)
    for root in roots:
        cached = root / "meta" / f"ctype_{symbol}.txt"
        if cached.is_file():
            return cached.read_text(encoding="utf-8").strip() or "4"
    cached = active_cache() / "meta" / f"ctype_{symbol}.txt"
    cached.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    text = ""
    for attempt in range(5):
        try:
            response = requests.get(
                "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index",
                params={"type": "web", "code": symbol.lower()},
                timeout=30,
            )
            response.raise_for_status()
            text = response.text
            break
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        if last_error:
            raise last_error
    marker = 'id="hidctype" value="'
    start = text.find(marker)
    value = text[start + len(marker) : start + len(marker) + 1] if start >= 0 else "4"
    if not value.isdigit():
        value = "4"
    cached.write_text(value, encoding="utf-8")
    return value


def _statement(symbol: str, kind: str, company_type: str, dates: list[str]) -> pd.DataFrame:
    endpoints = {"bs": "zcfzbAjaxNew", "is": "lrbAjaxNew", "cf": "xjllbAjaxNew"}
    frames = []
    for offset in range(0, len(dates), 5):
        batch = ",".join(dates[offset : offset + 5])
        payload = _get_json(
            f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/{endpoints[kind]}",
            {
                "companyType": company_type,
                "reportDateType": "0",
                "reportType": "1",
                "dates": batch,
                "code": symbol,
            },
        )
        rows = payload.get("data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(0.05)
    if not frames:
        raise RuntimeError(f"{symbol} {kind} 没有报表")
    return pd.concat(frames, ignore_index=True)


def _fundamental_one(code: str, exchange: str, dates: list[str]) -> pd.DataFrame:
    symbol = _em_symbol(code, exchange)
    company_type = "4"
    try:
        balance = _statement(symbol, "bs", company_type, dates)
    except RuntimeError:
        company_type = _company_type(symbol)
        balance = _statement(symbol, "bs", company_type, dates)
    income = _statement(symbol, "is", company_type, dates)
    cash = _statement(symbol, "cf", company_type, dates)
    from factorlab.data.cn_pit import statements_to_fundamentals

    frame = statements_to_fundamentals(code, balance, income, cash)
    if frame.empty:
        raise RuntimeError("公告日过滤后没有报表")
    return frame


def assemble_release() -> dict:
    """把已经落地的缓存收成 prices.parquet 和 fundamentals.parquet。"""

    symbols = target_symbols()
    prices = []
    funds = []
    missing_prices = []
    missing_funds = []
    for row in symbols.itertuples(index=False):
        price = _read("prices", f"{row.code}_panel")
        if price is None:
            missing_prices.append(row.code)
        else:
            prices.append(price)
        fund = _read("fundamentals", row.code)
        if fund is None:
            missing_funds.append(row.code)
        else:
            funds.append(fund)
    if not prices:
        raise RuntimeError("还没有行情缓存")
    release = active_release()
    release.mkdir(parents=True, exist_ok=True)
    price_frame = pd.concat(prices, ignore_index=True)
    price_frame["industry"] = _industry_asof(price_frame)
    price_frame.to_parquet(release / "prices.parquet", index=False)
    if funds:
        pd.concat(funds, ignore_index=True).to_parquet(release / "fundamentals.parquet", index=False)
    provenance = {
        "dataset_id": os.environ.get("FACTORLAB_DATASET", "csi300_pit_v1"),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "membership_source": os.environ.get("FACTORLAB_MEMBERSHIP_URL", SOURCE_URL),
        "membership_fetched": "2026-10-24",
        "price_source": "证券宝日行情。收益用前复权收盘价，计入分红和拆股；市值用未复权收盘价乘以成交量除以换手率",
        "fundamental_source": "东方财富按报告期资产负债表、利润表、现金流量表",
        "announcement_rule": "available_date = NOTICE_DATE，且必须不早于 REPORT_DATE；公告日缺失则丢弃该期",
        "universe_rule": os.environ.get("FACTORLAB_UNIVERSE_RULE", UNIVERSE_RULE),
        "sample_window": [SAMPLE_START, SAMPLE_END],
        "price_window": [PRICE_START, PRICE_END],
        "n_symbols": int(len(symbols)),
        "n_prices": len(prices),
        "n_fundamentals": len(funds),
        "missing_prices": missing_prices,
        "missing_fundamentals": missing_funds,
        "limitations": list(LIMITATIONS),
    }
    (release / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"assembled prices {len(prices)}/{len(symbols)} fundamentals {len(funds)}/{len(symbols)}",
        flush=True,
    )
    return provenance


def _industry_asof(prices: pd.DataFrame) -> pd.Series:
    import polars as pl

    left = (
        pl.from_pandas(prices.reset_index(drop=True)[["date", "ticker"]])
        .with_row_index("_i")
        .with_columns(pl.col("date").cast(pl.Utf8).str.to_date())
    )
    tapes = []
    for root in (active_cache(), CACHE):
        path = root / "industry_tape.parquet"
        if path.is_file():
            tapes.append(pl.read_parquet(path))
    if tapes:
        right = pl.concat(tapes, how="diagonal_relaxed")
        right = right.with_columns(pl.col("available_date").cast(pl.Utf8).str.to_date())
        right = right.drop_nulls(["available_date", "industry"]).unique(
            subset=["ticker", "available_date"],
            keep="first",
        ).sort(["ticker", "available_date"])
        joined = left.sort(["ticker", "date"]).join_asof(
            right,
            left_on="date",
            right_on="available_date",
            by="ticker",
            strategy="backward",
            check_sortedness=False,
        )
    else:
        joined = left.with_columns(pl.lit(None).alias("industry"))
    filled = _fill_industry_gaps(joined.sort("_i"))
    return pd.Series(filled["industry"].to_list(), index=prices.index)


def _read_industry_gap() -> "pl.DataFrame":
    import polars as pl

    path = Path(__file__).resolve().parents[3] / "data" / "reference" / "csi300_industry_gap.csv"
    empty = pl.DataFrame({"ticker": [], "gap_industry": []}).cast({"ticker": pl.Utf8, "gap_industry": pl.Utf8})
    if not path.is_file():
        return empty
    frame = pl.read_csv(path, schema_overrides={"ticker": pl.Utf8, "industry": pl.Utf8})
    if frame.is_empty() or "ticker" not in frame.columns or "industry" not in frame.columns:
        return empty
    return frame.select(
        pl.col("ticker").str.replace_all(r"\D", "").str.zfill(6),
        pl.col("industry").alias("gap_industry"),
    )


def _fill_industry_gaps(frame: "pl.DataFrame") -> "pl.DataFrame":
    """公告日之前沿用该公司最早的行业名。报表里始终没有的，用缺口表补。"""

    import polars as pl

    work = frame.with_columns(pl.col("industry").fill_null("未分类").cast(pl.Utf8))
    known = (
        work.filter(pl.col("industry") != "未分类")
        .sort("date")
        .group_by("ticker")
        .agg(pl.col("industry").first().alias("_first"))
    )
    gap = _read_industry_gap()
    return (
        work.join(known, on="ticker", how="left")
        .join(gap, on="ticker", how="left")
        .with_columns(
            pl.when(pl.col("industry") != "未分类")
            .then(pl.col("industry"))
            .when(pl.col("_first").is_not_null())
            .then(pl.col("_first"))
            .when(pl.col("gap_industry").is_not_null() & (pl.col("gap_industry") != ""))
            .then(pl.col("gap_industry"))
            .otherwise(pl.lit("未分类"))
            .alias("industry")
        )
        .sort("_i")
    )


def main() -> None:
    stage = os.environ.get("FACTORLAB_CSI300_STAGE", "all")
    symbols = target_symbols()
    print(f"symbols {len(symbols)} stage {stage}", flush=True)
    if stage in {"all", "prices"}:
        shared_tape = CACHE / "industry_tape.parquet"
        if active_cache().resolve() != CACHE.resolve() and shared_tape.is_file():
            print("industry tape reused", flush=True)
        else:
            fetch_industries()
        fetch_prices(symbols)
    if stage in {"all", "funds"}:
        fetch_fundamentals(symbols, workers=6)
    if stage == "all":
        assemble_release()


if __name__ == "__main__":
    main()
