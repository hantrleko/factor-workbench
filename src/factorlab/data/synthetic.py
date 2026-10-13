"""合成股票面板。

价格、成交和财报由同一组缓慢变化的潜变量驱动，并在下个月的收益里
放入温和的因子溢价。这样演示报告能看出统计链路，同时不会把溢价
写成市场结论。embed_premium=False 时只保留噪声。

财报可用日取季末当天或之前的最后一个工作日，与因子代码的
available_date <= 月末 规则一致。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

INDUSTRIES = ("银行", "食品饮料", "医药", "电子", "机械", "房地产", "公用事业", "交通运输")

# 每个潜变量 1 个截面标准差对应的下月预期收益。仅用于数据生成。
PREMIUM = {
    "accrual": -0.004,
    "capex": -0.003,
    "pm": 0.004,
    "at": 0.003,
    "em": -0.002,
    "illiq": 0.004,
    "ivol": -0.004,
    "reversal": -0.005,
}

_ACC, _CAP, _PM, _AT, _EM, _ILLIQ, _IVOL = range(7)


def make_synthetic_universe(
    n_stocks: int = 48,
    start: str = "2021-01-04",
    end: str = "2024-12-31",
    seed: int = 42,
    embed_premium: bool = True,
    n_industries: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if n_stocks < n_industries * 2:
        raise ValueError("股票数量至少要覆盖每个行业两只")
    if not 2 <= n_industries <= len(INDUSTRIES):
        raise ValueError(f"行业数必须在 2 到 {len(INDUSTRIES)} 之间")
    rng = np.random.default_rng(seed)
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    calendar = np.arange(
        np.datetime64(start_d),
        np.datetime64(end_d) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    days = calendar[np.is_busday(calendar)]
    if days.size < 40:
        raise ValueError("样本交易日太短")

    names = INDUSTRIES[:n_industries]
    industry_id = np.arange(n_stocks) % n_industries
    rng.shuffle(industry_id)
    tickers = np.array([f"S{i:04d}" for i in range(1, n_stocks + 1)])
    shares = rng.lognormal(mean=np.log(8e7), sigma=0.45, size=n_stocks)
    beta = rng.uniform(0.7, 1.35, size=n_stocks)
    price = rng.uniform(10.0, 60.0, size=n_stocks)

    hist_months = _month_grid(start_d.year - 2, end_d.year)
    lat = _ar1_latents(rng, hist_months.size, n_stocks)
    month_of = {np.datetime64(m, "M"): i for i, m in enumerate(hist_months)}

    day_months = days.astype("datetime64[M]")
    is_last = np.zeros(days.size, dtype=bool)
    is_last[:-1] = day_months[1:] != day_months[:-1]
    is_last[-1] = True
    month_ids = np.unique(day_months)
    days_in_month = {m: int(np.sum(day_months == m)) for m in month_ids}
    month_end = {m: days[day_months == m][-1] for m in month_ids}

    quarters = _quarters(start_d.year - 2, end_d.year)
    fundamentals = _build_fundamentals(
        rng, quarters, lat, month_of, tickers, shares, price, industry_id
    )

    n_days = days.size
    rets = np.empty(n_days * n_stocks)
    closes = np.empty(n_days * n_stocks)
    volumes = np.empty(n_days * n_stocks)
    amounts = np.empty(n_days * n_stocks)
    mktcaps = np.empty(n_days * n_stocks)
    last_month_ret = np.zeros(n_stocks)
    month_gross = np.ones(n_stocks)
    prev_month: np.datetime64 | None = None

    for di, day in enumerate(days):
        month = day_months[di]
        if prev_month is None or month != prev_month:
            month_gross = np.ones(n_stocks)
            prev_month = month
        mi_hist = month_of[month]
        if embed_premium and di > 0 and not _is_first_day_of_first_month(di, day_months):
            prev_m = _previous_month(month)
            drift = _monthly_drift(
                lat, month_of, quarters, month_end, prev_m, last_month_ret
            )
        else:
            drift = np.zeros(n_stocks)
        n_in_month = days_in_month[month]
        daily_drift = drift / n_in_month
        sig = 0.011 * (1.0 + 0.35 * np.tanh(lat[mi_hist, :, _IVOL]))
        turn = 0.012 * np.exp(-1.5 * np.tanh(lat[mi_hist, :, _ILLIQ]))
        mkt = float(rng.normal(0.00025, 0.009))
        ind_shock = rng.normal(0.0, 0.004, size=n_industries)
        eps = rng.normal(0.0, 1.0, size=n_stocks)
        r = daily_drift + beta * mkt + ind_shock[industry_id] + sig * eps
        price = price * (1.0 + r)
        vol = turn * shares
        sl = slice(di * n_stocks, (di + 1) * n_stocks)
        rets[sl] = r
        closes[sl] = price
        volumes[sl] = vol
        amounts[sl] = vol * price
        mktcaps[sl] = price * shares
        month_gross *= 1.0 + r
        if is_last[di]:
            last_month_ret = month_gross - 1.0

    industry_names = np.array(names, dtype=object)[industry_id]
    prices = pl.DataFrame(
        {
            "date": np.repeat(days, n_stocks),
            "ticker": np.tile(tickers, n_days),
            "close": closes,
            "volume": volumes,
            "amount": amounts,
            "ret": rets,
            "mktcap": mktcaps,
            "industry": np.tile(industry_names, n_days),
        }
    ).with_columns(pl.col("date").cast(pl.Date), pl.col("ticker").cast(pl.Utf8))
    return prices, fundamentals


def _is_first_day_of_first_month(di: int, day_months: np.ndarray) -> bool:
    return day_months[di] == day_months[0]


def _previous_month(month: np.datetime64) -> np.datetime64:
    return (month.astype("datetime64[M]") - np.timedelta64(1, "M")).astype("datetime64[M]")


def _month_grid(year0: int, year1: int) -> np.ndarray:
    start = np.datetime64(f"{year0}-01", "M")
    end = np.datetime64(f"{year1}-12", "M")
    return np.arange(start, end + np.timedelta64(1, "M"), dtype="datetime64[M]")


def _ar1_latents(rng: np.random.Generator, n_months: int, n_stocks: int) -> np.ndarray:
    lat = np.zeros((n_months, n_stocks, 7))
    lat[0] = rng.normal(0.0, 1.0, size=(n_stocks, 7))
    for t in range(1, n_months):
        lat[t] = 0.82 * lat[t - 1] + np.sqrt(1 - 0.82**2) * rng.normal(
            0.0, 1.0, size=(n_stocks, 7)
        )
    return lat


def _quarters(year0: int, year1: int) -> list[dict]:
    rows = []
    for year in range(year0, year1 + 1):
        for quarter, (mo, day) in enumerate(((3, 31), (6, 30), (9, 30), (12, 31)), start=1):
            qend = np.datetime64(f"{year:04d}-{mo:02d}-{day:02d}")
            avail = np.busday_offset(qend, 0, roll="backward")
            rows.append(
                {
                    "fiscal_year": year,
                    "fiscal_quarter": quarter,
                    "available": avail,
                    "month": np.datetime64(avail, "M"),
                }
            )
    return rows


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def _latest_quarter(quarters: list[dict], asof: np.datetime64) -> dict | None:
    chosen = None
    for row in quarters:
        if row["available"] <= asof:
            chosen = row
        else:
            break
    return chosen


def _monthly_drift(
    lat: np.ndarray,
    month_of: dict[np.datetime64, int],
    quarters: list[dict],
    month_end: dict[np.datetime64, np.datetime64],
    prev_month: np.datetime64,
    last_month_ret: np.ndarray,
) -> np.ndarray:
    if prev_month not in month_end or prev_month not in month_of:
        return np.zeros(last_month_ret.shape[0])
    q = _latest_quarter(quarters, month_end[prev_month])
    drift = np.zeros(last_month_ret.shape[0])
    if q is not None and q["month"] in month_of:
        published = lat[month_of[q["month"]]]
        drift += PREMIUM["accrual"] * _z(published[:, _ACC])
        drift += PREMIUM["capex"] * _z(published[:, _CAP])
        drift += PREMIUM["pm"] * _z(published[:, _PM])
        drift += PREMIUM["at"] * _z(published[:, _AT])
        drift += PREMIUM["em"] * _z(published[:, _EM])
    micro = lat[month_of[prev_month]]
    drift += PREMIUM["illiq"] * _z(micro[:, _ILLIQ])
    drift += PREMIUM["ivol"] * _z(micro[:, _IVOL])
    drift += PREMIUM["reversal"] * _z(last_month_ret)
    return drift


def _build_fundamentals(
    rng: np.random.Generator,
    quarters: list[dict],
    lat: np.ndarray,
    month_of: dict[np.datetime64, int],
    tickers: np.ndarray,
    shares: np.ndarray,
    price0: np.ndarray,
    industry_id: np.ndarray,
) -> pl.DataFrame:
    del industry_id
    n_q = len(quarters)
    n = tickers.size
    assets = np.zeros((n_q, n))
    assets[0] = shares * price0 * rng.uniform(0.7, 1.3, size=n)
    for q in range(1, n_q):
        assets[q] = assets[q - 1] * np.clip(1.012 + 0.008 * rng.normal(size=n), 0.92, 1.12)

    def lat_row(q: int) -> np.ndarray:
        return lat[month_of[quarters[q]["month"]]]

    cash = 0.10 * assets
    current_liab = 0.25 * assets
    short_debt = 0.07 * assets
    tax = 0.015 * assets
    dep = 0.012 * assets
    current_assets = np.zeros((n_q, n))
    current_assets[:4] = 0.42 * assets[:4]
    capex = np.zeros((n_q, n))
    revenue = np.zeros((n_q, n))
    net_income = np.zeros((n_q, n))
    equity = np.zeros((n_q, n))

    for q in range(n_q):
        L = lat_row(q)
        em = np.clip(1.8 + 0.45 * np.tanh(L[:, _EM]), 1.25, 3.5)
        at = np.clip(0.55 + 0.25 * np.tanh(L[:, _AT]), 0.25, 1.6)
        pm = np.clip(0.07 + 0.04 * np.tanh(L[:, _PM]), 0.02, 0.28)
        capex_ratio = np.clip(0.045 + 0.03 * np.tanh(L[:, _CAP]), 0.01, 0.16)
        if q < 4:
            equity[q] = assets[q] / em
            revenue[q] = at * assets[q] / 4.0
            net_income[q] = pm * revenue[q]
            capex[q] = capex_ratio * assets[q] / 4.0
            continue
        avg = 0.5 * (assets[q] + assets[q - 4])
        equity_q = 2.0 * avg / em - equity[q - 4]
        equity[q] = np.where(equity_q > 0.05 * assets[q], equity_q, assets[q] / em)
        rev_q = at * avg - revenue[q - 3 : q].sum(axis=0)
        revenue[q] = np.where(rev_q > 0.002 * assets[q], rev_q, at * assets[q] / 4.0)
        rev_ttm = revenue[q - 3 : q + 1].sum(axis=0)
        ni_q = pm * rev_ttm - net_income[q - 3 : q].sum(axis=0)
        net_income[q] = np.where(ni_q > 0, ni_q, pm * revenue[q])
        cap_q = capex_ratio * avg - capex[q - 3 : q].sum(axis=0)
        capex[q] = np.where(cap_q > 0, cap_q, capex_ratio * assets[q] / 4.0)
        dep_ttm = dep[q - 3 : q + 1].sum(axis=0)
        target = 0.06 * np.tanh(L[:, _ACC])
        numer = target * avg
        d_cash = cash[q] - cash[q - 4]
        d_cl = current_liab[q] - current_liab[q - 4]
        d_std = short_debt[q] - short_debt[q - 4]
        d_tp = tax[q] - tax[q - 4]
        d_ca = numer + d_cash + d_cl - d_std - d_tp + dep_ttm
        current_assets[q] = current_assets[q - 4] + d_ca

    frames = []
    for q, meta in enumerate(quarters):
        avail = dt.date.fromisoformat(str(meta["available"])[:10])
        frames.append(
            pl.DataFrame(
                {
                    "ticker": tickers,
                    "fiscal_year": np.full(n, meta["fiscal_year"], dtype=np.int32),
                    "fiscal_quarter": np.full(n, meta["fiscal_quarter"], dtype=np.int32),
                    "available_date": [avail] * n,
                    "total_assets": assets[q],
                    "current_assets": current_assets[q],
                    "cash": cash[q],
                    "current_liabilities": current_liab[q],
                    "short_term_debt": short_debt[q],
                    "tax_payable": tax[q],
                    "depreciation": dep[q],
                    "net_income": net_income[q],
                    "revenue": revenue[q],
                    "capex": capex[q],
                    "equity": equity[q],
                }
            )
        )
    return pl.concat(frames).with_columns(
        pl.col("available_date").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
