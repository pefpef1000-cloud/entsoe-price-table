"""How each spread delivered in the past - used by spread_dash.py (2026-10-21).

Peter: "on all spreads show how they have delivered in the past".

For every fully delivered period (month / quarter / year, the tenor of
leg 1) since 2020:
  Delivered  = the spread built from what was actually delivered
  Futures    = the same spread from EEX settlements on the last trading day
               before delivery started (only where settlements exist -
               the EEX database starts Aug 2026)
  Difference = delivered - futures (> 0: buying the spread paid)

Delivered legs
  Power  - mean ENTSO-E day-ahead price over the delivery hours
           (Base all hours, Peak 08-20 Mon-Fri, Europe/Berlin), via
           futures_vs_spot_dash.load_spot (entsoe_data.db in C:\\EnergyData).
           Period counted only with >= 95 % of its hours priced.
  Gas    - TTF front month (Yahoo TTF=F), average over the delivery days.
           (gas_month() = month-ahead price, kept for reference, not used.)
  EUA    - mean daily EUA December price over the delivery days
           (eex_dash.eua_prices: EEX settlements, EEX last trades, ICE).
  Coal   - API2 month: ICE settlement of that month after it expired, else
           Yahoo MTF=F front month on the last Friday of the month (the
           contract's expiry). EUR via ECB (ICE DB), else Yahoo EURUSD=X.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import eex_dash as E
import futures_vs_spot_dash as F

HERE = Path(__file__).resolve().parent
YF_DB = HERE.parent / "yfinance_scraper" / "yfinance_data.db"
COVER = 0.95                 # share of delivery hours that must have a price
START = "2020-01-01"
FREQ = {"Month": "M", "Quarter": "Q", "Year": "Y"}
MONTHS = {"Month": 1, "Quarter": 3, "Year": 12}

_lock = threading.Lock()
_cache: dict = {}


def _cached(key, stamp, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (stamp, val)
    return val


def _mtime(p: Path | None):
    try:
        return p.stat().st_mtime if p else None
    except OSError:
        return None


def period_end(start: date, tenor: str) -> date:
    m = start.month - 1 + MONTHS[tenor]
    nxt = date(start.year + m // 12, m % 12 + 1, 1)
    return nxt - timedelta(days=1)


def label(start: date, tenor: str) -> str:
    return F.label(start, tenor)


# ---------------------------------------------------------------------
# Power: delivered spot per period
# ---------------------------------------------------------------------
def spot_periods(market: str, tenor: str, load: str) -> pd.Series:
    """Mean day-ahead price per fully delivered period, index = start date."""
    zone = F.SPOT_ZONE.get(market)
    if not zone or F.SPOT_DB is None:
        return pd.Series(dtype=float)
    today = date.today()

    def build():
        s = F.load_spot(zone)
        if s.empty:
            return pd.Series(dtype=float)
        cal = F.calendar_hours(date.fromisoformat(START), today - timedelta(days=1))
        if load == "Peak":
            cal = cal[F.peak_mask(cal, False)]
        got = s.reindex(cal)
        per = cal.tz_localize(None).to_period(FREQ[tenor])
        g = pd.DataFrame({"p": got.values, "per": per})
        agg = g.groupby("per")["p"].agg(["mean", "count", "size"])
        out = {}
        for per_, r in agg.iterrows():
            start = per_.start_time.date()
            if period_end(start, tenor) >= today:
                continue                      # not fully delivered yet
            if r["count"] >= COVER * r["size"]:
                out[start] = r["mean"]
        return pd.Series(out, dtype=float).sort_index()

    return _cached(("spot", zone, tenor, load, today), F.stamp(F.SPOT_DB), build)


# ---------------------------------------------------------------------
# Futures before delivery (EEX settlements, last common day < start)
# ---------------------------------------------------------------------
def settled():
    return E.load()["settled"]


def settle_index() -> dict:
    """{(market, product, tenor, dkey): settlement by trade_date}, cached."""
    def build():
        s = settled()
        s = s[s["maturity_type"].isin(list(MONTHS))]
        g = s.groupby(["market", "product", "maturity_type", "dkey", "trade_date"])["settlement_price"].mean()
        return {k: v.droplevel([0, 1, 2, 3]) for k, v in g.groupby(level=[0, 1, 2, 3])}
    return _cached(("settle_index",), E.load()["mtime"], build)


def _last_common(series: list[pd.Series], max_apart=7):
    """Latest day all series have; else each series' own last day when they
    are at most `max_apart` days apart. (day text, [prices]) or (None, None)."""
    if not series or any(x.empty for x in series):
        return None, None
    common = sorted(set.intersection(*[set(x.index) for x in series]))
    if common:
        d = common[-1]
        return d, [float(x[d]) for x in series]
    days = [x.index[-1] for x in series]
    if (pd.Timestamp(max(days)) - pd.Timestamp(min(days))).days <= max_apart:
        return min(days), [float(x.iloc[-1]) for x in series]
    return None, None


def code_of(market: str, product: str, tenor: str):
    """EEX short code of a market/product/tenor family (DE Base Year -> DEBY)."""
    def build():
        s = settled()
        g = s[s["maturity_type"].isin(list(MONTHS))].drop_duplicates(["market", "product", "maturity_type"])
        return {(m, p, t): c for m, p, t, c in zip(g["market"], g["product"], g["maturity_type"], g["short_code"])}
    return _cached(("codes",), E.load()["mtime"], build).get((market, product, tenor))


def traded_series(leg) -> pd.Series:
    """Last traded price per day (EEX trade feed, from Nov 2025) of one leg."""
    market, product, tenor, dkey = leg
    code = code_of(market, product, tenor)
    none = pd.Series(dtype=float, index=pd.Index([], dtype=object))
    if not code:
        return none
    d = date.fromisoformat(dkey)

    def build():
        o = E.load_ohlc(code, f"{d.year}{d.month:02d}")
        if o.empty:
            return none
        o = o[o["last"].notna()].drop_duplicates("trade_date", keep="last").sort_values("trade_date")
        return pd.Series(o["last"].astype(float).values, index=o["trade_date"].astype(str).values)
    return _cached(("ohlc", code, dkey), E.load()["mtime"], build)


def futures_before(legs: list[tuple[str, str, str, str]], start: date):
    """legs = [(market, product, tenor, dkey)]. Settlement of each leg on the
    last trade date before `start` on which all legs settled. No settlements
    (EEX settlement data starts Aug 2026) -> last traded prices from the EEX
    trade feed (from Nov 2025); the day then reads "2025-12-29 (last trade)".
    Returns (trade_date, [prices]) or (None, None)."""
    idx = settle_index()
    empty = pd.Series(dtype=float)
    series = []
    for leg in legs:
        x = idx.get(leg)
        series.append(empty if x is None else x[x.index < start.isoformat()])
    d, px = _last_common(series)
    if d is not None:
        return d, px
    series = [x[x.index < start.isoformat()] for x in (traded_series(leg) for leg in legs)]
    d, px = _last_common(series)
    return (f"{d} (last trade)", px) if d is not None else (None, None)


# ---------------------------------------------------------------------
# Gas, EUA, coal: delivered per period
# ---------------------------------------------------------------------
def _yf(ticker: str) -> pd.Series:
    none = pd.Series(dtype=float, index=pd.Index([], dtype=object))   # empty: still date-comparable

    def build():
        if not YF_DB.exists():
            return none
        try:
            with closing(sqlite3.connect(f"{YF_DB.resolve().as_uri()}?mode=ro", uri=True)) as c:
                x = pd.read_sql_query("SELECT date, close FROM daily_prices WHERE ticker = ? "
                                      "AND close IS NOT NULL ORDER BY date", c, params=(ticker,))
        except Exception:
            return none
        return pd.Series(x["close"].values, index=x["date"].astype(str).str[:10])
    return _cached(("yf", ticker), _mtime(YF_DB), build)


def _on_or_before(s: pd.Series, day: str, max_gap=7):
    p = s.loc[:day]
    if p.empty or (pd.Timestamp(day) - pd.Timestamp(p.index[-1])).days > max_gap:
        return None
    return float(p.iloc[-1])


def gas_month(start: date):
    """TTF month-ahead for the month starting `start`: (EUR/MWh, source)."""
    s = settled()
    dkey = start.isoformat()
    g = s[(s["market"] == "Gas TTF") & (s["product"] == "Base") &
          (s["maturity_type"] == "Month") & (s["dkey"] == dkey) & (s["trade_date"] < dkey)]
    if not g.empty:
        return float(g.sort_values("trade_date")["settlement_price"].iloc[-1]), "EEX settle"
    o = E.load_ohlc("G3BM", f"{start.year}{start.month:02d}")
    if not o.empty:
        o = o[(o["trade_date"] < dkey) & o["last"].notna()]
        if not o.empty:
            return float(o.sort_values("trade_date")["last"].iloc[-1]), "EEX last trade"
    ttf = _yf("TTF=F")
    if not ttf.empty:
        expiry = (pd.Timestamp(start) - pd.offsets.BDay(2)).date().isoformat()
        v = _on_or_before(ttf, expiry)
        if v is not None:
            return v, "TTF=F"
    return None, None


def months_of(start: date, tenor: str) -> list[date]:
    return [date(start.year + (start.month - 1 + i) // 12, (start.month - 1 + i) % 12 + 1, 1)
            for i in range(MONTHS[tenor])]


def gas_period(start: date, tenor: str):
    """TTF front month (Yahoo TTF=F), average over the delivery days - tracks
    what gas actually cost during delivery (March 2026: month-ahead 31.76,
    delivery average ~53). Needs >= 60 % of the business days."""
    ttf = _yf("TTF=F")
    end = period_end(start, tenor)
    x = ttf[(ttf.index >= start.isoformat()) & (ttf.index <= end.isoformat())]
    bdays = len(pd.bdate_range(start, end))
    if bdays == 0 or len(x) < 0.6 * bdays:
        return None, None
    return float(x.mean()), "TTF=F avg"


def eua_daily() -> pd.Series:
    """Front December EUA per trade date."""
    def build():
        e = E.eua_prices()
        if e.empty:
            return pd.Series(dtype=float)
        e = e.copy()
        e["y"] = e["trade_date"].str[:4].astype(int)
        e = e[e["dec_year"] >= e["y"]].sort_values(["trade_date", "dec_year"])
        return e.groupby("trade_date")["eua"].first().astype(float)
    return _cached(("eua",), E.load()["mtime"], build)


def eua_period(start: date, tenor: str):
    e = eua_daily()
    end = period_end(start, tenor)
    x = e[(e.index >= start.isoformat()) & (e.index <= end.isoformat())]
    bdays = len(pd.bdate_range(start, end))
    if bdays == 0 or len(x) < 0.6 * bdays:
        return None
    return float(x.mean())


def fx_on(day: str):
    coal = E.load_coal()
    if coal and len(coal["fx"]):
        v = _on_or_before(coal["fx"], day)
        if v is not None:
            return v
    return _on_or_before(_yf("EURUSD=X"), day)


def coal_month(start: date, hub="API2"):
    """API2 USD/t for the month + EUR/MWh coal. (eur_mwh, usd, source)."""
    end = period_end(start, "Month")
    last_fri = end - timedelta(days=(end.weekday() - 4) % 7)
    coal = E.load_coal()
    strip = f"{E.MON[start.month - 1]}{start.year % 100:02d}"
    usd = src = None
    if coal:
        days = sorted(t for (p, t, s_) in coal["prices"] if p == hub and s_ == strip and t >= last_fri.isoformat())
        if days:
            usd, src, day = coal["prices"][(hub, days[-1], strip)], "ICE settle", days[-1]
    if usd is None and hub == "API2":
        v = _on_or_before(_yf("MTF=F"), last_fri.isoformat())
        if v is not None:
            usd, src, day = v, "MTF=F", last_fri.isoformat()
    if usd is None:
        return None, None, None
    fx = fx_on(day)
    if fx is None:
        return None, usd, src
    return usd / fx / E.COAL_MWH_PER_T, usd, src


def coal_period(start: date, tenor: str):
    vals = [coal_month(m) for m in months_of(start, tenor)]
    if any(v[0] is None for v in vals):
        return None, None, None
    return (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])),
            "/".join(sorted({v[2] for v in vals})))


# ---------------------------------------------------------------------
# One frame per spread type. Columns: start, Period, ..., Delivered,
# Futures, Futures day, Diff
# ---------------------------------------------------------------------
def _finish(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Diff"] = df["Delivered"] - df["Futures"]
    return df.sort_values("start", ascending=False)


def location(a: str, b: str, tenor: str, prod: str) -> pd.DataFrame:
    sa, sb = spot_periods(a, tenor, prod), spot_periods(b, tenor, prod)
    rows = []
    for start in sorted(set(sa.index) & set(sb.index)):
        d, px = futures_before([(a, prod, tenor, start.isoformat()), (b, prod, tenor, start.isoformat())], start)
        rows.append({"start": start, "Period": label(start, tenor), f"Spot {a}": sa[start],
                     f"Spot {b}": sb[start], "Delivered": sa[start] - sb[start],
                     "Futures": px[0] - px[1] if px else np.nan, "Futures day": d or ""})
    return _finish(rows)


def time_spread(area: str, t1: str, k1: str, t2: str, k2: str, prod: str) -> pd.DataFrame:
    """Same shape as the picked legs, shifted back in time: leg 2 starts the
    same number of months after leg 1 as in the pick."""
    d1, d2 = date.fromisoformat(k1), date.fromisoformat(k2)
    gap = (d2.year - d1.year) * 12 + d2.month - d1.month
    s1, s2 = spot_periods(area, t1, prod), spot_periods(area, t2, prod)
    rows = []
    for start in s1.index:
        m = start.month - 1 + gap
        start2 = date(start.year + m // 12, m % 12 + 1, 1)
        if start2 not in s2.index:
            continue
        d, px = futures_before([(area, prod, t1, start.isoformat()), (area, prod, t2, start2.isoformat())], start)
        rows.append({"start": start, "Period": f"{label(start, t1)} - {label(start2, t2)}",
                     "Spot leg 1": s1[start], "Spot leg 2": s2[start2],
                     "Delivered": s1[start] - s2[start2],
                     "Futures": px[0] - px[1] if px else np.nan, "Futures day": d or ""})
    return _finish(rows)


def _fuel_futures(area, prod, tenor, start, fuel, eff, cf, ceff=None, ccf=None):
    """Clean spark/dark spread from settlements on the last day before delivery."""
    dkey = start.isoformat()
    s = settled()
    p = s[(s["market"] == area) & (s["product"] == prod) & (s["maturity_type"] == tenor) &
          (s["dkey"] == dkey) & (s["trade_date"] < dkey)]
    if p.empty:
        return np.nan, ""
    p = p.groupby(["trade_date", "dkey", "ddate"], as_index=False)["settlement_price"].mean()
    e = E.eua_prices()
    if fuel == "gas":
        g = s[(s["market"] == "Gas TTF") & (s["product"] == "Base") & (s["maturity_type"] == tenor) &
              (s["dkey"] == dkey)]
        if g.empty:
            return np.nan, ""
        x = E.spark_frame(p, g, e, eff, cf).dropna(subset=["clean_spark"])
        return (float(x["clean_spark"].iloc[-1]), x["trade_date"].iloc[-1]) if not x.empty else (np.nan, "")
    m = p.rename(columns={"settlement_price": "power"})
    picks = [E.pick_eua(e, t, d.year) for t, d in zip(m["trade_date"], m["ddate"])]
    m["eua"] = [v for v, _, _ in picks]
    m["fuel_cost"] = m["carbon_cost"] = np.nan
    x = E.add_coal(m, E.load_coal(), "API2", tenor, ceff, ccf).dropna(subset=["cds"])
    return (float(x["cds"].iloc[-1]), x["trade_date"].iloc[-1]) if not x.empty else (np.nan, "")


def spark(area: str, tenor: str, prod: str, eff: float, cf: float) -> pd.DataFrame:
    sp = spot_periods(area, tenor, prod)
    rows = []
    for start in sp.index:
        gas, gsrc = gas_period(start, tenor)
        eua = eua_period(start, tenor)
        if gas is None or eua is None:
            continue
        fut, d = _fuel_futures(area, prod, tenor, start, "gas", eff, cf)
        rows.append({"start": start, "Period": label(start, tenor), "Power": sp[start],
                     "TTF": gas, "EUA": eua, "Prod. cost": gas / (eff / 100) + eua * cf,
                     "Delivered": sp[start] - gas / (eff / 100) - eua * cf,
                     "Futures": fut, "Futures day": d, "Gas source": gsrc})
    return _finish(rows)


def dark(area: str, tenor: str, prod: str, ceff: float, ccf: float) -> pd.DataFrame:
    sp = spot_periods(area, tenor, prod)
    rows = []
    for start in sp.index:
        coal, usd, csrc = coal_period(start, tenor)
        eua = eua_period(start, tenor)
        if coal is None or eua is None:
            continue
        fut, d = _fuel_futures(area, prod, tenor, start, "coal", None, None, ceff, ccf)
        rows.append({"start": start, "Period": label(start, tenor), "Power": sp[start],
                     "API2 USD/t": usd, "Coal €/MWh": coal, "EUA": eua,
                     "Prod. cost": coal / (ceff / 100) + eua * ccf,
                     "Delivered": sp[start] - coal / (ceff / 100) - eua * ccf,
                     "Futures": fut, "Futures day": d, "Coal source": csrc})
    return _finish(rows)
