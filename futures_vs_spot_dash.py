"""
Futures vs spot - Dash dashboard
================================

How did each EEX power future trade last before delivery, and what did the
day-ahead market actually settle at during the delivery period?

Pick a country, a period type (Day ahead, Weekend, Week, Month, Quarter, Cal) and a
load (Base, Peak, Off-peak). One row per delivery period:

    Last traded      last trade price on the last day the contract traded
                     before delivery started (EEX trade feed, eex_market_ohlc)
    Last settlement  EEX settlement on the last trading day before delivery
                     (eex_market_prices)
    Spot outcome     average day-ahead price over the delivery period
                     (ENTSO-E, entsoe_data.db) for the chosen load hours
    Spot - futures   positive = spot came out ABOVE the futures price
                     (a buyer of the future gained)

"Spread vs" (added 2026-10-18): pick a second country B and every number
becomes the spread A - B: futures spread before delivery vs the spread the
two day-ahead markets actually delivered (long A / short B gained when the
spot spread came out above the futures spread).

Load hours (CET/CEST):
    Peak       08:00-20:00 Mon-Fri (EEX peak). Weekend and ES "Mon-Sun" peak
               products: 08:00-20:00 every day.
    Off-peak   every other hour. EEX lists no off-peak futures, so the
               futures price is derived from Base and Peak:
               (Base x base hours - Peak x peak hours) / off-peak hours

Run it (Windows, in the folder this file sits in):

    python futures_vs_spot_dash.py

then open http://127.0.0.1:8057 . Options: --port, --host 0.0.0.0,
--db <eex_market_data.db>, --spot-db <entsoe_data.db>, --open, --debug.

Both databases are opened READ-ONLY and re-read when they change.
"""

from __future__ import annotations

import argparse
import calendar
import os
import re
import sqlite3
import threading
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

# ---------------------------------------------------------------------
# Where the databases are
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
EEX_NAME = "eex_market_data.db"
SPOT_NAME = "entsoe_data.db"


def _entsoe_db_default():
    """entsoe_data.db wherever it lives (see entsoe_scraper/entsoe_db.py)."""
    import sys
    folder = str(HERE.parent / "entsoe_scraper")
    if folder not in sys.path:
        sys.path.append(folder)
    try:
        from entsoe_db import entsoe_db_path
        return entsoe_db_path()
    except ImportError:
        return HERE.parent / "entsoe_scraper" / "entsoe_data.db"


EEX_CANDIDATES = [HERE / EEX_NAME, HERE / "Claude outputs" / EEX_NAME,
                  HERE.parent / "eex_scraper" / EEX_NAME, HERE.parent / EEX_NAME,
                  Path.cwd() / EEX_NAME]
SPOT_CANDIDATES = [_entsoe_db_default(), HERE.parent / "entsoe_scraper" / SPOT_NAME, HERE / SPOT_NAME,
                   HERE.parent / SPOT_NAME, Path.cwd() / SPOT_NAME]


def find(candidates, explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return next((c for c in candidates if c.exists()), None)


EEX_DB: Path | None = find(EEX_CANDIDATES)
SPOT_DB: Path | None = find(SPOT_CANDIDATES)

TZ = "Europe/Berlin"     # EEX delivery hours are CET/CEST

# ---------------------------------------------------------------------
# Look - same palette and CSS as eex_dash.py
# ---------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = {"text": "#1f2328", "muted": "#6b6a64", "grid": "#e8e7e1", "axis": "#c3c2b7",
       "bg": "#f6f5f1", "card": "#ffffff", "up": "#1a7f37", "down": "#c4314b"}

CSS = f"""
body {{ margin:0; background:{INK['bg']}; color:{INK['text']};
       font-family:'Segoe UI', Inter, Arial, sans-serif; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:16px 20px 40px; }}
h1 {{ font-size:22px; margin:0; }}
.sub {{ color:{INK['muted']}; font-size:13px; margin:2px 0 12px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; margin:12px 0; }}
.ctl {{ display:flex; flex-direction:column; min-width:150px; }}
.ctl label.cap {{ font-size:12px; color:{INK['muted']}; margin-bottom:3px; }}
.card {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
        padding:10px 12px; margin-bottom:12px; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 12px; }}
.kpi {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
       padding:8px 14px; min-width:150px; }}
.kpi-label {{ font-size:12px; color:{INK['muted']}; }}
.kpi-value {{ font-size:22px; font-weight:600; }}
.kpi-delta {{ font-size:13px; color:{INK['muted']}; }}
.note {{ color:{INK['muted']}; font-size:13px; margin:6px 0; }}
.warn {{ background:#fff4d6; border:1px solid #eda100; border-radius:6px;
        padding:8px 12px; font-size:13px; margin:6px 0; }}
.foot {{ color:{INK['muted']}; font-size:12px; margin-top:16px; }}
.seg {{ display:flex; border:1px solid {INK['axis']}; border-radius:6px; overflow:hidden;
        background:{INK['card']}; height:34px; }}
.seg label {{ display:flex; align-items:center; padding:0 14px; font-size:13px; cursor:pointer;
              margin:0 !important; color:{INK['muted']}; }}
.seg label + label {{ border-left:1px solid {INK['grid']}; }}
.seg input {{ display:none; }}
.seg label:has(input:checked) {{ background:{SERIES[0]}; }}
.seg label:has(input:checked), .seg label:has(input:checked) * {{ color:#fff !important; }}
details summary {{ cursor:pointer; color:{INK['muted']}; font-size:13px; margin:6px 0; }}
"""

# ---------------------------------------------------------------------
# Markets: EEX long name -> market code (same rule as eex_dash.py),
# then market -> ENTSO-E zone. None = no matching spot price in the DB.
# ---------------------------------------------------------------------
DEMONYM = {
    "German": "DE", "French": "FR", "Austrian": "AT", "Swiss": "CH",
    "Italian": "IT", "Spanish": "ES", "Dutch": "NL", "Belgian": "BE",
    "Polish": "PL", "Czech": "CZ", "Slovakian": "SK", "Hungarian": "HU",
    "Romanian": "RO", "Bulgarian": "BG", "Croatian": "HR", "Slovenian": "SI",
    "Serbian": "RS", "Greek": "GR", "Finnish": "FI", "GB": "GB",
    "Irish SEM": "IE",
}
ZONE_RE = re.compile(r"\b([A-Z]{2}\d)\b")

SPOT_ZONE = {
    "DE": "DE_LU", "FR": "FR", "AT": "AT", "CH": "CH", "ES": "ES", "NL": "NL",
    "BE": "BE", "PL": "PL", "CZ": "CZ", "SK": "SK", "HU": "HU", "RO": "RO",
    "BG": "BG", "HR": "HR", "SI": "SI", "RS": "RS", "GR": "GR", "FI": "FI",
    "IE": "IE_SEM", "Nordic SYS": "NORDIC_SYSTEM",
    "DK1": "DK_1", "DK2": "DK_2", "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3",
    "NO4": "NO_4", "NO5": "NO_5", "SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3",
    "SE4": "SE_4",
    # Not comparable with what is in entsoe_data.db:
    #   IT - EEX Italian futures settle on PUN (national price), not a zone
    #   GB - EEX GB futures are in GBP, ENTSO-E GB is stored in EUR
}


def market_of(long_name: str | None, code: str) -> str | None:
    """Market code for a POWER future, None for gas/options/Japan/etc."""
    if not long_name:
        return None
    name = re.sub(r"^EEX(-PXE)?\s+", "", long_name)
    if any(w in name for w in ("Freight", "Option", "Natural Gas", "Japan")):
        return None
    if name.startswith("Nordic System Price"):
        return "Nordic SYS"
    head = name.split(" Power")[0]
    z = ZONE_RE.search(head)
    if z:
        return z.group(1)
    return DEMONYM.get(head)


# ---------------------------------------------------------------------
# Delivery periods
# ---------------------------------------------------------------------
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PERIODS = [("Day", "Day ahead"), ("Weekend", "WE"), ("Week", "Weeks"), ("Month", "Months"),
           ("Quarter", "Q"), ("Year", "Cal")]
LOADS = ["Base", "Peak", "Off-peak"]
# Peak products whose peak runs every day, not just Mon-Fri.
PEAK_EVERY_DAY = ("ESP", "EWP")      # ES "Mon-Sun" peak week / weekend


def delivery_span(code: str, delivery: str, maturity: str) -> tuple[date, date] | None:
    """(first, last) delivery day, both inclusive."""
    t = str(delivery)
    if len(t) != 6 or not t.isdigit():
        return None
    y, m = int(t[:4]), int(t[4:])
    if maturity == "Day":
        # Day futures: code = family + 2-digit day (DB16 = 16th of the month).
        dd = code[-2:]
        if not dd.isdigit():
            return None
        try:
            d = date(y, m, int(dd))
        except ValueError:
            return None
        return d, d
    if maturity in ("Week", "Weekend"):
        n = code[-1]
        if not n.isdigit():
            return None
        first = date(y, m, 1)
        sunday = first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (int(n) - 1))
        if sunday.month != m:
            return None
        return (sunday - timedelta(days=6 if maturity == "Week" else 1), sunday)
    if maturity == "Month":
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    if maturity == "Quarter":
        end_m = m + 2
        return date(y, m, 1), date(y, end_m, calendar.monthrange(y, end_m)[1])
    if maturity == "Year":
        return date(y, 1, 1), date(y, 12, 31)
    return None


def label(start: date, maturity: str) -> str:
    yy = f"{start.year % 100:02d}"
    if maturity == "Day":
        return f"{start:%a} {start.day} {MON[start.month - 1]} {yy}"
    if maturity == "Year":
        return f"Cal-{yy}"
    if maturity == "Quarter":
        return f"Q{(start.month - 1) // 3 + 1}-{yy}"
    if maturity == "Month":
        return f"{MON[start.month - 1]}-{yy}"
    wk = start.isocalendar()[1]
    if maturity == "Week":
        return f"W{wk:02d}-{yy} · {start.day} {MON[start.month - 1]}"
    sun = start + timedelta(days=1)
    span = (f"{start.day}-{sun.day} {MON[start.month - 1]}" if sun.month == start.month
            else f"{start.day} {MON[start.month - 1]}-{sun.day} {MON[sun.month - 1]}")
    return f"WE{wk:02d}-{yy} · {span}"


def peak_mask(index: pd.DatetimeIndex, every_day: bool):
    m = (index.hour >= 8) & (index.hour < 20)
    if not every_day:
        m &= index.dayofweek < 5
    return m


def calendar_hours(start: date, end: date) -> pd.DatetimeIndex:
    """Every delivery hour in local time - 23/25-hour DST days handled."""
    return pd.date_range(pd.Timestamp(start, tz=TZ),
                         pd.Timestamp(end + timedelta(days=1), tz=TZ),
                         freq="h", inclusive="left")


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict = {}


def stamp(path: Path):
    wal = Path(str(path) + "-wal")
    s = path.stat()
    return (str(path), s.st_mtime, s.st_size, wal.stat().st_mtime if wal.exists() else 0)


def ro(path: Path):
    return closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True))


def load_contracts() -> pd.DataFrame:
    """One row per power contract + delivery that has a pre-delivery price.
    Columns: market, product, maturity, code, delivery, start, end,
    last_trade_date, last_px, trades, settle_date, settle_px, final_date, final_px."""
    key = stamp(EEX_DB)
    with _lock:
        if _cache.get("eex_key") == key:
            return _cache["eex"]
    with ro(EEX_DB) as conn:
        names = dict(conn.execute("SELECT short_code, long_name FROM contracts").fetchall())
        prices = pd.read_sql_query(
            """SELECT short_code, delivery, trade_date, product, maturity_type,
                      settlement_price, currency
               FROM eex_market_prices
               WHERE maturity_type IN ('Day','Weekend','Week','Month','Quarter','Year')
                 AND product IN ('Base','Peak')""", conn)
        ohlc = pd.read_sql_query(
            """SELECT short_code, delivery, trade_date, last, trades
               FROM eex_market_ohlc WHERE trades > 0 AND last IS NOT NULL""", conn)
    for f in (prices, ohlc):
        f["delivery"] = f["delivery"].astype(str)
        f["trade_date"] = f["trade_date"].astype(str)

    combos = prices[["short_code", "delivery", "product", "maturity_type", "currency"]] \
        .drop_duplicates(["short_code", "delivery"])
    rows = []
    for c in combos.itertuples(index=False):
        mkt = market_of(names.get(c.short_code), c.short_code)
        span = delivery_span(c.short_code, c.delivery, c.maturity_type)
        if mkt is None or span is None:
            continue
        # "cut" = first trade date that no longer counts as "before".
        # Day futures settle on the day-ahead auction, which runs on D-1, so
        # for them "before" means before the auction day (start - 1).
        cut = span[0] - timedelta(days=1) if c.maturity_type == "Day" else span[0]
        rows.append({"code": c.short_code, "delivery": c.delivery, "market": mkt,
                     "product": c.product, "maturity": c.maturity_type,
                     "currency": c.currency, "start": span[0].isoformat(),
                     "end": span[1].isoformat(), "cut": cut.isoformat(),
                     "fin_from": cut.isoformat() if c.maturity_type == "Day"
                     else span[1].isoformat()})
    meta = pd.DataFrame(rows)
    if meta.empty:
        out = meta
    else:
        k = ["code", "delivery"]
        # Last trade before delivery started.
        o = ohlc.merge(meta[k + ["cut"]], left_on=["short_code", "delivery"],
                       right_on=k)
        o = o[o["trade_date"] < o["cut"]].sort_values("trade_date")
        last = o.groupby(k).tail(1)[k + ["trade_date", "last", "trades"]] \
            .rename(columns={"trade_date": "last_trade_date", "last": "last_px"})
        # Settlements: last one before delivery, and the latest one after
        # delivery ended (EEX final settlement).
        s = prices[prices["settlement_price"].notna()].merge(
            meta[k + ["cut", "fin_from"]], left_on=["short_code", "delivery"], right_on=k)
        s = s.sort_values("trade_date")
        pre = s[s["trade_date"] < s["cut"]]
        pre_last = pre.groupby(k).tail(1)[k + ["trade_date", "settlement_price"]] \
            .rename(columns={"trade_date": "settle_date", "settlement_price": "settle_px"})
        fin = s[s["trade_date"] >= s["fin_from"]].groupby(k).tail(1)[k + ["trade_date", "settlement_price"]] \
            .rename(columns={"trade_date": "final_date", "settlement_price": "final_px"})
        out = meta.merge(last, on=k, how="left").merge(pre_last, on=k, how="left") \
            .merge(fin, on=k, how="left")
        # Every pre-delivery settlement, for deriving off-peak on a date
        # where Base AND Peak both settled.
        _cache["pre_settles"] = pre[k + ["trade_date", "settlement_price"]]
        out = out[out["last_px"].notna() | out["settle_px"].notna()]
    with _lock:
        _cache["eex_key"], _cache["eex"] = key, out
    return out


def load_spot(zone: str) -> pd.Series:
    """Hourly day-ahead prices for one zone, indexed in local time."""
    key = (zone,) + stamp(SPOT_DB)
    with _lock:
        if _cache.get(("spot", zone), (None,))[0] == key:
            return _cache[("spot", zone)][1]
    with ro(SPOT_DB) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, price_eur_mwh FROM day_ahead_prices "
            "WHERE zone = ? AND timestamp >= '2020-01-01'", conn, params=(zone,))
    if df.empty:
        s = pd.Series(dtype=float)
    else:
        # Floor to the hour in UTC (no DST ambiguity), then local time.
        # 15-minute prices (if any) become hourly means.
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h").dt.tz_convert(TZ)
        s = pd.Series(df["price_eur_mwh"].values, index=ts) \
            .groupby(level=0).mean().sort_index()
    with _lock:
        _cache[("spot", zone)] = (key, s)
    return s


# ---------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------

def spot_stats(spot: pd.Series, start: str, end: str, load: str, every_day: bool):
    """(average, hours with a price, calendar hours) for one delivery period."""
    s0, e0 = date.fromisoformat(start), date.fromisoformat(end)
    cal = calendar_hours(s0, e0)
    pk = peak_mask(cal, every_day)
    want = cal if load == "Base" else cal[pk] if load == "Peak" else cal[~pk]
    got = spot.reindex(want).dropna()
    return (got.mean() if len(got) else None), len(got), len(want), int(pk.sum()), len(cal)


def build(market: str, maturity: str, load: str, today: date | None = None):
    """Rows for the table (newest delivery first) + notes."""
    today = today or date.today()
    con = load_contracts()
    notes = []
    zone = SPOT_ZONE.get(market)
    f = con[(con["market"] == market) & (con["maturity"] == maturity)]
    base = f[f["product"] == "Base"].drop_duplicates("start").set_index("start")
    peak = f[f["product"] == "Peak"].drop_duplicates("start").set_index("start")
    spot = load_spot(zone) if (zone and SPOT_DB) else pd.Series(dtype=float)

    if load == "Base":
        src = base
    elif load == "Peak":
        src = peak
    else:
        src = base[base.index.isin(peak.index)]
    if load != "Base" and peak.empty:
        notes.append(f"EEX has no {maturity.lower()} peak futures for {market} in the "
                     f"database, so {load.lower()} can't be compared.")

    rows = []
    for start, r in src.iterrows():
        if start > today.isoformat():
            continue                                      # not delivering yet
        pcode = str(peak.loc[start, "code"]) if start in peak.index else str(r["code"])
        every_day = maturity in ("Weekend", "Day") or pcode.startswith(PEAK_EVERY_DAY)
        avg, n_spot, n_want, n_peak, n_all = spot_stats(spot, start, r["end"], load, every_day)

        if load == "Off-peak":
            p = peak.loc[start]
            n_off = n_all - n_peak
            px = lambda b, pk: (b * n_all - pk * n_peak) / n_off \
                if (b is not None and pk is not None and pd.notna(b) and pd.notna(pk)) else None
            last_px = px(r["last_px"], p["last_px"])
            d1, d2 = fmt_d(r["last_trade_date"]), fmt_d(p["last_trade_date"])
            last_date = d1 if d1 == d2 else " / ".join(x or "—" for x in (d1, d2))
            # Settlement: latest pre-delivery day where BOTH settled.
            ps = _cache.get("pre_settles")
            sb = ps[(ps.code == r["code"]) & (ps.delivery == r["delivery"])].set_index("trade_date")["settlement_price"]
            sp = ps[(ps.code == p["code"]) & (ps.delivery == p["delivery"])].set_index("trade_date")["settlement_price"]
            both = sb.index.intersection(sp.index)
            if len(both):
                dd = max(both)
                settle_px, settle_date = px(sb[dd], sp[dd]), dd
            else:
                settle_px, settle_date = None, None
            final_px = px(r["final_px"], p["final_px"])
            trades = None
        else:
            last_px, last_date = r["last_px"], fmt_d(r["last_trade_date"])
            settle_px, settle_date = r["settle_px"], r["settle_date"]
            final_px = r["final_px"]
            trades = r["trades"]

        delivered = r["end"] < today.isoformat()
        status = "Delivered" if delivered else "In delivery"
        if not zone:
            status += " · no spot"
        elif n_spot < n_want:
            status += f" · spot {n_spot}/{n_want} h"
        rows.append({
            "start": start,
            "Delivery": label(date.fromisoformat(start), maturity),
            "Status": status,
            "Last trade day": last_date or "—",
            "Trades": trades if trades is not None and pd.notna(trades) else None,
            "Last traded": num(last_px),
            "Last settle": num(settle_px),
            "Settle day": fmt_d(settle_date) if settle_date else "—",
            "Spot outcome": num(avg),
            "Spot − last traded": num(avg - last_px) if (avg is not None and last_px is not None and pd.notna(last_px)) else None,
            "Spot − last settle": num(avg - settle_px) if (avg is not None and settle_px is not None and pd.notna(settle_px)) else None,
            "EEX final settle": num(final_px),
            "complete": delivered and zone is not None and n_spot == n_want,
            "currency": r["currency"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("start", ascending=False)
        cur = set(out["currency"].dropna()) - {"EUR"}
        if cur:
            notes.append(f"Futures quoted in {', '.join(cur)} - spot is EUR, differences are not meaningful.")
    if zone is None:
        notes.append(f"No matching day-ahead price for {market} in entsoe_data.db "
                     "(IT futures settle on PUN; GB futures are in GBP).")
    return out, zone, notes


def build_spread(market: str, other: str, maturity: str, load: str, today: date | None = None):
    """Same rows as build(), but every price is the spread market - other.
    Only delivery periods both markets have. Extra columns: each leg's spot."""
    a, zone_a, notes_a = build(market, maturity, load, today)
    b, zone_b, notes_b = build(other, maturity, load, today)
    notes = [f"{market}: {n}" for n in notes_a] + [f"{other}: {n}" for n in notes_b]
    if a.empty or b.empty:
        return pd.DataFrame(), (zone_a, zone_b), notes
    m = a.merge(b, on="start", suffixes=("", "_b"))
    if m.empty:
        notes.append(f"{market} and {other} have no {maturity.lower()} delivery in common yet.")
        return pd.DataFrame(), (zone_a, zone_b), notes

    def sub(x, y):
        return num(x - y) if pd.notna(x) and pd.notna(y) else None

    rows = []
    for r in m.to_dict("records"):
        fut_t = sub(r["Last traded"], r["Last traded_b"])
        fut_s = sub(r["Last settle"], r["Last settle_b"])
        spot = sub(r["Spot outcome"], r["Spot outcome_b"])
        a_day, b_day = r["Last trade day"], r["Last trade day_b"]
        s_a, s_b = r["Status"], r["Status_b"]
        rows.append({
            "start": r["start"], "Delivery": r["Delivery"],
            "Status": s_a if s_a == s_b else f"{market}: {s_a} | {other}: {s_b}",
            "Last trade day": a_day if a_day == b_day else f"{a_day} / {b_day}",
            "Trades": None,
            "Last traded": fut_t,
            "Settle day": r["Settle day"] if r["Settle day"] == r["Settle day_b"]
                          else f"{r['Settle day']} / {r['Settle day_b']}",
            "Last settle": fut_s,
            "Spot A": r["Spot outcome"], "Spot B": r["Spot outcome_b"],
            "Spot outcome": spot,
            "Spot − last traded": sub(spot, fut_t) if spot is not None and fut_t is not None else None,
            "Spot − last settle": sub(spot, fut_s) if spot is not None and fut_s is not None else None,
            "EEX final settle": sub(r["EEX final settle"], r["EEX final settle_b"]),
            "complete": bool(r["complete"] and r["complete_b"]),
            "currency": r["currency"] if r["currency"] == r["currency_b"] else "mixed",
        })
    out = pd.DataFrame(rows).sort_values("start", ascending=False)
    return out, (zone_a, zone_b), notes


def num(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def fmt_d(d):
    if d is None or (not isinstance(d, str) and pd.isna(d)):
        return None
    x = date.fromisoformat(str(d)[:10])
    return f"{x:%a} {x.day} {MON[x.month - 1]} {x.year % 100:02d}"


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

def seg(id_, options, value):
    return dcc.RadioItems(id=id_, options=options, value=value, inline=True,
                          className="seg", persistence=True, persistence_type="local")


def ctl(caption, component):
    return html.Div([html.Label(caption, className="cap"), component], className="ctl")


def full_layout():
    return html.Div([
        html.H1("Futures vs spot"),
        html.Div("How each EEX future traded last before delivery, and what the "
                 "day-ahead market settled at afterwards.", className="sub"),
        html.Div([
            ctl("Country", dcc.Dropdown(id="market", clearable=False, persistence=True,
                                        persistence_type="local", style={"width": 230})),
            ctl("Spread vs", dcc.Dropdown(id="other", clearable=False, persistence=True,
                                          persistence_type="local", style={"width": 230})),
            ctl("Period", seg("maturity", [{"label": l, "value": v} for v, l in PERIODS], "Week")),
            ctl("Load", seg("load", LOADS, "Base")),
            ctl("Compare spot with", seg("basis", ["Last traded", "Last settle"], "Last traded")),
        ], className="controls"),
        dcc.Loading(html.Div(id="body"), type="dot", color=SERIES[0]),
        html.Div("Sources: EEX (futures, for information only - systematic republication "
                 "needs EEX's permission), ENTSO-E (day-ahead).", className="foot"),
    ], className="wrap")


def make_layout():
    missing = [(n, c) for p, n, c in ((EEX_DB, EEX_NAME, EEX_CANDIDATES),
                                      (SPOT_DB, SPOT_NAME, SPOT_CANDIDATES)) if p is None]
    if missing:
        return html.Div([html.H1("Futures vs spot")] + [
            html.Div(className="warn", children=[
                f"Can't find {n}. Looked in:", html.Ul([html.Li(str(x)) for x in c]),
                "Start with --db / --spot-db to point at it."]) for n, c in missing],
            className="wrap")
    return full_layout()


app = Dash(__name__, title="Futures vs spot")
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.layout = make_layout
app.validation_layout = full_layout()


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
@app.callback(Output("market", "options"), Output("market", "value"),
              Output("other", "options"), Output("other", "value"),
              Input("market", "id"), State("market", "value"), State("other", "value"))
def markets(_, current, other):
    con = load_contracts()
    have = set(con["market"])
    with_spot = sorted(m for m in have if SPOT_ZONE.get(m))
    without = sorted(m for m in have if not SPOT_ZONE.get(m))
    order = (["DE"] if "DE" in with_spot else []) + [m for m in with_spot if m != "DE"] + without
    opts = [{"label": m + (f"  (spot {SPOT_ZONE[m]})" if SPOT_ZONE.get(m) else "  (no spot)"),
             "value": m} for m in order]
    value = current if current in order else (order[0] if order else None)
    # Spread vs: "none" = plain futures vs spot for one country.
    other_opts = [{"label": "— none (one country) —", "value": "none"}] + \
        [{"label": o["label"], "value": o["value"]} for o in opts]
    other = other if other in ["none"] + order else "none"
    return opts, value, other_opts, other


def kpi(caption, value, sub=None):
    kids = [html.Div(caption, className="kpi-label"), html.Div(value, className="kpi-value")]
    if sub:
        kids.append(html.Div(sub, className="kpi-delta"))
    return html.Div(kids, className="kpi")


def f2(v, sign=False):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+,.2f}" if sign else f"{v:,.2f}"


@app.callback(Output("body", "children"),
              Input("market", "value"), Input("maturity", "value"),
              Input("load", "value"), Input("basis", "value"), Input("other", "value"))
def render(market, maturity, load, basis, other):
    if not market:
        return html.Div("No power futures in the database yet.", className="warn")
    spread = bool(other) and other not in ("none", market)
    if spread:
        df, zones, notes = build_spread(market, other, maturity, load)
        zone = f"{zones[0] or 'none'} − {zones[1] or 'none'}"
        if None in zones:
            zone = None
        name = f"{market} − {other}"
    else:
        df, zone, notes = build(market, maturity, load)
        name = market
    period_name = dict(PERIODS)[maturity]
    kids = [html.Div(n, className="warn") for n in notes]
    if spread:
        kids.append(html.Div(f"Spread {market} − {other}: every price below is {market} minus "
                             f"{other}. Spot − futures > 0 = the delivered spread came out above "
                             f"the futures spread (long {market} / short {other} gained).",
                             className="note"))
    if df.empty:
        kids.append(html.Div(
            f"No {period_name.lower()} for {name} {load.lower()} has a pre-delivery price yet. "
            "Settlements only go back to Aug 2026; older periods appear as the nightly "
            "trade-feed backfill (eex_ohlc_scraper.py) works back towards 2020.",
            className="warn"))
        return kids

    if basis == "Last traded" and df["Spot − last traded"].notna().sum() == 0 \
            and df["Spot − last settle"].notna().any():
        basis = "Last settle"
        kids.append(html.Div("No trades before delivery in EEX's trade feed for these "
                             "contracts - comparing with the last settlement instead.",
                             className="note"))
    diff_col = "Spot − last traded" if basis == "Last traded" else "Spot − last settle"
    fut_col = "Last traded" if basis == "Last traded" else "Last settle"
    done = df[df["complete"] & df[diff_col].notna()]
    if len(done):
        d = done[diff_col]
        kids.append(html.Div([
            kpi("Delivered periods compared", f"{len(done)}",
                f"{done['Delivery'].iloc[-1]} to {done['Delivery'].iloc[0]}"),
            kpi(f"Average spot spread − futures spread ({basis.lower()})" if spread
                else f"Average spot − {basis.lower()}", f2(d.mean(), True), "€/MWh"),
            kpi("Average miss (absolute)", f2(d.abs().mean()), "€/MWh"),
            kpi("Spot above futures", f"{int((d > 0).sum())} of {len(d)}",
                f"long {market} / short {other} gained" if spread else "buyer of the future gained"),
            kpi("Largest miss", f2(d.loc[d.abs().idxmax()], True),
                done.loc[d.abs().idxmax(), "Delivery"]),
        ], className="kpis"))
    else:
        kids.append(html.Div("No fully delivered period with spot prices yet - the "
                             "numbers below are still moving.", className="note"))

    # Chart: futures price vs spot outcome per delivery, oldest left.
    c = df.iloc[::-1]
    fig = go.Figure()
    what = "spread" if spread else ""
    fig.add_trace(go.Scatter(x=c["Delivery"], y=c[fut_col],
                             name=f"Futures {what} ({fut_col.lower()}, before delivery)" if spread
                             else f"{fut_col} (before delivery)",
                             mode="lines+markers", line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=c["Delivery"], y=c["Spot outcome"],
                             name="Spot spread (delivered)" if spread else "Spot outcome",
                             mode="lines+markers", line=dict(color=SERIES[1], width=2)))
    if spread:
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    fig.update_layout(
        height=340, margin=dict(l=60, r=20, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=INK["muted"]),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
        hovermode="x unified",
        title=dict(text=(f"{name} {load} {period_name.lower()}: futures spread vs spot spread (€/MWh)"
                         if spread else f"{market} {load} {period_name.lower()}: futures vs spot (€/MWh)"),
                   x=0, y=0.97, yanchor="top", font=dict(size=14, color=INK["text"])))
    fig.update_yaxes(gridcolor=INK["grid"], zeroline=False)
    fig.update_xaxes(type="category", showgrid=False)
    kids.append(html.Div(dcc.Graph(figure=fig, config={"displaylogo": False,
                                                       "displayModeBar": False}),
                         className="card"))

    cols = ["Delivery", "Status", "Last trade day", "Trades", "Last traded", "Settle day",
            "Last settle", "Spot outcome", "Spot − last traded", "Spot − last settle",
            "EEX final settle"]
    if load == "Off-peak" or spread:
        cols.remove("Trades")
    if spread:
        cols[cols.index("Spot outcome"):cols.index("Spot outcome")] = ["Spot A", "Spot B"]
    numeric = {"Trades", "Last traded", "Last settle", "Spot outcome", "Spot − last traded",
               "Spot − last settle", "EEX final settle", "Spot A", "Spot B"}
    # Column titles; in spread mode the same ids carry spreads.
    title_of = {c_: c_ for c_ in cols}
    if spread:
        title_of.update({"Last traded": f"Futures spread (last traded)",
                         "Last settle": "Futures spread (last settle)",
                         "Spot A": f"Spot {market}", "Spot B": f"Spot {other}",
                         "Spot outcome": "Spot spread",
                         "Spot − last traded": "Spot spread − futures (traded)",
                         "Spot − last settle": "Spot spread − futures (settle)",
                         "EEX final settle": "EEX final settle spread"})
    styles = []
    for col in ("Spot − last traded", "Spot − last settle"):
        styles += [
            {"if": {"filter_query": f"{{{col}}} > 0", "column_id": col}, "color": INK["up"]},
            {"if": {"filter_query": f"{{{col}}} < 0", "column_id": col}, "color": INK["down"]},
        ]
    styles.append({"if": {"column_id": diff_col}, "fontWeight": 600})
    styles.append({"if": {"filter_query": '{Status} contains "In delivery"'},
                   "backgroundColor": "#fbfaf6", "fontStyle": "italic"})
    kids.append(html.Div(dash_table.DataTable(
        data=df[cols].to_dict("records"),
        columns=[{"name": title_of[c_], "id": c_, "type": "numeric" if c_ in numeric else "text",
                  **({"format": {"specifier": ",.2f"}} if c_ in numeric - {"Trades"} else {})}
                 for c_ in cols],
        sort_action="native", page_size=25,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Segoe UI, Inter, Arial, sans-serif", "fontSize": 13,
                    "padding": "5px 9px", "textAlign": "right",
                    "border": f"1px solid {INK['grid']}"},
        style_cell_conditional=[{"if": {"column_id": c_}, "textAlign": "left"}
                                for c_ in ("Delivery", "Status", "Last trade day", "Settle day")],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"], "whiteSpace": "normal"},
        style_data_conditional=styles,
    ), className="card"))

    off = (" Off-peak futures price is derived: (Base × base hours − Peak × peak hours) "
           "÷ off-peak hours. Last trade days can differ between Base and Peak (both shown)."
           if load == "Off-peak" else "")
    kids.append(html.Details([
        html.Summary("How the numbers are made"),
        html.Div([
            html.P("Last traded = last trade price (EEX trade feed) on the last day the contract "
                   "traded before delivery started. Last settle = EEX settlement on the last "
                   "trading day before delivery." +
                   (" Day ahead: EEX day futures settle on the day-ahead auction (run on the "
                    "day before delivery), so 'before' means before the auction day - "
                    "trades and settlements from the auction day on are left out. "
                    "EEX final settle = the settlement from the auction day on."
                    if maturity == "Day" else "")),
            html.P((f"Spread mode: each price is {market} minus {other}, per delivery period both "
                    "have. " if spread else "") +
                   f"Spot outcome = average day-ahead price ({zone or 'none'}, ENTSO-E) over the "
                   "delivery hours. Peak = 08-20 Mon-Fri CET; weekend peak = 08-20 Sat-Sun; "
                   "day-ahead (Day futures) peak = 08-20 of that day. "
                   "Spot − futures > 0 means spot came out above the futures price." + off),
            html.P("EEX final settle = last settlement EEX published on or after the last "
                   "delivery day (should match the spot outcome when the zone matches). "
                   "Rows in italics are still in delivery: spot so far."),
        ], className="note"),
    ]))
    return kids


# ---------------------------------------------------------------------
def main():
    global EEX_DB, SPOT_DB
    ap = argparse.ArgumentParser(description="Futures vs spot (Dash)")
    ap.add_argument("--port", type=int, default=8057)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default=None, help="path to eex_market_data.db")
    ap.add_argument("--spot-db", default=None, help="path to entsoe_data.db")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the browser once it's up")
    args = ap.parse_args()
    EEX_DB = find(EEX_CANDIDATES, args.db)
    SPOT_DB = find(SPOT_CANDIDATES, args.spot_db)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print(f"EEX database:  {EEX_DB or 'NOT FOUND'}")
        print(f"Spot database: {SPOT_DB or 'NOT FOUND'}")
        print(f"Open {url}   (leave this window open; Ctrl+C stops it)")
        if args.open:
            import webbrowser
            threading.Timer(3.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
