"""
Energy futures viewer - Dash version
====================================

A local dashboard over EEX futures: power and gas from `eex_market_data.db`
(eex_market_scraper.py), plus TTF gas and EUA carbon from `eex_gas_eua.db`
(eex_gas_eua_scraper.py), which adds EUA and daily candles back to ~Nov 2020
- also for expired contracts.

Run it (Windows, in the folder this file sits in):

    python eex_dash.py

then open http://127.0.0.1:8050 in your browser. Leave the black window
open while you use it - closing it stops the dashboard.

Options:
    --port 8060            use another port
    --host 0.0.0.0         reachable from other devices (e.g. over Tailscale)
    --db  C:\\path\\to\\eex_market_data.db   use a specific database file
    --open                 open the browser automatically

First time only:

    pip install dash plotly pandas

Four tabs:
    Forward curve   settlement by delivery period on one trade date,
                    with an earlier date overlaid to show how it moved
    Area spreads    six A-B spread charts (own pair, product, tenor and
                    delivery each; default Base Cal front year) with KPIs,
                    then one delivery period across individual markets
    Correlation     correlation of daily % changes, all power markets + TTF,
                    API2/API4 and EUA for one tenor, N trading days (+/-)
    Production costs  gas and coal plant cost (fuel + carbon) per country and
                    tenor, clean spark (CSS) and clean dark (CDS) spreads,
                    grids of every country; gas defaults ICE (50 %, TTF), 0.184 tCO2/MWh gas,
                    coal 45 % API2 (Platts) (coal + EUR/USD from ice_market_data.db)
    Price history   one contract over time: price (line or daily candles),
                    volume, open interest, key numbers and a CSV download
    Data & runs     what is in the database, how fresh each maturity is,
                    and the scraper's run log

The database is opened READ-ONLY, so this can never disturb a scraper run.
It is re-read automatically whenever the file changes - just refresh the
page or touch a dropdown to see new data.
"""

from __future__ import annotations

import argparse
import os
import calendar
import re
import sqlite3
import threading
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

# ---------------------------------------------------------------------
# Where the database is
# ---------------------------------------------------------------------
DB_NAME = "eex_market_data.db"
HERE = Path(__file__).resolve().parent
DB_CANDIDATES = [
    HERE / DB_NAME,
    HERE / "Claude outputs" / DB_NAME,
    HERE.parent / DB_NAME,
    HERE.parent / "Claude outputs" / DB_NAME,
    Path.cwd() / DB_NAME,
]
DB_PATH: Path | None = None   # set just below, or by --db in main()


def find_db(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for c in DB_CANDIDATES:
        if c.exists():
            return c
    return None


DB_PATH = find_db()

# TTF + EUA (eex_gas_eua_scraper.py). Optional: the app works without it.
GAS_DB_NAME = "eex_gas_eua.db"


def gas_db() -> Path | None:
    """eex_gas_eua.db next to the main database, or next to this file."""
    for folder in ([DB_PATH.parent] if DB_PATH else []) + [HERE]:
        p = folder / GAS_DB_NAME
        if p.exists():
            return p
    return None


# Contracts from eex_gas_eua.db: long name, product, maturity, unit.
# EUA trades as December contracts -> own maturity "December", label Dec-26.
GAS_EUA = {
    "G3BM": ("EEX TTF Natural Gas Month Future", "Base", "Month", "EUR", "MWh"),
    "FEUA": ("EEX EUA Future", "Base", "December", "EUR", "tCO2"),
}


# ---------------------------------------------------------------------
# Colours - same colour-blind-checked palette as the Streamlit viewer.
# Assigned in fixed order and never cycled: max six series per chart.
# ---------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAX_SERIES = 6
ATR_DAYS = 10   # default ATR window (Peter, 2026-10-17); the user can change it
INK = {"text": "#1f2328", "muted": "#6b6a64", "grid": "#e8e7e1", "axis": "#c3c2b7",
       "bg": "#f6f5f1", "card": "#ffffff", "up": "#1a7f37", "down": "#c4314b"}

# ---------------------------------------------------------------------
# Market names
# ---------------------------------------------------------------------
# The scraper's own `area_hint` only knows ~16 prefixes, so the Streamlit
# viewer silently dropped ~145 contracts (France, NL, Belgium, Italy, CH,
# PL, CZ, ...) as area "?" - and labelled ZTP gas (GBB*) as "GB" power.
# Here the market comes from EEX's own long name instead, which every
# contract has.
DEMONYM = {
    "German": "DE", "French": "FR", "Austrian": "AT", "Swiss": "CH",
    "Italian": "IT", "Spanish": "ES", "Dutch": "NL", "Belgian": "BE",
    "Polish": "PL", "Czech": "CZ", "Slovakian": "SK", "Hungarian": "HU",
    "Romanian": "RO", "Bulgarian": "BG", "Croatian": "HR", "Slovenian": "SI",
    "Serbian": "RS", "Greek": "GR", "Finnish": "FI", "GB": "GB",
    "Irish SEM": "IE",
}
JAPAN = re.compile(r"Japan(?:ese)? Power (\w+)")
ZONE = re.compile(r"\b([A-Z]{2}\d)\b")          # DK1, NO2, SE3 ...


def market_of(long_name: str | None, short_code: str) -> tuple[str, str]:
    """(market label, group) from EEX's long contract name."""
    if not long_name:
        return short_code, "Other"
    name = re.sub(r"^EEX(-PXE)?\s+", "", long_name)
    if "Freight" in name:
        return "Capesize freight option", "Other"
    if "Option" in name:
        m = JAPAN.match(name)
        return (f"JP {m.group(1)} option" if m else name), "Other"
    if "EUA" in name:
        return "EUA", "Carbon"
    if "Natural Gas" in name:
        return "Gas " + name.split(" Natural Gas")[0], "Gas"
    if name.startswith("Nordic System Price"):
        return "Nordic SYS", "Power"
    m = JAPAN.match(name)
    if m:
        return f"JP {m.group(1)}", "Power"
    head = name.split(" Power")[0]
    z = ZONE.search(head)
    if z:
        return z.group(1), "Power"
    if head in DEMONYM:
        return DEMONYM[head], "Power"
    return head, "Power"


# ---------------------------------------------------------------------
# Delivery labels
# ---------------------------------------------------------------------
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_CODE = re.compile(r"^D[BP](\d{2})$")
MATURITY_ORDER = ["Year", "Quarter", "Month", "December", "Week", "Weekend", "Day"]


def delivery_date(short_code: str, delivery: str, maturity: str) -> date | None:
    """First day of delivery. For Day futures the day comes from the code
    (DB25 = the 25th) and the month from `delivery` (202609). For Week
    futures the code's last digit n picks the n-th Mon-Sun week whose
    Sunday falls in that month (DEB2 + 202610 = Mon 5 Oct, ISO week 41) -
    the rule EEX's own catalog follows, see power_contracts.py."""
    text = str(delivery)
    if len(text) != 6 or not text.isdigit():
        return None
    y, m = int(text[:4]), int(text[4:])
    if maturity == "Day":
        d = DAY_CODE.match(short_code)
        if not d:
            return None
        day = int(d.group(1))
        if day > calendar.monthrange(y, m)[1]:
            return None                      # DB31 in a 30-day month
        return date(y, m, day)
    if maturity in ("Week", "Weekend"):
        n = short_code[-1]
        if not n.isdigit():
            return None
        first = date(y, m, 1)
        sunday = first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (int(n) - 1))
        if sunday.month != m:
            return None
        # Week delivers Mon-Sun, Weekend Sat-Sun of that same week.
        return sunday - timedelta(days=6 if maturity == "Week" else 1)
    return date(y, m, 1)


def delivery_label(d: date | None, maturity: str, raw: str) -> str:
    """As a trader says it: Cal-28, Q2-27, Nov-26, W41 · 5 Oct,
    WE41 · 10-11 Oct, Fri 25 Sep."""
    if d is None:
        return str(raw)
    yy = f"{d.year % 100:02d}"
    if maturity == "Year":
        return f"Cal-{yy}"
    if maturity == "Quarter":
        return f"Q{(d.month - 1) // 3 + 1}-{yy}"
    if maturity in ("Month", "December"):
        return f"{MON[d.month - 1]}-{yy}"
    if maturity == "Week":
        return f"W{d.isocalendar()[1]:02d} · {d.day} {MON[d.month - 1]}"
    if maturity == "Weekend":
        sun = d + timedelta(days=1)
        if sun.month == d.month:
            return f"WE{d.isocalendar()[1]:02d} · {d.day}-{sun.day} {MON[d.month - 1]}"
        return f"WE{d.isocalendar()[1]:02d} · {d.day} {MON[d.month - 1]}-{sun.day} {MON[sun.month - 1]}"
    if maturity == "Day":
        return f"{d:%a} {d.day} {MON[d.month - 1]}"
    return str(raw)


# ---------------------------------------------------------------------
# Loading - whole table into memory (tens of thousands of rows, fast),
# re-read only when the file's modification time changes.
# ---------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict = {}


def load_gas_eua(path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Settlements and the list of contracts with candles from eex_gas_eua.db,
    shaped like eex_market_prices. Empty frames if the file isn't there."""
    cols = ["short_code", "delivery", "trade_date", "product", "maturity_type",
            "settlement_price", "volume", "gross_open_interest", "currency", "uom",
            "fetched_at", "long_name"]
    if path is None:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["short_code", "delivery"])
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            st = pd.read_sql_query(
                """SELECT short_code, delivery, trade_date, settle AS settlement_price,
                          volume, open_int AS gross_open_interest, fetched_at
                   FROM gas_eua_settlement""", conn)
            oh = pd.read_sql_query(
                """SELECT DISTINCT short_code, delivery FROM gas_eua_ohlc
                   WHERE trades > 0 AND open IS NOT NULL""", conn)
    except Exception:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["short_code", "delivery"])
    st = st[st["short_code"].isin(GAS_EUA)].copy()
    meta = pd.DataFrame(
        [(c, *v) for c, v in GAS_EUA.items()],
        columns=["short_code", "long_name", "product", "maturity_type", "currency", "uom"])
    st = st.merge(meta, on="short_code")[cols]
    oh = oh[oh["short_code"].isin(GAS_EUA)]
    return st, oh


def load() -> dict:
    path = DB_PATH
    gpath = gas_db()
    mtime = (path.stat().st_mtime, gpath.stat().st_mtime if gpath else None)
    with _lock:
        if _cache.get("key") == (str(path), mtime):
            return _cache["data"]
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            prices = pd.read_sql_query(
                """
                SELECT p.short_code, p.delivery, p.trade_date, p.product,
                       p.maturity_type, p.settlement_price, p.volume,
                       p.gross_open_interest, p.currency, p.uom, p.fetched_at,
                       c.long_name
                FROM eex_market_prices p
                LEFT JOIN contracts c ON c.short_code = p.short_code
                """, conn)
            try:
                runs = pd.read_sql_query(
                    "SELECT * FROM fetch_log ORDER BY run_started_at DESC LIMIT 30", conn)
            except Exception:
                runs = pd.DataFrame()

        prices["delivery"] = prices["delivery"].astype(str)
        # TTF / EUA from eex_gas_eua.db. Where both databases have the same
        # contract and day (TTF months), the power database's row is kept.
        g_set, g_ohlc = load_gas_eua(gpath)
        if not g_set.empty:
            g_set["delivery"] = g_set["delivery"].astype(str)
            have = set(zip(prices.short_code, prices.delivery, prices.trade_date.astype(str)))
            new = [k not in have for k in zip(g_set.short_code, g_set.delivery, g_set.trade_date)]
            prices = pd.concat([prices, g_set[new]], ignore_index=True)
        prices["delivery"] = prices["delivery"].astype(str)
        names = {c: market_of(n, c) for c, n in
                 prices[["short_code", "long_name"]].drop_duplicates().itertuples(index=False)}
        prices["market"] = prices["short_code"].map(lambda c: names[c][0])
        prices["group"] = prices["short_code"].map(lambda c: names[c][1])
        keys = prices[["short_code", "delivery", "maturity_type"]].drop_duplicates()
        dd = {(c, d, m): delivery_date(c, d, m) for c, d, m in keys.itertuples(index=False)}
        prices["ddate"] = [dd[k] for k in zip(prices.short_code, prices.delivery, prices.maturity_type)]
        prices = prices[prices["ddate"].notna()].copy()
        prices["dkey"] = prices["ddate"].map(lambda d: d.isoformat())
        prices["dlabel"] = [delivery_label(d, m, r) for d, m, r in
                            zip(prices.ddate, prices.maturity_type, prices.delivery)]
        prices["unit"] = prices["currency"].fillna("?") + "/" + prices["uom"].fillna("?")
        prices["trade_date"] = prices["trade_date"].astype(str)

        settled = prices[prices["settlement_price"].notna()]
        markets = (settled[["market", "group"]].drop_duplicates()
                   .assign(o=lambda f: f["group"].map({"Power": 0, "Gas": 1, "Carbon": 2}).fillna(3))
                   .sort_values(["o", "market"]))
        # Contracts with candles but no settlement stored (expired TTF months
        # and EUA Decembers from the 2020+ backfill): listed as deliveries too.
        older = []
        for code, dlv in g_ohlc.itertuples(index=False):
            dlv = str(dlv)
            if ((settled["short_code"] == code) & (settled["delivery"] == dlv)).any():
                continue
            long_name, product, maturity, cur, uom = GAS_EUA[code]
            dd_ = delivery_date(code, dlv, maturity)
            if dd_ is None:
                continue
            mk, grp = market_of(long_name, code)
            older.append(dict(short_code=code, delivery=dlv, product=product,
                              maturity_type=maturity, currency=cur, uom=uom,
                              long_name=long_name, market=mk, group=grp,
                              dkey=dd_.isoformat(), dlabel=delivery_label(dd_, maturity, dlv),
                              unit=f"{cur}/{uom}"))
        older = pd.DataFrame(older, columns=["short_code", "delivery", "product", "maturity_type",
                                             "currency", "uom", "long_name", "market", "group",
                                             "dkey", "dlabel", "unit"])
        data = {"prices": prices, "settled": settled, "runs": runs, "older": older,
                "markets": markets[["market", "group"]].values.tolist(),
                "path": path, "gas_path": gpath, "mtime": mtime}
        _cache["key"], _cache["data"] = (str(path), mtime), data
        return data


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def opts(values, labels=None):
    labels = labels or {}
    return [{"label": labels.get(v, v), "value": v} for v in values]


def keep(value, allowed, default=None):
    """Keep the current selection if still valid, else fall back."""
    if value in allowed:
        return value
    if default in allowed:
        return default
    return allowed[0] if allowed else None


def fmt(v, dec=2, sign=False):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+,.{dec}f}" if sign else f"{v:,.{dec}f}"


def kpi(label, value, delta=None, delta_val=None):
    children = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")]
    if delta is not None:
        cls = "kpi-delta"
        if delta_val is not None and not pd.isna(delta_val):
            cls += " up" if delta_val > 0 else " down" if delta_val < 0 else ""
        children.append(html.Div(delta, className=cls))
    return html.Div(children, className="kpi")


def table(df: pd.DataFrame, page_size=15):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(2)
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": str(c), "id": str(c)} for c in df.columns],
        page_size=page_size, sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "4px 8px",
                    "textAlign": "right", "border": f"1px solid {INK['grid']}"},
        style_cell_conditional=[{"if": {"column_type": "text"}, "textAlign": "left"}],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"]},
    )


def note(text):
    return html.Div(text, className="note")


def fig_base(title, ytitle, height=360, dates=True):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK["text"])),
        height=height, margin=dict(l=60, r=80, t=48, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=INK["muted"]),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", showlegend=True,
    )
    fig.update_yaxes(title=ytitle, gridcolor=INK["grid"], zeroline=False,
                     linecolor=INK["axis"])
    fig.update_xaxes(gridcolor=INK["grid"], linecolor=INK["axis"], showgrid=False)
    if dates:
        fig.update_xaxes(tickformat="%d %b", nticks=9)
    return fig


def empty_fig(message, height=300):
    fig = fig_base("", "", height=height, dates=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=INK["muted"]))
    return fig


def end_labels(fig, xs, ys, names, colours):
    """Direct label at the end of each line, staggered so labels on the
    same last x don't print over each other."""
    for i, (x, y, n, c) in enumerate(zip(xs, ys, names, colours)):
        fig.add_annotation(x=x, y=y, text=f"<b>{n}</b>", showarrow=False,
                           xanchor="left", xshift=8, yshift=(-10 if i % 2 else 10),
                           font=dict(color=c, size=12))


def graph(fig):
    return dcc.Graph(figure=fig, config={"displaylogo": False, "displayModeBar": False})


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
CSS = f"""
body {{ margin:0; background:{INK['bg']}; color:{INK['text']};
       font-family:'Segoe UI', Inter, Arial, sans-serif; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:16px 20px 40px; }}
h1 {{ font-size:22px; margin:0; }}
.sub {{ color:{INK['muted']}; font-size:13px; margin:2px 0 12px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end;
            margin:12px 0; }}
.ctl {{ display:flex; flex-direction:column; min-width:140px; }}
.ctl.wide {{ min-width:280px; flex:1; }}
.grp {{ align-self:center; font-size:12px; font-weight:600; color:{INK['text']};
        padding:0 2px 0 8px; border-left:3px solid {INK['grid']}; }}
.ctl label {{ font-size:12px; color:{INK['muted']}; margin-bottom:3px; }}
.card {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
        padding:10px 12px; margin-bottom:12px; }}
.row2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.row3 {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }}
@media (max-width:1100px) {{ .row3 {{ grid-template-columns:1fr; }} }}
@media (max-width:900px) {{ .row2 {{ grid-template-columns:1fr; }} }}
.kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 12px; }}
.kpi {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
       padding:8px 14px; min-width:140px; }}
.kpi-label {{ font-size:12px; color:{INK['muted']}; }}
.kpi-value {{ font-size:22px; font-weight:600; }}
.kpi-delta {{ font-size:13px; color:{INK['muted']}; }}
.kpi-delta.up {{ color:{INK['up']}; }} .kpi-delta.down {{ color:{INK['down']}; }}
.note {{ color:{INK['muted']}; font-size:13px; margin:6px 0; }}
.warn {{ background:#fff4d6; border:1px solid #eda100; border-radius:6px;
        padding:8px 12px; font-size:13px; margin:6px 0; }}
.foot {{ color:{INK['muted']}; font-size:12px; margin-top:16px; }}
.btn {{ background:{SERIES[0]}; color:white; border:0; border-radius:6px;
       padding:7px 14px; font-size:13px; cursor:pointer; }}
.seg {{ display:flex; border:1px solid {INK['axis']}; border-radius:6px; overflow:hidden;
        background:{INK['card']}; height:34px; }}
.seg label {{ display:flex; align-items:center; padding:0 14px; font-size:13px; cursor:pointer;
              margin:0 !important; color:{INK['muted']}; }}
.seg label + label {{ border-left:1px solid {INK['grid']}; }}
.seg input {{ display:none; }}
.seg label:has(input:checked) {{ background:{SERIES[0]}; }}
.seg label:has(input:checked), .seg label:has(input:checked) * {{ color:#fff !important; }}
/* Production costs tab: compact, so all tiles fit on one row */
#panel-spark .controls {{ gap:8px; margin:6px 0; }}
#panel-spark .controls {{ align-items:flex-end; }}
#panel-spark .ctl {{ min-width:96px; }}
#panel-spark .grp {{ align-self:flex-end; padding-bottom:7px; margin-left:4px; }}
#panel-spark .ctl label {{ font-size:11px; margin-bottom:2px; }}
#panel-spark .Select-control, #panel-spark .dash-dropdown {{ font-size:13px; }}
#panel-spark .kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(118px, 1fr));
                      gap:6px; margin:4px 0 10px; }}
#panel-spark .kpi {{ min-width:0; padding:5px 8px; }}
#panel-spark .kpi-label {{ font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
#panel-spark .kpi-value {{ font-size:17px; }}
#panel-spark .kpi-delta {{ font-size:11px; }}
#panel-spark .note {{ font-size:12px; margin:3px 0; }}
#panel-spark .btn {{ padding:5px 10px; font-size:12px; }}
/* Area spreads tab: 6 spread cards, 2 per row */
.spread-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin-top:8px; }}
@media (max-width:900px) {{ .spread-grid {{ grid-template-columns:1fr; }} }}
.spread-grid .card {{ margin-bottom:0; }}
.spread-grid .controls {{ gap:6px; margin:2px 0 6px; flex-wrap:nowrap; }}
.spread-grid .ctl {{ min-width:0; flex:1 1 0; }}
.spread-grid .ctl label {{ font-size:11px; margin-bottom:2px; }}
.spread-grid .Select-control, .spread-grid .dash-dropdown {{ font-size:13px; }}
.spread-grid .kpis {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(104px, 1fr));
                      gap:6px; margin:4px 0 6px; }}
.spread-grid .kpi {{ min-width:0; padding:4px 8px; }}
.spread-grid .kpi-label {{ font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.spread-grid .kpi-value {{ font-size:16px; }}
.spread-grid .kpi-delta {{ font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.spread-grid .note {{ font-size:12px; }}
h3.sec {{ font-size:15px; margin:22px 0 0; }}
.spread-grid.three {{ grid-template-columns:repeat(3, minmax(0, 1fr)); }}
@media (max-width:1250px) {{ .spread-grid.three {{ grid-template-columns:1fr; }} }}
details summary {{ cursor:pointer; color:{INK['muted']}; font-size:13px; margin:6px 0; }}
"""


def ctl(label, component, wide=False, narrow=False):
    return html.Div([html.Label(label), component],
                    className="ctl wide" if wide else "ctl",
                    style={"minWidth": 0} if narrow else None)


def dd(id_, multi=False, clearable=False):
    return dcc.Dropdown(id=id_, multi=multi, clearable=clearable)


def curve_tab():
    return html.Div([
        html.Div([
            ctl("Market", dd("c-market")), ctl("Product", dd("c-product")),
            ctl("Maturity", dd("c-maturity")), ctl("Trade date", dd("c-asof")),
            ctl("Compare with", dd("c-compare")),
        ], className="controls"),
        html.Div(id="c-body"),
    ])


# Default pairs for the six spread charts (Peter, 2026-10-17).
SPREAD_PAIRS = [("DE", "FR"), ("DE", "IT"), ("FR", "ES"),
                ("DE", "DK1"), ("IT", "FR"), ("DE", "NL")]
# Order of markets in the pair dropdowns; anything else follows A-Z.
MARKET_PREF = ["DE", "FR", "NL", "BE", "AT", "CH", "IT", "ES", "DK1", "DK2",
               "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5",
               "FI", "Nordic SYS", "PL", "CZ", "SK", "HU", "RO", "GB"]


# Three Area spreads tabs, same layout, different default tenor (Peter, 2026-10-17):
# (tab id, id prefix, default tenor, tab label). Default delivery = front contract.
SPREAD_TABS = [("spread", "", "Year", "Area spreads Cal"),
               ("spreadm", "m-", "Month", "Area spreads Month"),
               ("spreadq", "q-", "Quarter", "Area spreads Quarter")]


def pair_card(P, tenor, i, left, right):
    def pdd(kind, value=None):
        return dcc.Dropdown(id={"type": P + kind, "index": i}, value=value, clearable=False,
                            options=opts([value]) if value else [])
    return html.Div([
        html.Div([
            ctl("Area", pdd("ps-left", left)), ctl("minus", pdd("ps-right", right)),
            ctl("Product", pdd("ps-product", "Base")), ctl("Tenor", pdd("ps-tenor", tenor)),
            ctl("Delivery", pdd("ps-delivery")),
        ], className="controls"),
        html.Div(id={"type": P + "ps-body", "index": i}),
    ], className="card")


# Time spreads tab (Peter, 2026-10-17): one country, three cards -
# front vs next month, quarter and year. Spread = near - far.
TS_TENORS = ["Month", "Quarter", "Year"]
# One row of three cards per country; default countries (Peter, 2026-10-17).
TS_ROWS = ["DE", "FR", "IT", "HU"]


def ts_card(R, i, tenor):
    def pdd(kind, value=None):
        return dcc.Dropdown(id={"type": R + kind, "index": i}, value=value, clearable=False,
                            options=opts([value]) if value else [])
    return html.Div([
        html.Div([ctl("Tenor", pdd("-tenor", tenor)), ctl("Near", pdd("-near")),
                  ctl("minus", pdd("-far"))], className="controls"),
        html.Div(id={"type": R + "-body", "index": i}),
    ], className="card")


def time_spread_tab():
    rows = []
    for r, _ in enumerate(TS_ROWS):
        R = f"ts{r}"
        rows += [
            html.Div([ctl("Country", dd(R + "-market")), ctl("Product", dd(R + "-product"))],
                     className="controls", style={"marginTop": 22 if r else 12}),
            html.Div([ts_card(R, i, t) for i, t in enumerate(TS_TENORS)],
                     className="spread-grid three"),
        ]
    return html.Div(rows)


def spread_tab(P="", tenor="Year"):
    return html.Div([
        html.Div([pair_card(P, tenor, i, l, r) for i, (l, r) in enumerate(SPREAD_PAIRS)],
                 className="spread-grid"),
        html.H3("Individual markets", className="sec"),
        html.Div([
            ctl("Product", dd(P + "s-product")), ctl("Maturity", dd(P + "s-maturity")),
            ctl("Delivery period", dd(P + "s-delivery")),
            ctl(f"Markets (up to {MAX_SERIES})", dd(P + "s-markets", multi=True), wide=True),
        ], className="controls"),
        html.Div(id=P + "s-body"),
    ])


# ICE's EU spark spreads (German / Dutch / Italian Spark Spread (TTF), ICE Endex
# contract specs): "assumes a fuel efficiency of 50%", fuel cost = gas price
# divided by 0.5, no carbon. The clean spark adds EUA x carbon factor, where
# the carbon factor follows from natural gas's emission factor divided by the
# efficiency. TTF is priced per MWh of GROSS calorific value, so the factor is
# on that basis: 0.184 tCO2/MWh (Platts: 0.053942 tCO2/MMBtu HHV). Peter,
# 2026-10-14; was 0.202 (net basis), which overstated carbon by ~10 %.
ICE_EFFICIENCY = 50.0          # percent
GAS_EMISSION = 0.184           # tCO2 per MWh gas, gross calorific value (TTF basis)
SPARK_TENORS = ["Month", "Quarter", "Year"]


def carbon_factor(eff_pct):
    return round(GAS_EMISSION / (eff_pct / 100), 4)


# Coal: no ICE dark spread contract exists, so the Platts convention
# (European Electricity methodology): 6.978 MWh/t, 0.757 tCO2/MWh power at
# 45 % = 0.341 per MWh coal. Default efficiency 45 % as Platts (Peter,
# 2026-10-14), editable.
# API2/API4 are quoted in USD per tonne of 6,000 kcal/kg NAR coal
# = 25.12 GJ = 6.978 MWh. IPCC default for hard coal: 94.6 kg CO2/GJ
# = 0.341 tCO2 per MWh of coal.
COAL_EFFICIENCY = 45.0
COAL_EMISSION = 0.341
COAL_MWH_PER_T = 6.978
COAL_COLOR = "#6e5440"


def coal_factor(eff_pct):
    return round(COAL_EMISSION / (eff_pct / 100), 4)


def num_input(id_, value, step, width=80):
    return dcc.Input(id=id_, type="number", value=value, step=step, debounce=True,
                     style={"height": 36, "width": width, "padding": "0 6px", "boxSizing": "border-box",
                            "border": f"1px solid {INK['axis']}", "borderRadius": 6,
                            "fontSize": 14, "fontFamily": "inherit"})


def spark_tab():
    return html.Div([
        html.Div([
            ctl("Country", dd("sp-market")), ctl("Power", dd("sp-product")),
            ctl("Tenor", dd("sp-maturity")), ctl("Delivery", dd("sp-delivery")),
            ctl("Gas hub", dd("sp-gas")),
            html.Div("Gas plant", className="grp"),
            ctl("Efficiency %", num_input("sp-eff", ICE_EFFICIENCY, 0.1), narrow=True),
            ctl("tCO2/MWh", num_input("sp-cf", carbon_factor(ICE_EFFICIENCY), "any", 90), narrow=True),
            html.Div("Coal plant", className="grp"),
            ctl("Coal", dcc.Dropdown(id="sp-coal", options=["API2", "API4"], value="API2",
                                     clearable=False, style={"minWidth": 110})),
            ctl("Efficiency %", num_input("sp-ceff", COAL_EFFICIENCY, 0.1), narrow=True),
            ctl("tCO2/MWh", num_input("sp-ccf", coal_factor(COAL_EFFICIENCY), "any", 90), narrow=True),
            html.Div(html.Button("Reset defaults", id="sp-reset", className="btn"), className="ctl"),
        ], className="controls"),
        html.Div(f"Carbon factors follow the efficiency: {GAS_EMISSION} tCO2 per MWh gas, "
                 f"{COAL_EMISSION} per MWh coal, ÷ efficiency. Type over one to use your own.",
                 className="note"),
        html.Div(id="sp-body"),
    ])


# Correlation of daily % changes over the last N trading days. 30 = about
# six weeks: long enough to be stable, short enough to show regime changes,
# and inside the ~6 weeks of settlements EEX serves (Peter, 2026-10-14).
CORR_DAYS = 30


def corr_tab():
    small = {"height": 36, "width": 36, "padding": 0, "fontSize": 18, "lineHeight": "1"}
    return html.Div([
        html.Div([
            ctl("Power", dd("co-product")), ctl("Tenor", dd("co-maturity")),
            ctl("Delivery", dd("co-delivery")),
            ctl("Coal", dcc.Dropdown(id="co-coal", options=["API2", "API4"], value="API2",
                                     clearable=False)),
            ctl("Trading days", html.Div([
                html.Button("−", id="co-minus", className="btn", style=small),
                num_input("co-days", CORR_DAYS, 1, 70),
                html.Button("+", id="co-plus", className="btn", style=small),
            ], style={"display": "flex", "gap": "4px", "alignItems": "center"})),
            ctl("Hedge / rolling for", dd("co-focus")),
        ], className="controls"),
        html.Div(id="co-body"),
    ])


def history_tab():
    return html.Div([
        html.Div([
            ctl("Market", dd("h-market")), ctl("Product", dd("h-product")),
            ctl("Maturity", dd("h-maturity")), ctl("Delivery", dd("h-delivery")),
            ctl("Trade dates", dcc.DatePickerRange(id="h-dates", display_format="DD MMM YYYY")),
            ctl("Chart", dcc.RadioItems(
                id="h-style", options=["Line", "Candles"], value="Line", inline=True,
                className="seg", persistence=True, persistence_type="local")),
            html.Div([html.Label("ATR days"),
                      dcc.Input(id="h-atr", type="number", value=ATR_DAYS, min=2, max=250, step=1,
                                debounce=True, persistence=True, persistence_type="local",
                                style={"height": 36, "width": 80, "padding": "0 6px",
                                       "boxSizing": "border-box", "borderRadius": 6,
                                       "border": f"1px solid {INK['axis']}", "fontSize": 14,
                                       "fontFamily": "inherit"})],
                     className="ctl", style={"minWidth": 0}),
            html.Div([html.Button("Download CSV", id="h-csv-btn", className="btn"),
                      dcc.Download(id="h-csv")], className="ctl"),
        ], className="controls"),
        html.Div(id="h-body"),
    ])


TOP_N = 50


def top_tab():
    return html.Div([
        html.Div([
            ctl("Trade date", dcc.DatePickerSingle(id="t-date", display_format="DD MMM YYYY",
                                                   first_day_of_week=1)),
            ctl("Group", dd("t-group")),
            ctl("Country", dd("t-market")),
            ctl("Maturity", dd("t-maturity")),
            ctl("Contracts in delivery", dcc.RadioItems(
                id="t-delivering", options=["Hide", "Show"], value="Hide", inline=True,
                className="seg", persistence=True, persistence_type="local")),
        ], className="controls"),
        html.Div(id="t-body"),
    ])


def runs_tab():
    return html.Div([html.Button("Reload", id="r-reload", className="btn",
                                 style={"margin": "12px 0"}),
                     html.Div(id="r-body")])


def make_layout():
    if DB_PATH is None:
        return html.Div([
            html.H1("Energy futures"),
            html.Div(className="warn", children=[
                f"Can't find {DB_NAME}. It is created the first time "
                "eex_market_scraper.py runs. Looked in:",
                html.Ul([html.Li(str(c)) for c in DB_CANDIDATES]),
                "Start with:  python eex_dash.py --db C:\\path\\to\\eex_market_data.db",
            ]),
        ], className="wrap")
    return full_layout()


def full_layout():
    return html.Div([
        html.H1("Energy futures"),
        html.Div(id="header-line", className="sub"),
        # Tabs only switch which panel is visible. All four panels stay in
        # the page, so every dropdown keeps its selection when you switch
        # tabs and callbacks never point at a component that isn't there.
        # Opens on Price history (Peter, 2026-10-11): DE Base front month.
        dcc.Tabs(id="tabs", value="history", children=[
            dcc.Tab(label=label, value=value, style=TAB_STYLE, selected_style=TAB_SELECTED)
            for value, label in TABS]),
        html.Div(curve_tab(), id="panel-curve", style={"display": "none"}),
        *[html.Div(spread_tab(P, ten), id=f"panel-{key}", style={"display": "none"})
          for key, P, ten, _ in SPREAD_TABS],
        html.Div(time_spread_tab(), id="panel-tspread", style={"display": "none"}),
        html.Div(spark_tab(), id="panel-spark", style={"display": "none"}),
        html.Div(corr_tab(), id="panel-corr", style={"display": "none"}),
        html.Div(history_tab(), id="panel-history"),
        html.Div(top_tab(), id="panel-top", style={"display": "none"}),
        html.Div(runs_tab(), id="panel-runs", style={"display": "none"}),
        html.Div("Source: EEX. For information only - systematic republication "
                 "needs EEX's permission.", className="foot"),
    ], className="wrap")


TABS = [("curve", "Forward curve"), *[(key, lab) for key, _, _, lab in SPREAD_TABS], ("tspread", "Time spreads"),
        ("spark", "Production costs"),
        ("corr", "Correlation"), ("history", "Price history"), ("top", f"Top {TOP_N}"),
        ("runs", "Data & runs")]

TAB_STYLE = {"padding": "7px 12px", "fontSize": 14, "color": INK["muted"],
             "backgroundColor": INK["bg"], "border": "none",
             "borderBottom": f"1px solid {INK['grid']}"}
TAB_SELECTED = {**TAB_STYLE, "color": INK["text"], "fontWeight": 600,
                "borderBottom": f"3px solid {SERIES[0]}"}

app = Dash(__name__, title="Energy futures")
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.layout = make_layout   # a function: re-evaluated on every page load
# Callbacks are checked against the full page, even when the database
# wasn't found at import time (e.g. it's only given later with --db).
app.validation_layout = full_layout()


# ---------------------------------------------------------------------
# Selection logic shared by tabs
# ---------------------------------------------------------------------

def market_options(data):
    return [{"label": f"{m}" + ("" if g == "Power" else f"  ({g.lower()})"), "value": m}
            for m, g in data["markets"]]


def chain(s, market, product, maturity, default_maturity="Year"):
    """Given the settled rows, return valid products/maturities for a
    market and the narrowed frame."""
    f = s[s["market"] == market]
    # Base, Peak first; anything else (e.g. "6-22 Block" weeks) after.
    products = ([p for p in ("Base", "Peak") if p in set(f["product"])]
                + sorted(p for p in f["product"].unique() if p not in ("Base", "Peak")))
    product = keep(product, products, "Base")
    f = f[f["product"] == product]
    maturities = [m for m in MATURITY_ORDER if m in set(f["maturity_type"])]
    maturity = keep(maturity, maturities, default_maturity)
    f = f[f["maturity_type"] == maturity]
    return products, product, maturities, maturity, f


def front_delivery(f: pd.DataFrame):
    """The front contract: the nearest delivery that has NOT started
    delivering yet on the latest trade date. EEX keeps publishing a
    settlement for the month in delivery (Oct-26 on 9 Oct) but nothing
    trades in it any more - so on 9 Oct the front month is Nov-26, the
    front quarter Q1-27. None if nothing is quoted."""
    if f.empty:
        return None
    last_day = f["trade_date"].max()
    latest = f[f["trade_date"] == last_day]
    ahead = latest[latest["dkey"] > last_day]
    return (ahead if not ahead.empty else latest)["dkey"].min()


# ---------------------------------------------------------------------
# Header and tab switching
# ---------------------------------------------------------------------
@app.callback([Output(f"panel-{v}", "style") for v, _ in TABS], Input("tabs", "value"))
def switch_tab(tab):
    return [{"display": "block" if v == tab else "none"} for v, _ in TABS]


@app.callback(Output("header-line", "children"), Input("tabs", "value"))
def header(_):
    d = load()
    s = d["settled"]
    extra = f" + {d['gas_path'].name}" if d.get("gas_path") else ""
    return (f"{d['path'].name}{extra} · {len(s):,} settlements · {s['short_code'].nunique()} contracts · "
            f"{s['market'].nunique()} markets · trade dates {s['trade_date'].min()} to "
            f"{s['trade_date'].max()}")


# ---------------------------------------------------------------------
# 1. Forward curve
# ---------------------------------------------------------------------
@app.callback(
    Output("c-market", "options"), Output("c-market", "value"),
    Output("c-product", "options"), Output("c-product", "value"),
    Output("c-maturity", "options"), Output("c-maturity", "value"),
    Output("c-asof", "options"), Output("c-asof", "value"),
    Output("c-compare", "options"), Output("c-compare", "value"),
    Output("c-body", "children"),
    Input("c-market", "value"), Input("c-product", "value"),
    Input("c-maturity", "value"), Input("c-asof", "value"), Input("c-compare", "value"),
    Input("tabs", "value"),
)
def forward_curve(market, product, maturity, as_of, compare, tab):
    # Only build while visible: Plotly sizes a chart from its container, and
    # a hidden container has no width. Switching tab re-runs this.
    if tab != "curve":
        return [no_update] * 11
    d = load()
    s = d["settled"]
    m_opts = market_options(d)
    market = keep(market, [o["value"] for o in m_opts], "DE")
    products, product, maturities, maturity, f = chain(s, market, product, maturity)

    dates = sorted(f["trade_date"].unique(), reverse=True)
    as_of = keep(as_of, dates)
    earlier = [x for x in dates if as_of and x < as_of]
    compare_vals = ["none"] + earlier
    default_cmp = earlier[min(4, len(earlier) - 1)] if earlier else "none"
    if compare is None or compare not in compare_vals:
        compare = default_cmp
    cmp_opts = [{"label": "(none)", "value": "none"}] + opts(earlier)

    out = [m_opts, market, opts(products), product, opts(maturities), maturity,
           opts(dates), as_of, cmp_opts, compare]

    if f.empty or not as_of:
        return out + [graph(empty_fig(f"No {product} {maturity} settlements for {market}."))]

    wanted = [as_of] + ([compare] if compare != "none" else [])
    cur = f[f["trade_date"].isin(wanted)].sort_values("dkey")
    unit = cur["unit"].iloc[0]
    order = cur.drop_duplicates("dkey").sort_values("dkey")["dlabel"].tolist()

    fig = fig_base(f"{market} {product} {maturity} forward curve — {unit}", unit, 380, dates=False)
    xs, ys, ns, cs = [], [], [], []
    for i, td in enumerate(wanted):
        g = cur[cur["trade_date"] == td]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["dlabel"], y=g["settlement_price"], name=td, mode="lines+markers",
            line=dict(color=SERIES[i], width=2, dash="solid" if i == 0 else "dot"),
            marker=dict(size=7),
            customdata=g[["volume"]].values,
            hovertemplate="%{y:.2f}  (vol %{customdata[0]:,.0f})<extra>" + td + "</extra>"))
        xs.append(g["dlabel"].iloc[-1]); ys.append(g["settlement_price"].iloc[-1])
        ns.append(td); cs.append(SERIES[i])
    # Horizontal labels while they fit; a month of Day contracts doesn't.
    fig.update_xaxes(categoryorder="array", categoryarray=order,
                     tickangle=0 if len(order) <= 14 else -45)
    end_labels(fig, xs, ys, ns, cs)

    wide = (cur.pivot_table(index=["dkey", "dlabel"], columns="trade_date",
                            values="settlement_price").reset_index().sort_values("dkey"))
    kpis = []
    # "Front" = first delivery actually quoted on the chosen trade date.
    # Front = first delivery not yet started on the chosen trade date
    # (EEX keeps settling the month/week in delivery, but it no longer
    # trades). Falls back to the first quoted one.
    front = None
    if as_of in wide and wide[as_of].notna().any():
        quoted = wide[wide[as_of].notna()]
        ahead = quoted[quoted["dkey"] > as_of]
        front = (ahead if not ahead.empty else quoted).iloc[0]
    if compare != "none" and compare in wide and as_of in wide and front is not None:
        wide["change"] = wide[as_of] - wide[compare]
        front = wide.loc[front.name]
        moved = wide["change"].dropna()
        if not moved.empty:
            big = wide.loc[moved.abs().idxmax()]
            kpis = [
                kpi(f"Front {maturity.lower()} · {front['dlabel']}",
                    fmt(front[as_of]), f"{fmt(front['change'], sign=True)} vs {compare}",
                    front["change"]),
                kpi("Average shift", fmt(moved.mean(), sign=True), unit),
                kpi(f"Largest move · {big['dlabel']}", fmt(big["change"], sign=True), unit,
                    big["change"]),
            ]
    elif front is not None:
        kpis = [kpi(f"Front {maturity.lower()} · {front['dlabel']}", fmt(front[as_of]), unit)]

    body = [html.Div(kpis, className="kpis"), html.Div(graph(fig), className="card"),
            html.Details([html.Summary("Table — curve by delivery period"),
                          table(wide.drop(columns=["dkey"]).rename(columns={"dlabel": "delivery"}),
                                page_size=40)])]
    if maturity == "Day":
        body.insert(0, note("Day futures stop trading once delivered, so the curve only "
                            "shows days still ahead of the chosen trade date."))
    return out + [body]


# ---------------------------------------------------------------------
# 2. Area spreads
# ---------------------------------------------------------------------
def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 22nd ..."""
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def leg_series(f: pd.DataFrame) -> pd.Series:
    """Settlement by trade date for one market/product/delivery."""
    return f.groupby("trade_date")["settlement_price"].mean()


def register_spread_tab(key, P, TENOR):
    """Callbacks for one Area spreads tab (ids prefixed with P)."""
    @app.callback(
        Output({"type": P + "ps-left", "index": MATCH}, "options"),
        Output({"type": P + "ps-left", "index": MATCH}, "value"),
        Output({"type": P + "ps-right", "index": MATCH}, "options"),
        Output({"type": P + "ps-right", "index": MATCH}, "value"),
        Output({"type": P + "ps-product", "index": MATCH}, "options"),
        Output({"type": P + "ps-product", "index": MATCH}, "value"),
        Output({"type": P + "ps-tenor", "index": MATCH}, "options"),
        Output({"type": P + "ps-tenor", "index": MATCH}, "value"),
        Output({"type": P + "ps-delivery", "index": MATCH}, "options"),
        Output({"type": P + "ps-delivery", "index": MATCH}, "value"),
        Output({"type": P + "ps-body", "index": MATCH}, "children"),
        Input({"type": P + "ps-left", "index": MATCH}, "value"),
        Input({"type": P + "ps-right", "index": MATCH}, "value"),
        Input({"type": P + "ps-product", "index": MATCH}, "value"),
        Input({"type": P + "ps-tenor", "index": MATCH}, "value"),
        Input({"type": P + "ps-delivery", "index": MATCH}, "value"),
        Input("tabs", "value"),
    )
    def pair_spread(left, right, product, tenor, dkey, tab):
        if tab != key:
            return [no_update] * 11
        d = load()
        s = d["settled"][d["settled"]["group"] == "Power"]
        have = set(s["market"])
        mk = [m for m in MARKET_PREF if m in have] + sorted(have - set(MARKET_PREF))
        # Each card falls back to its own default pair, never to the first markets.
        idx = ctx.outputs_list[0]["id"]["index"]
        dl, dr = SPREAD_PAIRS[idx] if idx < len(SPREAD_PAIRS) else (None, None)
        left = keep(left, mk, dl)
        right = keep(right, [m for m in mk if m != left] or mk, dr)
        a, b = s[s["market"] == left], s[s["market"] == right]
        products = [p for p in ("Base", "Peak") if p in set(a["product"]) and p in set(b["product"])]
        product = keep(product, products, "Base")
        a, b = a[a["product"] == product], b[b["product"] == product]
        tenors = [m for m in ("Year", "Quarter", "Month", "Week")
                  if m in set(a["maturity_type"]) and m in set(b["maturity_type"])]
        tenor = keep(tenor, tenors, TENOR)
        a, b = a[a["maturity_type"] == tenor], b[b["maturity_type"] == tenor]
        common = sorted(set(a["dkey"]) & set(b["dkey"]))
        both = a[a["dkey"].isin(common)]
        labels = both.drop_duplicates("dkey").set_index("dkey")["dlabel"]
        del_opts = [{"label": labels[k], "value": k} for k in common]
        trig = ctx.triggered_id
        trig = trig.get("type") if isinstance(trig, dict) else trig
        # Default / tenor change: the front delivery (Year -> next Cal, today Cal-27).
        if trig == P + "ps-tenor" or dkey not in common:
            dkey = front_delivery(both)
        out = [opts(mk), left, opts(mk), right, opts(products), product,
               opts(tenors), tenor, del_opts, dkey]

        if left == right:
            return out + [note("Pick two different areas.")]
        if dkey is None:
            return out + [note(f"No {product} {tenor.lower()} delivery quoted for both {left} and {right}.")]
        a, b = a[a["dkey"] == dkey], b[b["dkey"] == dkey]
        ua, ub = a["unit"].iloc[0], b["unit"].iloc[0]
        if ua != ub:
            return out + [html.Div(f"{left} is in {ua}, {right} in {ub}. No FX conversion here.",
                                   className="warn")]
        unit = ua
        w = pd.concat([leg_series(a).rename("L"), leg_series(b).rename("R")], axis=1,
                      join="inner").sort_index()
        label = labels.get(dkey, "")
        if w.empty:
            return out + [note(f"{left} and {right} {label} never settled on the same day.")]
        w["spread"] = w["L"] - w["R"]
        sp = w["spread"]
        n = len(sp)
        last = sp.iloc[-1]
        d1 = last - sp.iloc[-2] if n > 1 else None
        d5 = last - sp.iloc[-6] if n > 5 else None
        mean, sd = sp.mean(), sp.std()
        z = (last - mean) / sd if n > 2 and sd > 0 else None
        pct = (sp <= last).mean() * 100
        vol = sp.diff().std() if n > 2 else None
        corr = w["L"].diff().corr(w["R"].diff()) if n > 3 else None
        lL, lR = w["L"].iloc[-1], w["R"].iloc[-1]
        dL = lL - w["L"].iloc[-2] if n > 1 else None
        dR = lR - w["R"].iloc[-2] if n > 1 else None
        rel = last / lR * 100 if lR else None
        day = w.index[-1]

        def sgn(v, dec=2):
            return fmt(v, dec, sign=True)
        tiles = [
            kpi(f"Spread · {day[5:]}", sgn(last), unit),
            kpi("1 day", sgn(d1), "change", d1),
            kpi("1 week", sgn(d5), "5 trading days", d5),
            kpi(f"Average · {n} d", sgn(mean), f"latest {sgn(last - mean)}", last - mean),
            kpi("Min", sgn(sp.min()), f"on {sp.idxmin()[5:]}"),
            kpi("Max", sgn(sp.max()), f"on {sp.idxmax()[5:]}"),
            kpi("Z-score", fmt(z, 1, sign=True), f"{ordinal(round(pct))} percentile"),
            kpi("Daily vol", fmt(vol), "sd of 1-day change"),
            kpi("Leg correlation", fmt(corr), "1-day changes"),
            kpi(f"{left} {label}", fmt(lL), f"{sgn(dL)} d/d", dL),
            kpi(f"{right} {label}", fmt(lR), f"{sgn(dR)} d/d", dR),
            kpi(f"Spread % of {right}", (f"{rel:+.1f} %" if rel is not None else "—"), "relative"),
        ]
        fig = fig_base(f"{left} − {right} · {product} {label} — {unit}", unit, 250)
        x = pd.to_datetime(w.index)
        fig.add_trace(go.Scatter(x=x, y=sp, name=f"{left} − {right}", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2), marker=dict(size=4),
                                 hovertemplate="%{y:+.2f}"))
        # Dynamic y scale: no fill to zero; zero line only when the spread crosses it.
        if sp.min() < 0 < sp.max():
            fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        pad = max((sp.max() - sp.min()) * 0.08, 0.5)
        fig.update_yaxes(range=[sp.min() - pad, sp.max() + pad], autorange=False)
        fig.add_hline(y=mean, line=dict(color=INK["muted"], width=1, dash="dot"),
                      annotation_text=f"avg {mean:+.2f}", annotation_position="top left",
                      annotation_font=dict(size=11, color=INK["muted"]))
        fig.update_layout(showlegend=False, margin=dict(l=50, r=20, t=40, b=30))
        tbl = w.reset_index().rename(columns={"index": "trade_date", "L": left, "R": right,
                                              "spread": f"{left} − {right}"})
        tbl = tbl.rename(columns={tbl.columns[0]: "trade_date"}).sort_values("trade_date", ascending=False)
        return out + [[html.Div(tiles, className="kpis"), graph(fig),
                       html.Details([html.Summary("Table"), table(tbl, page_size=10)])]]


    @app.callback(
        Output(P + "s-product", "options"), Output(P + "s-product", "value"),
        Output(P + "s-maturity", "options"), Output(P + "s-maturity", "value"),
        Output(P + "s-delivery", "options"), Output(P + "s-delivery", "value"),
        Output(P + "s-markets", "options"), Output(P + "s-markets", "value"),
        Output(P + "s-body", "children"),
        Input(P + "s-product", "value"), Input(P + "s-maturity", "value"),
        Input(P + "s-delivery", "value"), Input(P + "s-markets", "value"), Input("tabs", "value"),
    )
    def area_spreads(product, maturity, dkey, picked, tab):
        if tab != key:
            return [no_update] * 9
        d = load()
        s = d["settled"][d["settled"]["group"] == "Power"]
        products = [p for p in ("Base", "Peak") if p in set(s["product"])]
        product = keep(product, products, "Base")
        f = s[s["product"] == product]
        maturities = [m for m in MATURITY_ORDER if m in set(f["maturity_type"])]
        maturity = keep(maturity, maturities, TENOR)
        f = f[f["maturity_type"] == maturity]

        dels = f.drop_duplicates("dkey").sort_values("dkey")
        del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]
        if dkey not in set(dels["dkey"]) or ctx.triggered_id == P + "s-maturity":
            dkey = front_delivery(f)
        f = f[f["dkey"] == dkey]

        available = sorted(f["market"].unique())
        mk_opts = opts(available)
        picked = [m for m in (picked or []) if m in available]
        if not picked and ctx.triggered_id != P + "s-markets":
            picked = [m for m in ("DE", "FR", "DK1", "IT", "ES", "NL") if m in available] \
                or available[:4]
        warn = []
        if len(picked) > MAX_SERIES:
            warn.append(html.Div(f"Showing the first {MAX_SERIES} of {len(picked)} markets — beyond six "
                                 "the colours stop being reliably distinguishable.", className="warn"))
            picked = picked[:MAX_SERIES]
        if picked:
            unit0 = f[f["market"] == picked[0]]["unit"].iloc[0]
            other = [m for m in picked if f[f["market"] == m]["unit"].iloc[0] != unit0]
            if other:
                warn.append(html.Div(f"Left out {', '.join(other)}: priced in a different "
                                     f"currency/unit than {unit0}. No FX conversion here.",
                                     className="warn"))
                picked = [m for m in picked if m not in other]
        label = dels.set_index("dkey")["dlabel"].get(dkey, "")
        out = [opts(products), product, opts(maturities), maturity, del_opts, dkey, mk_opts, picked]
        if not picked:
            return out + [warn + [graph(empty_fig("Pick at least one market."))]]

        g = f[f["market"].isin(picked)].sort_values("trade_date")
        unit = g["unit"].iloc[0]
        fig = fig_base(f"{product} {label} settlement by market — {unit}", unit, 360)
        xs, ys, cs = [], [], []
        for i, m in enumerate(picked):
            r = g[g["market"] == m]
            if r.empty:
                continue
            fig.add_trace(go.Scatter(x=pd.to_datetime(r["trade_date"]), y=r["settlement_price"],
                                     name=m, mode="lines+markers",
                                     line=dict(color=SERIES[i], width=2), marker=dict(size=6),
                                     hovertemplate="%{y:.2f}"))
            xs.append(pd.to_datetime(r["trade_date"].iloc[-1])); ys.append(r["settlement_price"].iloc[-1])
            cs.append(SERIES[i])
        end_labels(fig, xs, ys, [m for m in picked if not g[g["market"] == m].empty], cs)
        wide = g.pivot_table(index="trade_date", columns="market", values="settlement_price").reset_index()
        return out + [warn + [html.Div(graph(fig), className="card"),
                              html.Details([html.Summary(f"Table — {label} by market"),
                                            table(wide.sort_values("trade_date", ascending=False))])]]


for _key, _P, _ten, _ in SPREAD_TABS:
    register_spread_tab(_key, _P, _ten)


def spread_view(w: pd.DataFrame, lname: str, rname: str, title: str, unit: str):
    """KPI tiles + chart + table for a spread. w: index trade_date,
    columns L and R (the two legs). Same tiles as the Area spreads cards."""
    w = w.copy()
    w["spread"] = w["L"] - w["R"]
    sp = w["spread"]
    n = len(sp)
    last = sp.iloc[-1]
    d1 = last - sp.iloc[-2] if n > 1 else None
    d5 = last - sp.iloc[-6] if n > 5 else None
    mean, sd = sp.mean(), sp.std()
    z = (last - mean) / sd if n > 2 and sd > 0 else None
    pct = (sp <= last).mean() * 100
    vol = sp.diff().std() if n > 2 else None
    corr = w["L"].diff().corr(w["R"].diff()) if n > 3 else None
    lL, lR = w["L"].iloc[-1], w["R"].iloc[-1]
    dL = lL - w["L"].iloc[-2] if n > 1 else None
    dR = lR - w["R"].iloc[-2] if n > 1 else None
    rel = last / lR * 100 if lR else None
    day = w.index[-1]

    def sgn(v, dec=2):
        return fmt(v, dec, sign=True)
    tiles = [
        kpi(f"Spread · {day[5:]}", sgn(last), unit),
        kpi("1 day", sgn(d1), "change", d1),
        kpi("1 week", sgn(d5), "5 trading days", d5),
        kpi(f"Average · {n} d", sgn(mean), f"latest {sgn(last - mean)}", last - mean),
        kpi("Min", sgn(sp.min()), f"on {sp.idxmin()[5:]}"),
        kpi("Max", sgn(sp.max()), f"on {sp.idxmax()[5:]}"),
        kpi("Z-score", fmt(z, 1, sign=True), f"{ordinal(round(pct))} percentile"),
        kpi("Daily vol", fmt(vol), "sd of 1-day change"),
        kpi("Leg correlation", fmt(corr), "1-day changes"),
        kpi(lname, fmt(lL), f"{sgn(dL)} d/d", dL),
        kpi(rname, fmt(lR), f"{sgn(dR)} d/d", dR),
        kpi(f"Spread % of {rname}", (f"{rel:+.1f} %" if rel is not None else "—"), "relative"),
    ]
    fig = fig_base(title + f" — {unit}", unit, 250)
    fig.add_trace(go.Scatter(x=pd.to_datetime(w.index), y=sp, name=title, mode="lines+markers",
                             line=dict(color=SERIES[0], width=2), marker=dict(size=4),
                             hovertemplate="%{y:+.2f}"))
    if sp.min() < 0 < sp.max():
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    pad = max((sp.max() - sp.min()) * 0.08, 0.5)
    fig.update_yaxes(range=[sp.min() - pad, sp.max() + pad], autorange=False)
    fig.add_hline(y=mean, line=dict(color=INK["muted"], width=1, dash="dot"),
                  annotation_text=f"avg {mean:+.2f}", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK["muted"]))
    fig.update_layout(showlegend=False, margin=dict(l=50, r=20, t=40, b=30))
    tbl = w.reset_index()
    tbl.columns = ["trade_date", lname, rname, f"{lname} − {rname}"]
    return [html.Div(tiles, className="kpis"), graph(fig),
            html.Details([html.Summary("Table"),
                          table(tbl.sort_values("trade_date", ascending=False), page_size=10)])]


def register_ts_row(r, DEF):
    """Callbacks for one Time spreads row (ids prefixed ts<r>)."""
    R = f"ts{r}"

    @app.callback(
        Output(R + "-market", "options"), Output(R + "-market", "value"),
        Output(R + "-product", "options"), Output(R + "-product", "value"),
        Input(R + "-market", "value"), Input(R + "-product", "value"), Input("tabs", "value"),
    )
    def ts_controls(market, product, tab):
        if tab != "tspread":
            return [no_update] * 4
        d = load()
        s = d["settled"]
        m_opts = market_options(d)
        market = keep(market, [o["value"] for o in m_opts], DEF)
        f = s[s["market"] == market]
        products = ([p for p in ("Base", "Peak") if p in set(f["product"])]
                    + sorted(p for p in f["product"].unique() if p not in ("Base", "Peak")))
        product = keep(product, products, "Base")
        return m_opts, market, opts(products), product


    @app.callback(
        Output({"type": R + "-tenor", "index": MATCH}, "options"),
        Output({"type": R + "-tenor", "index": MATCH}, "value"),
        Output({"type": R + "-near", "index": MATCH}, "options"),
        Output({"type": R + "-near", "index": MATCH}, "value"),
        Output({"type": R + "-far", "index": MATCH}, "options"),
        Output({"type": R + "-far", "index": MATCH}, "value"),
        Output({"type": R + "-body", "index": MATCH}, "children"),
        Input({"type": R + "-tenor", "index": MATCH}, "value"),
        Input({"type": R + "-near", "index": MATCH}, "value"),
        Input({"type": R + "-far", "index": MATCH}, "value"),
        Input(R + "-market", "value"), Input(R + "-product", "value"), Input("tabs", "value"),
    )
    def time_spread(tenor, near, far, market, product, tab):
        if tab != "tspread":
            return [no_update] * 7
        d = load()
        s = d["settled"]
        market, product = market or DEF, product or "Base"
        f = s[(s["market"] == market) & (s["product"] == product)]
        idx = ctx.outputs_list[0]["id"]["index"]
        tenors = [m for m in MATURITY_ORDER if m in set(f["maturity_type"]) and m not in ("Day", "Weekend")]
        tenor = keep(tenor, tenors, TS_TENORS[idx] if idx < len(TS_TENORS) else "Month")
        f = f[f["maturity_type"] == tenor]
        allk = sorted(f["dkey"].unique())
        labels = f.drop_duplicates("dkey").set_index("dkey")["dlabel"]
        dopts = [{"label": labels[k], "value": k} for k in allk]
        latest = f["trade_date"].max() if not f.empty else ""
        quoted = sorted(f[f["trade_date"] == latest]["dkey"].unique())
        ahead = [k for k in quoted if k > latest] or quoted
        trig = ctx.triggered_id
        trig = trig.get("type") if isinstance(trig, dict) else trig
        # Default / country, product or tenor change: front vs next.
        reset = trig in (R + "-tenor", R + "-market", R + "-product") or near not in allk
        if reset:
            near = ahead[0] if ahead else None
        later = [k for k in ahead if near and k > near] or [k for k in allk if near and k > near]
        if reset or far not in allk or (near and far <= near):
            far = later[0] if later else None
        out = [opts(tenors), tenor, dopts, near, dopts, far]
        if not near or not far:
            return out + [note(f"{market} {product}: need two {tenor.lower()} deliveries.")]
        a, b = f[f["dkey"] == near], f[f["dkey"] == far]
        w = pd.concat([leg_series(a).rename("L"), leg_series(b).rename("R")], axis=1,
                      join="inner").sort_index()
        if w.empty:
            return out + [note(f"{labels[near]} and {labels[far]} never settled on the same day.")]
        return out + [spread_view(w, labels[near], labels[far],
                                  f"{market} {product} {labels[near]} − {labels[far]}",
                                  a["unit"].iloc[0])]


for _r, _m in enumerate(TS_ROWS):
    register_ts_row(_r, _m)


# ---------------------------------------------------------------------
# 2b. Spark spread
# ---------------------------------------------------------------------
@app.callback(Output("sp-eff", "value"), Output("sp-cf", "value"),
              Input("sp-eff", "value"), Input("sp-reset", "n_clicks"),
              prevent_initial_call=True)
def spark_factor(eff, _):
    """New efficiency -> carbon factor recalculated. Reset -> ICE values.
    A carbon factor typed by hand stays until the efficiency changes."""
    if ctx.triggered_id == "sp-reset":
        return ICE_EFFICIENCY, carbon_factor(ICE_EFFICIENCY)
    if eff is None or eff <= 0 or eff > 100:
        return no_update, no_update
    return no_update, carbon_factor(eff)


@app.callback(Output("sp-ceff", "value"), Output("sp-ccf", "value"),
              Input("sp-ceff", "value"), Input("sp-reset", "n_clicks"),
              prevent_initial_call=True)
def coal_factor_cb(eff, _):
    if ctx.triggered_id == "sp-reset":
        return COAL_EFFICIENCY, coal_factor(COAL_EFFICIENCY)
    if eff is None or eff <= 0 or eff > 100:
        return no_update, no_update
    return no_update, coal_factor(eff)


ICE_DB = "ice_market_data.db"
_coal_cache: dict = {}


def ice_db() -> Path | None:
    """ice_market_data.db from ice_scraper (a sibling folder on the Desktop)."""
    roots = ([DB_PATH.parent] if DB_PATH else []) + [HERE]
    for r in roots:
        for c in (r.parent / "ice_scraper" / ICE_DB, r / ICE_DB):
            if c.exists():
                return c
    return None


def load_coal():
    """{'prices': {(product, trade_date, strip): usd}, 'fx': Series date->usd_per_eur}
    from ice_market_data.db, cached on file mtime. None if missing."""
    path = ice_db()
    if path is None:
        return None
    key = (str(path), path.stat().st_mtime)
    if _coal_cache.get("key") == key:
        return _coal_cache["data"]
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            px = pd.read_sql_query(
                "SELECT product, strip, trade_date, price FROM ice_settlement "
                "WHERE product IN ('API2', 'API4')", conn)
            try:
                fx = pd.read_sql_query("SELECT date, usd_per_eur FROM ecb_eurusd ORDER BY date", conn)
            except Exception:
                fx = pd.DataFrame(columns=["date", "usd_per_eur"])
    except Exception:
        return None
    data = {"prices": {(p, t, s): v for p, s, t, v in px.itertuples(index=False)},
            "fx": fx.set_index("date")["usd_per_eur"], "path": path}
    _coal_cache["key"], _coal_cache["data"] = key, data
    return data


def coal_strips(maturity: str, d) -> list[list[str]]:
    """ICE strip names for a power delivery, best first: the matching
    contract, then the sets whose average stands in for it."""
    yy = f"{d.year % 100:02d}"
    if maturity == "Month":
        return [[f"{MON[d.month - 1]}{yy}"]]
    if maturity == "Quarter":
        q = (d.month - 1) // 3 + 1
        return [[f"Q{q} {yy}"], [f"{MON[m - 1]}{yy}" for m in range(3 * q - 2, 3 * q + 1)]]
    return [[f"Cal {yy}"], [f"Q{q} {yy}" for q in range(1, 5)],
            [f"{MON[m]}{yy}" for m in range(12)]]


def add_coal(m: pd.DataFrame, coal, hub: str, maturity: str, ceff: float, ccf: float):
    """Add coal columns + CDS to a spark frame. Coal in EUR per MWh of coal
    = USD/t ÷ EUR/USD ÷ 6.978. EUR/USD: ECB rate of that day, else the last
    one up to 7 days before."""
    m = m.copy()
    usd, src, strip, fxv, fxd = [], [], [], [], []
    fx = coal["fx"] if coal else pd.Series(dtype=float)
    for r in m.itertuples(index=False):
        price = s_name = sr = None
        if coal:
            for i, names in enumerate(coal_strips(maturity, r.ddate)):
                vals = [coal["prices"].get((hub, r.trade_date, n)) for n in names]
                if all(v is not None for v in vals):
                    price = sum(vals) / len(vals)
                    s_name = names[0] if i == 0 else None
                    sr = "contract" if i == 0 else "average"
                    break
        rate = rdate = None
        if len(fx):
            prior = fx.loc[:r.trade_date]
            if len(prior) and (pd.Timestamp(r.trade_date) - pd.Timestamp(prior.index[-1])).days <= 7:
                rate, rdate = float(prior.iloc[-1]), prior.index[-1]
        usd.append(price); src.append(sr); strip.append(s_name); fxv.append(rate); fxd.append(rdate)
    m["coal_usd"] = pd.to_numeric(pd.Series(usd, index=m.index), errors="coerce")
    m["coal_src"], m["coal_strip"] = src, strip
    m["fx"] = pd.to_numeric(pd.Series(fxv, index=m.index), errors="coerce")
    m["fx_date"] = fxd
    m["coal_eur_th"] = m["coal_usd"] / m["fx"] / COAL_MWH_PER_T
    m["coal_fuel_cost"] = m["coal_eur_th"] / (ceff / 100)
    m["coal_carbon_cost"] = m["eua"].astype(float) * ccf
    m["coal_cost"] = m["coal_fuel_cost"] + m["coal_carbon_cost"]
    m["gas_cost"] = m["fuel_cost"] + m["carbon_cost"]
    m["cds"] = m["power"] - m["coal_cost"]
    return m


def spread_grid(grid: pd.DataFrame, col: str, title: str):
    cols = grid.drop_duplicates("dkey").sort_values("dkey")[["dkey", "dlabel"]]
    mat = grid.pivot_table(index="market", columns="dkey", values=col).reindex(columns=cols["dkey"])
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    lim = float(np.nanmax(np.abs(mat.values))) if mat.notna().values.any() else 1.0
    text = [[("" if pd.isna(v) else f"{v:.1f}") for v in row] for row in mat.values]
    hm = go.Figure(go.Heatmap(
        z=mat.values, x=cols["dlabel"].tolist(), y=mat.index.tolist(),
        zmin=-lim, zmax=lim, zmid=0,
        colorscale=[[0, INK["down"]], [0.5, "#f7f7f4"], [1, INK["up"]]],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} %{x}: %{z:.2f} EUR/MWh<extra></extra>",
        colorbar=dict(title="EUR/MWh", thickness=12)))
    hm.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK["text"])),
        height=max(260, 28 * len(mat) + 120), margin=dict(l=90, r=40, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=INK["muted"]))
    hm.update_yaxes(autorange="reversed")
    hm.update_xaxes(side="top", tickangle=0 if len(cols) <= 14 else -45)
    return hm


def eua_prices() -> pd.DataFrame:
    """EUA December prices per trade date: EEX settlement, else the day's
    last trade (eex_gas_eua.db candles). Columns trade_date, dec_year,
    eua, eua_src."""
    d = load()
    s = d["settled"]
    st = s[s["market"] == "EUA"][["trade_date", "ddate", "settlement_price"]].copy()
    st["dec_year"] = st["ddate"].map(lambda x: x.year)
    st = st.rename(columns={"settlement_price": "eua"}).drop(columns="ddate")
    st["eua_src"] = "settlement"
    rows = [st]
    have = set(zip(st.trade_date, st.dec_year))
    for dlv in sorted(set(d["older"].query("market == 'EUA'")["delivery"]) |
                      set(s[s["market"] == "EUA"]["delivery"])):
        o = load_ohlc("FEUA", dlv)
        if o.empty:
            continue
        y = int(str(dlv)[:4])
        o = o[[(t, y) not in have for t in o["trade_date"]]]
        if not o.empty:
            rows.append(pd.DataFrame({"trade_date": o["trade_date"], "dec_year": y,
                                      "eua": o["last"], "eua_src": "last trade"}))
    return pd.concat(rows, ignore_index=True)


def pick_eua(eua: pd.DataFrame, trade_date: str, year: int):
    """EUA for power delivered in `year` on one trade date: the Dec-`year`
    contract, else the first later December, else the furthest one quoted
    (power runs further out than EUA). Returns (price, dec_year, source)."""
    day = eua[eua["trade_date"] == trade_date]
    if day.empty:
        return None, None, None
    later = day[day["dec_year"] >= year].sort_values("dec_year")
    row = (later if not later.empty else day.sort_values("dec_year").tail(1)).iloc[0]
    return row["eua"], int(row["dec_year"]), row["eua_src"]


def spark_frame(power: pd.DataFrame, gas: pd.DataFrame, eua: pd.DataFrame,
                eff: float, cf: float) -> pd.DataFrame:
    """One row per trade date with both power and gas settled. EUA from the
    same day; if that day has none (EUA scraper not run yet), the latest EUA
    from up to 7 days before, marked 'carried'."""
    m = (power[["trade_date", "dkey", "ddate", "settlement_price"]]
         .rename(columns={"settlement_price": "power"})
         .merge(gas[["trade_date", "dkey", "settlement_price"]]
                .rename(columns={"settlement_price": "gas"}), on=["trade_date", "dkey"]))
    dates = sorted(eua["trade_date"].unique())
    out = []
    for r in m.itertuples(index=False):
        price, dec, src = pick_eua(eua, r.trade_date, r.ddate.year)
        if price is None:
            prior = [t for t in dates if t < r.trade_date and
                     (pd.Timestamp(r.trade_date) - pd.Timestamp(t)).days <= 7]
            if prior:
                price, dec, src = pick_eua(eua, prior[-1], r.ddate.year)
                src = f"carried from {prior[-1]}"
        out.append((price, dec, src))
    m["eua"], m["eua_dec"], m["eua_src"] = zip(*out) if out else ([], [], [])
    m["fuel_cost"] = m["gas"] / (eff / 100)
    m["carbon_cost"] = m["eua"].astype(float) * cf
    m["ice_spark"] = m["power"] - m["fuel_cost"]
    m["clean_spark"] = m["ice_spark"] - m["carbon_cost"]
    return m.sort_values(["dkey", "trade_date"])


def all_clean_sparks(power: pd.DataFrame, gas: pd.DataFrame, eua: pd.DataFrame,
                     eff: float, cf: float, maturity: str) -> pd.DataFrame:
    """Clean spark for every country and delivery of one tenor, on the latest
    trade date that has gas settlements. Only power rows of that day are
    used, so this stays fast."""
    power = power[power["maturity_type"] == maturity]
    if power.empty or gas.empty:
        return pd.DataFrame()
    td = min(power["trade_date"].max(), gas["trade_date"].max())
    p = power[power["trade_date"] == td]
    g = gas[gas["trade_date"] == td]
    rows = []
    for mk, pm in p.groupby("market"):
        fr = spark_frame(pm, g, eua, eff, cf)
        if not fr.empty:
            fr["market"] = mk
            rows.append(fr)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    labels = p.drop_duplicates("dkey").set_index("dkey")["dlabel"]
    out["dlabel"] = out["dkey"].map(labels)
    return out


@app.callback(
    Output("sp-market", "options"), Output("sp-market", "value"),
    Output("sp-product", "options"), Output("sp-product", "value"),
    Output("sp-maturity", "options"), Output("sp-maturity", "value"),
    Output("sp-delivery", "options"), Output("sp-delivery", "value"),
    Output("sp-gas", "options"), Output("sp-gas", "value"),
    Output("sp-body", "children"),
    Input("sp-market", "value"), Input("sp-product", "value"),
    Input("sp-maturity", "value"), Input("sp-delivery", "value"),
    Input("sp-gas", "value"), Input("sp-eff", "value"), Input("sp-cf", "value"),
    Input("sp-coal", "value"), Input("sp-ceff", "value"), Input("sp-ccf", "value"),
    Input("tabs", "value"),
)
def spark_spread(market, product, maturity, dkey, gas_hub, eff, cf, coal_hub, ceff, ccf, tab):
    if tab != "spark":
        return [no_update] * 11
    d = load()
    s = d["settled"]
    euro = s[s["unit"] == "EUR/MWh"]
    pw = euro[(euro["group"] == "Power") & euro["maturity_type"].isin(SPARK_TENORS)]
    gs = euro[(euro["group"] == "Gas") & (euro["product"] == "Base") &
              euro["maturity_type"].isin(SPARK_TENORS)]

    markets = sorted(pw["market"].unique())
    market = keep(market, markets, "DE")
    f = pw[pw["market"] == market]
    products = [p for p in ("Base", "Peak") if p in set(f["product"])]
    product = keep(product, products, "Base")
    f = f[f["product"] == product]
    maturities = [m for m in SPARK_TENORS
                  if m in set(f["maturity_type"]) and m in set(gs["maturity_type"])]
    maturity = keep(maturity, maturities, "Year")
    f = f[f["maturity_type"] == maturity]
    g_all = gs[gs["maturity_type"] == maturity]
    hubs = sorted(g_all["market"].unique())
    gas_hub = keep(gas_hub, hubs, "Gas TTF")
    g = g_all[g_all["market"] == gas_hub]

    # Only deliveries that both power and gas quote.
    both = sorted(set(f["dkey"]) & set(g["dkey"]))
    dels = f[f["dkey"].isin(both)].drop_duplicates("dkey").sort_values("dkey")
    del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]
    if dkey not in both or ctx.triggered_id in ("sp-maturity", "sp-market"):
        dkey = front_delivery(f[f["dkey"].isin(both)])
    hub_opts = [{"label": h.replace("Gas ", ""), "value": h} for h in hubs]
    out = [opts(markets), market, opts(products), product, opts(maturities), maturity,
           del_opts, dkey, hub_opts, gas_hub]

    if eff is None or not (0 < eff <= 100):
        return out + [html.Div("Efficiency must be between 0 and 100 %.", className="warn")]
    cf = 0.0 if cf is None else float(cf)
    if not dkey:
        return out + [graph(empty_fig(f"No {maturity.lower()} delivery that both {market} "
                                      f"{product} and {gas_hub} quote."))]

    eua = eua_prices()
    if eua.empty:
        return out + [html.Div("No EUA prices found - needs eex_gas_eua.db "
                               "(eex_gas_eua_scraper.py).", className="warn")]
    if ceff is None or not (0 < ceff <= 100):
        return out + [html.Div("Coal efficiency must be between 0 and 100 %.", className="warn")]
    ccf = 0.0 if ccf is None else float(ccf)
    coal = load_coal()
    coal_hub = coal_hub or "API2"

    allx = add_coal(spark_frame(f[f["dkey"].isin(both)], g, eua, eff, cf),
                    coal, coal_hub, maturity, ceff, ccf)
    hist = allx[allx["dkey"] == dkey]
    if hist.empty:
        return out + [graph(empty_fig("No trade date with both power and gas settled."))]
    label = dels.set_index("dkey")["dlabel"][dkey]
    gas_name = gas_hub.replace("Gas ", "")
    unit = "EUR/MWh"
    has_coal = hist["cds"].notna().any()

    # --- KPI tiles ---------------------------------------------------
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else None

    def dd(col):
        if prev is None or pd.isna(last[col]) or pd.isna(prev[col]):
            return None
        return last[col] - prev[col]

    def dtxt(v, base):
        return f"{fmt(v, sign=True)} day-on-day" if v is not None else base

    eua_txt = (f"Dec-{last['eua_dec'] % 100:02d}" if pd.notna(last["eua_dec"]) else "—")
    tiles = [
        kpi(f"CSS · {last['trade_date']}", fmt(last["clean_spark"]),
            dtxt(dd("clean_spark"), unit), dd("clean_spark")),
        kpi(f"CDS · {coal_hub}", fmt(last["cds"]),
            dtxt(dd("cds"), unit) if has_coal else "no coal price yet", dd("cds")),
        kpi(f"{market} {product} {label}", fmt(last["power"]), "EUR/MWh"),
        kpi(f"Gas plant · {eff:g} %", fmt(last["gas_cost"]),
            f"{fmt(last['fuel_cost'])} + {fmt(last['carbon_cost'])} CO2",
            None if dd("gas_cost") is None else -dd("gas_cost")),
        kpi(f"Coal plant · {ceff:g} %", fmt(last["coal_cost"]),
            f"{fmt(last['coal_fuel_cost'])} + {fmt(last['coal_carbon_cost'])} CO2",
            None if dd("coal_cost") is None else -dd("coal_cost")),
        kpi(f"{gas_name} {label}", fmt(last["gas"]), "EUR/MWh"),
        kpi(f"{coal_hub} {last['coal_strip'] or label}", fmt(last["coal_usd"]),
            f"USD/t = {fmt(last['coal_eur_th'])} €/MWh"),
        kpi(f"EUA {eua_txt}", fmt(last["eua"]), "EUR/t"),
        kpi("EUR/USD", fmt(last["fx"], 4), f"ECB {last['fx_date'] or '—'}"),
    ]
    tiles = html.Div(tiles, className="kpis")

    notes = [note(f"CSS = power − gas ÷ {eff:g} % − EUA × {cf:.3f}.   "
                  f"CDS = power − {coal_hub} (USD/t ÷ EUR/USD ÷ {COAL_MWH_PER_T} MWh/t) ÷ "
                  f"{ceff:g} % − EUA × {ccf:.3f}.   Gas: ICE defaults (50 %, TTF), "
                  f"{GAS_EMISSION} tCO2/MWh gas (gross CV). Coal: Platts defaults (45 %, "
                  f"6,000 kcal/kg, {COAL_EMISSION} tCO2/MWh coal).")]
    if coal is None:
        notes.append(html.Div("No coal prices found: needs ice_market_data.db with API2/API4 "
                              "(run ice_scraper.py --backfill --products API2,API4).",
                              className="warn"))
    elif hist["coal_usd"].notna().any() and hist["fx"].isna().all():
        notes.append(html.Div("No EUR/USD rate yet - CDS left blank. Run ice_scraper.py once "
                              "(it now also fetches the ECB rate).", className="warn"))
    elif not has_coal:
        notes.append(html.Div(f"No {coal_hub} price for {label} - CDS left blank. ICE lists "
                              "API2 months to ~2 years out, quarters and the next Cal only.",
                              className="warn"))
    if hist["eua"].isna().any():
        notes.append(html.Div(f"{int(hist['eua'].isna().sum())} day(s) without an EUA price: "
                              "clean spreads left blank there.", className="warn"))
    carried = hist["eua_src"].astype(str).str.startswith("carried").sum()
    if carried:
        notes.append(note(f"EUA: {carried} day(s) carry the previous day's EUA "
                          "(EUA scraper not run for that day yet)."))
    if has_coal and (hist["coal_src"] == "average").any():
        notes.append(note(f"{coal_hub} {label} not listed as one contract: average of its "
                          "quarters/months used."))

    x = pd.to_datetime(hist["trade_date"])
    xs = dict(rangebreaks=[dict(bounds=["sat", "mon"])])

    # --- 1. CSS + CDS (left) with production costs (right) ---------------
    f1 = fig_base(f"{market} {product} {label}: clean spark and clean dark spread, "
                  "with production costs", unit, 380)
    f1.add_trace(go.Scatter(x=x, y=hist["clean_spark"], name="CSS (left)", mode="lines+markers",
                            line=dict(color=SERIES[0], width=2.5), marker=dict(size=5),
                            hovertemplate="%{y:.2f}"))
    f1.add_trace(go.Scatter(x=x, y=hist["cds"], name="CDS (left)", mode="lines+markers",
                            line=dict(color=COAL_COLOR, width=2.5), marker=dict(size=5),
                            hovertemplate="%{y:.2f}"))
    f1.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    f1.add_trace(go.Scatter(x=x, y=hist["gas_cost"], name="Gas plant cost (right)",
                            mode="lines", yaxis="y2",
                            line=dict(color=SERIES[0], width=1.5, dash="dot"),
                            hovertemplate="%{y:.2f}"))
    f1.add_trace(go.Scatter(x=x, y=hist["coal_cost"], name="Coal plant cost (right)",
                            mode="lines", yaxis="y2",
                            line=dict(color=COAL_COLOR, width=1.5, dash="dot"),
                            hovertemplate="%{y:.2f}"))
    f1.update_layout(
        yaxis=dict(title=dict(text="Clean spread — EUR/MWh")),
        yaxis2=dict(title=dict(text="Production cost — EUR/MWh power"), overlaying="y",
                    side="right", showgrid=False, zeroline=False, linecolor=INK["axis"],
                    tickformat=".0f"),
        showlegend=True)
    f1.update_xaxes(**xs)

    # --- 2. Production cost, stacked, one chart per plant -----------------
    def cost_chart(title, fuel, carbon, fuel_name, fuel_rgba):
        fig = fig_base(title, unit, 320)
        fig.add_trace(go.Scatter(x=x, y=fuel, name=fuel_name, mode="lines", stackgroup="c",
                                 line=dict(width=1, color=fuel_rgba.replace("0.35", "1")),
                                 fillcolor=fuel_rgba, hovertemplate="%{y:.2f}"))
        fig.add_trace(go.Scatter(x=x, y=carbon, name="Carbon", mode="lines", stackgroup="c",
                                 line=dict(width=1, color=SERIES[3]),
                                 fillcolor="rgba(237,161,0,0.35)", hovertemplate="%{y:.2f}"))
        fig.add_trace(go.Scatter(x=x, y=hist["power"], name=f"Power {market}", mode="lines",
                                 line=dict(color=INK["text"], width=2), hovertemplate="%{y:.2f}"))
        fig.update_xaxes(**xs)
        return fig

    f2a = cost_chart(f"Gas plant ({gas_name}, {eff:g} %) — {unit}", hist["fuel_cost"],
                     hist["carbon_cost"], "Gas", "rgba(42,120,214,0.35)")
    f2b = cost_chart(f"Coal plant ({coal_hub}, {ceff:g} %) — {unit}", hist["coal_fuel_cost"],
                     hist["coal_carbon_cost"], "Coal", "rgba(110,84,64,0.35)")

    # --- 3. Curves on the latest trade date ------------------------------
    td = hist["trade_date"].max()
    cur = allx[allx["trade_date"] == td].sort_values("dkey").merge(dels[["dkey", "dlabel"]], on="dkey")
    order = dict(categoryorder="array", categoryarray=cur["dlabel"].tolist(),
                 tickangle=0 if len(cur) <= 14 else -45)
    f4 = fig_base(f"CSS and CDS by delivery on {td} — {unit}", unit, 340, dates=False)
    f4.add_trace(go.Bar(x=cur["dlabel"], y=cur["clean_spark"], name="CSS",
                        marker_color=SERIES[0], hovertemplate="%{y:.2f}"))
    f4.add_trace(go.Bar(x=cur["dlabel"], y=cur["cds"], name="CDS",
                        marker_color=COAL_COLOR, hovertemplate="%{y:.2f}"))
    f4.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    f4.update_layout(barmode="group")
    f4.update_xaxes(**order)
    f5 = fig_base(f"Production cost by delivery on {td} vs {market} {product} power — {unit}",
                  unit, 340, dates=False)
    f5.add_trace(go.Scatter(x=cur["dlabel"], y=cur["power"], name=f"Power {market}",
                            mode="lines+markers", line=dict(color=INK["text"], width=2.5),
                            marker=dict(size=8), hovertemplate="%{y:.2f}"))
    f5.add_trace(go.Scatter(x=cur["dlabel"], y=cur["gas_cost"], name="Gas plant cost",
                            mode="lines+markers", line=dict(color=SERIES[0], width=2, dash="dot"),
                            marker=dict(size=7), hovertemplate="%{y:.2f}"))
    f5.add_trace(go.Scatter(x=cur["dlabel"], y=cur["coal_cost"], name="Coal plant cost",
                            mode="lines+markers", line=dict(color=COAL_COLOR, width=2, dash="dot"),
                            marker=dict(size=7), hovertemplate="%{y:.2f}"))
    f5.update_xaxes(**order)

    # --- 4. All countries: CSS and CDS grids -------------------------------
    grid = all_clean_sparks(pw[pw["product"] == product], g_all[g_all["market"] == gas_hub],
                            eua, eff, cf, maturity)
    grid_block = []
    if not grid.empty:
        grid = add_coal(grid, coal, coal_hub, maturity, ceff, ccf)
        gd = grid["trade_date"].iloc[0]
        for col, name, sub in (("clean_spark", "clean spark spreads (CSS)",
                                f"{gas_name} gas, {eff:g} % / {cf:.3f}"),
                               ("cds", "clean dark spreads (CDS)",
                                f"{coal_hub} coal, {ceff:g} % / {ccf:.3f}")):
            if grid[col].notna().any():
                grid_block.append(html.Div(graph(spread_grid(
                    grid, col, f"All {name} — {product} {maturity.lower()}, {sub}, "
                               f"settlements {gd}")), className="card"))
        grid_block.append(note("Green = the plant earns more than fuel + carbon, red = it "
                               "doesn't. Same fuel price, efficiency and carbon factor for every "
                               "country. Rows sorted by average across the curve."))

    tbl = hist[["trade_date", "power", "gas", "fuel_cost", "coal_usd", "fx", "coal_eur_th",
                "coal_fuel_cost", "eua", "eua_dec", "carbon_cost", "coal_carbon_cost",
                "gas_cost", "coal_cost", "clean_spark", "cds"]].copy()
    tbl["eua_dec"] = tbl["eua_dec"].map(lambda v: f"Dec-{int(v) % 100:02d}" if pd.notna(v) else "")
    tbl = tbl.rename(columns={"clean_spark": "css", "fuel_cost": "gas_fuel_cost",
                              "carbon_cost": "gas_carbon_cost"})
    body = notes + [
        tiles,
        html.Div(graph(f1), className="card"),
        html.Div([html.Div(graph(f2a), className="card"), html.Div(graph(f2b), className="card")],
                 className="row2"),
        html.Div([html.Div(graph(f4), className="card"), html.Div(graph(f5), className="card")],
                 className="row2"),
        *grid_block,
        html.Details([html.Summary(f"Table — {market} {product} {label}, every trade date"),
                      table(tbl.sort_values("trade_date", ascending=False))]),
    ]
    return out + [body]


# ---------------------------------------------------------------------
# 2c. Correlation
# ---------------------------------------------------------------------
def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS with a constant. statsmodels if installed, else numpy (same
    numbers, no extra package needed). Returns coef, t-stats, R2, resid."""
    try:
        import statsmodels.api as sm
        res = sm.OLS(y, sm.add_constant(X)).fit()
        return {"coef": res.params.drop("const"), "t": res.tvalues.drop("const"),
                "r2": res.rsquared, "resid": res.resid, "fitted": res.fittedvalues,
                "engine": "statsmodels"}
    except ImportError:
        A = np.column_stack([np.ones(len(X)), X.values])
        beta, *_ = np.linalg.lstsq(A, y.values, rcond=None)
        fitted = A @ beta
        resid = y.values - fitted
        dof = max(len(y) - A.shape[1], 1)
        cov = (resid @ resid / dof) * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum() or 1)
        return {"coef": pd.Series(beta[1:], index=X.columns),
                "t": pd.Series(beta[1:] / se[1:], index=X.columns), "r2": r2,
                "resid": pd.Series(resid, index=y.index),
                "fitted": pd.Series(fitted, index=y.index), "engine": "numpy"}


HEDGE_UNITS = {"TTF": "MWh gas", "EUA": "t CO2", "API2": "t coal", "API4": "t coal"}


def hedge_block(wide: pd.DataFrame, focus: str, fuels: list, days: int, label: str,
                product: str):
    """Hedge ratios for one power market: regress its daily price change
    (EUR/MWh) on the daily changes of TTF (EUR/MWh), EUA (EUR/t) and coal
    (EUR/t) over the window. Coefficient = units of fuel per MWh of power."""
    if focus not in wide:
        return []
    chg = wide.diff().iloc[1:]
    cols = [focus] + [f for f in fuels if f in chg]
    win = chg[cols].tail(days).dropna()
    if len(win) < 10 or len(cols) < 2:
        return [note(f"Hedge ratios need at least 10 days where {focus} and all fuels "
                     f"have a price (have {len(win)}).")]
    y = win[focus]
    coal = next((f for f in fuels if f in ("API2", "API4")), None)
    models = [("TTF only", ["TTF"]), ("TTF + EUA", ["TTF", "EUA"]),
              (f"TTF + EUA + {coal}", ["TTF", "EUA", coal])]
    rows, fits = [], {}
    raw_sd = y.std()
    for name, xs in models:
        xs = [x for x in xs if x and x in win]
        if not xs or name in fits:
            continue
        r = ols(y, win[xs])
        fits[name] = r
        row = {"model": name, "R²": round(r["r2"], 2),
               "risk removed": f"{(1 - r['resid'].std() / raw_sd) * 100:.0f} %",
               "left-over risk €/MWh/day": round(r["resid"].std(), 2)}
        for fu in ["TTF", "EUA", coal]:
            if fu in r["coef"]:
                row[f"{fu} ({HEDGE_UNITS[fu]}/MWh)"] = (
                    f"{r['coef'][fu]:.3f}  (t {r['t'][fu]:.1f})")
            elif fu:
                row[f"{fu} ({HEDGE_UNITS[fu]}/MWh)"] = ""
        rows.append(row)
    # What a power plant at default settings would hedge with, for reference.
    gas_eua = carbon_factor(ICE_EFFICIENCY)
    coal_t = 1 / (COAL_EFFICIENCY / 100 * COAL_MWH_PER_T)
    ref = [{"model": f"Gas plant {ICE_EFFICIENCY:g} % (reference)", "R²": "", "risk removed": "",
            "left-over risk €/MWh/day": "",
            f"TTF ({HEDGE_UNITS['TTF']}/MWh)": f"{100 / ICE_EFFICIENCY:.3f}",
            f"EUA ({HEDGE_UNITS['EUA']}/MWh)": f"{gas_eua:.3f}"},
           {"model": f"Coal plant {COAL_EFFICIENCY:g} % (reference)", "R²": "", "risk removed": "",
            "left-over risk €/MWh/day": "",
            f"EUA ({HEDGE_UNITS['EUA']}/MWh)": f"{coal_factor(COAL_EFFICIENCY):.3f}"}]
    if coal:
        ref[1][f"{coal} ({HEDGE_UNITS[coal]}/MWh)"] = f"{coal_t:.3f}"
    tbl = pd.DataFrame(rows + ref).fillna("")

    # Chart: cumulative change of power vs the TTF + EUA hedge.
    best = fits.get("TTF + EUA") or next(iter(fits.values()))
    x = pd.to_datetime(win.index)
    cf = fig_base(f"{focus} {product} {label}: cumulative price change vs hedge "
                  f"(TTF + EUA fit) — EUR/MWh", "EUR/MWh", 320)
    cf.add_trace(go.Scatter(x=x, y=y.cumsum(), name=f"{focus} power", mode="lines",
                            line=dict(color=INK["text"], width=2.5), hovertemplate="%{y:.2f}"))
    cf.add_trace(go.Scatter(x=x, y=best["fitted"].cumsum(), name="Hedge portfolio",
                            mode="lines", line=dict(color=SERIES[0], width=2, dash="dot"),
                            hovertemplate="%{y:.2f}"))
    cf.add_trace(go.Scatter(x=x, y=best["resid"].cumsum(), name="Hedged position",
                            mode="lines", line=dict(color=SERIES[2], width=2),
                            hovertemplate="%{y:.2f}"))
    cf.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    cf.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # Chart: rolling TTF-only hedge ratio.
    rr = fig_base(f"{focus} {product} {label}: rolling {days}-day hedge ratio vs TTF "
                  "(MWh gas per MWh power)", "MWh gas / MWh", 280)
    both = chg[[focus, "TTF"]].dropna() if "TTF" in chg else pd.DataFrame()
    if len(both) > days:
        beta = (both[focus].rolling(days).cov(both["TTF"]) / both["TTF"].rolling(days).var()).dropna()
        rr.add_trace(go.Scatter(x=pd.to_datetime(beta.index), y=beta, mode="lines",
                                line=dict(color=SERIES[0], width=2), name="TTF ratio",
                                hovertemplate="%{y:.2f}"))
        rr.add_hline(y=100 / ICE_EFFICIENCY, line=dict(color=INK["axis"], width=1, dash="dash"),
                     annotation_text=f"gas plant {ICE_EFFICIENCY:g} %", annotation_position="top left")
        rr.update_layout(showlegend=False)
        rr.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    else:
        rr = empty_fig(f"Rolling hedge ratio needs more than {days} days of history "
                       f"(have {len(both)}).", 280)

    return [
        html.H3(f"Hedge ratios — {focus} {product} {label}, {len(win)} trading days",
                style={"fontSize": 15, "margin": "18px 0 4px"}),
        note("Regression of the daily change in the power price (EUR/MWh) on the daily changes "
             "in TTF (EUR/MWh), EUA (EUR/t) and coal (EUR/t, ECB rate). Each number is how much "
             "of that fuel to trade per MWh of power (sell power → buy fuel). t above ~2 = "
             "statistically solid. TTF and EUA move together, so with both in, the split between "
             "them is less certain than their sum. Engine: " + best["engine"] + "."),
        table(tbl, 10),
        html.Div([html.Div(graph(cf), className="card"), html.Div(graph(rr), className="card")],
                 className="row2"),
    ]


@app.callback(Output("co-days", "value"),
              Input("co-minus", "n_clicks"), Input("co-plus", "n_clicks"),
              State("co-days", "value"), prevent_initial_call=True)
def corr_days(_m, _p, days):
    days = int(days or CORR_DAYS)
    days += -1 if ctx.triggered_id == "co-minus" else 1
    return max(5, min(days, 500))


def corr_prices(product, maturity, dkey, coal_hub):
    """Daily prices for one delivery: every power market (EUR/MWh), TTF,
    coal (EUR/t via ECB EUR/USD) and EUA (Dec of the delivery year).
    Wide frame, one column per instrument, index = trade date."""
    d = load()
    s = d["settled"]
    s = s[(s["unit"] == "EUR/MWh") & (s["maturity_type"] == maturity) & (s["dkey"] == dkey)]
    pw = s[(s["group"] == "Power") & (s["product"] == product)]
    wide = pw.pivot_table(index="trade_date", columns="market", values="settlement_price")
    ttf = s[s["market"] == "Gas TTF"].set_index("trade_date")["settlement_price"]
    wide["TTF"] = ttf
    if wide.empty:
        return wide, None
    ddate = date.fromisoformat(dkey)
    coal = load_coal()
    if coal:
        vals = {}
        for td in wide.index:
            for names in coal_strips(maturity, ddate):
                v = [coal["prices"].get((coal_hub, td, n)) for n in names]
                if all(x is not None for x in v):
                    fx = coal["fx"].loc[:td]
                    if len(fx):
                        vals[td] = sum(v) / len(v) / float(fx.iloc[-1])
                    break
        wide[coal_hub] = pd.Series(vals)
    eua = eua_prices()
    if not eua.empty:
        wide["EUA"] = pd.Series({td: pick_eua(eua, td, ddate.year)[0] for td in wide.index},
                                dtype=float)
    wide = wide.sort_index()
    return wide, ddate


@app.callback(
    Output("co-product", "options"), Output("co-product", "value"),
    Output("co-maturity", "options"), Output("co-maturity", "value"),
    Output("co-delivery", "options"), Output("co-delivery", "value"),
    Output("co-focus", "options"), Output("co-focus", "value"),
    Output("co-body", "children"),
    Input("co-product", "value"), Input("co-maturity", "value"),
    Input("co-delivery", "value"), Input("co-coal", "value"),
    Input("co-days", "value"), Input("co-focus", "value"), Input("tabs", "value"),
)
def correlation(product, maturity, dkey, coal_hub, days, focus, tab):
    if tab != "corr":
        return [no_update] * 9
    d = load()
    s = d["settled"]
    pw = s[(s["group"] == "Power") & (s["unit"] == "EUR/MWh") & s["maturity_type"].isin(SPARK_TENORS)]
    products = [p for p in ("Base", "Peak") if p in set(pw["product"])]
    product = keep(product, products, "Base")
    pw = pw[pw["product"] == product]
    maturities = [m for m in SPARK_TENORS if m in set(pw["maturity_type"])]
    maturity = keep(maturity, maturities, "Year")
    f = pw[pw["maturity_type"] == maturity]
    ttf = s[(s["market"] == "Gas TTF") & (s["maturity_type"] == maturity)]
    both = sorted(set(f["dkey"]) & set(ttf["dkey"]))
    dels = f[f["dkey"].isin(both)].drop_duplicates("dkey").sort_values("dkey")
    # New tenor -> front delivery again (Cal-27 and Jan-27 share a start date).
    if dkey not in both or ctx.triggered_id == "co-maturity":
        de = f[(f["market"] == "DE") & f["dkey"].isin(both)]
        dkey = front_delivery(de if not de.empty else f[f["dkey"].isin(both)])
    del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]
    days = int(days) if days else CORR_DAYS
    days = max(5, days)
    coal_hub = coal_hub or "API2"

    wide, ddate = corr_prices(product, maturity, dkey, coal_hub) if dkey else (pd.DataFrame(), None)
    fuels = [c for c in ("TTF", coal_hub, "EUA") if c in wide.columns and wide[c].notna().any()]
    powers = sorted(c for c in wide.columns if c not in ("TTF", "API2", "API4", "EUA"))
    focus_opts = opts(powers)
    focus = keep(focus, powers, "DE")
    out = [opts(products), product, opts(maturities), maturity, del_opts, dkey, focus_opts, focus]
    if wide.empty:
        return out + [graph(empty_fig("No prices for that delivery."))]

    label = dels.set_index("dkey")["dlabel"].get(dkey, dkey)
    # Daily % changes. A non-positive price (possible for power) can't give a
    # % change, so those days drop out for that market only.
    rets = wide.where(wide > 0).pct_change(fill_method=None) * 100
    rets = rets.iloc[1:]
    avail = len(rets)
    win = rets.tail(days)
    min_obs = max(5, int(0.7 * len(win)))
    enough = [c for c in win.columns if win[c].notna().sum() >= min_obs]
    cols = [c for c in fuels if c in enough] + [c for c in powers if c in enough]
    win = win[cols]
    corr = win.corr(min_periods=min_obs)
    first, last = (win.index.min(), win.index.max()) if len(win) else ("—", "—")

    notes = [note(f"Pearson correlation of daily % changes in settlement prices, {product} "
                  f"{label}, last {len(win)} trading days ({first} → {last}). Fuels first, then "
                  f"power markets sorted by correlation with TTF. {coal_hub} converted to EUR "
                  f"with the ECB rate; EUA = Dec-{ddate.year % 100:02d} (or the next December quoted).")]
    if days > avail:
        notes.append(html.Div(f"Only {avail} trading days of changes stored for {label} - "
                              f"using all of them. Settlements go back to "
                              f"{wide.index.min()}.", className="warn"))
    dropped = [c for c in wide.columns if c not in cols]
    if dropped:
        notes.append(note(f"Left out (fewer than {min_obs} days with a price in the window): "
                          f"{', '.join(dropped)}."))
    if len(cols) < 2:
        return out + notes + [graph(empty_fig("Not enough overlapping prices."))]

    # Order: fuels, then power by correlation with TTF (strongest first).
    if "TTF" in corr:
        pw_sorted = sorted([c for c in cols if c in powers],
                           key=lambda c: -(corr.loc[c, "TTF"] if pd.notna(corr.loc[c, "TTF"]) else -9))
        order = [c for c in cols if c not in powers] + pw_sorted
        corr = corr.loc[order, order]

    # --- 1. Matrix ---------------------------------------------------
    text = [[("" if pd.isna(v) else f"{v:.2f}") for v in row] for row in corr.values]
    hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, INK["down"]], [0.5, "#f7f7f4"], [1, SERIES[0]]],
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="corr", thickness=12)))
    n = len(corr)
    hm.update_layout(
        title=dict(text=f"Correlation matrix — {product} {label}, {len(win)} trading days to {last}",
                   x=0, xanchor="left", font=dict(size=14, color=INK["text"])),
        height=max(420, 24 * n + 140), margin=dict(l=90, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=11, color=INK["muted"]))
    hm.update_yaxes(autorange="reversed")
    hm.update_xaxes(side="top", tickangle=-45)

    # --- 2. Each market vs the three fuels ---------------------------
    pm = [c for c in corr.index if c in powers]
    bars = fig_base(f"Power vs fuels — correlation of daily changes, {product} {label}",
                    "correlation", 340, dates=False)
    fuel_col = {"TTF": SERIES[0], coal_hub: COAL_COLOR, "EUA": SERIES[3]}
    for fu in [c for c in fuels if c in corr.columns]:
        bars.add_trace(go.Bar(x=pm, y=corr.loc[pm, fu], name=fu, marker_color=fuel_col[fu],
                              hovertemplate="%{x} vs " + fu + ": %{y:.2f}<extra></extra>"))
    bars.update_layout(barmode="group")
    bars.update_yaxes(range=[-1, 1])
    bars.update_xaxes(tickangle=-45)

    # --- 3. Rolling correlation of the chosen market vs the fuels ----
    roll = fig_base(f"{focus} {product} {label}: rolling {len(win)}-day correlation with fuels",
                    "correlation", 320)
    if focus in rets:
        for fu in [c for c in fuels if c in rets]:
            rc = rets[focus].rolling(len(win), min_periods=min_obs).corr(rets[fu]).dropna()
            if not rc.empty:
                roll.add_trace(go.Scatter(x=pd.to_datetime(rc.index), y=rc, name=fu, mode="lines",
                                          line=dict(color=fuel_col[fu], width=2),
                                          hovertemplate="%{y:.2f}"))
        roll.update_yaxes(range=[-1, 1])
        roll.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        roll.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if not roll.data:
        roll = empty_fig(f"Rolling correlation needs more than {len(win)} days of history "
                         f"(have {avail}). It fills in as the scrapers collect more.")

    # Tiles: the chosen market's correlations + realised volatility.
    vol = win.std() * np.sqrt(252)
    tiles = [kpi(f"{focus} vs {fu}", fmt(corr.loc[focus, fu]) if focus in corr.index else "—",
                 f"{len(win)} days") for fu in fuels if fu in corr.columns]
    tiles += [kpi(f"Volatility {c}", f"{vol[c]:.0f} %", "annualised, same days")
              for c in [focus] + [f for f in fuels if f in vol]]
    tbl = corr.round(2).reset_index().rename(columns={"index": ""})
    body = notes + [html.Div(tiles, className="kpis"),
                    *hedge_block(wide, focus, fuels, days, label, product),
                    html.Div(graph(hm), className="card"),
                    html.Div(graph(bars), className="card"),
                    html.Div(graph(roll), className="card"),
                    html.Details([html.Summary("Table — correlation matrix"), table(tbl, 40)])]
    return out + [body]


# ---------------------------------------------------------------------
# 3. Price history
# ---------------------------------------------------------------------
def history_frame(market, product, maturity, dkey):
    d = load()
    p = d["prices"]
    return p[(p["market"] == market) & (p["product"] == product) &
             (p["maturity_type"] == maturity) & (p["dkey"] == dkey)].sort_values("trade_date")


OHLC_COLS = ["trade_date", "open", "high", "low", "last", "trades", "tick_volume"]


def load_ohlc(short_code: str, delivery: str) -> pd.DataFrame:
    """Daily open/high/low/last for one contract, built by
    eex_ohlc_scraper.py from EEX's trade feed. Queried on demand (the table
    grows to hundreds of thousands of rows with the backfill, so it is
    not held in memory). Days without a priced trade are left out.
    Empty if the table doesn't exist yet."""
    frames = []
    try:
        uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            frames.append(pd.read_sql_query(
                f"SELECT {', '.join(OHLC_COLS)} FROM eex_market_ohlc "
                "WHERE short_code = ? AND delivery = ? AND trades > 0 AND open IS NOT NULL "
                "ORDER BY trade_date", conn, params=(short_code, str(delivery))))
    except Exception:
        pass
    # TTF / EUA history back to 2020 lives in eex_gas_eua.db.
    gpath = gas_db() if short_code in GAS_EUA else None
    if gpath:
        try:
            with closing(sqlite3.connect(f"{gpath.resolve().as_uri()}?mode=ro", uri=True)) as conn:
                frames.append(pd.read_sql_query(
                    "SELECT trade_date, open, high, low, last, trades, volume AS tick_volume "
                    "FROM gas_eua_ohlc WHERE short_code = ? AND delivery = ? "
                    "AND trades > 0 AND open IS NOT NULL", conn,
                    params=(short_code, str(delivery))))
        except Exception:
            pass
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=OHLC_COLS)
    # Same day in both databases: the first (power database) wins.
    return (pd.concat(frames, ignore_index=True).drop_duplicates("trade_date")
            .sort_values("trade_date").reset_index(drop=True))


def delivery_hours(dkey: str, maturity: str, product: str) -> int | None:
    """Hours in the delivery period (Europe/Berlin clock, so DST days are
    23/25 h). Peak = Mon-Fri 08-20. None if the period or product shape
    isn't known (EUA, blocks, ...)."""
    add = {"Year": pd.DateOffset(years=1), "Quarter": pd.DateOffset(months=3),
           "Month": pd.DateOffset(months=1), "Week": pd.DateOffset(days=7),
           "Weekend": pd.DateOffset(days=2), "Day": pd.DateOffset(days=1)}.get(maturity)
    if add is None or product not in ("Base", "Peak"):
        return None
    start = pd.Timestamp(dkey)
    idx = pd.date_range(start.tz_localize("Europe/Berlin"),
                        (start + add).tz_localize("Europe/Berlin"), freq="h", inclusive="left")
    if product == "Peak":
        idx = idx[(idx.weekday < 5) & (idx.hour >= 8) & (idx.hour < 20)]
    return len(idx) or None


def atr_frame(full: pd.DataFrame, ohlc: pd.DataFrame, n: int) -> pd.DataFrame:
    """Average True Range over n trading days, one row per trade date.
    Day's high/low = traded high/low (trade feed), widened to include the
    settlement; close = settlement (else last trade). True range = the
    bigger of high-low, |high - previous close|, |low - previous close|.
    Days without a trade use the settlement alone (TR = gap to yesterday).
    Simple n-day average of TR."""
    st = full.groupby("trade_date")["settlement_price"].mean().rename("settle")
    o = ohlc.set_index("trade_date")[["high", "low", "last"]] if not ohlc.empty else \
        pd.DataFrame(columns=["high", "low", "last"])
    df = pd.concat([st, o], axis=1).sort_index()
    df["close"] = df["settle"].fillna(df["last"])
    df = df.dropna(subset=["close"])
    hi = df[["high", "settle", "close"]].max(axis=1)
    lo = df[["low", "settle", "close"]].min(axis=1)
    prev = df["close"].shift()
    df["tr"] = (pd.concat([hi, prev], axis=1).max(axis=1)
                - pd.concat([lo, prev], axis=1).min(axis=1))
    df["atr"] = df["tr"].rolling(n, min_periods=n).mean()
    df.index.name = "trade_date"
    return df.reset_index()


def in_range(df, start, end):
    return df[(df["trade_date"] >= str(start)[:10]) & (df["trade_date"] <= str(end)[:10])]


def with_ohlc(settle: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Settlement rows and OHLC rows side by side, one row per trade date."""
    extra = ohlc[["trade_date", "open", "high", "low", "last", "trades"]]
    return settle.merge(extra, on="trade_date", how="outer").sort_values("trade_date")


@app.callback(
    Output("h-market", "options"), Output("h-market", "value"),
    Output("h-product", "options"), Output("h-product", "value"),
    Output("h-maturity", "options"), Output("h-maturity", "value"),
    Output("h-delivery", "options"), Output("h-delivery", "value"),
    Output("h-dates", "min_date_allowed"), Output("h-dates", "max_date_allowed"),
    Output("h-dates", "start_date"), Output("h-dates", "end_date"),
    Output("h-body", "children"),
    Input("h-market", "value"), Input("h-product", "value"),
    Input("h-maturity", "value"), Input("h-delivery", "value"),
    Input("h-dates", "start_date"), Input("h-dates", "end_date"), Input("tabs", "value"),
    Input("h-style", "value"), Input("h-atr", "value"),
)
def price_history(market, product, maturity, dkey, start, end, tab, style, atr_n):
    if tab != "history":
        return [no_update] * 13
    d = load()
    s = d["settled"]
    m_opts = market_options(d)
    market = keep(market, [o["value"] for o in m_opts], "DE")
    products, product, maturities, maturity, f = chain(s, market, product, maturity,
                                                       default_maturity="Month")
    # Expired TTF / EUA contracts have candles but no stored settlement.
    old = d["older"]
    old = old[(old["market"] == market) & (old["product"] == product) &
              (old["maturity_type"] == maturity)]
    dels = (pd.concat([f[["dkey", "dlabel"]], old[["dkey", "dlabel"]]])
            .drop_duplicates("dkey").sort_values("dkey"))
    if dkey not in set(dels["dkey"]):
        # Default: the front contract (nearest delivery still trading) -
        # liquid, and the one with candles. Kept until the user picks
        # another delivery or changes market/product/maturity.
        dkey = front_delivery(f)
    del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]

    allrows = history_frame(market, product, maturity, dkey)
    full = allrows.dropna(subset=["settlement_price"])
    if not full.empty:
        ref = full.iloc[-1]
    elif (old["dkey"] == dkey).any():
        ref = old[old["dkey"] == dkey].iloc[0]
    else:
        ref = None
    # Expired contract: candles only, whatever the toggle says.
    only_candles = full.empty and ref is not None
    candles = style == "Candles" or only_candles
    ohlc_all = (load_ohlc(ref["short_code"], ref["delivery"])
                if ref is not None else pd.DataFrame(columns=OHLC_COLS))
    # The trade feed goes back years (nightly backfill); settlements only
    # ~6 weeks. In candle mode the date picker spans both.
    dates = list(full["trade_date"]) + (list(ohlc_all["trade_date"]) if candles else [])
    lo, hi = (min(dates), max(dates)) if dates else (None, None)
    if ctx.triggered_id != "h-dates" or not start or not end:
        start, end = lo, hi
    out = [m_opts, market, opts(products), product, opts(maturities), maturity,
           del_opts, dkey, lo, hi, start, end]
    if ref is None or (full.empty and ohlc_all.empty):
        return out + [graph(empty_fig("Nothing stored for that delivery."))]

    h = in_range(full, start, end)
    o = in_range(ohlc_all, start, end)
    if h.empty and (not candles or o.empty):
        return out + [note("Nothing stored in the chosen trade-date range.")]

    code, unit, uom, label = ref["short_code"], ref["unit"], ref["uom"], ref["dlabel"]

    try:
        n_atr = int(atr_n) if atr_n else ATR_DAYS
    except (TypeError, ValueError):
        n_atr = ATR_DAYS
    n_atr = max(2, min(250, n_atr))
    af = atr_frame(full, ohlc_all, n_atr)
    a_in = in_range(af, start, end)
    hours = delivery_hours(dkey, maturity, product) if uom in ("MWh", "MWh/D") else None

    # Volume per trade date: EEX's reported volume, else the trade-feed total
    # (they match - checked on DEBY Cal-27, 2026-10-09). Shown in MW
    # (volume / hours in the delivery period) where the shape is known.
    vol = h.set_index("trade_date")["volume"]
    if candles and not o.empty:
        vol = o.set_index("trade_date")["tick_volume"].combine_first(vol)
        vol.update(h.set_index("trade_date")["volume"].dropna())
    vol = vol.sort_index()
    vol_unit = "MW" if hours else uom
    vol_mw = vol / hours if hours else vol

    tiles = []
    if not h.empty:
        last = h.iloc[-1]
        before = full[full["trade_date"] < last["trade_date"]]
        px = last["settlement_price"]
        chg = px - before["settlement_price"].iloc[-1] if not before.empty else None
        upto = full[full["trade_date"] <= last["trade_date"]]["settlement_price"]
        wk = px - upto.iloc[-6] if len(upto) > 5 else None
        move = px - h["settlement_price"].iloc[0]
        hi_, lo_ = h["settlement_price"].max(), h["settlement_price"].min()
        pos = (px - lo_) / (hi_ - lo_) * 100 if hi_ > lo_ else None
        rets = upto.where(upto > 0).pct_change().dropna()
        dvol = upto.diff().std() if len(upto) > 2 else None
        ann = rets.std() * np.sqrt(252) * 100 if len(rets) > 2 else None
        atr_now = af[af["trade_date"] <= last["trade_date"]]["atr"].dropna()
        atr_now = atr_now.iloc[-1] if not atr_now.empty else None
        oi_prev = before["gross_open_interest"].dropna()
        oi_chg = (last["gross_open_interest"] - oi_prev.iloc[-1]
                  if not oi_prev.empty and pd.notna(last["gross_open_interest"]) else None)
        v_last = vol_mw.get(last["trade_date"])
        v_avg = vol_mw[vol_mw.index <= last["trade_date"]].tail(n_atr).mean()
        tiles += [
            kpi(f"Settlement · {last['trade_date'][5:]}", fmt(px),
                f"{fmt(chg, sign=True)} d/d" if chg is not None else unit, chg),
            kpi("1 week", fmt(wk, sign=True), "5 trading days", wk),
            kpi("Move over range", fmt(move, sign=True), "first → last shown", move),
            kpi("High / low settle", f"{fmt(hi_)} / {fmt(lo_)}", "in range"),
            kpi("Position in range", (f"{pos:.0f} %" if pos is not None else "—"),
                "0 = low, 100 = high"),
            kpi(f"ATR {n_atr} d", fmt(atr_now),
                (f"{atr_now / px * 100:.1f} % of price" if atr_now and px else "not enough days")),
            kpi("Daily vol", fmt(dvol), (f"{ann:.0f} % annualised" if ann is not None else "sd of 1-day change")),
            kpi(f"Volume · {last['trade_date'][5:]}", fmt(v_last, 0), f"{vol_unit} · avg {n_atr} d {fmt(v_avg, 0)}"),
            kpi("Open interest", fmt(last["gross_open_interest"], 0),
                f"{fmt(oi_chg, 0, sign=True)} d/d" if oi_chg is not None else "lots (gross)", oi_chg),
        ]
    if candles and not o.empty:
        lo_row = o.iloc[-1]
        tiles.append(kpi(f"Last traded day · {lo_row['trade_date']}", fmt(lo_row["last"]),
                         f"O {fmt(lo_row['open'])} · H {fmt(lo_row['high'])} · "
                         f"L {fmt(lo_row['low'])} · {int(lo_row['trades'])} trades"))
    else:
        tiles.append(kpi("Trading days", f"{len(h)}", f"of {len(full)} stored"))
    kpis = html.Div(tiles, className="kpis")

    x = pd.to_datetime(h["trade_date"])
    xo = pd.to_datetime(o["trade_date"])
    if only_candles:
        notes = [note(f"{code} — {ref['long_name'] or ''}.  Expired before settlements were "
                      "collected, so this shows daily candles from EEX's trade feed only "
                      "(eex_gas_eua_scraper.py).")]
    else:
        notes = [note(f"{code} — {ref['long_name'] or ''}.  Settlements: EEX only serves ~6 weeks, "
                      "so they go as deep as your scraper's run history.")]

    if only_candles and style != "Candles":
        # Expired contract, Line chosen: the daily last traded price.
        pf = fig_base(f"{code} {label} — last traded price per day — {unit}", unit, 320)
        pf.add_trace(go.Scatter(x=xo, y=o["last"], name="Last", mode="lines+markers",
                                line=dict(color=SERIES[0], width=2), marker=dict(size=5),
                                hovertemplate="%{y:.2f}"))
        pf.update_layout(showlegend=False)
    elif candles and not o.empty:
        pf = fig_base(f"{code} {label} — daily open/high/low/last"
                      + ("" if h.empty else ", settlement dotted") + f" — {unit}", unit, 380)
        pf.add_trace(go.Candlestick(
            x=xo, open=o["open"], high=o["high"], low=o["low"], close=o["last"],
            name="Open/high/low/last",
            increasing=dict(line=dict(color=INK["up"], width=1), fillcolor=INK["up"]),
            decreasing=dict(line=dict(color=INK["down"], width=1), fillcolor=INK["down"])))
        if not h.empty:
            pf.add_trace(go.Scatter(x=x, y=h["settlement_price"], name="Settlement",
                                    mode="lines", line=dict(color=SERIES[0], width=1.5, dash="dot"),
                                    hovertemplate="%{y:.2f}"))
        pf.update_layout(xaxis_rangeslider_visible=False)
        notes.append(note("Candles: first, highest, lowest and last trade of each day, from "
                          "every trade on EEX (eex_ohlc_scraper.py; TTF/EUA also eex_gas_eua_scraper.py). Days without a trade have "
                          "no candle. Colour: green = closed above its open, red = below. "
                          "History grows further back as the nightly backfill runs."))
    else:
        if candles:
            notes.append(html.Div("No open/high/low/last for this contract in the chosen range "
                                  "yet — eex_ohlc_scraper.py fills it nightly. Showing the "
                                  "settlement line instead.", className="warn"))
        pf = fig_base(f"{code} {label} settlement — {unit}", unit, 320)
        pf.add_trace(go.Scatter(x=x, y=h["settlement_price"], name="Settlement",
                                mode="lines+markers", line=dict(color=SERIES[0], width=2),
                                marker=dict(size=6), hovertemplate="%{y:.2f}"))
        pf.update_layout(showlegend=False)

    # ATR band: close ± ATR, on the price chart (legend click hides it).
    band = a_in.dropna(subset=["atr"])
    if not band.empty:
        bx = pd.to_datetime(band["trade_date"])
        pf.add_trace(go.Scatter(x=bx, y=band["close"] + band["atr"], mode="lines",
                                line=dict(width=0), hoverinfo="skip", showlegend=False,
                                legendgroup="atr", name="+ATR"))
        pf.add_trace(go.Scatter(x=bx, y=band["close"] - band["atr"], mode="lines",
                                line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(235,104,52,0.14)", legendgroup="atr",
                                name=f"Close ± ATR {n_atr} d", hoverinfo="skip"))
        pf.update_layout(showlegend=True)

    # Volume in MW per trade date (bars), its own chart.
    vf = fig_base(f"Traded volume — {vol_unit} per day"
                  + (f" (MWh ÷ {hours:,} h)" if hours else ""), vol_unit, 230)
    vv = vol_mw[(vol_mw.index >= str(start)[:10]) & (vol_mw.index <= str(end)[:10])]
    vf.add_trace(go.Bar(x=pd.to_datetime(vv.index), y=vv.values, marker_color=SERIES[1],
                        name="Volume", hovertemplate="%{y:,.0f} " + vol_unit))
    vf.update_layout(showlegend=False)

    tf = fig_base(f"ATR {n_atr} d — {unit}", unit, 230)
    if not band.empty:
        tf.add_trace(go.Scatter(x=bx, y=band["atr"], mode="lines", name=f"ATR {n_atr} d",
                                line=dict(color=SERIES[3], width=2), hovertemplate="%{y:.2f}"))
        tf.add_trace(go.Bar(x=pd.to_datetime(a_in["trade_date"]), y=a_in["tr"],
                            name="True range (day)", marker_color="rgba(237,161,0,0.25)",
                            hovertemplate="%{y:.2f}"))
    else:
        tf = empty_fig(f"Need {n_atr} trading days for ATR {n_atr}.", 230)
    tf.update_layout(showlegend=True)

    of = fig_base("Gross open interest — lots", "lots", 230)
    of.add_trace(go.Scatter(x=x, y=h["gross_open_interest"], mode="lines+markers",
                            line=dict(color=SERIES[2], width=2), marker=dict(size=6),
                            name="Open interest", hovertemplate="%{y:,.0f}"))
    of.update_layout(showlegend=False)

    # Same x window on all three charts, weekends squeezed out.
    shown = list(x) + (list(xo) if candles else [])
    span = [min(shown) - pd.Timedelta(hours=18), max(shown) + pd.Timedelta(hours=18)]
    for fig in (pf, vf, of, tf):
        fig.update_xaxes(range=span, rangebreaks=[dict(bounds=["sat", "mon"])])

    cols = ["trade_date", "short_code", "dlabel", "settlement_price", "unit", "volume",
            "gross_open_interest", "fetched_at"]
    tbl = h[cols].rename(columns={"dlabel": "delivery"})
    if not o.empty:
        tbl = with_ohlc(tbl, o)
    body = notes + [
        kpis,
        html.Div(graph(pf), className="card"),
        (html.Div([html.Div(graph(vf), className="card"), html.Div(graph(tf), className="card")],
                  className="row2") if h.empty else
         html.Div([html.Div(graph(vf), className="card"), html.Div(graph(tf), className="card"),
                   html.Div(graph(of), className="card")], className="row3")),
        html.Details([html.Summary(f"Table — {code} {label}, every stored trading day"),
                      table(tbl.sort_values("trade_date", ascending=False))]),
    ]
    return out + [body]


@app.callback(
    Output("h-csv", "data"), Input("h-csv-btn", "n_clicks"),
    State("h-market", "value"), State("h-product", "value"),
    State("h-maturity", "value"), State("h-delivery", "value"),
    State("h-dates", "start_date"), State("h-dates", "end_date"),
    prevent_initial_call=True,
)
def download_csv(_, market, product, maturity, dkey, start, end):
    every = history_frame(market, product, maturity, dkey)
    if every.empty:
        # Expired TTF / EUA contract: candles only.
        old = load()["older"]
        old = old[(old["market"] == market) & (old["product"] == product) &
                  (old["maturity_type"] == maturity) & (old["dkey"] == dkey)]
        if old.empty:
            return no_update
        code, delivery = old["short_code"].iloc[0], old["delivery"].iloc[0]
        ohlc = load_ohlc(code, delivery)
        if start and end:
            ohlc = in_range(ohlc, start, end)
        if ohlc.empty:
            return no_update
        ohlc.insert(1, "short_code", code)
        ohlc.insert(2, "delivery", delivery)
        return dcc.send_data_frame(ohlc.to_csv, f"eex_{code}_{dkey}.csv",
                                   index=False, encoding="utf-8-sig")
    code, delivery = every["short_code"].iloc[0], every["delivery"].iloc[0]
    h = in_range(every, start, end) if start and end else every
    cols = ["trade_date", "short_code", "delivery", "dlabel", "settlement_price", "currency",
            "volume", "uom", "gross_open_interest", "fetched_at"]
    out = h[cols].rename(columns={"dlabel": "delivery_label"})
    ohlc = load_ohlc(code, delivery)
    if start and end:
        ohlc = in_range(ohlc, start, end)
    if not ohlc.empty:
        out = with_ohlc(out, ohlc)       # open/high/low/last columns added
    # utf-8-sig so Excel opens it with the right encoding.
    return dcc.send_data_frame(out.to_csv,
                               f"eex_{code}_{dkey}.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# Top 50 (Peter, 2026-10-17): on one trade date, the 50 contracts with the
# highest traded volume, the biggest change in open interest and the
# biggest open interest - all in LOTS (contracts), not MWh.
# ---------------------------------------------------------------------
EUA_LOT_T = 1000.0          # one EEX EUA future = 1,000 allowances


def lot_sizes() -> dict:
    """MWh (or t) per lot for every contract+delivery.

    EEX reports open interest twice: in lots (grossOpenInt) and in MWh
    (grossOpenIntSz), so their ratio IS the lot size - 1 MW x the hours of
    the delivery period (720 for a 30-day base month, 8,760 for Cal-27,
    168/169 for a week). Traded volume only comes in MWh, so lots traded =
    volume / lot size. Cached with the database file."""
    key = ("lots", str(DB_PATH), DB_PATH.stat().st_mtime)
    if _cache.get("lots_key") == key:
        return _cache["lots"]
    with closing(sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT short_code, delivery, MAX(gross_open_interest_size * 1.0 / gross_open_interest) "
            "FROM eex_market_prices WHERE gross_open_interest > 0 "
            "AND gross_open_interest_size > 0 GROUP BY short_code, delivery").fetchall()
    sizes = {(c, str(d)): v for c, d, v in rows if v}
    _cache["lots_key"], _cache["lots"] = key, sizes
    return sizes


def top_frame(day: str) -> pd.DataFrame:
    """Every contract quoted on `day`, with lots traded, open interest and
    the change in open interest since that contract's previous trade date."""
    p = load()["prices"]
    upto = p[p["trade_date"] <= day].sort_values("trade_date")
    last2 = upto.groupby(["short_code", "delivery"], sort=False).tail(2)
    prev = (last2[last2["trade_date"] < day]
            .groupby(["short_code", "delivery"])["gross_open_interest"].last()
            .rename("oi_prev"))
    f = p[p["trade_date"] == day].join(prev, on=["short_code", "delivery"])
    if f.empty:
        return f
    sizes = lot_sizes()

    def lot(r):
        if r.uom == "tCO2":
            return EUA_LOT_T
        v = sizes.get((r.short_code, r.delivery))
        if v:
            return v
        return delivery_hours(r.dkey, r.maturity_type, r.product)   # 1 MW x hours

    f = f.copy()
    f["lot"] = [lot(r) for r in f.itertuples(index=False)]
    f["lots"] = (f["volume"] / f["lot"]).round()
    f["oi"] = f["gross_open_interest"]
    f["d_oi"] = f["oi"] - f["oi_prev"]
    f["contract"] = f["market"] + " " + f["product"] + " " + f["dlabel"]
    return f


def top_table(df: pd.DataFrame, cols: list, key: str):
    from dash.dash_table.Format import Format, Group, Scheme, Sign
    num = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
    signed = Format(group=Group.yes, precision=0, scheme=Scheme.fixed, sign=Sign.positive)
    price = Format(precision=2, scheme=Scheme.fixed)
    spec = {"#": ("#", None), "contract": ("Contract", None), "short_code": ("Code", None),
            "lots": ("Lots", num), "d_oi": ("Δ OI", signed),
            "oi": ("OI", num), "settlement_price": ("Settle", price),
            "unit": ("Unit", None)}
    widths = {"#": "34px", "contract": "175px", "lots": "70px", "d_oi": "72px",
              "oi": "80px", "settlement_price": "62px"}
    columns = []
    for c in cols:
        name, f = spec[c]
        col = {"name": name, "id": c}
        if f is not None:
            col.update(type="numeric", format=f)
        columns.append(col)
    return dash_table.DataTable(
        id=f"t-tab-{key}",
        data=df[cols].to_dict("records"), columns=columns, cell_selectable=False,
        sort_action="native", page_action="none",
        fixed_rows={"headers": True}, style_table={"height": "620px", "overflowY": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": 12.5, "padding": "3px 6px",
                    "textAlign": "right", "border": f"1px solid {INK['grid']}",
                    "whiteSpace": "nowrap"},
        style_cell_conditional=[
            *[{"if": {"column_id": c}, "width": w, "minWidth": w, "maxWidth": w}
              for c, w in widths.items() if c in cols],
            {"if": {"column_id": "#"}, "color": INK["muted"]},
            {"if": {"column_id": ["contract", "short_code", "unit"]}, "textAlign": "left",
             "overflow": "hidden", "textOverflow": "ellipsis"},
            {"if": {"column_id": key}, "fontWeight": 600},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{d_oi} > 0", "column_id": "d_oi"}, "color": INK["up"]},
            {"if": {"filter_query": "{d_oi} < 0", "column_id": "d_oi"}, "color": INK["down"]},
            {"if": {"filter_query": "{d_oi} = 0", "column_id": "d_oi"}, "color": INK["muted"]},
        ],
        tooltip_data=[{"contract": f"{r['contract']} ({r['short_code']})"}
                      for r in df.to_dict("records")],
        tooltip_delay=300, tooltip_duration=None,
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"]},
    )


@app.callback(
    Output("t-date", "min_date_allowed"), Output("t-date", "max_date_allowed"),
    Output("t-date", "disabled_days"), Output("t-date", "date"),
    Output("t-group", "options"), Output("t-group", "value"),
    Output("t-market", "options"), Output("t-market", "value"),
    Output("t-maturity", "options"), Output("t-maturity", "value"),
    Output("t-body", "children"),
    Input("t-date", "date"), Input("t-group", "value"), Input("t-market", "value"),
    Input("t-maturity", "value"), Input("t-delivering", "value"), Input("tabs", "value"),
)
def top_contracts(day, group, market, maturity, delivering, tab):
    if tab != "top":
        return [no_update] * 11
    p = load()["prices"]
    traded = sorted(p.loc[p["volume"] > 0, "trade_date"].unique())
    if not traded:
        return [None, None, [], None, [], None, [], None, [], None,
                note("No traded volume stored yet.")]
    all_days = set(p["trade_date"].unique())
    lo, hi = traded[0], traded[-1]
    # Default: the last trading day with volume (Peter: "default last trading day").
    day = str(day)[:10] if day else hi
    if day not in all_days:
        day = hi
    span = pd.date_range(p["trade_date"].min(), p["trade_date"].max(), freq="D")
    disabled = [d.strftime("%Y-%m-%d") for d in span if d.strftime("%Y-%m-%d") not in all_days]

    f = top_frame(day)
    # A contract whose delivery has started (Oct-26 on 16 Oct) no longer
    # trades and EEX freezes its open interest - hidden by default so the
    # lists show what is actually tradable.
    hidden = 0
    if delivering != "Show":
        live = f["dkey"] > day
        hidden = int((~live).sum())
        f = f[live]
    groups = ["All"] + [g for g in ("Power", "Gas", "Carbon", "Other") if g in set(f["group"])]
    group = keep(group, groups, "All")
    if group != "All":
        f = f[f["group"] == group]
    # Country (Peter, 2026-10-17): "*" = no open interest on this date
    # (all contracts shown for that country have OI 0 or none reported).
    order = [m for m, _g in load()["markets"]]
    present = set(f["market"])
    countries = [m for m in order if m in present] + sorted(present - set(order))
    oi_by = f.groupby("market")["oi"].sum(min_count=1)
    no_oi = {m for m in countries if not (oi_by.get(m, 0) > 0)}
    market_opts = [{"label": "All", "value": "All"}] + [
        {"label": f"{m} *" if m in no_oi else m, "value": m} for m in countries]
    market = keep(market, ["All"] + countries, "All")
    if market != "All":
        f = f[f["market"] == market]
    mats = ["All"] + [m for m in MATURITY_ORDER if m in set(f["maturity_type"])]
    maturity = keep(maturity, mats, "All")
    if maturity != "All":
        f = f[f["maturity_type"] == maturity]
    out = [p["trade_date"].min(), p["trade_date"].max(), disabled, day,
           opts(groups), group, market_opts, market, opts(mats), maturity]
    if f.empty:
        return out + [note(f"Nothing quoted on {day} for that selection.")]

    def ranked(col, by_abs=False):
        g = f[f[col].notna() & (f[col] != 0)]
        order = g[col].abs() if by_abs else g[col]
        g = g.loc[order.sort_values(ascending=False).index].head(TOP_N).copy()
        g.insert(0, "#", range(1, len(g) + 1))
        return g

    vol, doi, oi = ranked("lots"), ranked("d_oi", by_abs=True), ranked("oi")
    prev_day = max((d for d in all_days if d < day), default=None)
    up, down = f.loc[f["d_oi"] > 0, "d_oi"].sum(), f.loc[f["d_oi"] < 0, "d_oi"].sum()
    tiles = html.Div([
        kpi(f"Lots traded · {day}", fmt(f["lots"].sum(), 0),
            f"{int((f['lots'] > 0).sum())} contracts traded"),
        kpi("Open interest", fmt(f["oi"].sum(), 0), "lots, all contracts shown"),
        kpi("OI change", fmt(up + down, 0, sign=True),
            f"+{up:,.0f} / {down:,.0f} vs {prev_day or 'previous day'}", up + down),
        kpi("Top contract", vol["contract"].iloc[0] if not vol.empty else "—",
            f"{vol['lots'].iloc[0]:,.0f} lots" if not vol.empty else None),
    ], className="kpis")

    def card(title, sub, df, cols, key):
        return html.Div([html.Div(title, style={"fontWeight": 600, "fontSize": 15}),
                         html.Div(sub, className="note", style={"margin": "2px 0 8px"}),
                         top_table(df, cols, key) if not df.empty else note("Nothing to rank.")],
                        className="card")

    body = [
        tiles,
        html.Div([
            card(f"Top {TOP_N} by lots traded", f"Volume on {day} in lots (contracts).",
                 vol, ["#", "contract", "lots", "d_oi", "settlement_price"], "lots"),
            card(f"Top {TOP_N} by change in open interest",
                 f"Biggest move up or down since each contract's previous trade date "
                 f"({prev_day or '—'}), in lots.",
                 doi, ["#", "contract", "d_oi", "oi", "lots"], "d_oi"),
            card(f"Top {TOP_N} by open interest", f"Open interest on {day} in lots.",
                 oi, ["#", "contract", "oi", "d_oi", "lots"], "oi"),
        ], className="row3"),
        note("Lots = contracts. EEX publishes traded volume in MWh, so lots = MWh ÷ lot size; "
             "the lot size (1 MW × hours of the delivery period, e.g. 720 for a 30-day month, "
             "8,760 for Cal-27) is EEX's own open interest in MWh ÷ open interest in lots. "
             "EUA: 1,000 t per lot. Lots of different products are not the same size - a Cal "
             "lot is 12× a month lot. Δ OI = change since the contract's previous trade date. "
             + (f"{hidden} contracts already in delivery are hidden (their open interest is "
                "frozen by EEX). " if hidden else "")
             + "Click a column header to re-sort; hover a contract for its EEX code."
             + (f" Country list: * = no open interest on {day} "
                f"({len(no_oi)} of {len(countries)})." if no_oi else "")),
    ]
    return out + [body]


# ---------------------------------------------------------------------
# 4. Data & runs
# ---------------------------------------------------------------------
@app.callback(Output("r-body", "children"), Input("r-reload", "n_clicks"), Input("tabs", "value"))
def data_and_runs(_, tab):
    if tab != "runs":
        return no_update
    d = load()
    p, s, runs = d["prices"], d["settled"], d["runs"]
    last_trade = s["trade_date"].max()
    last_fetch = str(p["fetched_at"].max())[:16].replace("T", " ")

    fresh = (s.groupby("maturity_type")
             .agg(last_trade_date=("trade_date", "max"),
                  contracts=("short_code", "nunique"),
                  settlements=("settlement_price", "size"))
             .reindex([m for m in MATURITY_ORDER if m in set(s["maturity_type"])]).reset_index())
    stale = fresh[fresh["last_trade_date"] < last_trade]["maturity_type"].tolist()

    by_market = (s.groupby(["group", "market"])
                 .agg(contracts=("short_code", "nunique"), settlements=("settlement_price", "size"),
                      first=("trade_date", "min"), last=("trade_date", "max"),
                      unit=("unit", "first")).reset_index())

    body = [html.Div([
        kpi("Settlements", f"{len(s):,}", f"{len(p):,} rows incl. empty"),
        kpi("Contracts", f"{s['short_code'].nunique()}", f"{s['market'].nunique()} markets"),
        kpi("Trade dates", f"{s['trade_date'].nunique()}", f"{s['trade_date'].min()} → {last_trade}"),
        kpi("Last fetched", last_fetch, "UTC, newest row written"),
    ], className="kpis")]

    if stale:
        body.append(html.Div(f"Not up to date: {', '.join(stale)} — last settlement is older than "
                             f"the newest trade date in the database ({last_trade}).",
                             className="warn"))
    if not runs.empty:
        last_logged = str(runs["finished_at"].iloc[0])[:16].replace("T", " ")
        if last_logged < last_fetch[:16]:
            body.append(html.Div(f"The run log's last entry is {last_logged}, but rows were written "
                                 f"as late as {last_fetch} — some runs are not being logged "
                                 "(e.g. stopped before finishing).", className="warn"))

    body += [
        html.H3("Freshness by maturity", style={"fontSize": 15}),
        table(fresh),
        html.H3("Scraper runs (fetch_log)", style={"fontSize": 15}),
        table(runs) if not runs.empty else note("No runs logged yet."),
        html.H3("Markets in the database", style={"fontSize": 15}),
        table(by_market, page_size=25),
        note(f"Database: {d['path']}"
             + (f"  ·  TTF/EUA: {d['gas_path']}" if d.get("gas_path") else
                "  ·  eex_gas_eua.db not found - no EUA, no TTF history before Aug 2026")),
    ]
    return body


# ---------------------------------------------------------------------
def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description="Energy futures viewer (Dash)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default=None, help="path to eex_market_data.db")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the browser once it's up")
    args = ap.parse_args()
    DB_PATH = find_db(args.db)
    if DB_PATH is None:
        print(f"Could not find {DB_NAME}. Pass it with --db. Looked in:")
        for c in DB_CANDIDATES:
            print("  ", c)
    else:
        print(f"Database: {DB_PATH}")
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    # use_reloader: a watcher process restarts the app whenever this file
    # (or anything it imports) changes, so an update shows up on the next
    # browser refresh - no closing and restarting by hand. Only the first
    # (watcher) process opens the browser; the restarted child has
    # WERKZEUG_RUN_MAIN set.
    child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not child:
        print(f"Open {url}   (leave this window open; Ctrl+C stops it)")
        print("Updates to eex_dash.py are picked up automatically - just refresh the page.")
        if args.open:
            import threading, webbrowser
            threading.Timer(3.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
