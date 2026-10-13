"""
EEX market data viewer - Dash version
=====================================

A local dashboard over `eex_market_data.db` (built by eex_market_scraper.py).
Replaces the two Streamlit viewers (eex_dashboard.py and
eex_market_dashboard.py) with one Dash app that does what both did.

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
    Area spreads    one delivery period across markets, plus A-B spread
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

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

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


# ---------------------------------------------------------------------
# Colours - same colour-blind-checked palette as the Streamlit viewer.
# Assigned in fixed order and never cycled: max six series per chart.
# ---------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAX_SERIES = 6
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
MATURITY_ORDER = ["Year", "Quarter", "Month", "Week", "Weekend", "Day"]


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
    if maturity == "Month":
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


def load() -> dict:
    path = DB_PATH
    mtime = path.stat().st_mtime
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
                   .assign(o=lambda f: f["group"].map({"Power": 0, "Gas": 1}).fillna(2))
                   .sort_values(["o", "market"]))
        data = {"prices": prices, "settled": settled, "runs": runs,
                "markets": markets[["market", "group"]].values.tolist(),
                "path": path, "mtime": mtime}
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
.ctl label {{ font-size:12px; color:{INK['muted']}; margin-bottom:3px; }}
.card {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
        padding:10px 12px; margin-bottom:12px; }}
.row2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
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
details summary {{ cursor:pointer; color:{INK['muted']}; font-size:13px; margin:6px 0; }}
"""


def ctl(label, component, wide=False):
    return html.Div([html.Label(label), component], className="ctl wide" if wide else "ctl")


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


def spread_tab():
    return html.Div([
        html.Div([
            ctl("Product", dd("s-product")), ctl("Maturity", dd("s-maturity")),
            ctl("Delivery period", dd("s-delivery")),
            ctl(f"Markets (up to {MAX_SERIES})", dd("s-markets", multi=True), wide=True),
        ], className="controls"),
        html.Div([
            ctl("Spread", dd("s-left")), ctl("minus", dd("s-right")),
        ], className="controls"),
        html.Div(id="s-body"),
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
            html.Div([html.Button("Download CSV", id="h-csv-btn", className="btn"),
                      dcc.Download(id="h-csv")], className="ctl"),
        ], className="controls"),
        html.Div(id="h-body"),
    ])


def runs_tab():
    return html.Div([html.Button("Reload", id="r-reload", className="btn",
                                 style={"margin": "12px 0"}),
                     html.Div(id="r-body")])


def make_layout():
    if DB_PATH is None:
        return html.Div([
            html.H1("EEX market data"),
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
        html.H1("EEX market data"),
        html.Div(id="header-line", className="sub"),
        # Tabs only switch which panel is visible. All four panels stay in
        # the page, so every dropdown keeps its selection when you switch
        # tabs and callbacks never point at a component that isn't there.
        # Opens on Price history (Peter, 2026-10-11): DE Base front month.
        dcc.Tabs(id="tabs", value="history", children=[
            dcc.Tab(label=label, value=value, style=TAB_STYLE, selected_style=TAB_SELECTED)
            for value, label in TABS]),
        html.Div(curve_tab(), id="panel-curve", style={"display": "none"}),
        html.Div(spread_tab(), id="panel-spread", style={"display": "none"}),
        html.Div(history_tab(), id="panel-history"),
        html.Div(runs_tab(), id="panel-runs", style={"display": "none"}),
        html.Div("Source: EEX. For information only - systematic republication "
                 "needs EEX's permission.", className="foot"),
    ], className="wrap")


TABS = [("curve", "Forward curve"), ("spread", "Area spreads"),
        ("history", "Price history"), ("runs", "Data & runs")]

TAB_STYLE = {"padding": "7px 12px", "fontSize": 14, "color": INK["muted"],
             "backgroundColor": INK["bg"], "border": "none",
             "borderBottom": f"1px solid {INK['grid']}"}
TAB_SELECTED = {**TAB_STYLE, "color": INK["text"], "fontWeight": 600,
                "borderBottom": f"3px solid {SERIES[0]}"}

app = Dash(__name__, title="EEX market data")
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
    return (f"{d['path'].name} · {len(s):,} settlements · {s['short_code'].nunique()} contracts · "
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
@app.callback(
    Output("s-product", "options"), Output("s-product", "value"),
    Output("s-maturity", "options"), Output("s-maturity", "value"),
    Output("s-delivery", "options"), Output("s-delivery", "value"),
    Output("s-markets", "options"), Output("s-markets", "value"),
    Output("s-left", "options"), Output("s-left", "value"),
    Output("s-right", "options"), Output("s-right", "value"),
    Output("s-body", "children"),
    Input("s-product", "value"), Input("s-maturity", "value"),
    Input("s-delivery", "value"), Input("s-markets", "value"),
    Input("s-left", "value"), Input("s-right", "value"), Input("tabs", "value"),
)
def area_spreads(product, maturity, dkey, picked, left, right, tab):
    if tab != "spread":
        return [no_update] * 13
    d = load()
    s = d["settled"][d["settled"]["group"] == "Power"]
    products = [p for p in ("Base", "Peak") if p in set(s["product"])]
    product = keep(product, products, "Base")
    f = s[s["product"] == product]
    maturities = [m for m in MATURITY_ORDER if m in set(f["maturity_type"])]
    maturity = keep(maturity, maturities, "Year")
    f = f[f["maturity_type"] == maturity]

    dels = f.drop_duplicates("dkey").sort_values("dkey")
    del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]
    # Default: the front delivery that has the most markets quoting it.
    if dkey not in set(dels["dkey"]):
        counts = f.groupby("dkey")["market"].nunique()
        dkey = counts.idxmax() if not counts.empty else None
    f = f[f["dkey"] == dkey]

    available = sorted(f["market"].unique())
    mk_opts = opts(available)
    picked = [m for m in (picked or []) if m in available]
    if not picked and ctx.triggered_id != "s-markets":
        picked = [m for m in ("DE", "FR", "DK1", "SE3", "NO2", "NL") if m in available][:4] \
            or available[:4]
    warn = []
    if len(picked) > MAX_SERIES:
        warn.append(html.Div(f"Showing the first {MAX_SERIES} of {len(picked)} markets — beyond six "
                             "the colours stop being reliably distinguishable.", className="warn"))
        picked = picked[:MAX_SERIES]
    # Only compare like with like: same currency and unit as the first pick.
    if picked:
        unit0 = f[f["market"] == picked[0]]["unit"].iloc[0]
        other = [m for m in picked if f[f["market"] == m]["unit"].iloc[0] != unit0]
        if other:
            warn.append(html.Div(f"Left out {', '.join(other)}: priced in a different "
                                 f"currency/unit than {unit0}. No FX conversion here.",
                                 className="warn"))
            picked = [m for m in picked if m not in other]

    left = keep(left, picked, picked[0] if picked else None)
    right = keep(right, picked, picked[1] if len(picked) > 1 else None)
    lr_opts = opts(picked)
    label = dels.set_index("dkey")["dlabel"].get(dkey, "")
    out = [opts(products), product, opts(maturities), maturity, del_opts, dkey,
           mk_opts, picked, lr_opts, left, lr_opts, right]

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

    body = warn + [html.Div(graph(fig), className="card")]
    wide = g.pivot_table(index="trade_date", columns="market", values="settlement_price").reset_index()
    if left and right and left != right and left in wide and right in wide:
        wide["spread"] = wide[left] - wide[right]
        sp = wide.dropna(subset=["spread"])
        sfig = fig_base(f"{left} − {right}, {product} {label} — {unit}", unit, 260)
        x = pd.to_datetime(sp["trade_date"])
        sfig.add_trace(go.Scatter(x=x, y=sp["spread"], name=f"{left} − {right}",
                                  mode="lines+markers", fill="tozeroy",
                                  fillcolor="rgba(42,120,214,0.15)",
                                  line=dict(color=SERIES[0], width=2), marker=dict(size=6),
                                  hovertemplate="%{y:+.2f}"))
        sfig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        sfig.update_layout(showlegend=False)
        if not sp.empty:
            body.append(html.Div([
                kpi(f"Latest spread · {sp['trade_date'].iloc[-1]}", fmt(sp["spread"].iloc[-1], sign=True), unit),
                kpi("Average", fmt(sp["spread"].mean(), sign=True), unit),
                kpi("Range", f"{fmt(sp['spread'].min(), sign=True)} … {fmt(sp['spread'].max(), sign=True)}", unit),
            ], className="kpis"))
        body.append(html.Div(graph(sfig), className="card"))
    else:
        body.append(note("Pick two different markets to see a spread."))
    body.append(html.Details([html.Summary(f"Table — {label} by market"),
                              table(wide.sort_values("trade_date", ascending=False))]))
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
    try:
        uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            return pd.read_sql_query(
                f"SELECT {', '.join(OHLC_COLS)} FROM eex_market_ohlc "
                "WHERE short_code = ? AND delivery = ? AND trades > 0 AND open IS NOT NULL "
                "ORDER BY trade_date", conn, params=(short_code, str(delivery)))
    except Exception:
        return pd.DataFrame(columns=OHLC_COLS)


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
    Input("h-style", "value"),
)
def price_history(market, product, maturity, dkey, start, end, tab, style):
    if tab != "history":
        return [no_update] * 13
    d = load()
    s = d["settled"]
    m_opts = market_options(d)
    market = keep(market, [o["value"] for o in m_opts], "DE")
    products, product, maturities, maturity, f = chain(s, market, product, maturity,
                                                       default_maturity="Month")
    dels = f.drop_duplicates("dkey").sort_values("dkey")
    if dkey not in set(dels["dkey"]):
        # Default: the front contract (nearest delivery still trading) -
        # liquid, and the one with candles. Kept until the user picks
        # another delivery or changes market/product/maturity.
        dkey = front_delivery(f)
    del_opts = [{"label": l, "value": k} for k, l in zip(dels["dkey"], dels["dlabel"])]

    allrows = history_frame(market, product, maturity, dkey)
    full = allrows.dropna(subset=["settlement_price"])
    candles = style == "Candles"
    ohlc_all = (load_ohlc(full["short_code"].iloc[0], full["delivery"].iloc[0])
                if not full.empty else pd.DataFrame(columns=OHLC_COLS))
    # The trade feed goes back years (nightly backfill); settlements only
    # ~6 weeks. In candle mode the date picker spans both.
    dates = list(full["trade_date"]) + (list(ohlc_all["trade_date"]) if candles else [])
    lo, hi = (min(dates), max(dates)) if dates else (None, None)
    if ctx.triggered_id != "h-dates" or not start or not end:
        start, end = lo, hi
    out = [m_opts, market, opts(products), product, opts(maturities), maturity,
           del_opts, dkey, lo, hi, start, end]
    if full.empty:
        return out + [graph(empty_fig("Nothing stored for that delivery."))]

    h = in_range(full, start, end)
    o = in_range(ohlc_all, start, end)
    if h.empty and (not candles or o.empty):
        return out + [note("Nothing stored in the chosen trade-date range.")]

    ref = full.iloc[-1]
    code, unit, uom, label = ref["short_code"], ref["unit"], ref["uom"], ref["dlabel"]

    tiles = []
    if not h.empty:
        last = h.iloc[-1]
        before = full[full["trade_date"] < last["trade_date"]]
        chg = last["settlement_price"] - before["settlement_price"].iloc[-1] if not before.empty else None
        move = last["settlement_price"] - h["settlement_price"].iloc[0]
        tiles += [
            kpi(f"Settlement · {last['trade_date']}", fmt(last["settlement_price"]),
                f"{fmt(chg, sign=True)} day-on-day" if chg is not None else unit, chg),
            kpi("Move over range", fmt(move, sign=True), unit, move),
            kpi("High / low settle", f"{fmt(h['settlement_price'].max())} / "
                f"{fmt(h['settlement_price'].min())}", unit),
            kpi(f"Volume · {last['trade_date']}", fmt(last["volume"], 0), uom),
            kpi("Open interest", fmt(last["gross_open_interest"], 0), "lots (gross)"),
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
    notes = [note(f"{code} — {ref['long_name'] or ''}.  Settlements: EEX only serves ~6 weeks, "
                  "so they go as deep as your scraper's run history.")]

    if candles and not o.empty:
        pf = fig_base(f"{code} {label} — daily open/high/low/last, settlement dotted — {unit}",
                      unit, 380)
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
                          "every trade on EEX (eex_ohlc_scraper.py). Days without a trade have "
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

    # Volume and open interest on their own charts - never a twin axis. In
    # candle mode volume comes from the trade feed (same total as EEX's
    # traded volume) so it covers the same days as the candles.
    vf = fig_base(f"Traded volume — {uom}", uom, 230)
    if candles and not o.empty:
        # EEX's reported volume where we have it, the trade-feed total for
        # older days (they match - checked on DEBY Cal-27, 2026-10-09).
        vol = (o.set_index("trade_date")["tick_volume"]
               .combine_first(h.set_index("trade_date")["volume"]))
        vol.update(h.set_index("trade_date")["volume"].dropna())
        vf.add_trace(go.Bar(x=pd.to_datetime(vol.index), y=vol.values, marker_color=SERIES[1],
                            name="Volume", hovertemplate="%{y:,.0f}"))
    else:
        vf.add_trace(go.Bar(x=x, y=h["volume"], marker_color=SERIES[1], name="Volume",
                            hovertemplate="%{y:,.0f}"))
    vf.update_layout(showlegend=False)
    of = fig_base("Gross open interest — lots", "lots", 230)
    of.add_trace(go.Scatter(x=x, y=h["gross_open_interest"], mode="lines+markers",
                            line=dict(color=SERIES[2], width=2), marker=dict(size=6),
                            name="Open interest", hovertemplate="%{y:,.0f}"))
    of.update_layout(showlegend=False)

    # Same x window on all three charts, weekends squeezed out.
    shown = list(x) + (list(xo) if candles else [])
    span = [min(shown) - pd.Timedelta(hours=18), max(shown) + pd.Timedelta(hours=18)]
    for fig in (pf, vf, of):
        fig.update_xaxes(range=span, rangebreaks=[dict(bounds=["sat", "mon"])])

    cols = ["trade_date", "short_code", "dlabel", "settlement_price", "unit", "volume",
            "gross_open_interest", "fetched_at"]
    tbl = h[cols].rename(columns={"dlabel": "delivery"})
    if not o.empty:
        tbl = with_ohlc(tbl, o)
    body = notes + [
        kpis,
        html.Div(graph(pf), className="card"),
        html.Div([html.Div(graph(vf), className="card"), html.Div(graph(of), className="card")],
                 className="row2"),
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
        return no_update
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
        note(f"Database: {d['path']}"),
    ]
    return body


# ---------------------------------------------------------------------
def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description="EEX market data viewer (Dash)")
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
