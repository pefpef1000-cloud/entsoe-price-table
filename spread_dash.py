"""Spread analyzer (Dash) - Peter, 2026-10-21.

Four spreads, one tab each, each with its own leg pickers:
  Location  - area A minus area B, same delivery    (default DE - FR, front month)
  Time      - one area, delivery A minus delivery B  (default DE front month - month 2)
  Spark     - clean spark spread                     (default DE, front month)
  Dark      - clean dark spread                      (default DE, front month)
Each card shows KPI tiles, the spread over trade dates and a table.
An overview table at the top lists all four.

Data and formulas come from eex_dash.py (imported, one source of truth):
  power, TTF, EUA  - eex_market_data.db + eex_gas_eua.db
  API2 + EUR/USD   - ..\\ice_scraper\\ice_market_data.db
  CSS = power - TTF / 50 % - EUA x 0.368
  CDS = power - (API2 USD/t / EURUSD / 6.978) / 45 % - EUA x 0.758

    python spread_dash.py            -> http://127.0.0.1:8062/
    flags: --port --host --db --open --debug
"""
from __future__ import annotations

import argparse
import os
import threading
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

import eex_dash as E   # loaders, front month rule, spark/dark formulas, styles
import futures_vs_spot_dash as F   # day-ahead spot loader (entsoe_data.db)
import spread_delivered as D       # how each spread delivered in the past

PORT = 8062
TENORS = ["Month", "Quarter", "Year"]
TENOR_SHORT = {"Month": "M", "Quarter": "Q", "Year": "Cal"}
PRODUCTS = ["Base", "Peak"]
INK = E.INK
BLUE, ORANGE, GREEN = E.SERIES[0], E.SERIES[1], E.SERIES[2]

GAS_EFF = E.ICE_EFFICIENCY            # 50 %
GAS_CF = E.carbon_factor(GAS_EFF)     # 0.368
COAL_EFF = E.COAL_EFFICIENCY          # 45 %
COAL_CF = E.coal_factor(COAL_EFF)     # 0.7578
COAL_HUB = "API2"

CARDS = [("loc", "Location spread"), ("time", "Time spread"),
         ("spark", "Clean spark spread"), ("dark", "Clean dark spread")]

# Live (delayed) prices on a button, same as Energy futures (2026-10-21).
# Fetching, storage and the job thread come from eex_dash / live_prices.
LIVE = E.LIVE_ENABLED
LIVE_COLOR = E.LIVE_COLOR


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def power() -> pd.DataFrame:
    """Power settlements, Month/Quarter/Year only."""
    s = E.load()["settled"]
    return s[(s["group"] == "Power") & s["maturity_type"].isin(TENORS)]


_eua_cache: dict = {}
_eua_lock = threading.Lock()


def eua() -> pd.DataFrame:
    """EUA per trade date (eex_dash.eua_prices), cached on the DB times."""
    key = E.load()["mtime"]
    with _eua_lock:
        if _eua_cache.get("key") != key:
            _eua_cache["key"], _eua_cache["data"] = key, E.eua_prices()
        return _eua_cache["data"]


def dval(maturity: str, dkey: str) -> str:
    return f"{maturity}|{dkey}"


def dsplit(v: str | None):
    if not v or "|" not in v:
        return None, None
    m, k = v.split("|", 1)
    return m, k


def delivery_options(f: pd.DataFrame) -> list[dict]:
    """Deliveries quoted on the latest trade date: months, then quarters,
    then years. Value 'Month|2026-11-01' (Cal-27 and Jan-27 share a dkey)."""
    if f.empty:
        return []
    last = f["trade_date"].max()
    live = f[(f["trade_date"] == last) & (f["dkey"] > last)]
    out = []
    for m in TENORS:
        g = live[live["maturity_type"] == m].drop_duplicates("dkey").sort_values("dkey")
        out += [{"label": l, "value": dval(m, k)} for k, l in zip(g["dkey"], g["dlabel"])]
    return out


def front(f: pd.DataFrame, n: int = 0) -> str | None:
    """n-th month from the front month (0 = front, 1 = month 2)."""
    mo = f[f["maturity_type"] == "Month"]
    k = E.front_delivery(mo)
    if k is None:
        return None
    last = mo["trade_date"].max()
    ks = sorted(mo[(mo["trade_date"] == last) & (mo["dkey"] >= k)]["dkey"].unique())
    return dval("Month", ks[n]) if n < len(ks) else None


def label_of(f: pd.DataFrame, v: str) -> str:
    m, k = dsplit(v)
    g = f[(f["maturity_type"] == m) & (f["dkey"] == k)]
    return g["dlabel"].iloc[0] if not g.empty else str(v)


def leg(f: pd.DataFrame, v: str) -> pd.DataFrame:
    """Rows of one delivery (all trade dates)."""
    m, k = dsplit(v)
    return f[(f["maturity_type"] == m) & (f["dkey"] == k)]


def opts(values):
    return [{"label": v, "value": v} for v in values]


# ---------------------------------------------------------------------
# Output pieces
# ---------------------------------------------------------------------
def fmt(v, dec=2, sign=False):
    return E.fmt(v, dec, sign)


def stats(sp: pd.Series) -> dict:
    """Latest, 1 d / 1 w change, min / max / average over the history."""
    sp = sp.dropna()
    if sp.empty:
        return {}
    last = sp.iloc[-1]
    return {"date": sp.index[-1], "last": last,
            "d1": last - sp.iloc[-2] if len(sp) > 1 else None,
            "w1": last - sp.iloc[-6] if len(sp) > 5 else None,
            "min": sp.min(), "max": sp.max(), "avg": sp.mean(), "n": len(sp)}


def tiles(st: dict, extra: list[tuple[str, str]] = (), live: dict | None = None):
    def col(v):
        if v is None or pd.isna(v):
            return INK["muted"]
        return INK["up"] if v > 0 else INK["down"] if v < 0 else INK["text"]
    items = []
    if live:
        v = live.get("value")
        items.append((f"Live · {live['time']}",
                      (fmt(v) + f' <span style="font-size:11px;color:{col(live.get("change"))}">'
                       f'{fmt(live.get("change"), sign=True)}</span>') if v is not None else "—",
                      LIVE_COLOR))
    items += [("Spread · settle" if live else "Spread", fmt(st.get("last")), col(st.get("last"))),
             ("1 day", fmt(st.get("d1"), sign=True), col(st.get("d1"))),
             ("1 week", fmt(st.get("w1"), sign=True), col(st.get("w1"))),
             ("Average", fmt(st.get("avg")), INK["text"]),
             ("Min", fmt(st.get("min")), INK["text"]),
             ("Max", fmt(st.get("max")), INK["text"])]
    items += [(a, b, INK["text"]) for a, b in extra]
    html_ = "".join(
        f'<div class="t"><div class="tl">{a}</div><div class="tv" style="color:{c}">{b}</div></div>'
        for a, b, c in items)
    return dcc.Markdown(f'<div class="tiles">{html_}</div>', dangerously_allow_html=True)


def chart(sp: pd.Series, title: str, name: str, colour=BLUE, live=None):
    fig = E.fig_base(title, "EUR/MWh", height=300)
    x = pd.to_datetime(sp.index)
    fig.add_trace(go.Scatter(x=x, y=sp.values, name=name, mode="lines+markers",
                             line=dict(color=colour, width=2), marker=dict(size=4),
                             hovertemplate="%{y:.2f}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    avg = sp.mean()
    fig.add_hline(y=avg, line=dict(color=INK["muted"], width=1, dash="dot"),
                  annotation_text=f"avg {avg:.2f}", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK["muted"]))
    fig.update_layout(showlegend=False, margin=dict(l=56, r=70, t=40, b=30))
    E.end_labels(fig, [x[-1]], [sp.iloc[-1]], [f"{sp.iloc[-1]:.2f}"], [colour])
    if live and live.get("value") is not None:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(date.today())], y=[live["value"]], name="Live",
                                 mode="markers", marker=dict(symbol="diamond", size=11, color=LIVE_COLOR,
                                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"live {live['time']}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return E.graph(fig)


def table(df: pd.DataFrame, sort=True):
    """Newest first, 2 decimals, spread column coloured by sign."""
    df = (df.sort_values("Date", ascending=False) if sort else df).copy()
    num = [c for c in df.columns if c not in ("Date", "Period") and pd.api.types.is_numeric_dtype(df[c])]
    for c in num:
        df[c] = df[c].astype(float).round(4 if c == "EUR/USD" else 2)
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c, "type": "numeric" if c in num else "text",
                  "format": {"specifier": ",.4f" if c == "EUR/USD" else ",.2f"} if c in num else None}
                 for c in df.columns],
        page_size=10, sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Segoe UI, Inter, Arial, sans-serif", "fontSize": 13,
                    "padding": "4px 8px", "textAlign": "right",
                    "border": f"1px solid {INK['grid']}"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                for c in ("Date", "Period", "Futures day", "Gas source", "Coal source")],
        style_data_conditional=[
            {"if": {"column_id": "Spread", "filter_query": "{Spread} > 0"}, "color": INK["up"], "fontWeight": 600},
            {"if": {"column_id": "Spread", "filter_query": "{Spread} < 0"}, "color": INK["down"], "fontWeight": 600},
            # today's live row on top ("2026-10-21 live 09:08")
            {"if": {"filter_query": '{Date} contains "live"'}, "color": LIVE_COLOR, "fontWeight": 600},
        ],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"]},
    )


def body(sp: pd.Series, tbl: pd.DataFrame, title: str, name: str, extra=(), notes=(), live=None):
    st = stats(sp)
    if live:
        notes = [f"Live = today's last trade (EEX trade feed / ICE bars, ~15 min delayed), "
                 f"fetched {live['time']}; {live.get('sub', '')}. Settlements unchanged."] + list(notes)
    if live and live.get("row"):
        tbl = pd.concat([pd.DataFrame([live["row"]]), tbl], ignore_index=True)
    return [tiles(st, extra, live),
            html.Div([html.Div(chart(sp, title, name, live=live), className="chart"),
                      html.Div(table(tbl), className="tbl")], className="split"),
            *[html.Div(n, className="note") for n in notes if n]], st


def same_period(df: pd.DataFrame, tenor: str, pick: str) -> pd.DataFrame:
    """Only the picked legs' own period in earlier years: Nov-26 -> every
    past November, Q1-27 -> every past Q1, Cal-27 -> every past Cal
    (Peter 2026-10-21: "it shall only show for legs picked")."""
    if df is None or df.empty or tenor == "Year":
        return df
    mon = date.fromisoformat(pick).month
    return df[[d.month == mon for d in df["start"]]]


def period_name(tenor: str, pick: str) -> str:
    d = date.fromisoformat(pick)
    return {"Month": E.MON[d.month - 1], "Quarter": f"Q{(d.month - 1) // 3 + 1}", "Year": "Cal"}[tenor]


def delivered_block(df: pd.DataFrame, what: str, tenor: str, empty_msg: str, notes=(), pick=None,
                    when=None):
    """'How it delivered': tiles, bar chart (delivered) + futures markers, table.
    With `pick` (leg 1 dkey) only the same period of earlier years is shown."""
    if pick:
        df = same_period(df, tenor, pick)
    when = when or (period_name(tenor, pick) if pick else tenor.lower() + "ly")
    head = html.H3(f"How it delivered · {what} · {when} in past years", className="dh")
    if df is None or df.empty:
        return [head, html.Div(empty_msg, className="warn")]
    d = df.sort_values("start")
    dl = d["Delivered"]
    fut = d[d["Futures"].notna()]
    diff = fut["Diff"]
    pos = (dl > 0).mean() * 100
    items = [("Periods", f"{len(d)}", INK["text"]),
             ("From", d["Period"].iloc[0].split(" - ")[0], INK["text"]),
             ("Average", fmt(dl.mean()), INK["up"] if dl.mean() > 0 else INK["down"]),
             ("Last", fmt(dl.iloc[-1]), INK["up"] if dl.iloc[-1] > 0 else INK["down"]),
             ("Min", fmt(dl.min()), INK["text"]), ("Max", fmt(dl.max()), INK["text"]),
             ("Positive", f"{pos:.0f} %", INK["text"]),
             ("vs futures", fmt(diff.mean(), sign=True) + f" ({len(diff)})" if len(diff) else "—",
              INK["muted"] if not len(diff) else INK["up"] if diff.mean() > 0 else INK["down"])]
    tl = "".join(f'<div class="t"><div class="tl">{a}</div><div class="tv" style="color:{c}">{b}</div></div>'
                 for a, b, c in items)
    fig = E.fig_base(f"{what}: delivered vs futures before delivery (EUR/MWh)", "EUR/MWh", height=320, dates=False)
    fig.add_trace(go.Bar(x=d["Period"], y=dl, name="Delivered",
                         marker_color=[INK["up"] if v > 0 else INK["down"] for v in dl],
                         hovertemplate="%{x}: %{y:.2f}<extra>Delivered</extra>"))
    if not fut.empty:
        fig.add_trace(go.Scatter(x=fut["Period"], y=fut["Futures"], name="Futures before delivery",
                                 mode="markers", marker=dict(symbol="diamond", size=10, color=ORANGE,
                                                             line=dict(color="white", width=1)),
                                 hovertemplate="%{x}: %{y:.2f}<extra>Futures</extra>"))
    fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    fig.update_xaxes(type="category", tickangle=-45, nticks=24)
    fig.update_layout(hovermode="closest", margin=dict(l=56, r=20, t=56, b=60), bargap=0.15,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"))
    tbl = d.sort_values("start", ascending=False).drop(columns=["start"])
    tbl = tbl.rename(columns={"Delivered": "Spread", "Diff": "Spread − futures"})
    tail = ["Futures", "Futures day", "Spread − futures"] if fut.shape[0] else []
    tbl = tbl[[c for c in tbl.columns if c not in ("Futures", "Futures day", "Spread − futures")] + tail]
    if not fut.shape[0]:
        notes = list(notes) + ["No futures price before delivery for these years (EEX settlements from Aug 2026, trades from Nov 2025)."]
    return [head,
            dcc.Markdown(f'<div class="tiles">{tl}</div>', dangerously_allow_html=True),
            html.Div([html.Div(E.graph(fig), className="chart"),
                      html.Div(table(tbl, sort=False), className="tbl")], className="split"),
            *[html.Div(n, className="note") for n in notes if n]]


def nodata(msg):
    return [html.Div(msg, className="warn")], {}


def summary(name, legs, st, unit="EUR/MWh"):
    return {"Spread": name, "Legs": legs, "Last": st.get("last"), "1 day": st.get("d1"),
            "1 week": st.get("w1"), "Average": st.get("avg"), "Min": st.get("min"),
            "Max": st.get("max"), "Date": st.get("date", "")}


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
CSS = E.CSS + f"""
.wrap {{ max-width:1600px; }}
.tabs-row {{ margin:10px 0 12px; }}
.scard {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px; padding:12px 14px; min-width:0; }}
.scard h2 {{ font-size:16px; margin:0 0 2px; }}
.scard .ctls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-end; margin:8px 0; }}
.scard .ctl {{ min-width:120px; flex:1; }}
.scard .ctl.sm {{ min-width:96px; flex:0 0 96px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(92px, 1fr)); gap:6px; margin:6px 0 4px; }}
.tiles .t {{ border:1px solid {INK['grid']}; border-radius:6px; padding:5px 8px; }}
.tiles .tl {{ font-size:11px; color:{INK['muted']}; }}
.tiles .tv {{ font-size:17px; font-weight:600; }}
.split {{ display:grid; grid-template-columns:minmax(0,1.3fr) minmax(0,1fr); gap:12px; align-items:start; }}
.split > div, #overview, .panel {{ min-width:0; }}
.dash-spreadsheet-container {{ max-width:100%; }}
.scard .ctl {{ position:relative; overflow:hidden; }}
.scard .ctl.lv {{ flex:0 1 auto; min-width:0; }}
.lvmsg {{ font-size:12px; color:{INK['muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.dh {{ font-size:15px; margin:18px 0 2px; padding-top:12px; border-top:1px solid {INK['grid']}; }}
.note {{ font-size:12px; color:{INK['muted']}; margin-top:4px; }}
.warn {{ background:#fff4d6; border:1px solid #eda100; border-radius:6px; padding:8px 10px; font-size:13px; margin:8px 0; }}
@media (max-width:1000px) {{ .split {{ grid-template-columns:1fr; }} }}
@media (max-width:720px) {{ .wrap {{ padding:12px 16px 32px; }} }}
"""


def ctl(label, id_, small=False):
    return html.Div([html.Label(label), dcc.Dropdown(id=id_, clearable=False)],
                    className="ctl sm" if small else "ctl")


def live_ctl(key):
    if not LIVE:
        return []
    return [html.Div([html.Label("Today (delayed)"),
                      html.Div([html.Button("Update prices", id=f"{key}-lv", className="btn",
                                            style={"height": 36, "whiteSpace": "nowrap"}),
                                html.Span(id=f"{key}-lvmsg", className="lvmsg")],
                               style={"display": "flex", "alignItems": "center", "gap": 8})],
                     className="ctl lv")]


def card(key, title, controls, sub):
    return html.Div(html.Div([html.H2(title), html.Div(sub, className="sub"),
                              html.Div(controls + live_ctl(key), className="ctls"),
                              dcc.Loading(html.Div(id=f"{key}-out"), type="dot", color=BLUE),
                              dcc.Store(id=f"{key}-sum")], className="scard"),
                    id=f"panel-{key}", className="panel")


TAB_STYLE = {"padding": "7px 14px", "fontSize": 14, "color": INK["muted"],
             "border": "none", "borderBottom": f"2px solid {INK['grid']}", "background": "none"}
TAB_SELECTED = {**TAB_STYLE, "color": INK["text"], "fontWeight": 600,
                "borderBottom": f"2px solid {BLUE}"}
TAB_LABEL = {"loc": "Location", "time": "Time", "spark": "Spark", "dark": "Dark"}


def make_layout():
    return html.Div([
        html.H1("Spread analyzer"),
        html.Div(id="hdr", className="sub"),
        html.Div(id="overview", className="card"),
        dcc.Store(id="live-ver", data=0),
        dcc.Interval(id="live-poll", interval=1000, disabled=True),
        dcc.Tabs(id="tabs", value="loc", className="tabs-row", persistence=True,
                 persistence_type="local",
                 children=[dcc.Tab(label=TAB_LABEL[k], value=k, style=TAB_STYLE,
                                   selected_style=TAB_SELECTED) for k, _ in CARDS]),
        html.Div([
            card("loc", "Location spread", [
                ctl("Leg 1 area", "loc-a"), ctl("Leg 2 area", "loc-b"),
                ctl("Delivery", "loc-d"), ctl("Load", "loc-p", small=True)],
                "Leg 1 minus leg 2, same delivery."),
            card("time", "Time spread", [
                ctl("Area", "time-area"), ctl("Leg 1 delivery", "time-a"),
                ctl("Leg 2 delivery", "time-b"), ctl("Load", "time-p", small=True)],
                "Leg 1 minus leg 2, same area."),
            card("spark", "Clean spark spread", [
                ctl("Area", "spark-area"), ctl("Delivery", "spark-d"),
                ctl("Load", "spark-p", small=True)],
                f"Power − TTF ÷ {GAS_EFF:.0f} % − EUA × {GAS_CF}."),
            card("dark", "Clean dark spread", [
                ctl("Area", "dark-area"), ctl("Delivery", "dark-d"),
                ctl("Load", "dark-p", small=True)],
                f"Power − {COAL_HUB} (USD/t ÷ EUR/USD ÷ {E.COAL_MWH_PER_T}) ÷ {COAL_EFF:.0f} % "
                f"− EUA × {COAL_CF:.3f}."),
        ]),
    ], className="wrap")


app = Dash(__name__, title="Spread analyzer")
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.layout = make_layout


@app.callback([Output(f"panel-{k}", "style") for k, _ in CARDS], Input("tabs", "value"))
def switch_tab(tab):
    return [{"display": "block" if k == tab else "none"} for k, _ in CARDS]


def shown(out, content, summ, key, tab):
    """Content only for the visible tab (Plotly needs a visible box); the
    overview row is always updated."""
    return out + [content if tab == key else no_update, summ]


@app.callback(Output("hdr", "children"), Input("loc-sum", "data"))
def header(_):
    s = E.load()["settled"]
    return (f"EEX settlements up to {s['trade_date'].max()} · TTF from EEX · EUA from EEX (ICE as backup) · "
            f"{COAL_HUB} + EUR/USD from ICE / ECB · all EUR/MWh")


# ---------------------------------------------------------------------
# Location: area A - area B, same delivery
# ---------------------------------------------------------------------
@app.callback(
    Output("loc-a", "options"), Output("loc-a", "value"),
    Output("loc-b", "options"), Output("loc-b", "value"),
    Output("loc-d", "options"), Output("loc-d", "value"),
    Output("loc-p", "options"), Output("loc-p", "value"),
    Output("loc-out", "children"), Output("loc-sum", "data"),
    Input("loc-a", "value"), Input("loc-b", "value"),
    Input("loc-d", "value"), Input("loc-p", "value"), Input("tabs", "value"), Input("live-ver", "data"))
def location(a, b, dv, prod, tab=None, _live=None):
    pw = power()
    pw = pw[pw["unit"] == "EUR/MWh"]          # no FX mixing (GB ICE is GBP)
    areas = sorted(pw["market"].unique(), key=lambda m: (m not in E.MARKET_PREF,
                   E.MARKET_PREF.index(m) if m in E.MARKET_PREF else 0, m))
    a = E.keep(a, areas, "DE")
    b = E.keep(b, areas, "FR")
    prod = E.keep(prod, PRODUCTS, "Base")
    fa = pw[(pw["market"] == a) & (pw["product"] == prod)]
    fb = pw[(pw["market"] == b) & (pw["product"] == prod)]
    oa = delivery_options(fa)
    ok_b = {o["value"] for o in delivery_options(fb)}
    dopts = [o for o in oa if o["value"] in ok_b]
    vals = [o["value"] for o in dopts]
    if dv not in vals:
        dv = front(fa) if front(fa) in vals else (vals[0] if vals else None)
    out = [opts(areas), a, opts(areas), b, dopts, dv, opts(PRODUCTS), prod]
    if a == b:
        return shown(out, *nodata("Pick two different areas."), "loc", tab)
    if not dv:
        return shown(out, *nodata(f"No delivery that both {a} and {b} {prod} quote."), "loc", tab)
    la, lb = E.leg_series(leg(fa, dv)), E.leg_series(leg(fb, dv))
    w = pd.concat([la.rename(a), lb.rename(b)], axis=1, join="inner").dropna()
    if w.empty:
        return shown(out, *nodata("No trade date with both legs settled."), "loc", tab)
    w["Spread"] = w[a] - w[b]
    lab = label_of(fa, dv)
    tbl = w.reset_index().rename(columns={"trade_date": "Date"})
    name = f"{a}-{b} {prod} {lab}"
    live = live_two_legs(leg(fa, dv), leg(fb, dv), w[[a, b]], a, b)
    if live and live.get("value") is not None:
        live["row"] = {"Date": E.live_label(live["time"]), a: live["L"], b: live["R"], "Spread": live["value"]}
    content, st = body(w["Spread"], tbl, f"{name} (EUR/MWh)", name,
                       extra=[(f"{a}", fmt(w[a].iloc[-1])), (f"{b}", fmt(w[b].iloc[-1]))], live=live)
    if tab == "loc":
        m = dsplit(dv)[0]
        content += delivered_block(
            D.location(a, b, m, prod), f"{a}-{b} {prod}", m,
            f"No earlier {period_name(m, dsplit(dv)[1])} with day-ahead prices for both {a} and {b}.",
            notes=["Delivered = mean day-ahead price over the delivery hours (ENTSO-E). Futures = EEX settlement on the last day before delivery (from Aug 2026), else last traded price (from Nov 2025)."],
            pick=dsplit(dv)[1])
    return shown(out, content, summary("Location", name, st), "loc", tab)


# ---------------------------------------------------------------------
# Time: one area, delivery A - delivery B
# ---------------------------------------------------------------------
@app.callback(
    Output("time-area", "options"), Output("time-area", "value"),
    Output("time-a", "options"), Output("time-a", "value"),
    Output("time-b", "options"), Output("time-b", "value"),
    Output("time-p", "options"), Output("time-p", "value"),
    Output("time-out", "children"), Output("time-sum", "data"),
    Input("time-area", "value"), Input("time-a", "value"),
    Input("time-b", "value"), Input("time-p", "value"), Input("tabs", "value"), Input("live-ver", "data"))
def time_spread(area, da, db, prod, tab=None, _live=None):
    pw = power()
    areas = sorted(pw["market"].unique(), key=lambda m: (m not in E.MARKET_PREF,
                   E.MARKET_PREF.index(m) if m in E.MARKET_PREF else 0, m))
    area = E.keep(area, areas, "DE")
    f0 = pw[pw["market"] == area]
    prods = [p for p in PRODUCTS if p in set(f0["product"])]
    prod = E.keep(prod, prods, "Base")
    f = f0[f0["product"] == prod]
    dopts = delivery_options(f)
    vals = [o["value"] for o in dopts]
    if da not in vals:
        da = front(f, 0) if front(f, 0) in vals else (vals[0] if vals else None)
    if db not in vals:
        db = front(f, 1) if front(f, 1) in vals else (vals[1] if len(vals) > 1 else None)
    unit = f["unit"].iloc[0] if not f.empty else "EUR/MWh"
    out = [opts(areas), area, dopts, da, dopts, db, opts(prods), prod]
    if not da or not db:
        return shown(out, *nodata(f"{area} {prod}: fewer than two deliveries quoted."), "time", tab)
    if da == db:
        return shown(out, *nodata("Pick two different deliveries."), "time", tab)
    a_l, b_l = label_of(f, da), label_of(f, db)
    w = pd.concat([E.leg_series(leg(f, da)).rename(a_l), E.leg_series(leg(f, db)).rename(b_l)],
                  axis=1, join="inner").dropna()
    if w.empty:
        return shown(out, *nodata("No trade date with both legs settled."), "time", tab)
    w["Spread"] = w[a_l] - w[b_l]
    tbl = w.reset_index().rename(columns={"trade_date": "Date"})
    name = f"{area} {prod} {a_l} - {b_l}"
    live = live_two_legs(leg(f, da), leg(f, db), w[[a_l, b_l]], a_l, b_l)
    if live and live.get("value") is not None:
        live["row"] = {"Date": E.live_label(live["time"]), a_l: live["L"], b_l: live["R"], "Spread": live["value"]}
    content, st = body(w["Spread"], tbl, f"{name} ({unit})", name,
                       extra=[(a_l, fmt(w[a_l].iloc[-1])), (b_l, fmt(w[b_l].iloc[-1]))],
                       notes=[f"Prices in {unit}." if unit != "EUR/MWh" else None], live=live)
    if tab == "time":
        (m1, k1), (m2, k2) = dsplit(da), dsplit(db)
        content += delivered_block(
            D.time_spread(area, m1, k1, m2, k2, prod), f"{area} {prod}", m1,
            f"No earlier {period_name(m1, k1)} - {period_name(m2, k2)} pair with day-ahead prices for {area}.",
            pick=k1, when=f"{period_name(m1, k1)} - {period_name(m2, k2)}",
            notes=["Each row: the picked leg 1 period of an earlier year minus the leg 2 period the same distance later.",
                   "Delivered = mean day-ahead price over the delivery hours (ENTSO-E). Futures = EEX settlement on the last day before delivery (from Aug 2026), else last traded price (from Nov 2025)."])
    return shown(out, content, summary("Time", name, st), "time", tab)


# ---------------------------------------------------------------------
# Spark and dark: area + delivery
# ---------------------------------------------------------------------
def fuel_controls(area, dv, prod):
    pw = power()
    pw = pw[pw["unit"] == "EUR/MWh"]
    areas = sorted(pw["market"].unique(), key=lambda m: (m not in E.MARKET_PREF,
                   E.MARKET_PREF.index(m) if m in E.MARKET_PREF else 0, m))
    area = E.keep(area, areas, "DE")
    f0 = pw[pw["market"] == area]
    prods = [p for p in PRODUCTS if p in set(f0["product"])]
    prod = E.keep(prod, prods, "Base")
    f = f0[f0["product"] == prod]
    dopts = delivery_options(f)
    vals = [o["value"] for o in dopts]
    if dv not in vals:
        dv = front(f) if front(f) in vals else (vals[0] if vals else None)
    return f, areas, area, prods, prod, dopts, dv


def power_frame(f, dv):
    p = leg(f, dv)
    return p.groupby(["trade_date", "dkey", "ddate"], as_index=False)["settlement_price"].mean()


@app.callback(
    Output("spark-area", "options"), Output("spark-area", "value"),
    Output("spark-d", "options"), Output("spark-d", "value"),
    Output("spark-p", "options"), Output("spark-p", "value"),
    Output("spark-out", "children"), Output("spark-sum", "data"),
    Input("spark-area", "value"), Input("spark-d", "value"), Input("spark-p", "value"), Input("tabs", "value"),
    Input("live-ver", "data"))
def spark(area, dv, prod, tab=None, _live=None):
    f, areas, area, prods, prod, dopts, dv = fuel_controls(area, dv, prod)
    out = [opts(areas), area, dopts, dv, opts(prods), prod]
    if not dv:
        return shown(out, *nodata(f"No {area} {prod} delivery quoted."), "spark", tab)
    m, k = dsplit(dv)
    s = E.load()["settled"]
    gas = s[(s["market"] == "Gas TTF") & (s["product"] == "Base") &
            (s["maturity_type"] == m) & (s["dkey"] == k)]
    if gas.empty:
        return shown(out, *nodata(f"No TTF {label_of(f, dv)} settlement - no spark spread."), "spark", tab)
    e = eua()
    if e.empty:
        return shown(out, *nodata("No EUA prices (eex_gas_eua.db)."), "spark", tab)
    x = E.spark_frame(power_frame(f, dv), gas, e, GAS_EFF, GAS_CF).dropna(subset=["clean_spark"])
    if x.empty:
        return shown(out, *nodata("No trade date with power, TTF and EUA."), "spark", tab)
    x = x.set_index("trade_date")
    lab = label_of(f, dv)
    tbl = pd.DataFrame({"Date": x.index, "Power": x["power"], "TTF": x["gas"],
                        "EUA": x["eua"].astype(float), "Gas cost": x["fuel_cost"],
                        "CO2 cost": x["carbon_cost"], "Prod. cost": x["fuel_cost"] + x["carbon_cost"],
                        "Spread": x["clean_spark"]})
    last = x.iloc[-1]
    name = f"{area} {prod} {lab} CSS"
    notes = []
    if any(str(v).startswith("carried") for v in x["eua_src"]):
        notes.append("Some days use EUA carried from an earlier day.")
    if (x["eua_src"] == "ICE settlement").any():
        notes.append("EUA from ICE on days EEX has not published yet.")
    if (x["eua_dec"].astype(float) != x["ddate"].map(lambda d: d.year)).any():
        notes.append(f"EUA: Dec-{int(last['eua_dec']) % 100:02d} used.")
    live = live_fuel("spark", area, prod, m, k, last, x["clean_spark"].iloc[-1])
    content, st = body(x["clean_spark"], tbl, f"{name} (EUR/MWh)", name,
                       extra=[("Power", fmt(last["power"])), ("TTF", fmt(last["gas"])),
                              ("EUA", fmt(last["eua"]))], notes=notes, live=live)
    if tab == "spark":
        content += delivered_block(
            D.spark(area, m, prod, GAS_EFF, GAS_CF), f"{area} {prod} CSS", m,
            f"No earlier {period_name(m, k)} with day-ahead, TTF and EUA prices for {area}.",
            notes=["Delivered CSS = day-ahead power − TTF ÷ 50 % − EUA × 0.368, TTF (front month, Yahoo TTF=F) and EUA "
                   "averaged over the delivery days. History starts where EUA history starts (grows with the EEX EUA backfill)."], pick=k)
    return shown(out, content, summary("Spark", name, st), "spark", tab)


@app.callback(
    Output("dark-area", "options"), Output("dark-area", "value"),
    Output("dark-d", "options"), Output("dark-d", "value"),
    Output("dark-p", "options"), Output("dark-p", "value"),
    Output("dark-out", "children"), Output("dark-sum", "data"),
    Input("dark-area", "value"), Input("dark-d", "value"), Input("dark-p", "value"), Input("tabs", "value"),
    Input("live-ver", "data"))
def dark(area, dv, prod, tab=None, _live=None):
    f, areas, area, prods, prod, dopts, dv = fuel_controls(area, dv, prod)
    out = [opts(areas), area, dopts, dv, opts(prods), prod]
    if not dv:
        return shown(out, *nodata(f"No {area} {prod} delivery quoted."), "dark", tab)
    m, _ = dsplit(dv)
    coal = E.load_coal()
    if coal is None:
        return shown(out, *nodata("ice_market_data.db not found (..\\ice_scraper)."), "dark", tab)
    e = eua()
    p = power_frame(f, dv).rename(columns={"settlement_price": "power"})
    # EUA: same day, else carried from up to 7 days before (as eex_dash.spark_frame)
    dates = sorted(e["trade_date"].unique())
    picks = []
    for t, d in zip(p["trade_date"], p["ddate"]):
        v, y, src = E.pick_eua(e, t, d.year)
        if v is None:
            prior = [x for x in dates if x < t and (pd.Timestamp(t) - pd.Timestamp(x)).days <= 7]
            if prior:
                v, y, src = E.pick_eua(e, prior[-1], d.year)
                src = f"carried from {prior[-1]}"
        picks.append((v, y, src))
    p["eua"] = [v for v, _, _ in picks]
    p["eua_dec"] = [y for _, y, _ in picks]
    p["eua_src"] = [s for _, _, s in picks]
    p["fuel_cost"] = float("nan")       # gas columns not used here
    p["carbon_cost"] = float("nan")
    x = E.add_coal(p, coal, COAL_HUB, m, COAL_EFF, COAL_CF).dropna(subset=["cds"])
    lab = label_of(f, dv)
    if x.empty:
        return shown(out, *nodata(f"No {COAL_HUB} price for {lab} (ICE lists months ~2 years, "
                                 "quarters, next Cal only) or no EUR/USD rate."), "dark", tab)
    x = x.set_index("trade_date").sort_index()
    tbl = pd.DataFrame({"Date": x.index, "Power": x["power"], f"{COAL_HUB} USD/t": x["coal_usd"],
                        "EUR/USD": x["fx"], "Coal €/MWh": x["coal_eur_th"],
                        "EUA": x["eua"].astype(float), "Coal cost": x["coal_fuel_cost"],
                        "CO2 cost": x["coal_carbon_cost"], "Prod. cost": x["coal_cost"],
                        "Spread": x["cds"]})
    last = x.iloc[-1]
    name = f"{area} {prod} {lab} CDS"
    notes = []
    if any(str(v).startswith("carried") for v in x["eua_src"]):
        notes.append("Some days use EUA carried from an earlier day.")
    if (x["eua_src"] == "ICE settlement").any():
        notes.append("EUA from ICE on days EEX has not published yet.")
    if (x["coal_src"] == "average").any():
        notes.append(f"{COAL_HUB}: no exact {lab} contract - average of its quarters/months.")
    live = live_fuel("dark", area, prod, m, dsplit(dv)[1], last, x["cds"].iloc[-1])
    content, st = body(x["cds"], tbl, f"{name} (EUR/MWh)", name,
                       extra=[("Power", fmt(last["power"])), (f"{COAL_HUB} $/t", fmt(last["coal_usd"])),
                              ("EUA", fmt(last["eua"]))], notes=notes, live=live)
    if tab == "dark":
        content += delivered_block(
            D.dark(area, m, prod, COAL_EFF, COAL_CF), f"{area} {prod} CDS", m,
            "No coal history yet. API2 front month (Yahoo MTF=F) and EUR/USD from 2020 "
            "arrive with the next nightly run (yfinance scraper, added 21 Oct).",
            notes=["Delivered CDS = day-ahead power − API2 month (USD/t ÷ EUR/USD ÷ 6.978) ÷ 45 % − EUA × 0.758. "
                   "API2 month = ICE final settle, else Yahoo MTF=F on the contract's last Friday."], pick=dsplit(dv)[1])
    return shown(out, content, summary("Dark", name, st), "dark", tab)



# ---------------------------------------------------------------------
# Live (delayed) prices: one "Update prices" button per tab
# ---------------------------------------------------------------------
def live_two_legs(rows_a, rows_b, w2, name_a, name_b):
    """Live spread of two power legs (eex_dash.live_spread_info). w2: the
    two settled legs (2 columns, by trade date)."""
    if not LIVE:
        return None
    w = pd.DataFrame({"L": w2.iloc[:, 0], "R": w2.iloc[:, 1]})
    w["spread"] = w["L"] - w["R"]
    return E.live_spread_info(rows_a, rows_b, w, name_a, name_b)


def fuel_roles(kind, area, prod, m, k):
    roles = E.live_spark_specs(area, prod, m, k, "Gas TTF", COAL_HUB)
    skip = "coal" if kind == "spark" else "gas"
    return [(r, x) for r, x in roles if r != skip]


def live_fuel(kind, area, prod, m, k, last, settled_value):
    """Live CSS / CDS: each input live where it traded today, else its
    settlement. None when nothing was fetched today."""
    if not LIVE:
        return None
    roles = fuel_roles(kind, area, prod, m, k)
    q = E.LP.read([x for _, x in roles])
    if not q:
        return None
    got = {}
    for role, x in roles:
        got.setdefault(role, []).append(q.get(E.live_key(x), {}))

    def pick(role, settled):
        vs = got.get(role, [])
        if vs and all(v.get("last") is not None for v in vs):
            return sum(v["last"] for v in vs) / len(vs), True
        return settled, False
    pw, pw_l = pick("power", last["power"])
    eu, eu_l = pick("eua", last["eua"])
    eu = float(eu)
    fetched = max(v["fetched_at"] for v in q.values())[11:16]
    day = E.live_label(fetched)
    if kind == "spark":
        fu, fu_l = pick("gas", last["gas"])
        fuel, co2 = fu / (GAS_EFF / 100), eu * GAS_CF
        value = pw - fuel - co2
        row = {"Date": day, "Power": pw, "TTF": fu, "EUA": eu, "Gas cost": fuel, "CO2 cost": co2,
               "Prod. cost": fuel + co2, "Spread": value}
    else:
        fu, fu_l = pick("coal", last["coal_usd"])
        th = fu / last["fx"] / E.COAL_MWH_PER_T
        fuel, co2 = th / (COAL_EFF / 100), eu * COAL_CF
        value = pw - fuel - co2
        row = {"Date": day, "Power": pw, f"{COAL_HUB} USD/t": fu, "EUR/USD": last["fx"],
               "Coal €/MWh": th, "EUA": eu, "Coal cost": fuel, "CO2 cost": co2,
               "Prod. cost": fuel + co2, "Spread": value}
    names = [n for n, l in (("power", pw_l), ("TTF" if kind == "spark" else COAL_HUB, fu_l),
                            ("EUA", eu_l)) if l]
    if not names:
        return {"value": None, "time": fetched, "sub": "no trades today"}
    chg = value - settled_value
    return {"value": value, "time": fetched, "change": chg, "row": row,
            "sub": f"{fmt(chg, sign=True)} vs settle · live: {', '.join(names)}"
                   + ("" if len(names) == 3 else ", rest settle")}


if LIVE:
    LIVE_KEYS = [k for k, _ in CARDS]

    @app.callback(
        Output("live-poll", "disabled"),
        [Input(f"{k}-lv", "n_clicks") for k in LIVE_KEYS],
        State("loc-a", "value"), State("loc-b", "value"), State("loc-d", "value"), State("loc-p", "value"),
        State("time-area", "value"), State("time-a", "value"), State("time-b", "value"), State("time-p", "value"),
        State("spark-area", "value"), State("spark-d", "value"), State("spark-p", "value"),
        State("dark-area", "value"), State("dark-d", "value"), State("dark-p", "value"),
        prevent_initial_call=True)
    def live_start(_l, _t, _s, _d, la, lb, ld, lp, ta, t1, t2, tp, sa, sd, sp, da, dd, dp):
        key = str(ctx.triggered_id or "").replace("-lv", "")
        pw = power()
        specs = []

        def rows(market, prod, v):
            return E.live_row_specs(leg(pw[(pw["market"] == market) & (pw["product"] == prod)], v).tail(1))
        if key == "loc" and ld:
            specs = rows(la, lp, ld) + rows(lb, lp, ld)
        elif key == "time" and t1 and t2:
            specs = rows(ta, tp, t1) + rows(ta, tp, t2)
        elif key in ("spark", "dark"):
            area, v, prod = (sa, sd, sp) if key == "spark" else (da, dd, dp)
            if v:
                m, k = dsplit(v)
                specs = [x for _, x in fuel_roles(key, area, prod, m, k)]
        return E._live_begin(key, specs)

    @app.callback(
        [Output(f"{k}-lv", "children") for k in LIVE_KEYS]
        + [Output(f"{k}-lv", "disabled") for k in LIVE_KEYS]
        + [Output(f"{k}-lvmsg", "children") for k in LIVE_KEYS]
        + [Output("live-ver", "data"), Output("live-poll", "disabled", allow_duplicate=True)],
        Input("live-poll", "n_intervals"), State("live-ver", "data"),
        prevent_initial_call=True)
    def live_progress(_, ver):
        with E._live_lock:
            job = {**E.LIVE_JOB, "msg": dict(E.LIVE_JOB["msg"]), "vers": dict(E.LIVE_JOB["vers"])}
        labels = ["Update prices"] * len(LIVE_KEYS)
        msgs = [job["msg"].get(k, "") for k in LIVE_KEYS]
        if job["running"]:
            i = LIVE_KEYS.index(job["tab"]) if job["tab"] in LIVE_KEYS else 0
            labels[i] = f"Updating {job['done']}/{job['total']} …"
            msgs[i] = job["text"]
            return labels + [True] * len(LIVE_KEYS) + msgs + [no_update, False]
        total = sum(job["vers"].get(k, 0) for k in LIVE_KEYS)
        return labels + [False] * len(LIVE_KEYS) + msgs + [total if total != ver else no_update, True]


# ---------------------------------------------------------------------
# Overview: all four spreads in one table
# ---------------------------------------------------------------------
@app.callback(Output("overview", "children"),
              [Input(f"{k}-sum", "data") for k, _ in CARDS])
def overview(*sums):
    rows = [s for s in sums if s]
    if not rows:
        return html.Div("Loading…", className="note")
    df = pd.DataFrame(rows)[["Spread", "Legs", "Last", "1 day", "1 week",
                             "Average", "Min", "Max", "Date"]]
    num = ["Last", "1 day", "1 week", "Average", "Min", "Max"]
    return [html.Div("Overview", style={"fontWeight": 600, "marginBottom": 6}),
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c, "type": "numeric" if c in num else "text",
                          "format": {"specifier": "+,.2f" if c in ("1 day", "1 week") else ",.2f"}
                          if c in num else None} for c in df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Segoe UI, Inter, Arial, sans-serif", "fontSize": 13,
                            "padding": "4px 8px", "textAlign": "right",
                            "border": f"1px solid {INK['grid']}"},
                style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                        for c in ("Spread", "Legs", "Date")],
                style_data_conditional=[
                    {"if": {"column_id": c, "filter_query": f"{{{c}}} > 0"}, "color": INK["up"]}
                    for c in ("Last", "1 day", "1 week")] + [
                    {"if": {"column_id": c, "filter_query": f"{{{c}}} < 0"}, "color": INK["down"]}
                    for c in ("Last", "1 day", "1 week")],
                style_header={"fontWeight": 600, "backgroundColor": INK["bg"]})]


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Spread analyzer (Dash)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default=None, help="path to eex_market_data.db")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the browser once it's up")
    args = ap.parse_args()
    E.DB_PATH = E.find_db(args.db)
    if E.DB_PATH is None:
        print(f"Could not find {E.DB_NAME}. Pass it with --db.")
        return
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print(f"Database: {E.DB_PATH}")
        print(f"Open {url}   (leave this window open; Ctrl+C stops it)")
        if args.open:
            import webbrowser
            threading.Timer(3.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
