"""
ENTSO-E day-ahead price dashboard - Dash version
================================================

Dash version of `dashboard.py` (Streamlit): pick up to 5 zones and a
day / week / month view, see a chart, the averages and a table. Optionally
also each zone's load forecast (MW). Same calculations as dashboard.py
(copied unchanged). dashboard.py is left as it was.

One difference on purpose: the load forecast gets its OWN chart under the
price chart, instead of a second y-axis in the same chart (the project's
chart rule: never two y-scales in one chart). Same zone, same colour in
both charts; forecast lines dashed.

Run it (Windows, in the folder this file sits in):

    python entsoe_dash.py

then open http://127.0.0.1:8056 in your browser. Or double-click
`start_entsoe_dash.bat`. Leave the black window
open while you use it - closing it stops the dashboard.

Options:
    --port 8056            use another port
    --host 0.0.0.0         reachable from other devices (e.g. over Tailscale)
    --db  C:\\path\\to\\entsoe_data.db   use a specific database file
    --open                 open the browser automatically
    --debug                Dash's error overlay (slower - only to read an error)

First time only:

    pip install dash plotly pandas

The database is opened READ-ONLY. Results are kept until the database
file changes, then re-read - refresh the page or touch a picker.

Since 2026-10-14 this is also the "Day-ahead charts" tab of the Day-ahead
prices app (price_table_dash.py, port 8051) - Peter: "merge hourly price
table and day ahead prices". price_table_dash imports this file and uses
make_layout(title=False) + register_callbacks(). It still runs on its own
too (port 8056), but Dash Hub only starts the merged app now.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Group, Scheme

DEFAULT_PORT = 8056
DB_NAME = "entsoe_data.db"
DB_PATH = Path(__file__).resolve().parent / DB_NAME   # main() replaces it with --db
NONE_OPTION = "None"  # shown in zone pickers 2-5; means "not selected"
NO_DB_TEXT = None     # the online app sets its own "no data yet" text here

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FONT = "'Segoe UI', Inter, Arial, sans-serif"
INK = {"text": "#1f2328", "muted": "#6b6a64", "grid": "#e8e7e1", "axis": "#c3c2b7",
       "bg": "#f6f5f1", "card": "#ffffff"}


# ---------------------------------------------------------------------
# Cache: same job as Streamlit's @st.cache_data(ttl=60), but kept until
# the database file changes. Results are copied on the way out.
# ---------------------------------------------------------------------
_cache: dict = {}
_cache_lock = threading.Lock()


def db_stamp() -> tuple:
    stamp = []
    for path in (DB_PATH, Path(f"{DB_PATH}-wal")):
        try:
            s = path.stat()
            stamp.append((s.st_mtime_ns, s.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def cached(fn):
    def wrapper(*args):
        key = (fn.__name__, args, db_stamp())
        with _cache_lock:
            hit = _cache.get(key, _cache)
        if hit is _cache:
            hit = fn(*args)
            with _cache_lock:
                if len(_cache) > 400:
                    _cache.clear()
                _cache[key] = hit
        return hit.copy() if isinstance(hit, (pd.DataFrame, pd.Series, list)) else hit
    wrapper.__name__ = fn.__name__
    return wrapper


def _connect() -> sqlite3.Connection:
    """Read-only, so this can never change or lock the scraper's database."""
    return sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)


# =====================================================================
# DATA - the loaders of dashboard.py, unchanged except:
#   @st.cache_data(ttl=60) -> @cached, sqlite3.connect(DB_PATH) -> _connect(),
#   and one zone's whole price history is read (and its timestamps
#   converted) once and kept, instead of once per picked day.
# Two DST fixes (bugs in dashboard.py, NOT fixed there - found 2026-10-12):
#   - Day view = the real calendar day in CET (23 hours in March, 25 in
#     October). dashboard.py takes 24 clock hours from midnight, so the
#     March day also shows the next day's 00:00 (at the top) and the
#     October day loses 23:00 and shows "02:00" twice. Here the repeated
#     October hour is labelled "02:00 (2nd)".
#   - Week view of the load forecast: the last hour of the October DST
#     week landed in a one-hour "week" of its own, so that ISO week showed
#     up twice. Here weeks are taken on local clock time.
# If a calculation changes in dashboard.py, make the same change here.
# =====================================================================

@cached
def load_zones() -> list[str]:
    conn = _connect()
    zones = [
        row[0]
        for row in conn.execute("SELECT DISTINCT zone FROM day_ahead_prices ORDER BY zone")
    ]
    conn.close()
    return zones


@cached
def _read_prices(zone: str) -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT timestamp, price_eur_mwh FROM day_ahead_prices WHERE zone = ?",
        conn,
        params=(zone,),
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")
    df["day"] = local_day(df["timestamp"])
    return df


def local_day(ts: pd.Series) -> pd.Series:
    """The CET calendar day of each timestamp (local clock time)."""
    return ts.dt.tz_localize(None).dt.normalize()


def hour_labels(ts: pd.Series) -> pd.Index:
    """ "HH:MM", with the hour that repeats when clocks go back in October
    shown as "02:00 (2nd)" so every hour of the day keeps its own row."""
    labels = ts.dt.strftime("%H:%M")
    return pd.Index(labels.where(~labels.duplicated(), labels + " (2nd)"))


def load_hourly(zone: str, date) -> pd.Series:
    """All hourly prices for one zone, one calendar day. Index = "HH:MM"."""
    df = _read_prices(zone)
    out = df.loc[df["day"] == pd.Timestamp(date)].sort_values("timestamp")
    return pd.Series(out["price_eur_mwh"].values, index=hour_labels(out["timestamp"]), name=zone)


@cached
def _read_sources(zone: str) -> pd.DataFrame | None:
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, source FROM day_ahead_prices WHERE zone = ?",
            conn,
            params=(zone,),
        )
    except pd.errors.DatabaseError:
        return None
    finally:
        conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")
    df["day"] = local_day(df["timestamp"])
    return df


def load_hourly_provisional_hours(zone: str, date) -> list[str]:
    """Which "HH:MM" hours for this zone/day are still provisional (Nord
    Pool, not yet confirmed by ENTSO-E). [] if none, or if the database
    has no "source" column."""
    df = _read_sources(zone)
    if df is None or df.empty:
        return []
    day = df.loc[df["day"] == pd.Timestamp(date)].sort_values("timestamp")
    labels = pd.Series(hour_labels(day["timestamp"]), index=day.index)
    return sorted(labels[day["source"] == "nordpool"])


@cached
def load_all_weekly(zone: str) -> pd.Series:
    """Every weekly average on record for one zone. Index = "2026-W02"."""
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT week_start, avg_price_eur_mwh FROM weekly_prices WHERE zone = ? ORDER BY week_start",
        conn,
        params=(zone,),
    )
    conn.close()
    iso = pd.to_datetime(df["week_start"]).dt.isocalendar()
    labels = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return pd.Series(df["avg_price_eur_mwh"].values, index=labels, name=zone)


@cached
def load_all_monthly(zone: str) -> pd.Series:
    """Every monthly average on record for one zone. Index = month_start."""
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT month_start, avg_price_eur_mwh FROM monthly_prices WHERE zone = ? ORDER BY month_start",
        conn,
        params=(zone,),
    )
    conn.close()
    return pd.Series(df["avg_price_eur_mwh"].values, index=df["month_start"], name=zone)


@cached
def _read_forecast(zone: str) -> pd.DataFrame:
    """Every hourly load forecast row on record for one zone."""
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT timestamp, forecast_mw FROM load_forecast_mw WHERE zone = ?",
        conn,
        params=(zone,),
    )
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")
    df["day"] = local_day(df["timestamp"])
    return df


def load_hourly_forecast(zone: str, date) -> pd.Series:
    df = _read_forecast(zone)
    if df.empty:
        return pd.Series(dtype=float, name=zone)
    out = df.loc[df["day"] == pd.Timestamp(date)].sort_values("timestamp")
    return pd.Series(out["forecast_mw"].values, index=hour_labels(out["timestamp"]), name=zone)


@cached
def load_all_weekly_forecast(zone: str) -> pd.Series:
    """Hourly forecast averaged into Monday-Sunday weeks, labelled like
    load_all_weekly (ISO week)."""
    df = _read_forecast(zone)
    if df.empty:
        return pd.Series(dtype=float, name=zone)
    local = df["timestamp"].dt.tz_localize(None)   # local clock time (DST fix)
    week_start = (local - pd.to_timedelta(local.dt.weekday, unit="D")).dt.normalize()
    weekly = df.assign(week_start=week_start).groupby("week_start")["forecast_mw"].mean().round(2)
    iso = weekly.index.isocalendar()
    labels = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return pd.Series(weekly.values, index=labels, name=zone)


@cached
def load_all_monthly_forecast(zone: str) -> pd.Series:
    df = _read_forecast(zone)
    if df.empty:
        return pd.Series(dtype=float, name=zone)
    month_start = df["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    monthly = df.assign(month_start=month_start).groupby("month_start")["forecast_mw"].mean().round(2)
    return pd.Series(monthly.values, index=monthly.index.strftime("%Y-%m-%d"), name=zone)


# Peter's default zone picks, in Zone-1..5 order - shown when the page
# loads. A zone not in the database falls back to the first zone (Zone 1)
# or "None" (Zones 2-5).
_DEFAULT_ZONES = ["DE_LU", "FR", "AT", "NL", "BE"]


def build_view(selected_zones, view, pick_date, show_forecast):
    """Everything the page shows, as plain data - the same steps as
    dashboard.py's main() after the pickers."""
    result = {"title": None, "data": None, "provisional": None, "forecast": None,
              "forecast_note": None}
    if view == "Day":
        series_list = [load_hourly(z, pick_date) for z in selected_zones]
        result["title"] = f"{pick_date} - hourly prices"
    elif view == "Week":
        series_list = [load_all_weekly(z) for z in selected_zones]
        result["title"] = "Weekly averages (every week on record)"
    else:  # Month
        series_list = [load_all_monthly(z) for z in selected_zones]
        result["title"] = "Monthly averages (every month on record)"

    data = pd.concat(series_list, axis=1).sort_index().round(2)
    data.index.name = "period"
    result["data"] = data
    if data.empty:
        return result

    if view == "Day":
        provisional = {z: load_hourly_provisional_hours(z, pick_date) for z in selected_zones}
        provisional = {z: hours for z, hours in provisional.items() if hours}
        if provisional:
            details = "; ".join(f"{z} ({len(hours)}h)" for z, hours in provisional.items())
            result["provisional"] = (
                f"⏳ Still provisional (Nord Pool, not yet confirmed by ENTSO-E): {details}. "
                "These will update automatically once fetch_data.py picks up the official price."
            )

    if show_forecast:
        if view == "Day":
            forecast_list = [load_hourly_forecast(z, pick_date) for z in selected_zones]
        elif view == "Week":
            forecast_list = [load_all_weekly_forecast(z) for z in selected_zones]
        else:  # Month
            forecast_list = [load_all_monthly_forecast(z) for z in selected_zones]

        forecast_data = pd.concat(forecast_list, axis=1).sort_index()
        forecast_data.index.name = "period"
        missing = [
            z for z in selected_zones
            if z not in forecast_data.columns or forecast_data[z].isna().all()
        ]
        forecast_data = forecast_data.dropna(axis=1, how="all")
        if forecast_data.empty:
            result["forecast_note"] = ("No load forecast data yet for these zones - run "
                                       "fetch_load_forecast.py (or run_all.py) to see it here too.")
        else:
            result["forecast"] = forecast_data
            if missing:
                result["forecast_note"] = f"No load forecast data for: {', '.join(missing)}."
    return result


# =====================================================================
# PAGE
# =====================================================================

def note(text, cls="note"):
    return html.Div(text, className=cls)


def kpi(label, value):
    return html.Div([html.Div(label, className="kpi-label"),
                     html.Div(value, className="kpi-value")], className="kpi")


def table(df: pd.DataFrame):
    """period + one column per zone, always 2 decimals (like the
    Streamlit table's NumberColumn format)."""
    df = df.reset_index()
    columns = [{"name": "period", "id": "period", "type": "text"}] + [
        {"name": str(c), "id": str(c), "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.fixed, group=Group.no)}
        for c in df.columns[1:]]
    return html.Div(style={"display": "inline-block", "maxWidth": "100%"}, children=dash_table.DataTable(
        data=df.rename(columns=str).to_dict("records"), columns=columns,
        sort_action="native", page_action="none", fixed_rows={"headers": True},
        style_table={"maxHeight": 440, "overflowY": "auto", "overflowX": "auto"},
        style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "4px 10px",
                    "border": f"1px solid {INK['grid']}", "minWidth": 90, "width": 90},
        style_cell_conditional=[{"if": {"column_id": "period"}, "textAlign": "left",
                                 "backgroundColor": INK["bg"]}],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"], "fontFamily": FONT},
    ))


def line_chart(df: pd.DataFrame, order: list, colour_of: dict, y_title: str,
               title: str, dashed=False, unit=""):
    """One line per zone over the shared period order (HH:MM, YYYY-Wnn or
    YYYY-MM-DD - all sort correctly as text)."""
    fig = go.Figure()
    # An empty trace over every period, so the price and forecast charts
    # always share the same x axis (Plotly only shows categories with data).
    fig.add_trace(go.Scatter(x=order, y=[None] * len(order), mode="lines",
                             showlegend=False, hoverinfo="skip"))
    for zone in df.columns:
        s = df[zone]
        fig.add_trace(go.Scatter(
            x=list(s.index), y=s.values, name=str(zone), mode="lines",
            line=dict(color=colour_of.get(zone, INK["muted"]), width=2,
                      dash="dash" if dashed else "solid"),
            hovertemplate="%{y:,.2f} " + unit + "<extra>" + str(zone) + "</extra>"))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK["text"])),
        height=360, margin=dict(l=70, r=20, t=56, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=INK["muted"]),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_yaxes(title=y_title, gridcolor=INK["grid"], zeroline=False, linecolor=INK["axis"])
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=order,
                     showgrid=False, linecolor=INK["axis"], nticks=24)
    return dcc.Graph(figure=fig, config={"displaylogo": False, "displayModeBar": False})


CSS = f"""
body {{ margin:0; background:{INK['bg']}; color:{INK['text']}; font-family:{FONT}; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:16px 20px 40px; }}
h1 {{ font-size:24px; margin:0 0 6px; }}
h2 {{ font-size:19px; margin:14px 0 4px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; margin:10px 0; }}
.ctl {{ display:flex; flex-direction:column; min-width:150px; flex:1; max-width:240px; }}
.ctl label.head {{ font-size:12px; color:{INK['muted']}; margin-bottom:3px; }}
.note {{ color:{INK['muted']}; font-size:13px; margin:6px 0; }}
.info {{ background:#e9f1fb; border:1px solid #b9d2f0; border-radius:6px;
        padding:8px 12px; font-size:13px; margin:8px 0; }}
.warn {{ background:#fff4d6; border:1px solid #eda100; border-radius:6px;
        padding:8px 12px; font-size:13px; margin:6px 0; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 4px; }}
.kpi {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
       padding:8px 14px; min-width:150px; }}
.kpi-label {{ font-size:13px; color:{INK['muted']}; }}
.kpi-value {{ font-size:24px; font-weight:600; }}
.card {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
        padding:6px 10px; margin:10px 0; }}
.seg {{ display:flex; border:1px solid {INK['axis']}; border-radius:6px; overflow:hidden;
        background:{INK['card']}; height:34px; width:max-content; }}
.seg label {{ display:flex; align-items:center; padding:0 16px; font-size:13px; cursor:pointer;
              margin:0 !important; color:{INK['muted']}; }}
.seg label + label {{ border-left:1px solid {INK['grid']}; }}
.seg input {{ display:none; }}
.seg label:has(input:checked) {{ background:{SERIES[0]}; }}
.seg label:has(input:checked), .seg label:has(input:checked) * {{ color:#fff !important; }}
.check label {{ display:flex !important; align-items:center; gap:8px; cursor:pointer; }}
.check input {{ width:16px; height:16px; margin:0; }}
@media (max-width:700px) {{ .ctl {{ min-width:45%; }} }}
"""


def ctl(label, component, id_=None):
    kw = {"id": id_} if id_ else {}
    return html.Div([html.Label(label, className="head"), component], className="ctl", **kw)


def full_layout(zones=None, title=True):
    """title=False: without the headline (as a tab of the Day-ahead prices app)."""
    zones = zones or []
    pickers = []
    for i in range(1, 6):
        wanted = _DEFAULT_ZONES[i - 1]
        if i == 1:
            options = zones
            value = wanted if wanted in zones else (zones[0] if zones else None)
        else:
            options = [NONE_OPTION] + zones
            value = wanted if wanted in zones else NONE_OPTION
        pickers.append(ctl(f"Zone {i}", dcc.Dropdown(id=f"zone-{i}", options=options,
                                                     value=value, clearable=False)))
    today = pd.Timestamp.now(tz="CET").date().isoformat()
    return html.Div([
        *([html.H1("ENTSO-E day-ahead price dashboard")] if title else []),
        html.Div(pickers, className="controls"),
        html.Div([
            ctl("View", dcc.RadioItems(id="view", options=["Day", "Week", "Month"], value="Day",
                                       inline=True, className="seg")),
            ctl("Pick a date", dcc.DatePickerSingle(id="date", date=today,
                                                    display_format="YYYY-MM-DD",
                                                    first_day_of_week=1, clearable=False),
                id_="date-box"),
        ], className="controls"),
        dcc.Checklist(id="forecast", options=[{"label": "Also show load forecast (MW)",
                                               "value": "on"}],
                      value=[], className="check"),
        dcc.Loading(html.Div(id="body"), type="dot", delay_show=400),
    ], className="wrap")


def make_layout(title=True):
    """Built on every page load, so the date always starts on today."""
    head = [html.H1("ENTSO-E day-ahead price dashboard")] if title else []
    if not Path(DB_PATH).exists():
        text = NO_DB_TEXT or (f"No {DB_NAME} found at {DB_PATH} - run fetch_data.py "
                              "first, then reload this page.")
        return html.Div([*head, note(text, "warn")], className="wrap")
    zones = load_zones()
    if not zones:
        return html.Div([*head, note("No data yet - run fetch_data.py first, then reload "
                                     "this page.", "warn")], className="wrap")
    return full_layout(zones, title=title)


def date_box(view):
    # Only "Day" needs a date - Week/Month show every period on record.
    return {} if view == "Day" else {"display": "none"}


def update(z1, z2, z3, z4, z5, view, day, forecast):
    choices = [z1, z2, z3, z4, z5]
    # Picking the same zone twice just shows it once.
    selected_zones = list(dict.fromkeys(z for z in choices if z and z != NONE_OPTION))
    if not selected_zones:
        return note("Pick at least one zone above.", "info")
    pick_date = None
    if view == "Day":
        if not day:
            return note("Pick a date.", "info")
        pick_date = pd.Timestamp(str(day)[:10]).date()
    # Colour follows the picker the zone sits in (Zone 1 is always blue).
    colour_of = {}
    for i, z in enumerate(choices):
        if z and z != NONE_OPTION and z not in colour_of:
            colour_of[z] = SERIES[i]

    try:
        r = build_view(selected_zones, view, pick_date, bool(forecast))
    except Exception as exc:   # e.g. a day the data can't be lined up for
        return note(f"Could not build this view: {exc}", "warn")
    data = r["data"]
    body = [html.H2(r["title"])]
    if data.empty:
        return body + [note("No data for these zones and this period.", "info")]
    if r["provisional"]:
        body.append(note(r["provisional"]))
    body.append(html.Div([kpi(f"{zone} avg (EUR/MWh)", f"{data[zone].mean():.2f}")
                          for zone in selected_zones], className="kpis"))
    forecast_data = r["forecast"]
    if r["forecast_note"]:
        body.append(note(r["forecast_note"]))
    # One shared left-to-right order for both charts, built from whichever
    # periods either measure has (all labels sort correctly as text).
    order = list(dict.fromkeys(data.index))
    if forecast_data is not None and not forecast_data.empty:
        order = sorted(set(order) | set(forecast_data.index))
    body.append(html.Div(line_chart(data, order, colour_of, "Price (EUR/MWh)", "Price",
                                    unit="EUR/MWh"), className="card"))
    if forecast_data is not None:
        body.append(html.Div(line_chart(forecast_data, order, colour_of, "Load forecast (MW)",
                                        "Load forecast", dashed=True, unit="MW"),
                             className="card"))
    body.append(table(data))
    if forecast_data is not None:
        body += [note("Load forecast values (MW):"), table(forecast_data)]
    return body


_UPDATE_INPUTS = [*[Input(f"zone-{i}", "value") for i in range(1, 6)],
                  Input("view", "value"), Input("date", "date"), Input("forecast", "value")]


def register_callbacks(target_app, active_when=None) -> None:
    """Hook the callbacks onto an app: this file's own, or the Day-ahead prices
    app (price_table_dash.py). active_when=(tabs id, tab value): only draw
    while that tab is open - the charts read 5 zones' whole history, which
    the price table tab doesn't need."""
    target_app.callback(Output("date-box", "style"), Input("view", "value"))(date_box)
    if active_when is None:
        target_app.callback(Output("body", "children"), *_UPDATE_INPUTS)(update)
        return
    tabs_id, wanted = active_when

    def update_when_open(z1, z2, z3, z4, z5, view, day, forecast, tab):
        if tab != wanted:
            raise PreventUpdate
        return update(z1, z2, z3, z4, z5, view, day, forecast)

    target_app.callback(Output("body", "children"), *_UPDATE_INPUTS,
                        Input(tabs_id, "value"))(update_when_open)


app = Dash(__name__, title="ENTSO-E price dashboard", update_title=None)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.layout = make_layout
app.validation_layout = full_layout()
register_callbacks(app)


# ---------------------------------------------------------------------
def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description="ENTSO-E day-ahead price dashboard (Dash)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from the phone / Tailscale")
    ap.add_argument("--db", default=None, help=f"path to {DB_NAME}")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the browser once it's up")
    args = ap.parse_args()
    if args.db:
        DB_PATH = Path(args.db).expanduser()
    print(f"Database: {DB_PATH}" + ("" if DB_PATH.exists() else "   (not found)"))
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    # use_reloader: restarts the app whenever this file changes, so an
    # update shows up on the next browser refresh. Only the first (watcher)
    # process prints and opens the browser.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print(f"Open {url}   (leave this window open; Ctrl+C stops it)")
        print("Updates to entsoe_dash.py are picked up automatically - just refresh the page.")
        if args.open:
            import webbrowser
            threading.Timer(3.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
