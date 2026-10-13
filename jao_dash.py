"""
JAO auction data - Dash version
===============================

Dash version of `jao_dashboard.py` (Streamlit) over `jao_data.db`, built by
`jao_scraper.py`. Same queries, same calculations (copied unchanged), same
four tabs. The Streamlit file is left as it was.

- Corridor explorer: one corridor + horizon + product - every auction's
  volume and price, with the opposite direction alongside. For the
  Monthly horizon also year-by-month tables of price and volume.
- Buyers by month: one horizon + month - who won each corridor.
- Buyer lookup: one buyer - everything they have won.
- Top buyers: leaderboard by auctions won - all time, one year or one month.

Run it (Windows, in the folder this file sits in):

    python jao_dash.py

then open http://127.0.0.1:8055 in your browser. Or double-click
`start_jao_dash.bat`. Leave the black window
open while you use it - closing it stops the dashboard.

Options:
    --port 8055            use another port
    --host 0.0.0.0         reachable from other devices (e.g. over Tailscale)
    --db  C:\\path\\to\\jao_data.db   use a specific database file
    --open                 open the browser automatically
    --debug                Dash's error overlay (slower - only to read an error)

First time only:

    pip install dash plotly pandas

The database is opened READ-ONLY. Results are kept until the database
file changes, then re-read - refresh the page or touch a picker.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Group, Scheme

DEFAULT_PORT = 8055
DB_NAME = "jao_data.db"
DB_PATH = Path(__file__).resolve().parent / DB_NAME   # main() replaces it with --db

# Calendar order for the monthly pivot table's rows - pivot_table's default
# alphabetical sort would otherwise put "Apr" before "Feb".
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# How many rows the "Top buyers" tab's leaderboard shows.
TOP_BUYERS_LIMIT = 20

# Horizons excluded from the ENTIRE dashboard - the auctions that run every
# day (Peter: "take all daily functions out of the dashboard for now", then
# "dont show any 'every day' auctions"). Same set as jao_dashboard.py - see
# the long note there. Display-only: nothing is deleted from the database.
EXCLUDED_DASHBOARD_HORIZONS = {"daily", "intraday"}

FONT = "'Segoe UI', Inter, Arial, sans-serif"
INK = {"text": "#1f2328", "muted": "#6b6a64", "grid": "#e8e7e1", "axis": "#c3c2b7",
       "bg": "#f6f5f1", "card": "#ffffff", "accent": "#2a78d6"}


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
                if len(_cache) > 300:
                    _cache.clear()
                _cache[key] = hit
        return hit.copy() if isinstance(hit, (pd.DataFrame, list)) else hit
    wrapper.__name__ = fn.__name__
    return wrapper


# =====================================================================
# DATA - copied from jao_dashboard.py unchanged, except:
#   @st.cache_data(ttl=60) -> @cached, and _connect() uses a file URI.
# If a calculation changes there, make the same change here.
# =====================================================================

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)


def _horizon_exclusion(column: str = "horizon_name") -> tuple[str, tuple]:
    if not EXCLUDED_DASHBOARD_HORIZONS:
        return "1=1", ()
    placeholders = ", ".join("?" for _ in EXCLUDED_DASHBOARD_HORIZONS)
    return f"LOWER({column}) NOT IN ({placeholders})", tuple(EXCLUDED_DASHBOARD_HORIZONS)


def _order_horizons(horizons) -> list[str]:
    rank = {h: i for i, h in enumerate(
        ("yearly", "seasonal", "quarterly", "monthly", "weekly", "daily", "intraday")
    )}
    return sorted(horizons, key=lambda h: (rank.get(str(h).strip().lower(), len(rank)), str(h)))


def opposite_corridor(corridor_code: str) -> str | None:
    """The same interconnector in the other direction ("DE-FR" -> "FR-DE",
    "EL1-FR-GB" -> "EL1-GB-FR"), or None (e.g. "CEEXD")."""
    parts = corridor_code.split("-")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0]}"
    if len(parts) == 3:
        prefix, a, b = parts
        return f"{prefix}-{b}-{a}"
    return None


@cached
def load_corridors() -> list[str]:
    frag, params = _horizon_exclusion()
    conn = _connect()
    rows = conn.execute(
        f"SELECT DISTINCT corridor_code FROM auctions WHERE {frag} ORDER BY corridor_code",
        params,
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@cached
def load_horizons(corridor: str) -> list[str]:
    frag, params = _horizon_exclusion()
    conn = _connect()
    rows = conn.execute(
        f"SELECT DISTINCT horizon_name FROM auctions WHERE corridor_code = ? AND {frag} "
        "ORDER BY horizon_name",
        (corridor, *params),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@cached
def load_products(corridor: str, horizon: str) -> list[str]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT DISTINCT r.product_identification
        FROM auction_results r
        JOIN auctions a ON a.identification = r.identification
        WHERE a.corridor_code = ? AND a.horizon_name = ?
        ORDER BY r.product_identification
        """,
        (corridor, horizon),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@cached
def load_all_horizons() -> list[str]:
    frag, params = _horizon_exclusion()
    conn = _connect()
    rows = conn.execute(
        f"SELECT DISTINCT horizon_name FROM auctions WHERE {frag} ORDER BY horizon_name",
        params,
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@cached
def load_available_months(horizon: str) -> list[str]:
    """Months ("YYYY-MM") with an auction for this horizon. market_period_start
    is stored one calendar day early (CET midnight in UTC), hence +1 day."""
    conn = _connect()
    rows = conn.execute(
        "SELECT DISTINCT market_period_start FROM auctions WHERE horizon_name = ?",
        (horizon,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    shifted = pd.to_datetime([r[0] for r in rows], utc=True) + pd.Timedelta(days=1)
    return sorted(set(shifted.strftime("%Y-%m")))


@cached
def load_buyers_by_month(horizon: str, year_month: str) -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT
            a.identification,
            a.corridor_code,
            a.market_period_start,
            w.eic_code,
            n.party_name
        FROM auctions a
        LEFT JOIN auction_winners w ON w.identification = a.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE a.horizon_name = ?
        """,
        conn,
        params=(horizon,),
    )
    conn.close()
    if df.empty:
        return df

    df["period_shifted"] = pd.to_datetime(df["market_period_start"], utc=True) + pd.Timedelta(days=1)
    df = df[df["period_shifted"].dt.strftime("%Y-%m") == year_month]
    if df.empty:
        return df

    df["buyer"] = df["party_name"].fillna(df["eic_code"])

    def _combine_buyers(buyers: pd.Series) -> str:
        names = sorted({b for b in buyers if pd.notna(b)})
        return ", ".join(names) if names else "-"

    grouped = (
        df.groupby("corridor_code")
        .agg(period=("period_shifted", "min"), buyer=("buyer", _combine_buyers))
        .reset_index()
        .sort_values("corridor_code")
    )
    grouped["period"] = grouped["period"].dt.strftime("%Y-%m-%d")
    return grouped


@cached
def load_all_buyers() -> list[str]:
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT DISTINCT w.eic_code, n.party_name
        FROM auction_winners w
        JOIN auctions a ON a.identification = w.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE {frag}
        """,
        conn,
        params=params,
    )
    conn.close()
    if df.empty:
        return []
    df["label"] = df["party_name"].fillna(df["eic_code"])
    return sorted(df["label"].unique())


@cached
def load_purchases_for_buyer(buyer_label: str) -> pd.DataFrame:
    """Everything one buyer has won, newest auction close first.
    bid_gate_closure is a real timestamp - no +1 day shift."""
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT
            a.corridor_code,
            a.horizon_name,
            a.market_period_start,
            a.bid_gate_closure,
            w.eic_code,
            n.party_name
        FROM auction_winners w
        JOIN auctions a ON a.identification = w.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE {frag}
        """,
        conn,
        params=params,
    )
    conn.close()
    if df.empty:
        return df

    df["buyer_label"] = df["party_name"].fillna(df["eic_code"])
    df = df[df["buyer_label"] == buyer_label]
    if df.empty:
        return df

    df["period"] = (pd.to_datetime(df["market_period_start"], utc=True) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df["auction_closed"] = pd.to_datetime(df["bid_gate_closure"], utc=True, errors="coerce")
    out = (
        df[["horizon_name", "period", "corridor_code", "auction_closed"]]
        .drop_duplicates(subset=["horizon_name", "period", "corridor_code"])
        .sort_values("auction_closed", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out["auction_closed"] = out["auction_closed"].dt.strftime("%Y-%m-%d %H:%M")
    return out


@cached
def load_buyer_monthly_pivot(buyer_label: str) -> pd.DataFrame:
    """Auctions won per month (rows) by year x horizon (columns), counted by
    when the auction closed (bid_gate_closure), each auction once."""
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT a.identification, a.horizon_name, a.bid_gate_closure, w.eic_code, n.party_name
        FROM auction_winners w
        JOIN auctions a ON a.identification = w.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE {frag}
        """,
        conn,
        params=params,
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()

    df["buyer_label"] = df["party_name"].fillna(df["eic_code"])
    df = df[df["buyer_label"] == buyer_label]
    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["identification"])
    closed = pd.to_datetime(df["bid_gate_closure"], utc=True, errors="coerce")
    df = df[closed.notna()].copy()
    closed = closed[closed.notna()]
    if df.empty:
        return pd.DataFrame()

    df["year"] = closed.dt.year
    df["month"] = closed.dt.strftime("%b")

    counts = (
        df.groupby(["month", "year", "horizon_name"])
        .size()
        .reset_index(name="wins")
    )
    pivot = counts.pivot_table(
        index="month", columns=["year", "horizon_name"], values="wins", aggfunc="sum"
    )
    pivot = pivot.reindex(MONTH_ORDER)

    years_present = sorted(pivot.columns.get_level_values(0).unique())
    horizons_present = _order_horizons(pivot.columns.get_level_values(1).unique())
    ordered_cols = [
        (year, horizon)
        for year in years_present
        for horizon in horizons_present
        if (year, horizon) in pivot.columns
    ]
    return pivot.reindex(columns=ordered_cols)


@cached
def load_available_years() -> list[int]:
    frag, params = _horizon_exclusion()
    conn = _connect()
    rows = conn.execute(
        f"SELECT DISTINCT bid_gate_closure FROM auctions WHERE {frag}", params
    ).fetchall()
    conn.close()
    if not rows:
        return []
    closed = pd.to_datetime([r[0] for r in rows], utc=True, errors="coerce").dropna()
    return sorted(set(closed.year.tolist()))


@cached
def load_available_year_months() -> list[str]:
    frag, params = _horizon_exclusion()
    conn = _connect()
    rows = conn.execute(
        f"SELECT DISTINCT bid_gate_closure FROM auctions WHERE {frag}", params
    ).fetchall()
    conn.close()
    if not rows:
        return []
    closed = pd.to_datetime([r[0] for r in rows], utc=True, errors="coerce").dropna()
    return sorted(set(closed.strftime("%Y-%m")))


@cached
def load_top_buyers(scope: str, value: str | None) -> pd.DataFrame:
    """Buyers by auctions won, one column per horizon plus "total" (the
    ranking). Scoped by bid_gate_closure ("all", "year" = "2026" or
    "month" = "2026-09") - see jao_dashboard.py for why not the delivery
    period."""
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT a.bid_gate_closure, a.horizon_name, w.eic_code, n.party_name
        FROM auction_winners w
        JOIN auctions a ON a.identification = w.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE {frag}
        """,
        conn,
        params=params,
    )
    conn.close()
    if df.empty:
        return df

    if scope in ("year", "month"):
        closed = pd.to_datetime(df["bid_gate_closure"], utc=True, errors="coerce")
        if scope == "year":
            df = df[closed.dt.strftime("%Y") == value]
        else:
            df = df[closed.dt.strftime("%Y-%m") == value]
    if df.empty:
        return df

    df["buyer"] = df["party_name"].fillna(df["eic_code"])

    counts = (
        df.groupby(["buyer", "horizon_name"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=_order_horizons(counts.columns))
    counts["total"] = counts.sum(axis=1)
    counts = (
        counts.sort_values("total", ascending=False)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    counts.insert(0, "rank", counts.index + 1)
    return counts


@cached
def load_series(corridor: str, horizon: str, product: str) -> pd.DataFrame:
    """Every auction for this corridor+horizon+product, oldest first. Period
    shifted +1 day (market_period_start is stored one day early)."""
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT
            a.market_period_start,
            a.cancelled,
            r.product_hour,
            r.offered_capacity,
            r.requested_capacity,
            r.allocated_capacity,
            r.auction_price
        FROM auction_results r
        JOIN auctions a ON a.identification = r.identification
        WHERE a.corridor_code = ? AND a.horizon_name = ? AND r.product_identification = ?
        ORDER BY a.market_period_start ASC
        """,
        conn,
        params=(corridor, horizon, product),
    )
    conn.close()
    if df.empty:
        return df
    df["market_period_start"] = pd.to_datetime(df["market_period_start"], utc=True) + pd.Timedelta(days=1)
    df["period"] = df["market_period_start"].dt.strftime("%Y-%m-%d")
    return df


def render_monthly_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Years across, months (Jan-Dec) down - plain values, no colours."""
    pivot_source = df.copy()
    pivot_source["year"] = pivot_source["market_period_start"].dt.year
    pivot_source["month"] = pivot_source["market_period_start"].dt.strftime("%b")
    pivot = pivot_source.pivot_table(index="month", columns="year", values=value_col, aggfunc="mean")
    return pivot.reindex(MONTH_ORDER)

# =====================================================================
# PAGE
# =====================================================================


def note(text, cls="note"):
    return html.Div(text, className=cls)


def kpi(label, value):
    return html.Div([html.Div(label, className="kpi-label"),
                     html.Div(value, className="kpi-value")], className="kpi")


def fixed(decimals, nully=""):
    return Format(precision=decimals, scheme=Scheme.fixed, group=Group.no, nully=nully)


def table(df: pd.DataFrame, names: dict | None = None, formats: dict | None = None,
          order: list | None = None, page_size: int | None = 50, tooltips: dict | None = None,
          nully: str = ""):
    """A plain table. names: column -> header; formats: column -> decimals;
    order: which columns, in which order. Shows everything when it fits on
    one page, else pages of 50 rows. Click a header to sort."""
    names, formats = names or {}, formats or {}
    cols = order or list(df.columns)
    df = df[cols]
    columns = []
    for c in cols:
        col = {"name": names.get(c, str(c)), "id": str(c)}
        if pd.api.types.is_numeric_dtype(df[c]):
            col["type"] = "numeric"
            if c in formats:
                col["format"] = fixed(formats[c], nully)
        else:
            col["type"] = "text"
        columns.append(col)
    records = df.rename(columns=str).to_dict("records")
    paged = page_size is not None and len(records) > page_size
    return dash_table.DataTable(
        data=records, columns=columns, sort_action="native",
        page_action="native" if paged else "none", page_size=page_size or len(records) or 1,
        tooltip_header=tooltips or {},
        style_table={"overflowX": "auto", "maxWidth": "100%"},
        style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "4px 10px",
                    "border": f"1px solid {INK['grid']}", "whiteSpace": "normal",
                    "height": "auto", "minWidth": 60, "maxWidth": 420},
        style_cell_conditional=[{"if": {"column_type": "text"}, "textAlign": "left"}],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"], "fontFamily": FONT},
        css=[{"selector": ".dash-table-container", "rule": "width: max-content; max-width: 100%;"}],
    )


def pivot_table(pivot: pd.DataFrame, decimals: int):
    """Months down, years (or year x horizon) across. Empty cells show "-",
    like the Streamlit tables."""
    df = pivot.copy()
    multi = isinstance(df.columns, pd.MultiIndex)
    ids = [f"c{i}" for i in range(len(df.columns))]
    if multi:
        # Invisible marks keep the same horizon name in two neighbouring
        # years from being merged into one header cell across both years.
        years = list(dict.fromkeys(a for a, _ in df.columns))
        heads = [[str(a), str(b) + "\u200b" * (years.index(a) % 2)] for a, b in df.columns]
        first = ["year", "month"]
    else:
        heads = [str(c) for c in df.columns]
        first = "month"
    df.columns = ids
    df = df.reset_index().rename(columns={df.index.name or "index": "month"})
    columns = [{"name": first, "id": "month", "type": "text"}] + [
        {"name": h, "id": i, "type": "numeric", "format": fixed(decimals, "-")}
        for h, i in zip(heads, ids)]
    return html.Div(style={"display": "inline-block", "maxWidth": "100%"}, children=dash_table.DataTable(
        data=df.to_dict("records"), columns=columns, merge_duplicate_headers=multi,
        page_action="none",
        style_table={"overflowX": "auto", "maxWidth": "100%"},
        style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "4px 10px",
                    "border": f"1px solid {INK['grid']}", "minWidth": 56},
        style_cell_conditional=[{"if": {"column_id": "month"}, "textAlign": "left",
                                 "backgroundColor": INK["bg"], "fontWeight": 500}],
        style_header={"fontWeight": 600, "backgroundColor": INK["bg"], "fontFamily": FONT,
                      "textAlign": "center"},
    ))


def num(value, decimals):
    return f"{value:.{decimals}f}" if pd.notna(value) else "-"


CSS = f"""
body {{ margin:0; background:{INK['bg']}; color:{INK['text']}; font-family:{FONT}; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:16px 20px 40px; }}
h1 {{ font-size:24px; margin:0; }}
h3 {{ font-size:16px; margin:18px 0 6px; }}
.sub {{ color:{INK['muted']}; font-size:13px; margin:2px 0 10px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; margin:12px 0; }}
.ctl {{ display:flex; flex-direction:column; min-width:220px; }}
.ctl.wide {{ min-width:380px; }}
.ctl label.head {{ font-size:12px; color:{INK['muted']}; margin-bottom:3px; }}
.note {{ color:{INK['muted']}; font-size:13px; margin:6px 0; }}
.info {{ background:#e9f1fb; border:1px solid #b9d2f0; border-radius:6px;
        padding:8px 12px; font-size:13px; margin:8px 0; }}
.err {{ background:#fdecee; border:1px solid #e8a3ad; border-radius:6px;
       padding:8px 12px; font-size:13px; margin:8px 0; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 4px; }}
.kpi {{ background:{INK['card']}; border:1px solid {INK['grid']}; border-radius:8px;
       padding:8px 14px; min-width:200px; }}
.kpi-label {{ font-size:13px; color:{INK['muted']}; }}
.kpi-value {{ font-size:24px; font-weight:600; }}
.pair {{ display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; }}
.pair > div {{ min-width:0; max-width:100%; }}
.seg {{ display:flex; border:1px solid {INK['axis']}; border-radius:6px; overflow:hidden;
        background:{INK['card']}; height:34px; width:max-content; }}
.seg label {{ display:flex; align-items:center; padding:0 14px; font-size:13px; cursor:pointer;
              margin:0 !important; color:{INK['muted']}; }}
.seg label + label {{ border-left:1px solid {INK['grid']}; }}
.seg input {{ display:none; }}
.seg label:has(input:checked) {{ background:{INK['accent']}; }}
.seg label:has(input:checked), .seg label:has(input:checked) * {{ color:#fff !important; }}
@media (max-width:700px) {{ .ctl, .ctl.wide {{ min-width:100%; }} }}
"""

TABS = [("corridor", "Corridor explorer"), ("bymonth", "Buyers by month"),
        ("lookup", "Buyer lookup"), ("top", "Top buyers")]
TAB_STYLE = {"padding": "7px 12px", "fontSize": 14, "color": INK["muted"],
             "backgroundColor": INK["bg"], "border": "none",
             "borderBottom": f"1px solid {INK['grid']}"}
TAB_SELECTED = {**TAB_STYLE, "color": INK["text"], "fontWeight": 600,
                "borderBottom": f"3px solid {INK['accent']}"}


def ctl(label, component, wide=False, id_=None):
    kw = {"id": id_} if id_ else {}
    return html.Div([html.Label(label, className="head"), component],
                    className="ctl wide" if wide else "ctl", **kw)


def dd(id_):
    return dcc.Dropdown(id=id_, clearable=False)


def full_layout():
    caption = (f"Note: {', '.join(sorted(EXCLUDED_DASHBOARD_HORIZONS))}-horizon auctions are "
               "excluded from this whole dashboard for now.") if EXCLUDED_DASHBOARD_HORIZONS else ""
    return html.Div([
        html.H1("JAO auction data"),
        html.Div(caption, className="sub"),
        dcc.Tabs(id="tabs", value="corridor", children=[
            dcc.Tab(label=label, value=value, style=TAB_STYLE, selected_style=TAB_SELECTED)
            for value, label in TABS]),
        # Tabs only switch which panel is visible; every picker keeps its value.
        html.Div(id="panel-corridor", children=[
            html.Div([ctl("Corridor", dd("c-corridor")), ctl("Horizon", dd("c-horizon")),
                      ctl("Product", dd("c-product"))], className="controls"),
            dcc.Loading(html.Div(id="c-body"), type="dot", delay_show=400)]),
        html.Div(id="panel-bymonth", style={"display": "none"}, children=[
            html.Div([ctl("Horizon", dd("m-horizon")), ctl("Month", dd("m-month")),
                      ctl("Corridor", dd("m-corridor"))], className="controls"),
            dcc.Loading(html.Div(id="m-body"), type="dot", delay_show=400)]),
        html.Div(id="panel-lookup", style={"display": "none"}, children=[
            html.Div([ctl("Buyer", dd("l-buyer"), wide=True)], className="controls"),
            dcc.Loading(html.Div(id="l-body"), type="dot", delay_show=400)]),
        html.Div(id="panel-top", style={"display": "none"}, children=[
            html.Div([
                ctl("Scope", dcc.RadioItems(id="t-scope", options=["All time", "Year", "Month"],
                                            value="All time", inline=True, className="seg")),
                ctl("Year", dd("t-year"), id_="t-year-box"),
                ctl("Month", dd("t-month"), id_="t-month-box"),
            ], className="controls"),
            dcc.Loading(html.Div(id="t-body"), type="dot", delay_show=400)]),
    ], className="wrap")


def make_layout():
    if not DB_PATH.exists():
        return html.Div([html.H1("JAO auction data"), note(
            f"Can't find {DB_PATH.name} - run python jao_scraper.py at least once first.",
            "err")], className="wrap")
    if not load_corridors():
        return html.Div([html.H1("JAO auction data"), note(
            "No auctions saved yet - run python jao_scraper.py first.", "info")],
            className="wrap")
    return full_layout()


app = Dash(__name__, title="JAO auction data", update_title=None)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.layout = make_layout
app.validation_layout = full_layout()


def keep(value, allowed, fallback=None):
    """The current choice if it still exists, else the given fallback,
    else the first option (what a Streamlit selectbox does)."""
    if value in allowed:
        return value
    if fallback in allowed:
        return fallback
    return allowed[0] if allowed else None


@app.callback([Output(f"panel-{v}", "style") for v, _ in TABS], Input("tabs", "value"))
def switch_tab(tab):
    return [{"display": "block" if v == tab else "none"} for v, _ in TABS]


# ---------------------------------------------------------------------
# 1. Corridor explorer
# ---------------------------------------------------------------------
@app.callback(
    Output("c-corridor", "options"), Output("c-corridor", "value"),
    Output("c-horizon", "options"), Output("c-horizon", "value"),
    Output("c-product", "options"), Output("c-product", "value"),
    Output("c-body", "children"),
    Input("c-corridor", "value"), Input("c-horizon", "value"), Input("c-product", "value"),
    Input("tabs", "value"),
)
def corridor_explorer(corridor, horizon, product, tab):
    if tab != "corridor":
        return [no_update] * 7
    corridors = load_corridors()
    corridor = keep(corridor, corridors)
    horizons = load_horizons(corridor)
    horizon = keep(horizon, horizons)
    if not horizon:
        return (corridors, corridor, [], None, [], None,
                note(f"No auctions saved yet for {corridor}.", "info"))
    products = load_products(corridor, horizon)
    product = keep(product, products)
    out = [corridors, corridor, horizons, horizon, products, product]
    if not product:
        return out + [note(f"No results saved yet for {corridor} / {horizon}.", "info")]

    df = load_series(corridor, horizon, product)
    if df.empty:
        return out + [note("No results for this combination yet - the auction(s) found may "
                           "still be open (no results published).", "info")]

    # The same interconnector the other way round, always shown alongside
    # the picked one where it has data (Peter: "always also show the
    # opposite corridor when a corridor is picked").
    opposite = opposite_corridor(corridor)
    opp_df = (
        load_series(opposite, horizon, product)
        if opposite and opposite in load_corridors()
        else pd.DataFrame()
    )

    body = []
    latest = df.iloc[-1]
    if opp_df.empty:
        body.append(html.Div([
            kpi(f"{corridor} latest price (EUR/MW)", num(latest["auction_price"], 2)),
            kpi(f"{corridor} latest allocated (MW)", num(latest["allocated_capacity"], 0)),
            kpi("Auctions shown", str(len(df))),
        ], className="kpis"))
        if opposite:
            body.append(note(f"No {opposite} auctions saved for this horizon/product, "
                             f"so only {corridor} is shown."))
    else:
        opp_latest = opp_df.iloc[-1]
        body.append(html.Div([
            kpi(f"{corridor} latest price (EUR/MW)", num(latest["auction_price"], 2)),
            kpi(f"{opposite} latest price (EUR/MW)", num(opp_latest["auction_price"], 2)),
            kpi(f"{corridor} latest allocated (MW)", num(latest["allocated_capacity"], 0)),
            kpi(f"{opposite} latest allocated (MW)", num(opp_latest["allocated_capacity"], 0)),
        ], className="kpis"))

    if horizon == "Monthly":
        for title, col, dec in (("Price by month/year (EUR/MW)", "auction_price", 2),
                                ("Allocated volume by month/year (MW)", "allocated_capacity", 0)):
            body.append(html.H3(title))
            if opp_df.empty:
                body.append(pivot_table(render_monthly_pivot(df, col), dec))
            else:
                body.append(html.Div([
                    html.Div([note(corridor), pivot_table(render_monthly_pivot(df, col), dec)]),
                    html.Div([note(opposite), pivot_table(render_monthly_pivot(opp_df, col), dec)]),
                ], className="pair"))

    body.append(html.H3("All auctions"))
    if opp_df.empty:
        body.append(table(
            df, order=["period", "product_hour", "offered_capacity", "requested_capacity",
                       "allocated_capacity", "auction_price", "cancelled"],
            names={"period": "Period", "product_hour": "Product hour",
                   "offered_capacity": "Offered (MW)", "requested_capacity": "Requested (MW)",
                   "allocated_capacity": "Allocated (MW)", "auction_price": "Price (EUR/MW)",
                   "cancelled": "Cancelled"},
            formats={"offered_capacity": 2, "requested_capacity": 2,
                     "allocated_capacity": 2, "auction_price": 2}))
        return out + [body]

    # Both directions, paired metric by metric. Outer join on
    # period+product_hour so a period only one direction auctioned still
    # shows up (blank on the other side).
    merged = pd.merge(
        df[["period", "product_hour", "offered_capacity", "allocated_capacity", "auction_price"]],
        opp_df[["period", "product_hour", "offered_capacity", "allocated_capacity", "auction_price"]],
        on=["period", "product_hour"],
        how="outer",
        suffixes=("_a", "_b"),
    ).sort_values(["period", "product_hour"]).reset_index(drop=True)

    body.append(note(f"{corridor} vs. {opposite} - each metric shown for both directions, "
                     "side by side."))
    body.append(table(
        merged,
        order=["period", "product_hour", "auction_price_a", "auction_price_b",
               "allocated_capacity_a", "allocated_capacity_b",
               "offered_capacity_a", "offered_capacity_b"],
        names={"period": "Period", "product_hour": "Product hour",
               "auction_price_a": f"Price {corridor}", "auction_price_b": f"Price {opposite}",
               "allocated_capacity_a": f"Allocated {corridor}",
               "allocated_capacity_b": f"Allocated {opposite}",
               "offered_capacity_a": f"Offered {corridor}",
               "offered_capacity_b": f"Offered {opposite}"},
        formats={c: 2 for c in ("auction_price_a", "auction_price_b", "allocated_capacity_a",
                                "allocated_capacity_b", "offered_capacity_a",
                                "offered_capacity_b")}))
    return out + [body]


# ---------------------------------------------------------------------
# 2. Buyers by month
# ---------------------------------------------------------------------
@app.callback(
    Output("m-horizon", "options"), Output("m-horizon", "value"),
    Output("m-month", "options"), Output("m-month", "value"),
    Output("m-corridor", "options"), Output("m-corridor", "value"),
    Output("m-body", "children"),
    Input("m-horizon", "value"), Input("m-month", "value"), Input("m-corridor", "value"),
    Input("tabs", "value"),
)
def buyers_by_month(horizon, year_month, buyers_corridor, tab):
    if tab != "bymonth":
        return [no_update] * 7
    all_horizons = load_all_horizons()
    if not all_horizons:
        return [], None, [], None, [], None, note("No auctions saved yet.", "info")
    horizon = keep(horizon, all_horizons)
    months = load_available_months(horizon)
    if not months:
        return (all_horizons, horizon, [], None, [], None,
                note(f"No auctions saved yet for {horizon}.", "info"))
    # Newest month by default, and again whenever the horizon changes.
    if ctx.triggered_id == "m-horizon":
        year_month = None
    year_month = keep(year_month, months, months[-1])

    buyers_df = load_buyers_by_month(horizon, year_month)
    if buyers_df.empty:
        return (all_horizons, horizon, months, year_month, [], None,
                note(f"No auctions found for {horizon} in {year_month}.", "info"))

    # Corridor filter - starts over at "All corridors" whenever the
    # horizon or month changes (like the Streamlit picker's key).
    corridor_options = ["All corridors"] + sorted(buyers_df["corridor_code"].unique())
    if ctx.triggered_id in ("m-horizon", "m-month"):
        buyers_corridor = "All corridors"
    buyers_corridor = keep(buyers_corridor, corridor_options)
    display_df = (
        buyers_df
        if buyers_corridor == "All corridors"
        else buyers_df[buyers_df["corridor_code"] == buyers_corridor]
    )
    body = [
        note("Buyer shown as the resolved company name where known, otherwise the raw EIC "
             "code. Run python eic_lookup.py to resolve more names. \"-\" means no winner is "
             "recorded yet (auction may still be open)."),
        table(display_df, order=["corridor_code", "period", "buyer"],
              names={"corridor_code": "Corridor", "period": "Period", "buyer": "Buyer(s)"},
              page_size=None),
    ]
    return (all_horizons, horizon, months, year_month, corridor_options, buyers_corridor, body)


# ---------------------------------------------------------------------
# 3. Buyer lookup
# ---------------------------------------------------------------------
@app.callback(
    Output("l-buyer", "options"), Output("l-buyer", "value"), Output("l-body", "children"),
    Input("l-buyer", "value"), Input("tabs", "value"),
)
def buyer_lookup(buyer, tab):
    if tab != "lookup":
        return no_update, no_update, no_update
    buyers = load_all_buyers()
    if not buyers:
        return [], None, note("No winners recorded yet - run python eic_lookup.py after a "
                              "backfill.", "info")
    buyer = keep(buyer, buyers)
    purchases = load_purchases_for_buyer(buyer)
    if purchases.empty:
        return buyers, buyer, note(f"No purchases found for {buyer}.", "info")

    body = []
    pivot = load_buyer_monthly_pivot(buyer)
    if not pivot.empty:
        body += [note(f"Auctions won by {buyer}, by month - split by year and horizon, counted "
                      "by when the auction actually closed (not the delivery period)."),
                 pivot_table(pivot, 0)]
    body.append(html.H3(f"Everything {buyer} has won ({len(purchases):,})"))
    body.append(table(
        purchases, order=["horizon_name", "period", "corridor_code", "auction_closed"],
        names={"horizon_name": "Horizon", "period": "Period", "corridor_code": "Corridor",
               "auction_closed": "Auction closed (UTC)"},
        tooltips={"auction_closed": "When bidding actually closed and the auction cleared - "
                                    "not the delivery period."}))
    return buyers, buyer, body


# ---------------------------------------------------------------------
# 4. Top buyers
# ---------------------------------------------------------------------
@app.callback(
    Output("t-year-box", "style"), Output("t-month-box", "style"),
    Output("t-year", "options"), Output("t-year", "value"),
    Output("t-month", "options"), Output("t-month", "value"),
    Output("t-body", "children"),
    Input("t-scope", "value"), Input("t-year", "value"), Input("t-month", "value"),
    Input("tabs", "value"),
)
def top_buyers(scope_label, year, year_month, tab):
    hide, show = {"display": "none"}, {"display": "flex"}
    if tab != "top":
        return [no_update] * 7
    year_style, month_style = hide, hide
    year_opts, month_opts = no_update, no_update
    if scope_label == "Year":
        years = load_available_years()
        if not years:
            return show, hide, [], None, no_update, no_update, note("No auctions saved yet.", "info")
        year = keep(year, years, years[-1])
        year_style, year_opts = show, years
        counts = load_top_buyers("year", str(year))
        caption_scope = f"in {year}"
    elif scope_label == "Month":
        year_months = load_available_year_months()
        if not year_months:
            return hide, show, no_update, no_update, [], None, note("No auctions saved yet.", "info")
        year_month = keep(year_month, year_months, year_months[-1])
        month_style, month_opts = show, year_months
        counts = load_top_buyers("month", year_month)
        caption_scope = f"in {year_month}"
    else:
        counts = load_top_buyers("all", None)
        caption_scope = "of all time"
    head = [year_style, month_style, year_opts,
            year if scope_label == "Year" else no_update, month_opts,
            year_month if scope_label == "Month" else no_update]

    if counts.empty:
        return head + [note(f"No auctions won {caption_scope}.", "info")]

    total_buyers = len(counts)
    counts = counts.head(TOP_BUYERS_LIMIT)
    shown_note = (
        f"top {len(counts)} of {total_buyers} buyers"
        if total_buyers > TOP_BUYERS_LIMIT
        else f"all {total_buyers} buyer(s)"
    )
    horizon_cols = [c for c in counts.columns if c not in ("rank", "buyer", "total")]
    body = [
        note(f"Buyers ranked by total auctions won {caption_scope}, across every corridor - "
             f"{shown_note}, split by horizon. Buyer shown as the resolved company name where "
             "known, otherwise the raw EIC code."),
        table(counts, order=["rank", "buyer", *horizon_cols, "total"],
              names={"rank": "Rank", "buyer": "Buyer", "total": "Total won"}, page_size=None),
    ]
    return head + [body]


# ---------------------------------------------------------------------
def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description="JAO auction data (Dash)")
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
        print("Updates to jao_dash.py are picked up automatically - just refresh the page.")
        if args.open:
            import webbrowser
            threading.Timer(3.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
