"""
JAO auction data - Dash version
===============================

Dash version of `jao_dashboard.py` (Streamlit, since deleted - see
`dashboards/00-standard.md`) over `jao_data.db`, built by `jao_scraper.py`.
The first four tabs are the original queries/calculations copied unchanged;
"Buyer lookup map" (added 2026-10-13) is new, Dash-only.

- Corridor explorer: one corridor + horizon + product - every auction's
  volume and price, with the opposite direction alongside. For the
  Monthly horizon also year-by-month tables of price and volume.
- Buyers by month: one horizon + month - who won each corridor.
- Buyer lookup: one buyer - everything they have won.
- Buyer lookup map: one buyer + a time scope (all time/year/month) - a map
  of Europe with a line for each corridor direction they won, sized and
  coloured by how many times.
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
import json
import math
from html import escape as html_escape
import os
import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
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
# Buyer lookup map - geography for JAO's corridor codes. One point per
# country/zone token that actually appears in Peter's 103 real corridor
# codes (checked against jao_data.db, 2026-10-13) - approximate, for a
# schematic line map, not survey-accurate. D1/D2/N2 are JAO's own tokens
# for DK1/DK2/NO2 (confirmed in dashboards/year-comparison-streamlit.md's
# ZONE_TO_JAO_TOKEN and scrapers/jao-ceexd-bundle.md's sibling notes on
# JAO's Danish/Norwegian corridors - JAO never uses "DK1"/"DK2"/"NO2").
# lat, lon, display name.
GEO = {
    # Real country centroids (or, for DK1/DK2/NO2, the real centre of that
    # bidding zone's own sub-region) - as close to the actual geography as
    # a single point per zone can get, not a schematic spread.
    "AT": (47.60, 14.55, "Austria"), "BE": (50.50, 4.47, "Belgium"),
    "BG": (42.73, 25.49, "Bulgaria"), "CH": (46.82, 8.23, "Switzerland"),
    "CZ": (49.82, 15.47, "Czechia"), "D1": (56.05, 9.30, "Denmark (DK1)"),
    "D2": (55.50, 11.90, "Denmark (DK2)"), "DE": (51.17, 10.45, "Germany"),
    "EE": (58.60, 25.01, "Estonia"),
    "ES": (40.46, -3.75, "Spain"), "FI": (61.92, 25.75, "Finland"),
    "FR": (46.23, 2.21, "France"), "GB": (54.00, -2.50, "Great Britain"),
    "GR": (39.07, 21.82, "Greece"), "HR": (45.10, 15.20, "Croatia"),
    "HU": (47.16, 19.50, "Hungary"), "IT": (42.80, 12.80, "Italy"),
    "LV": (56.88, 24.60, "Latvia"), "ME": (42.71, 19.37, "Montenegro"),
    "MK": (41.61, 21.75, "North Macedonia"), "N2": (58.70, 7.50, "Norway (NO2)"),
    "NL": (52.13, 5.29, "Netherlands"), "PL": (51.92, 19.15, "Poland"),
    "PT": (39.40, -8.22, "Portugal"), "RO": (45.94, 24.97, "Romania"),
    "RS": (44.02, 21.01, "Serbia"), "SI": (46.15, 15.00, "Slovenia"),
    "SK": (48.67, 19.70, "Slovakia"), "UA": (48.38, 31.17, "Ukraine"),
}

# Named-cable prefixes (3-part corridor codes, e.g. "IF1-FR-GB") - just for
# a nicer hover label, the geography comes from GEO regardless.
CABLE_NAMES = {"EL1": "ElecLink", "IF1": "IFA", "IF2": "IFA2",
               "NLL": "Nemo Link", "NSL": "North Sea Link", "VKL": "Viking Link"}

# Where each country pair's line crosses the REAL border - the direction
# arrows sit here (Peter, 2026-10-13, drawing on the map: two arrows side by
# side on each border, one per direction, e.g. BE-FR up by Chimay, not
# halfway between Paris and Brussels). Land pairs: where the straight
# centroid-to-centroid line crosses the two countries' shared border in
# Natural Earth 1:50m - or, where that line misses the shared border (e.g.
# HU-UA, whose straight line runs through Romania), the nearest point on it;
# the line is then drawn bent through this point. Sea / same-country pairs
# (cables, DK1-DK2): halfway between the real link's two landfalls.
# Key = the two GEO tokens in alphabetical order. Any pair missing here
# falls back to the midpoint of the two centroids (_border_point).
BORDER_POINTS = {
    "AT-CH": (47.02, 9.82), "AT-CZ": (48.98, 15.12), "AT-DE": (48.55, 13.46),
    "AT-HU": (47.42, 16.56), "AT-IT": (46.52, 13.72), "AT-SI": (46.59, 14.86),
    "BE-DE": (50.68, 6.11), "BE-FR": (49.96, 4.18), "BE-NL": (51.41, 4.93),
    "BG-GR": (41.53, 24.28),
    "BG-MK": (41.94, 22.86), "BG-RO": (43.68, 25.34), "BG-RS": (43.60, 22.48),
    "CH-DE": (47.72, 8.69), "CH-FR": (46.62, 6.17), "CH-IT": (46.09, 9.06),
    "CZ-DE": (50.48, 13.01), "CZ-SK": (49.11, 18.10), "D1-DE": (54.85, 9.58),
    "DE-FR": (49.12, 7.03), "EE-LV": (57.99, 24.87),
    # DE-NL: on the real border but further north (near Emmen) than where the
    # straight line crosses (51.88, 6.63), so its arrows are clear of the
    # BE-NL / NL-BE ones (Peter, 2026-10-13: "move the de nl arrows up").
    "DE-NL": (52.77, 7.06),
    "ES-FR": (43.01, -1.12), "ES-PT": (39.58, -7.47), "FR-IT": (44.68, 6.99),
    "HR-HU": (46.08, 17.24), "HR-RS": (44.87, 18.99), "HR-SI": (45.45, 15.13),
    "HU-RO": (46.72, 21.49), "HU-RS": (46.16, 19.98), "HU-SI": (46.49, 16.54),
    "HU-SK": (48.22, 19.64), "HU-UA": (47.93, 22.86), "IT-SI": (45.61, 13.87),
    "ME-RS": (43.10, 19.85), "MK-RS": (42.25, 21.55), "PL-UA": (50.48, 24.03),
    "RO-RS": (44.68, 22.37), "RO-UA": (47.99, 26.16), "SK-UA": (48.61, 22.19),
    # Sea cables / same country - halfway between the two landfalls.
    "BE-GB": (51.33, 2.30),    # Nemo Link, Zeebrugge - Richborough
    "D1-D2": (56.00, 10.90),   # Great Belt link - placed north of the real
                               # crossing (55.35, 10.95) on purpose, so it
                               # doesn't crowd D1-DE / D2-DE (Peter, 2026-10-13)
    "D1-GB": (54.40, 4.20),    # Viking Link, Jutland coast - Lincolnshire
    "D2-DE": (54.75, 13.00),   # Kriegers Flak link, east of Kontek - further
                               # right so it's clear of D1-DE (Peter, 2026-10-13)
    "D1-NL": (54.45, 7.50),    # COBRAcable, Jutland coast - Eemshaven (JAO: "DK-NL")
    "EE-FI": (59.80, 25.00),   # Estlink 1+2, Gulf of Finland
    "FR-GB": (50.00, -1.90),   # IFA / IFA2 / ElecLink - moved west from the
                               # Strait of Dover (51.00, 1.45) to the western
                               # Channel, clear of Nemo (Peter, 2026-10-13:
                               # "move ifa ifa2 more to the left", "a bit more")
    "GB-N2": (57.20, 1.95),    # North Sea Link, Blyth - Norwegian coast
    "GR-IT": (39.83, 19.37),   # GRITA, Otranto - Epirus
    "IT-ME": (42.38, 16.48),   # MONITA, Pescara - Kotor
}

# Arrow colours (Peter, 2026-10-13: "if all is selected one colour if more
# are selected separate colours for the tenors picked"). Tenor colours are
# the dashboard standard's fixed colour-blind-safe palette, one per tenor in
# TENOR_ORDER, so a tenor keeps its colour whatever else is picked. "All"
# is a neutral dark, so it never reads as one particular tenor.
ALL_TENORS_COLOR = "#1f2328"
TENOR_COLORS = {"Yearly": "#2a78d6", "Seasonal": "#eb6834", "Quarterly": "#1baf7a",
                "Monthly": "#eda100", "Weekly": "#e87ba4"}


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
        key = (fn.__name__, args, db_stamp(), eex_stamp())
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


# JAO's corridor "DK-NL" (COBRAcable) lands in DK1, so on the map "DK" is
# shown as D1 (Peter, 2026-10-13: "Call DK D1"). Display only - the
# database and the other tabs keep JAO's own "DK-NL".
TOKEN_ALIAS = {"DK": "D1"}


def display_code(corridor_code: str) -> str:
    """corridor_code with TOKEN_ALIAS applied to each part ("DK-NL" ->
    "D1-NL"); everything else unchanged."""
    return "-".join(TOKEN_ALIAS.get(p, p) for p in corridor_code.split("-"))


def corridor_endpoints(corridor_code: str) -> tuple[str, str, str] | None:
    """(origin token, destination token, display label) for a corridor code,
    using the same two/three-part shapes as opposite_corridor() above - or
    None if it has no single geographic direction ("CEEXD", a bundled
    multi-border auction - see scrapers/jao-ceexd-bundle.md) or either end
    isn't in GEO."""
    if corridor_code == "CEEXD":
        return None
    shown = display_code(corridor_code)
    parts = shown.split("-")
    if len(parts) == 2:
        a, b = parts
        label = shown
    elif len(parts) == 3:
        prefix, a, b = parts
        label = f"{shown} ({CABLE_NAMES.get(prefix, prefix)})"
    else:
        return None
    a0, b0 = re.sub(r"\(.*\)", "", a), re.sub(r"\(.*\)", "", b)
    if a0 not in GEO or b0 not in GEO:
        return None
    return a0, b0, label


def _border_point(a: str, b: str) -> tuple[float, float]:
    """(lat, lon) where the a-b line crosses the real border (BORDER_POINTS),
    or the midpoint of the two centroids if the pair isn't listed there."""
    key = "-".join(sorted((a, b)))
    if key in BORDER_POINTS:
        return BORDER_POINTS[key]
    return (GEO[a][0] + GEO[b][0]) / 2, (GEO[a][1] + GEO[b][1]) / 2


# Pairs whose arrows should point straight at one country's dot instead of
# following the centroid-border-centroid line (Peter, 2026-10-13: "de-d2
# more to the right and point towards d2").
ARROW_AIM = {"D2-DE": "D2"}


# Direction-arrow size, in "local degrees" (1 = one degree of latitude, and
# longitude is squeezed by cos(lat) first so the arrow isn't stretched
# sideways). ~1.4 long on the full-Europe view = roughly a thumb's width.
ARROW_HALF_LEN = 0.7
ARROW_HEAD_LEN = 0.35     # length of each of the two arrowhead strokes
ARROW_SIDE = 0.4      # how far each arrow sits off the line, to its own side
LABEL_GAP = 0.4       # corridor-name label: this much further out than the arrow
ARROW_WIDTH = 2           # px - thin line arrows (Peter, 2026-10-13: "make
                          # arrowheads only lines and arrow thinner")
ARROW_HEAD_ANGLE = 28     # degrees between the shaft and each head stroke


def _arrow_label(items: list[tuple[str, int]], o: str, d: str) -> str:
    """Small corridor name next to an arrow (Peter, 2026-10-13: "write the
    name of the corridor in small next to the arrows"). A plain corridor
    shows its own code ("FR-ES", "DE(TenneT)-CZ"); cable corridors sharing
    one arrow show the direction plus the cable names ("FR-GB IFA, IFA2")."""
    cables = [lab.split("(", 1)[1].rstrip(")") for lab, _ in items
              if " (" in lab and lab.count("-") == 2]
    if cables:
        return f"{o}-{d} " + ", ".join(cables)
    return ", ".join(lab for lab, _ in items)


def _flow_arrow(lat_c, lon_c, ux, uy, half_len, head_len, side, n_parts=1):
    """One direction's arrow at a border, drawn with thin lines only: a
    shaft plus an open arrowhead (two short strokes from the tip - no
    filled triangle; Peter, 2026-10-13: "make arrowheads only lines and
    arrow thinner"). Centred on (lat_c, lon_c) but moved `side` to the
    RIGHT of travel, pointing along (ux, uy) (a unit vector in the local
    frame, x = lon * cos(lat), y = lat), so A->B and B->A land side by
    side on opposite sides of the border.

    With several tenors the shaft is cut lengthwise into `n_parts` equal
    pieces, tail to tip, one per tenor colour - still one arrow with more
    colours. Returns (parts, head): parts = list of (lats, lons) shaft
    pieces from tail to tip; head = (lats, lons) of the open arrowhead
    (left stroke end, tip, right stroke end)."""
    k = math.cos(math.radians(lat_c)) or 1.0
    nx, ny = uy, -ux                       # right-hand side of travel
    cx, cy = lon_c * k + nx * side, lat_c + ny * side
    tail = (cx - ux * half_len, cy - uy * half_len)
    tip = (cx + ux * half_len, cy + uy * half_len)

    def geo(pts):
        return [p[1] for p in pts], [p[0] / k for p in pts]

    parts = []
    for i in range(n_parts):
        a, b = i / n_parts, (i + 1) / n_parts
        parts.append(geo([(tail[0] + (tip[0] - tail[0]) * a, tail[1] + (tip[1] - tail[1]) * a),
                          (tail[0] + (tip[0] - tail[0]) * b, tail[1] + (tip[1] - tail[1]) * b)]))
    t = math.radians(ARROW_HEAD_ANGLE)
    back = (-ux, -uy)

    def wing(sign):                        # rotate "back" by +/- the head angle
        c, s = math.cos(sign * t), math.sin(sign * t)
        return (tip[0] + (back[0] * c - back[1] * s) * head_len,
                tip[1] + (back[0] * s + back[1] * c) * head_len)

    head = geo([wing(1), tip, wing(-1)])
    return parts, head


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
def load_purchases_for_buyer(buyer_label: str, show: str = "fut") -> pd.DataFrame:
    """Everything one buyer has won, newest auction close first.
    bid_gate_closure is a real timestamp - no +1 day shift."""
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT
            a.identification,
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
    # One row per won auction (several EIC codes of one company = 1 row).
    # Was one row per horizon + period + corridor, which hid a peak auction
    # won next to the base one - now that each row has its own price, both show.
    out = (
        df[["identification", "horizon_name", "period", "corridor_code", "auction_closed"]]
        .drop_duplicates(subset=["identification"])
        .sort_values("auction_closed", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out["auction_closed"] = out["auction_closed"].dt.strftime("%Y-%m-%d %H:%M")
    # Auction price (EUR/MW) of each win (Peter, 2026-10-18: "add the auction
    # price of the corridor in the winning list"). Bundles / peak / not yet
    # published: see price_note.
    # In brackets: futures spread and/or settle, as the tick box says
    # (Peter, 2026-10-26) - see FUT_NOTE / SETTLE_NOTE.
    details = _auction_price_details(tuple(out["identification"]), show)
    out["price"] = [px_text(details[i][0], show, details[i][3], details[i][2])
                    for i in out["identification"]]
    out["price_note"] = [details[i][1] for i in out["identification"]]
    return out.drop(columns="identification")


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


def _expand_to_delivery_months(df: pd.DataFrame) -> pd.DataFrame:
    """Spread each row over every calendar month its delivery window
    [market_period_start, market_period_stop) touches, in CET/CEST local
    time (JAO stores local midnight as UTC, e.g. "2025-12-31T23:00Z" =
    1 Jan 2026 00:00 CET; stop is exclusive, hence -1s) - the exact same
    date handling as entsoe_scraper/year_comparison_dash.py's
    _load_jao_tenor_prices()/_load_jao_tenor_winners(), the already-
    verified JAO delivery-month expansion, reused here rather than
    re-derived. Needs market_period_start, market_period_stop and
    horizon_name columns (Weekly/Daily products are shorter than a month
    and count only for the month they start in, same special case those
    functions make). Adds year/month (1-12) columns; every other column
    is repeated as-is, so one input row can become several output rows."""
    if df.empty:
        return df.assign(year=pd.Series(dtype="int64"), month=pd.Series(dtype="int64"))
    start = pd.to_datetime(df["market_period_start"], utc=True).dt.tz_convert("CET")
    stop = pd.to_datetime(df["market_period_stop"], utc=True).dt.tz_convert("CET") - pd.Timedelta(seconds=1)
    first_month = (start.dt.year * 12 + start.dt.month - 1).to_numpy()
    last_month = (stop.dt.year * 12 + stop.dt.month - 1).fillna(-1).astype(int).to_numpy()
    sub_month = df["horizon_name"].isin(("Weekly", "Daily")).to_numpy()
    last_month = np.where(sub_month, first_month, last_month)
    n_months = np.maximum(last_month - first_month + 1, 1)

    out = df.loc[df.index.repeat(n_months)].reset_index(drop=True)
    month_index = np.repeat(first_month, n_months) + np.concatenate([np.arange(n) for n in n_months])
    out["year"] = month_index // 12
    out["month"] = month_index % 12 + 1
    return out


# Same canonical order as _order_horizons() (Daily/Intraday are excluded
# dashboard-wide already - EXCLUDED_DASHBOARD_HORIZONS - so they are not
# offered as a tenor to pick here either). Peter: "pick/unpick each
# tenor" - the Buyer lookup map's only filter besides Buyer and Month
# (Peter, 2026-10-13: "not scope and tenor. only tenor." - the earlier
# Year/Quarter/Season/Month Scope control is gone; there is always
# exactly one picked month, see load_delivery_months() below, and Tenor
# is the one remaining way to narrow what's shown for it).
TENOR_ORDER = ["Yearly", "Seasonal", "Quarterly", "Monthly", "Weekly"]


@cached
def load_buyer_map_data(buyer_label: str, year_month: str, tenors: str = "All",
                        show: str = "fut") -> pd.DataFrame:
    """One row per corridor + tenor (horizon_name) this buyer HOLDS
    capacity for during the picked delivery month ("2026-10"), with the
    win count and win date(s) (the day each auction closed), filtered by
    `tenors` ("All", or a comma-joined subset of TENOR_ORDER - a plain
    string rather than a list/set because @cached needs a hashable
    argument).

    Scoped by DELIVERY period (market_period_start/stop, via
    _expand_to_delivery_months()), NOT by bid_gate_closure like Top
    Buyers (load_top_buyers) - a Yearly auction won in November still
    counts for every month it delivers in. That's deliberately different:
    this map answers "what does this buyer hold in October", Top Buyers'
    Month scope answers "which auctions did they win in October" - two
    different questions that happen to share a month picker's shape.
    (Peter, 2026-10-13, after the first bid_gate_closure-scoped version
    showed only 2 corridors for a month where he knew there were more:
    "they have more than 2 for oct".)

    One win per auction regardless of how many EIC codes resolve to this
    buyer's name, and regardless of how many delivery months it spans
    overall (nunique on identification, not a row count - a Yearly
    auction touches 12 expanded rows across the year but is still 1 win
    in whichever single one of those months is picked here)."""
    frag, params = _horizon_exclusion("a.horizon_name")
    conn = _connect()
    df = pd.read_sql_query(
        f"""
        SELECT a.identification, a.corridor_code, a.horizon_name,
               a.market_period_start, a.market_period_stop, a.bid_gate_closure,
               w.eic_code, n.party_name
        FROM auction_winners w
        JOIN auctions a ON a.identification = w.identification
        LEFT JOIN eic_names n ON n.eic_code = w.eic_code
        WHERE {frag}
        """,
        conn,
        params=params,
    )
    conn.close()
    empty = pd.DataFrame(columns=["corridor_code", "horizon_name", "wins", "win_date", "price"])
    if df.empty:
        return empty

    df["buyer_label"] = df["party_name"].fillna(df["eic_code"])
    df = df[df["buyer_label"] == buyer_label]
    if df.empty:
        return empty
    df = df.drop_duplicates(subset=["identification"])

    if tenors != "All":
        df = df[df["horizon_name"].isin(tenors.split(","))]
        if df.empty:
            return empty

    expanded = _expand_to_delivery_months(df)
    y, m = year_month.split("-")
    mask = (expanded["year"] == int(y)) & (expanded["month"] == int(m))
    expanded = expanded[mask]
    if expanded.empty:
        return empty

    # One row per corridor + tenor (an auction has exactly one tenor, so
    # summing these over tenors gives the per-corridor total again).
    counts = (expanded.groupby(["corridor_code", "horizon_name"])["identification"]
              .nunique().reset_index(name="wins"))

    # Win date = the day the auction closed (bid_gate_closure, CET) - when
    # the capacity was won (Peter, 2026-10-13: "add win date to the
    # table"). One date per auction; several wins on the same day show as
    # "2026-09-16 (x2)".
    won = expanded.drop_duplicates("identification").copy()
    won["day"] = (pd.to_datetime(won["bid_gate_closure"], utc=True)
                  .dt.tz_convert("CET").dt.strftime("%Y-%m-%d"))

    def fmt_days(days):
        tally = days.dropna().value_counts()
        return ", ".join(d if tally[d] == 1 else f"{d} (x{tally[d]})"
                         for d in sorted(tally.index))

    dates = (won.groupby(["corridor_code", "horizon_name"])["day"]
             .apply(fmt_days).reset_index(name="win_date"))
    counts = counts.merge(dates, on=["corridor_code", "horizon_name"], how="left")

    # Auction price (EUR/MW) of each won auction, in the same order as the
    # win dates (Peter, 2026-10-18: "add the auction price of the corridor in
    # the winning list"). Several auctions on one day: " / " between them.
    texts = _auction_price_texts(tuple(won["identification"]), show)
    won["price"] = won["identification"].map(texts).fillna("-")

    def fmt_prices(group):
        return ", ".join(" / ".join(sub["price"]) for _, sub in group.groupby("day", sort=True))

    prices = (won.groupby(["corridor_code", "horizon_name"])[["day", "price"]]
              .apply(fmt_prices).reset_index(name="price"))
    counts = counts.merge(prices, on=["corridor_code", "horizon_name"], how="left")
    rank = {t: i for i, t in enumerate(TENOR_ORDER)}
    counts["_t"] = counts["horizon_name"].map(rank).fillna(len(rank))
    counts = counts.sort_values(["corridor_code", "_t"]).drop(columns="_t")
    return counts.reset_index(drop=True)


# ---------------------------------------------------------------------
# Settle next to every auction price (Peter, 2026-10-26: "whenever auction
# price is shown, put settlement price in parentis next to it" - "in general
# in LTCAP tool"). Shown as "4.63 (5.12)": auction price, then in brackets
# what that capacity settled at in the day-ahead market over the auction's
# own delivery period. Settle = the Corridor margins tab's definition: for
# corridor A-B (flow A -> B), the hours B was pricier than A (B minus A,
# else 0), averaged over ALL hours of the period both zones have a price.
# Peak products: only Mon-Fri 08:00-20:00 (CET) hours. A delivery period
# still running: so far. Not started yet: no bracket.
# Prices come from entsoe_data.db (the day-ahead prices); the zone mapping
# is the one year_comparison_dash.py uses (_corridor_zones) - keep in sync.
# ---------------------------------------------------------------------
def _default_entsoe_db() -> Path:
    folder = Path(__file__).resolve().parent.parent / "entsoe_scraper"
    try:
        if str(folder) not in sys.path:
            sys.path.append(str(folder))  # appended: never shadows this folder's modules
        from entsoe_db import entsoe_db_path
        return Path(entsoe_db_path())
    except Exception:  # noqa: BLE001 - online copy / no entsoe_scraper next to it
        return folder / "entsoe_data.db"


ENTSOE_DB_PATH = _default_entsoe_db()   # the LTCAP tool / online_app may replace it
SETTLE_TOKEN_TO_ZONE = {"DE": "DE_LU", "D1": "DK_1", "D2": "DK_2", "DK": "DK_1",
                        "N2": "NO_2", "IE": "IE_SEM", "IT": "IT_NORD"}
SETTLE_ZONE_OVERRIDES = {"GR-IT": ("GR", "IT_SUD"), "IT-GR": ("IT_SUD", "GR"),
                         "ME-IT": ("ME", "IT_CSUD"), "IT-ME": ("IT_CSUD", "ME")}
SETTLE_NOTE = ("(...) = settle: what that capacity earned in the day-ahead market over "
               "its delivery period - hours the importing side was pricier, averaged over "
               "all hours (peak products: peak hours). Running periods: so far. Not "
               "delivered yet: no bracket.")
_SETTLE_EPOCH = pd.Timestamp("2019-12-31", tz="UTC").as_unit("ns").value // 3_600_000_000_000  # hours


def settle_zones(corridor: str) -> tuple:
    """(from, to) ENTSO-E zones of a JAO corridor code, or (None, None)."""
    if corridor in SETTLE_ZONE_OVERRIDES:
        return SETTLE_ZONE_OVERRIDES[corridor]
    parts = [part.split("(")[0] for part in corridor.split("-")]
    if len(parts) == 3:
        parts = parts[1:]
    if len(parts) != 2:
        return None, None
    return (SETTLE_TOKEN_TO_ZONE.get(parts[0], parts[0]),
            SETTLE_TOKEN_TO_ZONE.get(parts[1], parts[1]))


def _entsoe_stamp() -> tuple:
    out = []
    for path in (Path(ENTSOE_DB_PATH), Path(f"{ENTSOE_DB_PATH}-wal")):
        try:
            st = path.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return (str(ENTSOE_DB_PATH), *out)


def _zone_prices(zone: str) -> pd.Series:
    """Hourly day-ahead prices of one zone (UTC index). Empty if none."""
    conn = sqlite3.connect(Path(ENTSOE_DB_PATH).resolve().as_uri() + "?mode=ro", uri=True,
                           timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, price_eur_mwh FROM day_ahead_prices "
            "WHERE zone = ? AND timestamp >= '2019-12-31'", conn, params=(zone,))
    finally:
        conn.close()
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["timestamp"], utc=True)
    return pd.Series(df["price_eur_mwh"].to_numpy(dtype=float), index=idx).groupby(level=0).mean()


def _settle_cache_get(key):
    # Kept in the same _cache dict as @cached, so the online app's memory
    # guard (which empties each app's _cache) covers these too.
    with _cache_lock:
        return _cache.get(key)


def _settle_cache_put(key, value):
    with _cache_lock:
        if len(_cache) > 300:
            _cache.clear()
        _cache[key] = value


def _zone_prices_cached(zone: str) -> pd.Series:
    """_zone_prices(), kept until entsoe_data.db changes (a zone is in many corridors)."""
    key = ("settle-zone", zone, _entsoe_stamp())
    series = _settle_cache_get(key)
    if series is None:
        series = _zone_prices(zone)
        _settle_cache_put(key, series)
    return series


def _spread_hours(corridor: str):
    """The corridor's hourly max(P_to - P_from, 0) over the hours both zones
    have a price, compact: (hour numbers int32, spread float32, peak-hour
    flags). None if the corridor has no two priced zones."""
    zone_from, zone_to = settle_zones(corridor)
    if not zone_from or not zone_to or not Path(ENTSOE_DB_PATH).exists():
        return None
    key = ("settle-pair", zone_from, zone_to, _entsoe_stamp())
    hit = _settle_cache_get(key)
    if hit is not None:
        return hit or None
    try:
        a, b = _zone_prices_cached(zone_from), _zone_prices_cached(zone_to)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return None
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if both.empty:
        result = ()
    else:
        local = both.index.tz_convert("CET")
        # as_unit("ns"): pandas 3 parses to microseconds - asi8 must be ns here.
        result = ((both.index.as_unit("ns").asi8 // 3_600_000_000_000 - _SETTLE_EPOCH)
                  .astype(np.int32),
                  (both["b"] - both["a"]).clip(lower=0).to_numpy(dtype=np.float32),
                  np.asarray((local.weekday < 5) & (local.hour >= 8) & (local.hour < 20)))
    _settle_cache_put(key, result)
    return result or None


def settle_for(corridor: str, start, stop, peak: bool = False):
    """Settle (EUR/MWh) of `corridor` over [start, stop) - market_period_
    start/stop as stored by JAO (UTC instants of local midnight). None when
    nothing of the period has a price yet (or the corridor has no zones)."""
    data = _spread_hours(corridor)
    if data is None or start is None or stop is None:
        return None
    try:
        t0, t1 = (pd.Timestamp(x) for x in (start, stop))
        t0 = t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")
        t1 = t1.tz_convert("UTC") if t1.tzinfo else t1.tz_localize("UTC")
    except (ValueError, TypeError):
        return None
    hours, spread, peak_hours = data
    h0 = t0.as_unit("ns").value // 3_600_000_000_000 - _SETTLE_EPOCH
    h1 = t1.as_unit("ns").value // 3_600_000_000_000 - _SETTLE_EPOCH
    i, j = np.searchsorted(hours, [h0, h1])
    chunk = spread[i:j]
    if peak:
        chunk = chunk[peak_hours[i:j]]
    return float(chunk.astype(np.float64).mean()) if len(chunk) else None


def with_settle(price, settle) -> str:
    """ "4.63 (5.12)" - auction price, settle in brackets (none if unknown)."""
    if price is None or pd.isna(price):
        return ""
    if settle is None or pd.isna(settle):
        return f"{price:.2f}"
    return f"{price:.2f} ({settle:.2f})"


# ---------------------------------------------------------------------
# Futures spread next to every auction price (Peter, 2026-10-26: "add also
# the price spread of the futures contracts for the corridor countries at
# settlement on auction day. also everywhere where auction price is. tick
# box decides what is shown, default is spread future contracts").
# For corridor A-B (flow A -> B): the EEX future of B minus the one of A,
# for the auction's own delivery product (Yearly -> Cal, Quarterly -> Q,
# Monthly -> that month; base, or peak for a peak auction), at its
# settlement price on the day the auction's bidding closed (CET). No
# settlement that day (weekend / holiday): the last one before, up to
# FUT_MAX_DAYS_BACK days. EEX settlements are only stored from Aug 2026;
# older days use that day's last traded price (eex_market_ohlc), marked "*".
# Italy: the Italian (PUN) future for every Italian zone. No EUR future at
# EEX for one side (GB is in GBP, Baltics, PT, ME, ...), no peak future
# (Nordic zones, BE, GR, IE) or only a Mon-Sun peak (ES): no futures spread.
# ---------------------------------------------------------------------
def _default_eex_db() -> Path:
    return Path(__file__).resolve().parent.parent / "eex_scraper" / "eex_market_data.db"


EEX_DB_PATH = _default_eex_db()   # online_app replaces it
FUT_MAX_DAYS_BACK = 7
FUT_HORIZON_MATURITY = {"Yearly": "Year", "Quarterly": "Quarter", "Monthly": "Month"}
FUT_MONTHS = {"Year": 12, "Quarter": 3, "Month": 1}
# Market names in EEX's contract names (same list as futures_vs_spot_dash.py,
# GB left out: GBP).
FUT_DEMONYM = {
    "German": "DE", "French": "FR", "Austrian": "AT", "Swiss": "CH",
    "Italian": "IT", "Spanish": "ES", "Dutch": "NL", "Belgian": "BE",
    "Polish": "PL", "Czech": "CZ", "Slovakian": "SK", "Hungarian": "HU",
    "Romanian": "RO", "Bulgarian": "BG", "Croatian": "HR", "Slovenian": "SI",
    "Serbian": "RS", "Greek": "GR", "Finnish": "FI", "Irish SEM": "IE",
}
FUT_ZONE_RE = re.compile(r"\b([A-Z]{2}\d)\b")     # "Danish DK1", "Norwegian NO2"
# ENTSO-E zone (settle_zones()) -> EEX market.
FUT_ZONE_TO_MARKET = {
    "DE_LU": "DE", "FR": "FR", "AT": "AT", "CH": "CH", "ES": "ES", "NL": "NL",
    "BE": "BE", "PL": "PL", "CZ": "CZ", "SK": "SK", "HU": "HU", "RO": "RO",
    "BG": "BG", "HR": "HR", "SI": "SI", "RS": "RS", "GR": "GR", "FI": "FI",
    "IE_SEM": "IE", "DK_1": "DK1", "DK_2": "DK2", "NO_1": "NO1", "NO_2": "NO2",
    "NO_3": "NO3", "NO_4": "NO4", "NO_5": "NO5", "SE_1": "SE1", "SE_2": "SE2",
    "SE_3": "SE3", "SE_4": "SE4",
    **{z: "IT" for z in ("IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD", "IT_SICI",
                         "IT_SARD", "IT_CALA")},
}


def eex_stamp() -> tuple:
    out = []
    for path in (Path(EEX_DB_PATH), Path(f"{EEX_DB_PATH}-wal")):
        try:
            st = path.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return (str(EEX_DB_PATH), *out)


def _fut_market(long_name: str | None) -> str | None:
    """EEX market of a plain power future name, None for anything else."""
    if not long_name:
        return None
    name = re.sub(r"^EEX(-PXE)?\s+", "", long_name)
    if " Power" not in name or any(w in name for w in (
            "Option", "Japan", "Block", "Off-Peak", "Mon-Sun", "Week", "Day")):
        return None
    head = name.split(" Power")[0]
    zone = FUT_ZONE_RE.search(head)
    return zone.group(1) if zone else FUT_DEMONYM.get(head)


def _fut_data():
    """({(market, "Base"/"Peak", maturity): short code},
    {(code, delivery): (days, prices, last_traded flags)}) - month, quarter
    and year futures, days = days since 1970 (sorted). Kept until the EEX
    file changes. None without the EEX database."""
    path = Path(EEX_DB_PATH)
    if not path.exists():
        return None
    key = ("fut-data", eex_stamp())
    hit = _settle_cache_get(key)
    if hit is not None:
        return hit or None
    result = ()
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            codes = {}
            for code, name, product, maturity in conn.execute(
                    "SELECT short_code, long_name, product, maturity_type FROM contracts "
                    "ORDER BY short_code"):
                if product in ("Base", "Peak") and maturity in FUT_MONTHS:
                    market = _fut_market(name)
                    if market:
                        codes.setdefault((market, product, maturity), code)
            wanted = sorted(set(codes.values()))
            marks = ",".join("?" * len(wanted))
            frames = [pd.read_sql_query(
                "SELECT short_code, delivery, trade_date, settlement_price AS price, 0 AS last "
                f"FROM eex_market_prices WHERE short_code IN ({marks}) "
                "AND settlement_price IS NOT NULL AND COALESCE(currency, 'EUR') = 'EUR'",
                conn, params=wanted)]
            try:
                frames.append(pd.read_sql_query(
                    "SELECT short_code, delivery, trade_date, last AS price, 1 AS last "
                    f"FROM eex_market_ohlc WHERE short_code IN ({marks}) "
                    "AND trades > 0 AND last IS NOT NULL AND COALESCE(currency, 'EUR') = 'EUR'",
                    conn, params=wanted))
            except (sqlite3.Error, pd.errors.DatabaseError):
                pass   # no trade table: settlements only
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        _settle_cache_put(key, result)
        return None
    both = pd.concat(frames, ignore_index=True)
    both["day"] = pd.to_datetime(both["trade_date"], errors="coerce")
    both = both.dropna(subset=["day", "price"])
    series = {}
    if not both.empty:
        both["day"] = both["day"].to_numpy().astype("datetime64[D]").astype(np.int64)
        both["delivery"] = both["delivery"].astype(str)
        # A settlement wins over the last trade of the same day.
        both = (both.sort_values(["short_code", "delivery", "day", "last"])
                    .drop_duplicates(["short_code", "delivery", "day"], keep="first"))
        for (code, delivery), g in both.groupby(["short_code", "delivery"], sort=False):
            series[(code, delivery)] = (g["day"].to_numpy(dtype=np.int64),
                                        g["price"].to_numpy(dtype=float),
                                        g["last"].to_numpy(dtype=bool))
    result = (codes, series)
    _settle_cache_put(key, result)
    return result


def _fut_product(horizon, start, stop):
    """(maturity, delivery "YYYYMM") of the EEX future that delivers exactly
    the auction's period, or None (Seasonal / Weekly, or a period that is not
    a whole calendar month / quarter / year)."""
    maturity = FUT_HORIZON_MATURITY.get(horizon)
    if maturity is None or start is None or stop is None:
        return None
    try:
        t0, t1 = (pd.Timestamp(x) for x in (start, stop))
        t0 = (t0 if t0.tzinfo else t0.tz_localize("UTC")).tz_convert("CET")
        t1 = (t1 if t1.tzinfo else t1.tz_localize("UTC")).tz_convert("CET")
    except (ValueError, TypeError):
        return None
    if (t0.day, t0.hour, t0.minute) != (1, 0, 0) or (t1.day, t1.hour, t1.minute) != (1, 0, 0):
        return None
    if (t1.year * 12 + t1.month) - (t0.year * 12 + t0.month) != FUT_MONTHS[maturity]:
        return None
    if maturity == "Quarter" and t0.month not in (1, 4, 7, 10):
        return None
    if maturity == "Year" and t0.month != 1:
        return None
    return maturity, f"{t0.year}{t0.month:02d}"


def _fut_leg(data, market, product, maturity, delivery, day):
    """(price, last_traded) of one future on `day` (or the last day before,
    up to FUT_MAX_DAYS_BACK), or None."""
    code = data[0].get((market, product, maturity))
    found = data[1].get((code, delivery)) if code else None
    if found is None:
        return None
    days, prices, last = found
    i = int(np.searchsorted(days, day, side="right")) - 1
    if i < 0 or day - days[i] > FUT_MAX_DAYS_BACK:
        return None
    return float(prices[i]), bool(last[i])


def futures_spread_for(corridor: str, horizon, start, stop, gate, peak: bool = False):
    """(F_to - F_from, last_traded) in EUR/MWh for one auction - see the
    note above - or None. start/stop = market_period_start/stop as stored by
    JAO, gate = bid_gate_closure."""
    product = _fut_product(horizon, start, stop)
    if product is None or gate is None or (isinstance(gate, float) and math.isnan(gate)):
        return None
    zone_from, zone_to = settle_zones(corridor)
    m_from, m_to = FUT_ZONE_TO_MARKET.get(zone_from), FUT_ZONE_TO_MARKET.get(zone_to)
    if not m_from or not m_to or m_from == m_to:
        return None
    data = _fut_data()
    if not data:
        return None
    try:
        g = pd.Timestamp(gate)
        g = (g if g.tzinfo else g.tz_localize("UTC")).tz_convert("CET")
    except (ValueError, TypeError):
        return None
    day = (g.date() - pd.Timestamp("1970-01-01").date()).days
    kind = "Peak" if peak else "Base"
    a = _fut_leg(data, m_from, kind, *product, day)
    b = _fut_leg(data, m_to, kind, *product, day) if a else None
    if a is None or b is None:
        return None
    return round(b[0] - a[0], 2), a[1] or b[1]


# ---- the tick box: what goes in brackets after an auction price ---------
PX_FUT, PX_SETTLE = "fut", "settle"
PX_OPTIONS = [{"label": "Futures spread", "value": PX_FUT},
              {"label": "Settle", "value": PX_SETTLE}]
PX_DEFAULT = [PX_FUT]                     # Peter: "default is spread future contracts"
PX_TABS = {"corridor", "lookup", "bmap", "cal", "yc", "cm"}   # tabs that show auction prices
FUT_NOTE = ("futures spread: EEX futures price of the importing country minus the exporting "
            "one, same product (month / quarter / year, base or peak), settlement on the day "
            "the auction closed. * = last traded price that day (EEX settlements only stored "
            "from Aug 2026). Italy: PUN futures. None for GB, Baltics etc. (no EUR future) "
            "and for peak where a country has no peak future.")


def px_show(value) -> str:
    """The tick box's value as a hashable key: "fut", "settle",
    "fut,settle" or "" (nothing ticked). None (no tick box) = the default."""
    if value is None:
        value = PX_DEFAULT
    return ",".join(k for k in (PX_FUT, PX_SETTLE) if k in value)


def _px_keys(show: str) -> list:
    return [k for k in (show or "").split(",") if k]


def px_inside(show: str, fut=None, settle=None) -> str:
    """What goes in the brackets: "4.10", "4.10*", "4.10 / 5.12", "- / 5.12"
    or "" (nothing known). fut = (spread, last_traded) or None."""
    parts = []
    for k in _px_keys(show):
        if k == PX_FUT:
            ok = fut is not None and fut[0] is not None and not pd.isna(fut[0])
            parts.append(f"{fut[0]:.2f}" + ("*" if fut[1] else "") if ok else None)
        elif k == PX_SETTLE:
            ok = settle is not None and not pd.isna(settle)
            parts.append(f"{settle:.2f}" if ok else None)
    if not any(parts):
        return ""
    return " / ".join(p or "-" for p in parts)


def px_text(price, show: str, fut=None, settle=None) -> str:
    """ "4.63 (4.10)" - auction price, then the tick box's pick in brackets."""
    if price is None or pd.isna(price):
        return ""
    inside = px_inside(show, fut, settle)
    return f"{price:.2f} ({inside})" if inside else f"{price:.2f}"


def px_head(show: str) -> str:
    """ "(futures spread)", "(settle)", "(futures spread / settle)" or ""."""
    names = {PX_FUT: "futures spread", PX_SETTLE: "settle"}
    keys = _px_keys(show)
    return f"({' / '.join(names[k] for k in keys)})" if keys else ""


def px_label(prefix: str, show: str, unit: str = ", EUR/MW") -> str:
    head = px_head(show)
    return f"{prefix} {head}{unit}" if head else f"{prefix}{unit}"


def px_note(show: str) -> str:
    keys = _px_keys(show)
    settle_text = SETTLE_NOTE.split("= settle: ", 1)[-1]
    if keys == [PX_FUT]:
        return "(...) = " + FUT_NOTE
    if keys == [PX_SETTLE]:
        return SETTLE_NOTE
    if keys:
        return ("(futures spread / settle). " + FUT_NOTE[0].upper() + FUT_NOTE[1:]
                + " Settle: " + settle_text)
    return ""


def px_notes(show: str) -> list:
    text = px_note(show)
    return [note(text)] if text else []


def bracket_picker():
    """The tick box (here and in the LTCAP tool, under the tabs)."""
    return html.Div(id="px-box", className="pxbar", children=[
        html.Span("In brackets after auction prices:", className="pxlabel"),
        dcc.Checklist(id="px-show", options=PX_OPTIONS, value=list(PX_DEFAULT), inline=True,
                      className="seg", persistence=True, persistence_type="local"),
    ])


def _auction_price_details(identifications, show: str = PX_FUT) -> dict:
    """{auction: (price, note, settle, fut)} from auction_results (EUR/MW),
    the day-ahead prices (settle, see SETTLE_NOTE) and the EEX futures (fut =
    (spread, last_traded), see FUT_NOTE). Settle / fut are only worked out
    when the tick box shows them (`show`, see px_show()), else None. price
    is None when there is no single price: no result published yet (note
    "no result yet"), or a bundled CEEXD auction, which has one price per
    border (note lists the borders that got capacity, with their brackets:
    "CZ-DE(50Hz) 1.38 (0.90) · DE(50Hz)-CZ 5.15 (6.02)"). A peak auction
    gets the note "peak" and a peak-hours settle / peak futures."""
    want = _px_keys(show)
    ids = list(dict.fromkeys(identifications))
    rows, periods = [], {}
    conn = _connect()
    try:
        for i in range(0, len(ids), 900):  # SQLite's limit on "?" per query
            chunk = ids[i:i + 900]
            marks = ",".join("?" * len(chunk))
            rows += conn.execute(
                "SELECT identification, corridor_code, product_identification, "
                "allocated_capacity, auction_price FROM auction_results "
                f"WHERE identification IN ({marks})", chunk).fetchall()
            for ident, corridor, start, stop, horizon, gate in conn.execute(
                    "SELECT identification, corridor_code, market_period_start, "
                    "market_period_stop, horizon_name, bid_gate_closure "
                    f"FROM auctions WHERE identification IN ({marks})", chunk):
                periods[ident] = (corridor, start, stop, horizon, gate)
    finally:
        conn.close()
    by_auction: dict = {}
    for ident, corridor, product, allocated, price in rows:
        by_auction.setdefault(ident, []).append((corridor, product or "", allocated, price))
    out = {}
    for ident in ids:
        results = by_auction.get(ident)
        corridor, start, stop, horizon, gate = periods.get(ident, (None,) * 5)
        if not results:
            out[ident] = (None, "no result yet", None, None)
            continue
        peak = any("PEAK" in r[1].upper() for r in results)

        def brackets(code):
            return (settle_for(code, start, stop, peak) if PX_SETTLE in want else None,
                    futures_spread_for(code, horizon, start, stop, gate, peak)
                    if PX_FUT in want else None)

        notes, settle, fut = [], None, None
        if len(results) == 1:
            price = results[0][3]
            if price is None:
                notes.append("no result yet")
            elif corridor:
                settle, fut = brackets(corridor)
        else:
            price = None
            parts = []
            for c, _, a, p in results:
                if p is not None and (a or 0) > 0:
                    s, f = brackets(c)
                    parts.append(f"{display_code(c)} {px_text(p, show, f, s)}")
            notes.append(" · ".join(parts) if parts else "all borders 0.00")
        if peak:
            notes.append("peak")
        out[ident] = (price, ", ".join(notes), settle, fut)
    return out


def _auction_price_texts(identifications: tuple, show: str = PX_FUT) -> dict:
    """{auction: its price as text} for the Buyer lookup map table:
    "12.34 (10.50)" (tick box's pick in brackets), "12.34 (10.50) peak",
    the CEEXD border list, or "-" (no result yet)."""
    out = {}
    for ident, (price, note_, settle, fut) in _auction_price_details(identifications, show).items():
        if price is not None:
            out[ident] = px_text(price, show, fut, settle) + (" peak" if note_ == "peak" else "")
        elif note_ == "no result yet":
            out[ident] = "-"
        else:
            out[ident] = note_.replace(", peak", " peak")
    return out


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
def _all_auctions_by_delivery_month() -> pd.DataFrame:
    """Every non-excluded-horizon auction, one row per calendar month it
    DELIVERS in (_expand_to_delivery_months()) - the universe the Buyer
    lookup map's Month picker is built from. Not buyer-filtered and not
    the same thing as load_available_years()/load_available_year_months()
    (those stay bid_gate_closure-based, for Top Buyers - see
    load_buyer_map_data()'s docstring for why the two tabs deliberately
    use different date columns)."""
    frag, params = _horizon_exclusion()
    conn = _connect()
    df = pd.read_sql_query(
        f"SELECT identification, corridor_code, horizon_name, market_period_start, "
        f"market_period_stop FROM auctions WHERE {frag}",
        conn,
        params=params,
    )
    conn.close()
    return _expand_to_delivery_months(df)


@cached
def load_delivery_months() -> list[str]:
    expanded = _all_auctions_by_delivery_month()
    return sorted({f"{y}-{m:02d}" for y, m in zip(expanded["year"], expanded["month"])})


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def delivery_label(horizon: str, start: pd.Timestamp, stop: pd.Timestamp) -> str:
    """Trader-style name of a delivery period (start/stop already in CET,
    stop exclusive): Cal-27, Q1-27, Sum-27 / Win-26, Nov-26, or the dates
    for anything shorter ("03-04 Oct")."""
    if pd.isna(start) or pd.isna(stop):
        return "-"
    end = stop - pd.Timedelta(seconds=1)
    yy = f"{start.year % 100:02d}"
    if horizon == "Yearly":
        return f"Cal-{yy}"
    if horizon == "Quarterly":
        return f"Q{(start.month - 1) // 3 + 1}-{yy}"
    if horizon == "Monthly":
        return f"{_MONTH_ABBR[start.month - 1]}-{yy}"
    if horizon == "Seasonal" and start.month in (4, 10):
        return f"{'Sum' if start.month == 4 else 'Win'}-{yy}"
    if start.date() == end.date():
        return f"{start.day:02d} {_MONTH_ABBR[start.month - 1]}"
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.day:02d}-{end.day:02d} {_MONTH_ABBR[start.month - 1]}"
    return (f"{start.day:02d} {_MONTH_ABBR[start.month - 1]} - "
            f"{end.day:02d} {_MONTH_ABBR[end.month - 1]}")


@cached
def load_auction_calendar(first_day: str, show: str = "fut") -> pd.DataFrame:
    """Every non-daily auction (dashboard-wide exclusions) whose bid gate
    CLOSES on or after `first_day` ("2026-10-01", CET), one row per
    auction slot: same close time, open time, tenor and delivery period ->
    one row, with all its corridors. Columns: closes, opens (CET
    Timestamps), horizon_name, delivery, corridors (list), cancelled (list
    of the corridors JAO marked cancelled), prices / price_notes (lists in
    the same order as corridors - _auction_price_details(); None / "no
    result yet" until JAO publishes the result). The Auction calendar tab
    (Peter, 2026-10-13: "make a new tab with the auction calendar"; prices
    2026-10-18: "add the auction price after status")."""
    frag, params = _horizon_exclusion()
    since = pd.Timestamp(first_day, tz="CET").tz_convert("UTC")
    conn = _connect()
    df = pd.read_sql_query(
        f"""SELECT identification, corridor_code, horizon_name, cancelled, bid_gate_opening,
                   bid_gate_closure, market_period_start, market_period_stop
            FROM auctions
            WHERE {frag} AND bid_gate_closure >= ?""",
        conn, params=params + (since.strftime("%Y-%m-%dT%H:%M"),))
    conn.close()
    cols = ["closes", "opens", "horizon_name", "delivery", "corridors", "cancelled",
            "prices", "price_notes", "settles", "futs"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["closes"] = pd.to_datetime(df["bid_gate_closure"], utc=True).dt.tz_convert("CET")
    df["opens"] = pd.to_datetime(df["bid_gate_opening"], utc=True).dt.tz_convert("CET")
    df = df[df["closes"] >= since.tz_convert("CET")]
    start = pd.to_datetime(df["market_period_start"], utc=True).dt.tz_convert("CET")
    stop = pd.to_datetime(df["market_period_stop"], utc=True).dt.tz_convert("CET")
    df["delivery"] = [delivery_label(h, a, b) for h, a, b in zip(df["horizon_name"], start, stop)]
    df["dstart"] = start
    df = df.sort_values("corridor_code")
    df["cancel_code"] = df["corridor_code"].where(df["cancelled"].fillna(0).astype(int) == 1)
    details = _auction_price_details(tuple(df["identification"]), show)
    df["price"] = [details[i][0] for i in df["identification"]]
    df["price_note"] = [details[i][1] for i in df["identification"]]
    df["settle"] = [details[i][2] for i in df["identification"]]
    df["fut"] = [details[i][3] for i in df["identification"]]
    out = (df.groupby(["closes", "opens", "horizon_name", "delivery", "dstart"], dropna=False)
             .agg(corridors=("corridor_code", list),
                  cancelled=("cancel_code", lambda s: [c for c in s if isinstance(c, str)]),
                  prices=("price", list), price_notes=("price_note", list),
                  settles=("settle", list), futs=("fut", list))
             .reset_index())
    rank = {t: i for i, t in enumerate(TENOR_ORDER)}
    out["_t"] = out["horizon_name"].map(rank).fillna(len(rank))
    out = out.sort_values(["closes", "_t", "dstart"]).reset_index(drop=True)
    return out[cols]


# "Update database" button (Peter, 2026-10-13: "add a button that if pressed
# updates the database"). Runs `jao_scraper.py --upcoming` in the background
# - the quick refresh of upcoming auctions, ~8 min at JAO's pace - and shows
# its progress from the status file every scraper run writes next to the
# database (jao_update_status.json, nightly runs included). Only shown where
# the scraper and its API key sit next to this file, i.e. on Peter's PC -
# never in the online copy.
SCRAPER_PATH = Path(__file__).resolve().parent / "jao_scraper.py"
CAN_UPDATE = SCRAPER_PATH.exists() and (SCRAPER_PATH.parent / "api_key.txt").exists()
UPDATE_STATUS_FILE = "jao_update_status.json"
UPDATE_STALE_MINUTES = 10        # "running" but silent this long = it died
UPDATE_LOGS_KEPT = 20
_update_lock = threading.Lock()
_update_proc = None              # the run this dashboard started, if any


def _scraper_python() -> str:
    """The scraper's own venv (what the nightly run uses), else this Python."""
    for p in (SCRAPER_PATH.parent / ".venv" / "Scripts" / "python.exe",
              SCRAPER_PATH.parent / ".venv" / "bin" / "python"):
        if p.exists():
            return str(p)
    return sys.executable


def read_update_status() -> dict | None:
    try:
        return json.loads(DB_PATH.with_name(UPDATE_STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _status_fresh(st: dict | None) -> bool:
    """A run that says "running" and wrote within UPDATE_STALE_MINUTES."""
    if not st or st.get("state") != "running":
        return False
    try:
        last = pd.Timestamp(st.get("updated") or st.get("started"))
    except (TypeError, ValueError):
        return False
    return pd.Timestamp.now(tz="UTC") - last < pd.Timedelta(minutes=UPDATE_STALE_MINUTES)


def update_is_running() -> bool:
    with _update_lock:
        own = _update_proc is not None and _update_proc.poll() is None
    return own or _status_fresh(read_update_status())


def start_update() -> bool:
    """Start `jao_scraper.py --upcoming` in the background, output to
    logs/update_button_<time>.log. False if a run is already going (this
    button's, or the nightly one)."""
    global _update_proc
    with _update_lock:
        if _update_proc is not None and _update_proc.poll() is None:
            return False
        if _status_fresh(read_update_status()):
            return False
        log_dir = SCRAPER_PATH.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        old = sorted(log_dir.glob("update_button_*.log"))
        for f in old[:max(0, len(old) - UPDATE_LOGS_KEPT + 1)]:
            try:
                f.unlink()
            except OSError:
                pass
        log = log_dir / f"update_button_{pd.Timestamp.now(tz='CET'):%Y-%m-%d_%H%M%S}.log"
        flags = 0
        if os.name == "nt":       # no console window, not killed with the dashboard's
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        with open(log, "w", encoding="utf-8") as fh:
            _update_proc = subprocess.Popen(
                [_scraper_python(), "-u", str(SCRAPER_PATH), "--upcoming", "--db", str(DB_PATH)],
                cwd=str(SCRAPER_PATH.parent), stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=flags)
        return True


def _cet_hhmm(iso: str | None) -> str:
    try:
        t = pd.Timestamp(iso).tz_convert("CET")
    except (TypeError, ValueError):
        return "?"
    today = pd.Timestamp.now(tz="CET").normalize()
    return f"{t:%H:%M}" if t.normalize() == today else f"{t.day} {_MONTH_ABBR[t.month - 1]} {t:%H:%M}"


def update_status_text() -> str:
    st = read_update_status()
    if st is None:
        return "No update run recorded yet."
    who = "Nightly update" if st.get("mode") == "full" else "Update"
    state = st.get("state")
    if state == "running" and _status_fresh(st):
        done, total = st.get("done", 0), st.get("total", 0)
        pct = f" ({100 * done // total}%)" if total else ""
        return f"{who} running since {_cet_hhmm(st.get('started'))}: {done}/{total}{pct}."
    if state == "running":
        return (f"{who} from {_cet_hhmm(st.get('started'))} stopped without finishing "
                f"({st.get('done', 0)}/{st.get('total', 0)}).")
    when = _cet_hhmm(st.get("updated"))
    if state == "done":
        return f"Last update {when} ({who.lower()}): {st.get('message', '')}."
    return f"{who} at {when} {state}: {st.get('message', '')}"


@cached
def load_last_fetch() -> str | None:
    """When the scraper last saved auctions (CET, "13 Oct 04:14"), or None."""
    conn = _connect()
    row = conn.execute("SELECT MAX(fetched_at) FROM auctions").fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    t = pd.Timestamp(row[0]).tz_convert("CET")
    return f"{t.day} {_MONTH_ABBR[t.month - 1]} {t:%H:%M}"


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
            a.market_period_start AS raw_start,
            a.market_period_stop AS raw_stop,
            a.bid_gate_closure AS gate,
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
    text_cols = {i for i in ids if df[i].dtype == object}   # e.g. "4.63 (5.12)"
    for i in text_cols:
        df[i] = df[i].fillna("-").replace("", "-")
    columns = [{"name": first, "id": "month", "type": "text"}] + [
        ({"name": h, "id": i, "type": "text"} if i in text_cols else
         {"name": h, "id": i, "type": "numeric", "format": fixed(decimals, "-")})
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


def _calendar_price_cell(r, now: pd.Timestamp, show: str = "fut") -> str:
    """Auction calendar, price column (Peter, 2026-10-18: "add the auction
    price after status"): each corridor's clearing price, "AT-DE 1.32 ·
    DE-AT 24.39", once JAO has published the result. Empty while the slot
    is upcoming / open; "result pending" once closed but not published;
    cancelled corridors left out. A bundle (CEEXD) shows its border prices."""
    if now < r.closes:
        return ""
    esc = html_escape
    parts = []
    for code, price, note, settle, fut in zip(r.corridors, r.prices, r.price_notes, r.settles,
                                              r.futs):
        if code in r.cancelled:
            continue
        peak = " peak" if "peak" in (note or "") else ""
        if price is not None and not pd.isna(price):
            # Brackets: the tick box's pick (Peter, 2026-10-26).
            parts.append(f'<span class="code pc">{esc(code)}</span> '
                         f'{esc(px_text(price, show, fut, settle))}{peak}')
        elif note and not note.startswith("no result"):
            parts.append(f'<span class="code pc">{esc(code)}</span> {esc(note)}')
    if parts:
        return " · ".join(parts)
    if all(c in r.cancelled for c in r.corridors):
        return ""
    return '<span class="cnt">result pending</span>'


def render_calendar_html(cal: pd.DataFrame, now: pd.Timestamp, show: str = "fut") -> str:
    """The Auction calendar as one HTML table (one text, not one Dash
    component per cell - see the dashboard standard), painted by CAL_PAINT_JS.
    A header row per day, then that day's auction slots. Days before today
    and slots that already closed are greyed. The first day from today on
    carries data-anchor="today" - the painter scrolls there, so the tab opens
    on today with the rest of the month above it."""
    esc = html_escape
    today = now.normalize()
    out = ['<table class="cal"><thead><tr><th>Closes</th><th>Opens</th><th>Tenor</th>'
           '<th>Delivery</th><th>Corridors</th><th>Status</th>'
           '<th>Auction price (settle), EUR/MW</th></tr></thead><tbody>']
    anchored = False
    for day, rows in cal.groupby(cal["closes"].dt.normalize(), sort=True):
        cls = ["day"]
        if day < today:
            cls.append("past")
        if day == today:
            cls.append("today")
        anchor = ""
        if not anchored and day >= today:
            anchor, anchored = ' data-anchor="today"', True
        label = f"{day:%A} {day.day} {_MONTH_ABBR[day.month - 1]} {day.year}"
        if day == today:
            label += " - today"
        out.append(f'<tr class="{" ".join(cls)}"{anchor}><th colspan="7">{esc(label)}</th></tr>')
        for r in rows.itertuples():
            live = [c for c in r.corridors if c not in r.cancelled]
            if not live:
                status = '<span class="st st-cancel">Cancelled</span>'
            elif pd.notna(r.opens) and now < r.opens:
                status = '<span class="st st-up">Upcoming</span>'
            elif now < r.closes:
                status = '<span class="st st-open">Open now</span>'
            else:
                status = '<span class="st st-closed">Closed</span>'
            opens = ("-" if pd.isna(r.opens) else
                     f"{r.opens.day} {_MONTH_ABBR[r.opens.month - 1]} {r.opens:%H:%M}")
            color = TENOR_COLORS.get(r.horizon_name, INK["muted"])
            corr = ", ".join(f'<s class="code">{esc(c)}</s>' if c in r.cancelled
                             else f'<span class="code">{esc(c)}</span>' for c in r.corridors)
            row_cls = ' class="past"' if r.closes < now else ""
            out.append(
                f"<tr{row_cls}><td>{r.closes:%H:%M}</td><td>{esc(opens)}</td>"
                f'<td><span class="tdot" style="background:{color}"></span>{esc(r.horizon_name)}</td>'
                f"<td>{esc(r.delivery)}</td>"
                f'<td class="corr">{corr} <span class="cnt">({len(r.corridors)})</span></td>'
                f"<td>{status}</td>"
                f'<td class="price">{_calendar_price_cell(r, now, show)}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def render_buyer_map(counts: pd.DataFrame, split_tenors: bool) -> html.Div:
    """counts: corridor_code, horizon_name, wins (load_buyer_map_data).

    Arrows only, no lines (Peter, 2026-10-13: "loose the lines, they have
    no relevance"). Every arrow sits on the real border crossing
    (BORDER_POINTS), moved to the right-hand side of its own direction, so
    A->B and B->A land side by side on opposite sides (_flow_arrow).

    split_tenors=False ("All" picked): one arrow per direction, all in
    ALL_TENORS_COLOR. split_tenors=True (one or more tenors picked): still
    ONE arrow per direction, split lengthwise into one coloured strip per
    tenor that direction was won in (TENOR_COLORS, in TENOR_ORDER) - a
    direction won only as Yearly is all blue, Yearly + Monthly is half blue
    half orange.

    A small corridor-name label sits just outside each arrow. Several cable corridors in the same direction (FR->GB
    via IFA, IFA2, ElecLink) share one arrow; the hover lists each. CEEXD
    (a bundled multi-border auction) can't be placed - only counted in the
    caption/table."""
    borders, unmapped_wins = {}, 0
    for _, row in counts.iterrows():
        ends = corridor_endpoints(row["corridor_code"])
        if ends is None:
            unmapped_wins += int(row["wins"])
            continue
        a, b, label = ends
        tenor = row["horizon_name"] if split_tenors else "All"
        key = "-".join(sorted((a, b)))
        entry = borders.setdefault(key, {"ends": tuple(sorted((a, b))), "dirs": {}})
        per_tenor = entry["dirs"].setdefault((a, b), {})
        per_tenor.setdefault(tenor, []).append((label, int(row["wins"])))

    def unit(dx, dy):
        m = math.hypot(dx, dy) or 1.0
        return dx / m, dy / m

    def color_of(tenor):
        return TENOR_COLORS.get(tenor, ALL_TENORS_COLOR) if split_tenors else ALL_TENORS_COLOR

    fig = go.Figure()
    tenors_used = set()
    lab_lats, lab_lons, lab_text, lab_pos, lab_col = [], [], [], [], []
    for entry in borders.values():
        t1, t2 = entry["ends"]
        lat1, lon1 = GEO[t1][0], GEO[t1][1]
        lat2, lon2 = GEO[t2][0], GEO[t2][1]
        plat, plon = _border_point(t1, t2)

        # Direction t1 -> t2 AT the border crossing, in the local frame
        # (x = lon * cos(lat)): average of centroid->border and
        # border->centroid, so it still lines up where the crossing is off
        # the straight line (e.g. HU-UA).
        k = math.cos(math.radians(plat))
        leg1 = ((plon - lon1) * k, plat - lat1)
        leg2 = ((lon2 - plon) * k, lat2 - plat)
        u1, u2 = unit(*leg1), unit(*leg2)
        ux, uy = unit(u1[0] + u2[0], u1[1] + u2[1])
        aim = ARROW_AIM.get(f"{t1}-{t2}")
        if aim in (t1, t2):
            vx, vy = unit((GEO[aim][1] - plon) * k, GEO[aim][0] - plat)
            ux, uy = (vx, vy) if aim == t2 else (-vx, -vy)
        # Short legs (BE-FR's crossing is close to Belgium's centre) get a
        # shorter arrow so it doesn't run past the country dot.
        half_len = min(ARROW_HALF_LEN, 0.8 * min(math.hypot(*leg1), math.hypot(*leg2)))
        head_len = min(ARROW_HEAD_LEN, half_len * 0.9)

        for (o, d), per_tenor in entry["dirs"].items():
            sx, sy = (ux, uy) if (o, d) == (t1, t2) else (-ux, -uy)
            order = [t for t in ["All"] + TENOR_ORDER if t in per_tenor] + \
                    sorted(t for t in per_tenor if t not in TENOR_ORDER and t != "All")
            parts, (hlats, hlons) = _flow_arrow(
                plat, plon, sx, sy, half_len, head_len, ARROW_SIDE, n_parts=len(order))
            all_items, lines_ = [], []
            for tenor in order:
                items = per_tenor[tenor]
                all_items += items
                wins = sum(w for _, w in items)
                what = f"{tenor}: " if split_tenors else ""
                lines_.append(f"{what}{wins} win{'' if wins == 1 else 's'}")
                if len(items) > 1 or items[0][0] != f"{o}-{d}":
                    lines_ += [f"  {lab}: {w}" for lab, w in items]
            hover = f"<b>{o} -> {d}</b><br>" + "<br>".join(lines_)
            for tenor, (lats, lons) in zip(order, parts):
                tenors_used.add(tenor)
                fig.add_trace(go.Scattergeo(
                    lat=lats, lon=lons, mode="lines",
                    line=dict(width=ARROW_WIDTH, color=color_of(tenor)),
                    hoverinfo="text", text=hover, showlegend=False))
            # open arrowhead, in the colour of the piece it sits on
            fig.add_trace(go.Scattergeo(
                lat=hlats, lon=hlons, mode="lines",
                line=dict(width=ARROW_WIDTH, color=color_of(order[-1])),
                hoverinfo="text", text=hover, showlegend=False))

            # Name label just outside the arrow, text running away from the
            # border so the two directions' labels don't sit on each other.
            nx, ny = sy, -sx
            off = ARROW_SIDE + LABEL_GAP
            lab_lats.append(plat + ny * off)
            lab_lons.append(plon + nx * off / k)
            if nx > 0.35:
                lab_pos.append("middle right")
            elif nx < -0.35:
                lab_pos.append("middle left")
            else:
                lab_pos.append("top center" if ny > 0 else "bottom center")
            # merge the same corridor across tenors before labelling
            seen = {}
            for lab, w in all_items:
                seen[lab] = seen.get(lab, 0) + w
            lab_text.append(_arrow_label(list(seen.items()), o, d))
            lab_col.append(color_of(order[0]) if len(order) == 1 else INK["text"])

    if lab_text:
        fig.add_trace(go.Scattergeo(
            lat=lab_lats, lon=lab_lons, mode="text", text=lab_text,
            textposition=lab_pos, textfont=dict(size=9, color=lab_col),
            hoverinfo="skip", showlegend=False))

    touched = sorted({t for e in borders.values() for t in e["ends"]})
    if touched:
        fig.add_trace(go.Scattergeo(
            lat=[GEO[t][0] for t in touched], lon=[GEO[t][1] for t in touched],
            mode="markers+text", text=touched, textposition="top center",
            marker=dict(size=6, color="#ffffff", line=dict(width=1.5, color=INK["muted"])),
            textfont=dict(size=10, color=INK["muted"]),
            hovertext=[GEO[t][2] for t in touched], hoverinfo="text", showlegend=False))

    # Legend only when the colours mean something (tenors split).
    if split_tenors:
        for tenor in [t for t in TENOR_ORDER if t in tenors_used]:
            fig.add_trace(go.Scattergeo(
                lat=[None], lon=[None], mode="markers",
                marker=dict(size=11, symbol="triangle-right", color=TENOR_COLORS[tenor]),
                name=tenor, showlegend=True))

    fig.update_geos(scope="europe", resolution=50, showland=True, landcolor="#eeede7",
                    showcountries=True, countrycolor=INK["grid"], showcoastlines=True,
                    coastlinecolor=INK["grid"], showframe=False, bgcolor="rgba(0,0,0,0)",
                    lataxis_range=[33, 65], lonaxis_range=[-12, 33])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=560, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK["text"]), showlegend=split_tenors,
        legend=dict(title=dict(text="Tenor"), orientation="h", y=-0.03))

    children = [dcc.Graph(figure=fig, config={"displayModeBar": False})]
    if unmapped_wins:
        children.append(note(
            f"{unmapped_wins} win(s) via CEEXD (a bundled multi-border auction, not a "
            "single corridor - see the Corridor explorer tab) are not shown on the map, "
            "only in the table below.", "note"))
    return html.Div(children)

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
.pxbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0 2px; }}
.pxlabel {{ font-size:13px; color:{INK['muted']}; }}
.pxbar .seg label {{ padding:0 12px; }}
.updbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:2px 0 10px; }}
.updbtn {{ padding:5px 14px; border:1px solid {INK['accent']}; background:{INK['accent']}; color:#fff;
          border-radius:6px; cursor:pointer; font-size:13px; font-family:{FONT}; }}
.updbtn:disabled {{ opacity:.5; cursor:default; }}
.updstatus {{ font-size:13px; color:{INK['muted']}; }}
.calbox {{ max-height:70vh; overflow:auto; background:{INK['card']};
           border:1px solid {INK['grid']}; border-radius:6px; margin-top:8px; }}
table.cal {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.cal thead th {{ position:sticky; top:0; z-index:2; background:{INK['bg']}; text-align:left;
                     font-weight:600; padding:6px 10px; border-bottom:1px solid {INK['axis']}; }}
table.cal td {{ padding:5px 10px; border-bottom:1px solid {INK['grid']}; vertical-align:top; }}
table.cal tr.day th {{ text-align:left; padding:12px 10px 4px; font-weight:600;
                      border-bottom:1px solid {INK['grid']}; }}
table.cal tr.day.today th {{ color:{INK['accent']}; }}
table.cal tr.past td, table.cal tr.day.past th {{ color:{INK['muted']}; opacity:.75; }}
table.cal td {{ white-space:nowrap; }}
table.cal td.corr {{ white-space:normal; max-width:560px; }}
table.cal td.price {{ white-space:normal; max-width:420px; font-variant-numeric:tabular-nums; }}
table.cal .pc {{ color:{INK['muted']}; }}
table.cal .code {{ white-space:nowrap; }}
.tdot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
.st {{ display:inline-block; padding:1px 8px; border-radius:9px; font-size:12px; white-space:nowrap; }}
.st-open {{ background:#1baf7a; color:#fff; }}
.st-up {{ border:1px solid {INK['axis']}; color:{INK['text']}; }}
.st-closed {{ color:{INK['muted']}; }}
.st-cancel {{ color:#c0392b; }}
.cnt {{ color:{INK['muted']}; }}
@media (max-width:700px) {{ .ctl, .ctl.wide {{ min-width:100%; }} }}
"""

# Callbacks are recorded here and registered on an app by register_callbacks(),
# so the same tabs can run on their own (this file's `app`) AND inside the
# LTCAP tool (year_comparison_dash.py imports this file and registers them on
# its app - Peter, 2026-10-13: "merge jao auction data app and LTCAP tool").
_CALLBACKS: list = []


def _callback(*args, **kwargs):
    def record(func):
        _CALLBACKS.append(("server", args, kwargs, func))
        return func
    return record


def _clientside_callback(*args, **kwargs):
    _CALLBACKS.append(("client", args, kwargs, None))


def register_callbacks(target_app) -> None:
    """Attach every callback of these tabs to `target_app`."""
    for kind, args, kwargs, func in _CALLBACKS:
        if kind == "server":
            target_app.callback(*args, **kwargs)(func)
        else:
            target_app.clientside_callback(*args, **kwargs)


TABS = [("corridor", "Corridor explorer"), ("bymonth", "Buyers by month"),
        ("lookup", "Buyer lookup"), ("bmap", "Buyer lookup map"), ("top", "Top buyers"),
        ("cal", "Auction calendar")]
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


def header_parts():
    """Caption + "Update database" bar - above the tabs, here and in the LTCAP tool."""
    caption = (f"Note: {', '.join(sorted(EXCLUDED_DASHBOARD_HORIZONS))}-horizon auctions are "
               "excluded from the JAO tabs for now.") if EXCLUDED_DASHBOARD_HORIZONS else ""
    return [
        html.Div(caption, className="sub"),
        html.Div([
            html.Button("Update database", id="upd-btn", n_clicks=0, className="updbtn",
                        title="Re-fetch upcoming auctions from JAO now (~8 min). "
                              "The full update still runs every night at 04:00."),
            html.Span(id="upd-status", className="updstatus"),
            dcc.Interval(id="upd-tick", interval=15000, disabled=not CAN_UPDATE),
            dcc.Store(id="upd-state"),
            dcc.Store(id="db-refresh", data=0),
        ], className="updbar", style=None if CAN_UPDATE else {"display": "none"}),
    ]


HIDDEN = {"display": "none"}


def panels(first_visible: bool = True):
    """The six tab panels (Corridor explorer ... Auction calendar). Only the
    first is visible at the start, and only when these tabs come first."""
    return [
        # Tabs only switch which panel is visible; every picker keeps its value.
        html.Div(id="panel-corridor", style=None if first_visible else HIDDEN, children=[
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
        html.Div(id="panel-bmap", style={"display": "none"}, children=[
            html.Div([
                ctl("Buyer", dd("bm-buyer"), wide=True),
                ctl("Month", dd("bm-month")),
                ctl("Tenor", dcc.Checklist(
                    id="bm-tenor", options=["All"] + TENOR_ORDER, value=["All"],
                    inline=True, className="seg"), wide=True),
                dcc.Store(id="bm-tenor-prev", data=["All"]),
            ], className="controls"),
            dcc.Loading(html.Div(id="bm-body"), type="dot", delay_show=400)]),
        html.Div(id="panel-top", style={"display": "none"}, children=[
            html.Div([
                ctl("Scope", dcc.RadioItems(id="t-scope", options=["All time", "Year", "Month"],
                                            value="All time", inline=True, className="seg")),
                ctl("Year", dd("t-year"), id_="t-year-box"),
                ctl("Month", dd("t-month"), id_="t-month-box"),
            ], className="controls"),
            dcc.Loading(html.Div(id="t-body"), type="dot", delay_show=400)]),
        html.Div(id="panel-cal", style={"display": "none"}, children=[
            html.Div(id="cal-head"),
            dcc.Store(id="cal-html"),
            html.Div(id="cal-box", className="calbox")]),   # filled by CAL_PAINT_JS
    ]


def full_layout():
    return html.Div([
        html.H1("JAO auction data"),
        *header_parts(),
        dcc.Tabs(id="tabs", value="corridor", children=[
            dcc.Tab(label=label, value=value, style=TAB_STYLE, selected_style=TAB_SELECTED)
            for value, label in TABS]),
        bracket_picker(),
        *panels(),
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

# Opening any dropdown puts the cursor straight in its search box, so you can
# type at once instead of clicking "Search" first (Peter, 2026-10-13). Dash's
# own dropdown focuses the first option on open; this waits a moment and
# moves focus to the search input whenever a dropdown's list appears.
DROPDOWN_FOCUS_JS = """
new MutationObserver(function (muts) {
  muts.forEach(function (m) {
    m.addedNodes.forEach(function (n) {
      if (!(n instanceof HTMLElement)) return;
      var inp = n.matches('input.dash-dropdown-search') ? n
                : n.querySelector('input.dash-dropdown-search');
      if (inp) setTimeout(function () { inp.focus(); }, 60);
    });
  });
}).observe(document.documentElement, {childList: true, subtree: true});
"""
app.index_string = app.index_string.replace(
    "</body>", "<script>" + DROPDOWN_FOCUS_JS + "</script></body>")
app.layout = make_layout
app.validation_layout = full_layout()


# Buyer picked when a buyer tab first opens (Peter, 2026-10-13: "orsted
# should be default pick"). JAO spells it without the slash-o.
DEFAULT_BUYER = "Orsted Salg & Service A /S"


def default_buyer(buyers: list[str]) -> str | None:
    """DEFAULT_BUYER if it's in the list, else any other Orsted entity,
    else None (keep() then falls back to the first buyer)."""
    if DEFAULT_BUYER in buyers:
        return DEFAULT_BUYER
    return next((b for b in buyers if "orsted" in b.lower() or "ørsted" in b.lower()), None)


def keep(value, allowed, fallback=None):
    """The current choice if it still exists, else the given fallback,
    else the first option (what a Streamlit selectbox does)."""
    if value in allowed:
        return value
    if fallback in allowed:
        return fallback
    return allowed[0] if allowed else None


@_callback([Output(f"panel-{v}", "style") for v, _ in TABS], Input("tabs", "value"))
def switch_tab(tab):
    return [{"display": "block" if v == tab else "none"} for v, _ in TABS]


@_callback(Output("px-box", "style"), Input("tabs", "value"))
def px_box_visible(tab):
    """The bracket tick box only on tabs that show auction prices."""
    return None if tab in PX_TABS else {"display": "none"}


# ---------------------------------------------------------------------
# 1. Corridor explorer
# ---------------------------------------------------------------------
@_callback(
    Output("c-corridor", "options"), Output("c-corridor", "value"),
    Output("c-horizon", "options"), Output("c-horizon", "value"),
    Output("c-product", "options"), Output("c-product", "value"),
    Output("c-body", "children"),
    Input("c-corridor", "value"), Input("c-horizon", "value"), Input("c-product", "value"),
    Input("tabs", "value"), Input("px-show", "value"),
)
def corridor_explorer(corridor, horizon, product, tab, px_value=None):
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

    # In brackets next to every auction price: futures spread and/or settle,
    # as the tick box says (Peter, 2026-10-26) - see FUT_NOTE / SETTLE_NOTE.
    peak = "PEAK" in str(product).upper()
    show = px_show(px_value)
    want = _px_keys(show)

    def add_brackets(frame, code):
        if frame.empty:
            return frame
        frame = frame.copy()
        spans = list(zip(frame["raw_start"], frame["raw_stop"], frame["gate"]))
        settles = [settle_for(code, a, b, peak) if PX_SETTLE in want else None
                   for a, b, _ in spans]
        futs = [futures_spread_for(code, horizon, a, b, g, peak) if PX_FUT in want else None
                for a, b, g in spans]
        frame["settle"] = np.array([np.nan if v is None else v for v in settles], dtype=float)
        frame["fut"] = np.array([np.nan if f is None else f[0] for f in futs], dtype=float)
        frame["fut_last"] = np.array([np.nan if f is None else float(f[1]) for f in futs],
                                     dtype=float)
        frame["price_text"] = [px_text(p, show, f, st)
                               for p, f, st in zip(frame["auction_price"], futs, settles)]
        return frame

    df = add_brackets(df, corridor)
    opp_df = add_brackets(opp_df, opposite) if opposite else opp_df

    def cell(pivot, r, c):
        return pivot.at[r, c] if r in pivot.index and c in pivot.columns else np.nan

    def monthly(frame, col):
        if frame[col].notna().any():
            return render_monthly_pivot(frame, col)
        return pd.DataFrame(index=MONTH_ORDER)

    def price_pivot(frame):
        prices = render_monthly_pivot(frame, "auction_price")
        settles, futs, lasts = (monthly(frame, c) for c in ("settle", "fut", "fut_last"))
        out = prices.copy().astype(object)
        for r in prices.index:
            for c in prices.columns:
                f = cell(futs, r, c)
                fut = None if pd.isna(f) else (f, cell(lasts, r, c) > 0)
                out.at[r, c] = px_text(prices.at[r, c], show, fut, cell(settles, r, c)) or "-"
        return out

    price_name = px_label("Price", show)
    body = []
    latest = df.iloc[-1]
    if opp_df.empty:
        body.append(html.Div([
            kpi(px_label(f"{corridor} latest price", show), latest["price_text"] or "-"),
            kpi(f"{corridor} latest allocated (MW)", num(latest["allocated_capacity"], 0)),
            kpi("Auctions shown", str(len(df))),
        ], className="kpis"))
        if opposite:
            body.append(note(f"No {opposite} auctions saved for this horizon/product, "
                             f"so only {corridor} is shown."))
    else:
        opp_latest = opp_df.iloc[-1]
        body.append(html.Div([
            kpi(px_label(f"{corridor} latest price", show), latest["price_text"] or "-"),
            kpi(px_label(f"{opposite} latest price", show), opp_latest["price_text"] or "-"),
            kpi(f"{corridor} latest allocated (MW)", num(latest["allocated_capacity"], 0)),
            kpi(f"{opposite} latest allocated (MW)", num(opp_latest["allocated_capacity"], 0)),
        ], className="kpis"))

    if horizon == "Monthly":
        for title, col, dec in ((px_label("Price", show, "") + " by month/year (EUR/MW)",
                                 "auction_price", 2),
                                ("Allocated volume by month/year (MW)", "allocated_capacity", 0)):
            body.append(html.H3(title))
            make = price_pivot if col == "auction_price" else \
                (lambda frame, c=col: render_monthly_pivot(frame, c))
            if opp_df.empty:
                body.append(pivot_table(make(df), dec))
            else:
                body.append(html.Div([
                    html.Div([note(corridor), pivot_table(make(df), dec)]),
                    html.Div([note(opposite), pivot_table(make(opp_df), dec)]),
                ], className="pair"))

    body.append(html.H3("All auctions"))
    if opp_df.empty:
        body.append(table(
            df, order=["period", "product_hour", "offered_capacity", "requested_capacity",
                       "allocated_capacity", "price_text", "cancelled"],
            names={"period": "Period", "product_hour": "Product hour",
                   "offered_capacity": "Offered (MW)", "requested_capacity": "Requested (MW)",
                   "allocated_capacity": "Allocated (MW)",
                   "price_text": price_name, "cancelled": "Cancelled"},
            formats={"offered_capacity": 2, "requested_capacity": 2,
                     "allocated_capacity": 2}))
        body += px_notes(show)
        return out + [body]

    # Both directions, paired metric by metric. Outer join on
    # period+product_hour so a period only one direction auctioned still
    # shows up (blank on the other side).
    merged = pd.merge(
        df[["period", "product_hour", "offered_capacity", "allocated_capacity", "price_text"]],
        opp_df[["period", "product_hour", "offered_capacity", "allocated_capacity", "price_text"]],
        on=["period", "product_hour"],
        how="outer",
        suffixes=("_a", "_b"),
    ).sort_values(["period", "product_hour"]).reset_index(drop=True)

    body.append(note(f"{corridor} vs. {opposite} - each metric shown for both directions, "
                     "side by side."))
    body.append(table(
        merged,
        order=["period", "product_hour", "price_text_a", "price_text_b",
               "allocated_capacity_a", "allocated_capacity_b",
               "offered_capacity_a", "offered_capacity_b"],
        names={"period": "Period", "product_hour": "Product hour",
               "price_text_a": px_label("Price", show, f" {corridor}"),
               "price_text_b": px_label("Price", show, f" {opposite}"),
               "allocated_capacity_a": f"Allocated {corridor}",
               "allocated_capacity_b": f"Allocated {opposite}",
               "offered_capacity_a": f"Offered {corridor}",
               "offered_capacity_b": f"Offered {opposite}"},
        formats={c: 2 for c in ("allocated_capacity_a",
                                "allocated_capacity_b", "offered_capacity_a",
                                "offered_capacity_b")}))
    body += px_notes(show)
    return out + [body]


# ---------------------------------------------------------------------
# 2. Buyers by month
# ---------------------------------------------------------------------
@_callback(
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
@_callback(
    Output("l-buyer", "options"), Output("l-buyer", "value"), Output("l-body", "children"),
    Input("l-buyer", "value"), Input("tabs", "value"), Input("px-show", "value"),
)
def buyer_lookup(buyer, tab, px_value=None):
    if tab != "lookup":
        return no_update, no_update, no_update
    buyers = load_all_buyers()
    if not buyers:
        return [], None, note("No winners recorded yet - run python eic_lookup.py after a "
                              "backfill.", "info")
    buyer = keep(buyer, buyers, default_buyer(buyers))
    show = px_show(px_value)
    purchases = load_purchases_for_buyer(buyer, show)
    if purchases.empty:
        return buyers, buyer, note(f"No purchases found for {buyer}.", "info")

    body = []
    pivot = load_buyer_monthly_pivot(buyer)
    if not pivot.empty:
        body += [note(f"Auctions won by {buyer}, by month - split by year and horizon, counted "
                      "by when the auction actually closed (not the delivery period)."),
                 pivot_table(pivot, 0)]
    body.append(html.H3(f"Everything {buyer} has won ({len(purchases):,})"))
    # "Price note" only when a row needs it (peak, CEEXD bundle, no result yet).
    order = ["horizon_name", "period", "corridor_code", "price", "auction_closed"]
    if (purchases["price_note"] != "").any():
        order.insert(4, "price_note")
    body.append(table(
        purchases, order=order,
        names={"horizon_name": "Horizon", "period": "Period", "corridor_code": "Corridor",
               "price": px_label("Auction price", show), "price_note": "Price note",
               "auction_closed": "Auction closed (UTC)"},
        tooltips={"auction_closed": "When bidding actually closed and the auction cleared - "
                                    "not the delivery period.",
                  "price": "The auction's clearing price; in brackets what the tick box "
                           "picks (futures spread / settle). A bundled CEEXD auction has one "
                           "price per border - see Price note."}))
    body += px_notes(show)
    return buyers, buyer, body


# ---------------------------------------------------------------------
# 3b. Buyer lookup map
# ---------------------------------------------------------------------
@_callback(
    Output("bm-tenor", "value"), Output("bm-tenor-prev", "data"),
    Input("bm-tenor", "value"), State("bm-tenor-prev", "data"),
)
def bm_tenor_pick(value, prev):
    """Make "All" and the single tenors exclude each other (Peter,
    2026-10-13: "if year is selected it only shows the year corridors").
    Before this, clicking Yearly while All was still ticked gave
    ["All", "Yearly"] - which still meant everything, so the map didn't
    change. Now: ticking a tenor drops All; ticking All drops the tenors;
    unticking the last one goes back to All."""
    value, prev = list(value or []), list(prev or ["All"])
    added = [v for v in value if v not in prev]
    if not value or "All" in added:
        new = ["All"]
    else:
        new = [t for t in TENOR_ORDER if t in value]
        if not new:
            new = ["All"]
    if new == value:
        return no_update, new
    return new, new


@_callback(
    Output("bm-buyer", "options"), Output("bm-buyer", "value"),
    Output("bm-month", "options"), Output("bm-month", "value"),
    Output("bm-body", "children"),
    Input("bm-buyer", "value"), Input("bm-month", "value"),
    Input("bm-tenor", "value"), Input("tabs", "value"), Input("db-refresh", "data"),
    Input("px-show", "value"),
)
def buyer_lookup_map(buyer, year_month, tenor_value, tab, _refresh=None, px_value=None):
    if tab != "bmap":
        return [no_update] * 5
    buyers = load_all_buyers()
    if not buyers:
        return ([], None, no_update, no_update,
                note("No winners recorded yet - run python eic_lookup.py after a "
                     "backfill.", "info"))
    buyer = keep(buyer, buyers, default_buyer(buyers))

    year_months = load_delivery_months()
    if not year_months:
        return (buyers, buyer, [], None, note("No auctions saved yet.", "info"))
    # Peter: "default month is always current" - today's own month (CET,
    # matching the delivery-month convention used throughout this tab),
    # not just the latest month with any data at all.
    today_month = pd.Timestamp.now(tz="CET").strftime("%Y-%m")
    default_month = today_month if today_month in year_months else year_months[-1]
    year_month = keep(year_month, year_months, default_month)

    # "All" (or nothing) = every tenor, one colour. Any tenors ticked =
    # only those, one colour each - even if all five are ticked by hand.
    # Comma-joined and canonically ordered because @cached needs a
    # hashable argument, not the raw list from the Checklist.
    picked = [t for t in TENOR_ORDER if t in (tenor_value or [])]
    split = bool(picked) and "All" not in (tenor_value or [])
    if split:
        tenors_arg, tenor_clause = ",".join(picked), f", {' + '.join(picked)} only"
    else:
        tenors_arg, tenor_clause = "All", ""

    head = [buyers, buyer, year_months, year_month]
    show = px_show(px_value)
    counts = load_buyer_map_data(buyer, year_month, tenors_arg, show)

    if counts.empty:
        return head + [note(f"No corridors held by {buyer} in {year_month}"
                             f"{tenor_clause}.", "info")]

    n_corridors = counts["corridor_code"].nunique()
    n_wins = int(counts["wins"].sum())
    headline = (f"{buyer} holds {n_corridors} corridor direction(s) ({n_wins} "
                f"win(s) behind them) in {year_month}{tenor_clause} - by DELIVERY "
                "period, not when the auction closed, so a multi-year auction won "
                "long ago still counts (unlike Top buyers' Month/Year, which count "
                "by when the auction closed).")
    if split:
        how = ("One arrow per direction, on the real border crossing, "
               "coloured by tenor (legend under the map) - an arrow won in "
               "several of the picked tenors is striped in their colours.")
    else:
        how = ("All tenors in one colour - tick tenors to see only those, one "
               "colour each. One arrow per direction, on the real border "
               "crossing; two side by side means both ways.")
    body = [
        note(f"{headline} {how} Hover an arrow for its win count."),
        render_buyer_map(counts, split),
        table(counts.assign(corridor_code=counts["corridor_code"].map(display_code))
              .sort_values("corridor_code", kind="stable"),
              order=["corridor_code", "horizon_name", "win_date", "price", "wins"],
              names={"corridor_code": "Corridor", "horizon_name": "Tenor",
                     "win_date": "Win date", "price": px_label("Auction price", show),
                     "wins": "Wins"},
              page_size=None),
        note("Auction price = the auction's clearing price, one per win, in the same "
             "order as the win dates. " + px_note(show)),
    ]
    return head + [body]


# ---------------------------------------------------------------------
# 4. Top buyers
# ---------------------------------------------------------------------
@_callback(
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
# ---------------------------------------------------------------------
# 6. Auction calendar
# ---------------------------------------------------------------------
@_callback(
    Output("upd-status", "children"), Output("upd-btn", "disabled"),
    Output("upd-tick", "interval"), Output("upd-state", "data"), Output("db-refresh", "data"),
    Input("upd-btn", "n_clicks"), Input("upd-tick", "n_intervals"),
    State("upd-state", "data"), State("db-refresh", "data"),
)
def update_database(n_clicks, _tick, last_state, refresh):
    """The "Update database" button and its status line. Polls every 2 s
    while a run is going, every 15 s otherwise. When a run finishes, bumps
    db-refresh so the open calendar / map redraw with the new data."""
    if not CAN_UPDATE:
        return "", True, 60000, no_update, no_update
    prefix = ""
    if ctx.triggered_id == "upd-btn" and n_clicks:
        if not start_update():
            prefix = "Already running. "
    if update_is_running():
        text = update_status_text()
        if not text.endswith("%).") and "running" not in text:
            text = "Starting the update..."
        return prefix + text, True, 2000, "running", no_update
    bump = (refresh or 0) + 1 if last_state == "running" else no_update
    return prefix + update_status_text(), False, 15000, "idle", bump


@_callback(Output("cal-head", "children"), Output("cal-html", "data"),
              Input("tabs", "value"), Input("db-refresh", "data"), Input("px-show", "value"))
def auction_calendar(tab, _refresh, px_value=None):
    """All non-daily auctions from the 1st of this month on (Peter,
    2026-10-13: "all nondaily auctions starting with today on onwards. you
    can scroll back to earlier auctions but only for current month"). The
    box opens scrolled to today; scrolling up shows the rest of this month,
    nothing older."""
    if tab != "cal":
        return no_update, no_update
    now = pd.Timestamp.now(tz="CET")
    first_day = now.normalize().replace(day=1)
    show = px_show(px_value)
    cal = load_auction_calendar(first_day.strftime("%Y-%m-%d"), show)
    if cal.empty:
        return note("No auctions in the database from this month on.", "info"), ""
    upcoming = cal[cal["closes"] >= now.normalize()]
    n_auctions = int(upcoming["corridors"].map(len).sum())
    last = load_last_fetch()
    fetched = f" Auction list as of the last scrape ({last} CET)." if last else ""
    head = note(
        f"{n_auctions} auctions from today to {cal['closes'].max():%d %b %Y}, "
        f"grouped by when the bidding closes. Opens on today - scroll up for "
        f"earlier this month (from {first_day.day} {_MONTH_ABBR[first_day.month - 1]}). "
        f"Times CET. Daily and intraday not included. Struck-through corridors "
        f"were cancelled by JAO.{fetched}" + (f" Auction price {px_note(show)}"
                                               if px_note(show) else ""))
    return head, render_calendar_html(cal, now, show)


CAL_PAINT_JS = """
function (html_text) {
    var box = document.getElementById('cal-box');
    if (!box || typeof html_text !== 'string') { return window.dash_clientside.no_update; }
    box.innerHTML = html_text;
    var go = function () {
        var a = box.querySelector('[data-anchor="today"]');
        var head = box.querySelector('thead');
        if (a) {
            box.scrollTop += a.getBoundingClientRect().top - box.getBoundingClientRect().top
                             - (head ? head.offsetHeight : 0);
        }
    };
    requestAnimationFrame(function () { requestAnimationFrame(go); });
    return window.dash_clientside.no_update;
}
"""
_clientside_callback(CAL_PAINT_JS, Output("cal-html", "storage_type"), Input("cal-html", "data"))


register_callbacks(app)      # this file's own app (run on its own, port 8055)


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
