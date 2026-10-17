"""
price_table_dash.py - the "Day-ahead prices" app (Dash), port 8051.

Two tabs (merged 2026-10-14, Peter: "merge hourly price table and day ahead
prices"):
  - Hourly price table: one day, every country side by side (this file).
  - Day-ahead charts: up to 5 zones, day / week / month, load forecast -
    that is entsoe_dash.py (same folder), shown here as a tab. Its charts
    are only drawn once the tab is opened.

HOURLY PRICE TABLE

Same table as price_table.py (the Streamlit one), same numbers, same colours:

  - rows: the 24 hours (00:00 .. 23:00), one column per country/zone
  - then three summary rows: Base, Peak, Off-peak

  Base     = average of all hourly prices of the day
  Peak     = average of the 12 peak hours, 08:00-20:00
  Off-peak = 2 x Base - Peak

Controls (top to bottom), same as the Streamlit version:
  - Date. Starts on TOMORROW every time the page is loaded.
  - "Nordic/Baltic zones only": SYS, DK, NO, SE, FI, EE, LV, LT twice - first
    the prices, then each price minus the system price (SYS).
  - "Short list": DE_LU, FR, CH, AT, SI, HU and all Italian zones.
  - Tick box "Show Nordic prices": hides the Nordic columns in the full table.
  - Reference zone + range: every price within the reference zone's price
    +/- the range (same hour) gets a black frame.

Below the table a warning lists the zones that have no prices for the day
yet (compared with the 7 days before) and the zones with only some hours.

Colours: each cell is half its RANK among all prices on screen, half the
price itself (see COLOUR_RANK_SHARE). Red = cheapest, dark green = dearest.

Run it:
    python price_table_dash.py            -> http://127.0.0.1:8051/
    python price_table_dash.py --open     (also opens the browser)
    python price_table_dash.py --host 0.0.0.0   (phone / Tailscale)
or double-click start_price_table_dash.bat, or start it from Dash Hub.

The database is opened read-only. It is re-read when the scraper writes new
data - just change the date or reload the page.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import webbrowser
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html

# The "Day-ahead charts" tab = entsoe_dash.py from the same folder. If it
# can't be loaded, the price table still works on its own (no tabs then).
try:
    import entsoe_dash as ed
except Exception as exc:  # noqa: BLE001
    print(f"Day-ahead charts tab left out - entsoe_dash.py could not be loaded: {exc}",
          file=sys.stderr, flush=True)
    ed = None

APP_TITLE = "Day-ahead prices"
TABS = [("table", "Hourly price table"), ("charts", "Day-ahead charts")]

DB_NAME = "entsoe_data.db"
try:  # where the database lives (entsoe_db_location.txt), see entsoe_db.py
    from entsoe_db import entsoe_db_path as _edp
    _ENTSOE_DB = _edp()
except ImportError:
    _ENTSOE_DB = None
DEFAULT_PORT = 8051

PEAK_START_HOUR = 8   # inclusive
PEAK_END_HOUR = 20    # exclusive -> peak hours are 08:00 through 19:00 (12 hours)

# How much of each cell's colour comes from its RANK (0 = all from the price
# itself, 1 = all from the rank). Same value as price_table.py.
COLOUR_RANK_SHARE = 0.5

# Column order per Peter's own layout, not alphabetical. Any zone not in this
# list (e.g. the Italian sub-zones) is appended afterwards, A-Z.
PREFERRED_ZONE_ORDER = [
    "DE_LU", "AT", "CH", "FR", "BE", "NL", "ES", "PT", "PL",
    "CZ", "SK", "HU", "HR", "NORDIC_SYSTEM", "DK_1", "DK_2", "NO_1",
    "NO_2", "NO_3", "NO_4", "NO_5", "SE_1", "SE_2", "SE_3", "SE_4",
    "FI", "LV", "LT", "SI", "EE", "BG", "RS", "RO", "GR", "MK", "XK",
    "ME", "AL", "IE_SEM",
]
NORDIC_BALTIC_ORDER = [
    "SYS", "DK_1", "DK_2", "NO_1", "NO_2", "NO_3", "NO_4", "NO_5",
    "SE_1", "SE_2", "SE_3", "SE_4", "FI", "EE", "LV", "LT",
]
# Columns the "Show Nordic prices" tick box hides. TEL = Nord Pool's Telemark
# area. Peter counts the Baltics (EE, LV, LT) as Nordic here.
NORDIC_ZONES = NORDIC_BALTIC_ORDER + ["TEL"]
CORE_ZONES = ["DE_LU", "FR", "CH", "AT", "SI", "HU"]
ITALY_ORDER = [
    "IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD", "IT_CALA", "IT_SICI", "IT_SARD",
]
SUMMARY_ROWS = ("Base", "Peak", "Off-peak")

LABEL_NORDIC = ("Show Nordic/Baltic zones only, with a second block of each "
                "price minus the system price (SYS)")
LABEL_CORE = "Show only DE_LU, FR, CH, AT, SI, HU and all Italian zones"
LABEL_SWITCH = "Show Nordic prices (SYS, DK, NO, SE, FI, EE, LV, LT, TEL)"



# ---------------------------------------------------------------------
# Database (read-only, cached until the file changes)
# ---------------------------------------------------------------------
def find_db(cli_path=None):
    """--db if given, else entsoe_data.db next to this script, in
    'Claude outputs', one folder up, or in the folder you started from."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        *([_ENTSOE_DB] if _ENTSOE_DB else []),
        here / DB_NAME,
        here / "Claude outputs" / DB_NAME,
        here.parent / DB_NAME,
        Path.cwd() / DB_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _connect():
    """Read-only, so this viewer can never change or lock the scraper's
    database. Falls back to a normal connection if read-only is refused."""
    try:
        conn = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=10)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
        return conn
    except sqlite3.Error:
        return sqlite3.connect(str(DB_PATH), timeout=10)


def _db_stamp():
    """Changes whenever the scraper writes (the .db file or its -wal file)."""
    stamp = []
    for path in (DB_PATH, Path(str(DB_PATH) + "-wal")):
        try:
            st = path.stat()
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def _find_price_column(conn):
    """Which column of day_ahead_prices holds the price. Different builds of
    this project have used different names, so look instead of assuming."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(day_ahead_prices)")]
    candidates = [c for c in cols if c not in {"zone", "timestamp"}]
    if len(candidates) == 1:
        return candidates[0]
    for guess in ("price", "price_eur_mwh", "eur_mwh", "value", "price_eur"):
        if guess in cols:
            return guess
    raise RuntimeError(
        "Could not figure out which column in day_ahead_prices holds the "
        f"price. Columns found: {cols}."
    )


# Walks the zone list through the (zone, timestamp) index instead of reading
# the whole table: a fraction of a second on the 650 MB database.
_ZONES_CTE = """
    WITH RECURSIVE z(zone) AS (
        SELECT MIN(zone) FROM day_ahead_prices
        UNION ALL
        SELECT (SELECT MIN(zone) FROM day_ahead_prices WHERE zone > z.zone)
        FROM z WHERE z.zone IS NOT NULL
    )
"""

_cache = {}
_cache_lock = threading.Lock()


def _cached(kind, day, loader):
    key = (kind, day, _db_stamp())
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    value = loader()
    with _cache_lock:
        if len(_cache) > 40:
            _cache.clear()
        _cache[key] = value
    return value


def load_day(day):
    """All prices whose stored timestamp starts with this date (the stored
    timestamps start with YYYY-MM-DD, so comparing them as text with the
    plain dates gives exactly that day)."""
    def _load():
        conn = _connect()
        try:
            price_col = _find_price_column(conn)
            df = pd.read_sql_query(
                _ZONES_CTE + f"""
                SELECT d.zone AS zone, d.timestamp AS timestamp, d.{price_col} AS price
                FROM z JOIN day_ahead_prices d
                  ON d.zone = z.zone AND d.timestamp >= ? AND d.timestamp < ?
                WHERE z.zone IS NOT NULL
                """,
                conn,
                params=(day.isoformat(), (day + timedelta(days=1)).isoformat()),
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")
        df["hour"] = df["timestamp"].dt.strftime("%H:%M")
        df["hour_num"] = df["timestamp"].dt.hour
        return df
    return _cached("day", day, _load)


def load_expected_zones(day):
    """Zones that had at least one price in the 7 days BEFORE this day. A
    zone on this list with no rows on the picked day is reported as missing."""
    def _load():
        try:
            conn = _connect()
            try:
                rows = conn.execute(
                    _ZONES_CTE + """
                    SELECT zone FROM z
                    WHERE zone IS NOT NULL
                      AND EXISTS (SELECT 1 FROM day_ahead_prices d
                                  WHERE d.zone = z.zone
                                    AND d.timestamp >= ? AND d.timestamp < ?)
                    """,
                    ((day - timedelta(days=7)).isoformat(), day.isoformat()),
                ).fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 - the warning is a nice-to-have, never break the table
            return []
        return [r[0] for r in rows]
    return _cached("expected", day, _load)


# ---------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------
def build_tables(df):
    """(table, full_table): the hourly table (rows = hour, columns = zone)
    and the same with Base / Peak / Off-peak added underneath."""
    table = df.pivot_table(index="hour", columns="zone", values="price")
    available = list(table.columns)
    ordered = [z for z in PREFERRED_ZONE_ORDER if z in available]
    leftover = sorted(z for z in available if z not in PREFERRED_ZONE_ORDER)
    table = table.reindex(ordered + leftover, axis=1)
    table = table.rename(columns={"NORDIC_SYSTEM": "SYS"})
    table = table.sort_index()

    hour_lookup = df.drop_duplicates("hour").set_index("hour")["hour_num"]
    peak_hours = [h for h in table.index
                  if PEAK_START_HOUR <= hour_lookup[h] < PEAK_END_HOUR]
    base = table.mean()
    peak = table.loc[peak_hours].mean() if peak_hours else pd.Series(dtype=float)
    off_peak = 2 * base - peak
    summary = pd.DataFrame({"Base": base, "Peak": peak, "Off-peak": off_peak}).T
    return table, pd.concat([table, summary])


def is_core_zone(zone):
    return zone in CORE_ZONES or zone.startswith("IT")


def core_columns(columns):
    """CORE_ZONES in that order, then the Italian zones (the 7 market zones
    first, then any other IT* that has data, A-Z)."""
    columns = list(columns)
    italy = [z for z in ITALY_ORDER if z in columns]
    italy += sorted(z for z in columns if z.startswith("IT") and z not in ITALY_ORDER)
    return [z for z in CORE_ZONES if z in columns] + italy


def colour_map(values_table):
    """Where each cell sits on the colour scale, 0 (red) .. 1 (dark green):
    COLOUR_RANK_SHARE x its place in the price ranking over every cell in the
    table + the rest x its price between the cheapest and the dearest cell.
    Equal prices get the same number, empty cells stay empty."""
    vals = values_table.to_numpy(dtype=float)
    flat = np.sort(vals[~np.isnan(vals)])
    if flat.size == 0:
        return np.full(vals.shape, np.nan)
    left = np.searchsorted(flat, vals, side="left")
    right = np.searchsorted(flat, vals, side="right")
    rank = (left + right) / 2 / flat.size
    lo, hi = flat[0], flat[-1]
    price = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    mix = COLOUR_RANK_SHARE * rank + (1 - COLOUR_RANK_SHARE) * price
    return np.where(np.isnan(vals), np.nan, mix)


# The red - yellow - green scale ("RdYlGn"), the same one the Streamlit table
# uses. Written out here so the app needs no extra package for it.
_RDYLGN = np.array([
    (165, 0, 38), (215, 48, 39), (244, 109, 67), (253, 174, 97),
    (254, 224, 139), (255, 255, 191), (217, 239, 139), (166, 217, 106),
    (102, 189, 99), (26, 152, 80), (0, 104, 55),
], dtype=float)


def cell_colours(position):
    """(background, text colour) for a position 0..1 on the scale. Dark cells
    get light text so the number stays readable."""
    # 256 colour steps, the same steps the Streamlit table's scale has.
    step = min(int(min(max(float(position), 0.0), 1.0) * 256), 255)
    x = step / 255 * (len(_RDYLGN) - 1)
    i = min(int(x), len(_RDYLGN) - 2)
    rgb = _RDYLGN[i] + (_RDYLGN[i + 1] - _RDYLGN[i]) * (x - i)
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb / 255]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    text = "#f1f1f1" if luminance < 0.408 else "#000000"
    return "rgb({:.0f},{:.0f},{:.0f})".format(*rgb), text


# ---------------------------------------------------------------------
# Page building
# ---------------------------------------------------------------------
def _warning(children):
    return html.Div(children, className="pt-warning")


def _html_table(blocks, row_labels, full_table, ref_zone, highlight_range):
    """One HTML table from one or two blocks of columns.

    blocks: list of (group label or None, DataFrame, colour positions array
            or None). Colour positions are 0..1 per cell (NaN = no colour).
    Highlight frame (ref_zone / highlight_range) is drawn when
    highlight_range is not None.
    """
    grouped = any(label for label, _, _ in blocks)
    n_cols = sum(frame.shape[1] for _, frame, _ in blocks)

    head_rows = []
    if grouped:
        head_rows.append(html.Tr(
            [html.Th("", className="pt-corner")]
            + [html.Th(label, colSpan=frame.shape[1], className="pt-group")
               for label, frame, _ in blocks]
        ))
    head_rows.append(html.Tr(
        [html.Th("", className="pt-corner")]
        + [html.Th(col, className="pt-group-start" if grouped and j == 0 and b > 0 else None)
           for b, (_, frame, _) in enumerate(blocks)
           for j, col in enumerate(frame.columns)]
    ))

    body = []
    for r, label in enumerate(row_labels):
        is_summary = label in SUMMARY_ROWS
        ref = np.nan
        if highlight_range is not None and ref_zone in full_table.columns:
            ref = full_table.loc[label, ref_zone]  # also works if that column is hidden
        cells = [html.Th(label, className="pt-rowhead")]
        col_no = 0
        for b, (_, frame, positions) in enumerate(blocks):
            for j, col in enumerate(frame.columns):
                val = frame.iat[r, j]
                style = {}
                if positions is not None and not np.isnan(positions[r, j]):
                    style["backgroundColor"], style["color"] = cell_colours(positions[r, j])
                # Black frame: price within the reference zone's price +/- range.
                if (highlight_range is not None and col != ref_zone
                        and pd.notna(val) and pd.notna(ref)
                        and abs(round(val, 2) - round(ref, 2)) <= highlight_range + 1e-9):
                    style["border"] = "3px solid black"
                # Fat frame around the Base / Peak / Off-peak block.
                if is_summary:
                    if label == "Base":
                        style["borderTop"] = "4px solid black"
                    if label == "Off-peak":
                        style["borderBottom"] = "4px solid black"
                    if col_no == 0:
                        style["borderLeft"] = "4px solid black"
                    if col_no == n_cols - 1:
                        style["borderRight"] = "4px solid black"
                cls = "pt-group-start" if grouped and j == 0 and b > 0 else None
                cells.append(html.Td("" if pd.isna(val) else f"{val:.2f}",
                                     style=style, className=cls))
                col_no += 1
        body.append(html.Tr(cells, className="pt-summary" if is_summary else None))

    return html.Div(
        html.Table([html.Thead(head_rows), html.Tbody(body)], className="pt-table"),
        className="pt-table-wrap",
    )


def render(day, nordic_only, core_only, show_nordic_columns, ref_zone, highlight_range):
    """Everything below the controls: (warnings, table, caption)."""
    caption = (
        f"Peak = average price {PEAK_START_HOUR:02d}:00-{PEAK_END_HOUR:02d}:00. "
        "Off-peak = 2 x Base - Peak. Prices in EUR/MWh. "
        "Colours: half by rank, half by price - red = cheapest on screen, "
        "dark green = dearest."
    )
    if nordic_only:
        caption += (" 'vs SYS' = that zone's price minus the system price (SYS) "
                    "for the same hour/row.")
    try:
        updated = datetime.fromtimestamp(DB_PATH.stat().st_mtime)
        caption += f" Database last written {updated:%Y-%m-%d %H:%M}."
    except OSError:
        pass

    if not DB_PATH.is_file():
        return _warning([
            html.B("Database not found. "),
            f"Looked for {DB_PATH}. Start the app from the entsoe_scraper folder "
            "or give the file with --db.",
        ]), None, ""

    df = load_day(day)
    if df.empty:
        return _warning(
            f"No price data stored for {day.isoformat()} yet. Day-ahead prices "
            "for tomorrow are usually only published in the early afternoon "
            "(CET) - run fetch_data.py again a bit later if it's still too early."
        ), None, caption

    table, full_table = build_tables(df)
    core_only = core_only and not nordic_only
    hide_nordic = not show_nordic_columns and not nordic_only and not core_only

    notes = []
    blocks = None
    if nordic_only:
        available_nb = [z for z in NORDIC_BALTIC_ORDER if z in full_table.columns]
        if not available_nb:
            notes.append(_warning(
                "None of the Nordic/Baltic zones (SYS, DK, NO, SE, FI, EE, LV, "
                "LT) are in the data for this date."))
        else:
            price_block = full_table[available_nb]
            if "SYS" in price_block.columns:
                spread_block = price_block.sub(price_block["SYS"], axis=0)
            else:
                spread_block = price_block.copy()
                notes.append(html.Div(
                    "SYS wasn't found in the data for this date, so the second "
                    "block below shows raw prices instead of a spread to SYS.",
                    className="pt-caption"))
            # Price block: the usual colours (padded 0.1 at both ends, like
            # the Streamlit table). "vs SYS" block: centred on zero, so the
            # system price itself is the middle colour.
            price_pos = (colour_map(price_block) + 0.1) / 1.2
            spread_vals = spread_block.to_numpy(dtype=float)
            finite = spread_vals[~np.isnan(spread_vals)]
            extreme = float(np.max(np.abs(finite))) if finite.size else 0.0
            spread_pos = (spread_vals + extreme) / (2 * extreme) if extreme > 0 else None
            blocks = [("Price", price_block, price_pos),
                      ("vs SYS", spread_block, spread_pos)]
    else:
        if core_only:
            cols = core_columns(full_table.columns)
            if not cols:
                notes.append(_warning(
                    "None of DE_LU, FR, CH, AT, SI, HU or the Italian zones are "
                    "in the data for this date."))
        elif hide_nordic:
            cols = [c for c in full_table.columns if c not in NORDIC_ZONES]
        else:
            cols = list(full_table.columns)
        if cols:
            display = full_table[cols]
            blocks = [(None, display, (colour_map(display) + 0.1) / 1.2)]

    # Which zones are missing for this day? Only for the zones of the view
    # that is switched on.
    if nordic_only:
        relevant = lambda z: z in NORDIC_BALTIC_ORDER  # noqa: E731
    elif core_only:
        relevant = is_core_zone
    elif hide_nordic:
        relevant = lambda z: z not in NORDIC_ZONES  # noqa: E731
    else:
        relevant = lambda z: True  # noqa: E731

    def _label(zone):
        return "SYS" if zone == "NORDIC_SYSTEM" else zone

    n_hours = len(table.index)
    no_prices = sorted({_label(z) for z in load_expected_zones(day)} - set(table.columns))
    no_prices += [z for z in table.columns if table[z].isna().all()]
    only_some = [
        f"{z} ({int(table[z].notna().sum())} of {n_hours} hours)"
        for z in table.columns if 0 < table[z].notna().sum() < n_hours
    ]
    no_prices = [z for z in no_prices if relevant(z)]
    only_some = [t for t in only_some if relevant(t.split(" ")[0])]
    if no_prices or only_some:
        lines = []
        if no_prices:
            lines.append(html.P([
                html.B(f"Missing for {day.isoformat()} - no prices for "
                       f"{len(no_prices)} zone(s): "),
                ", ".join(no_prices)]))
        if only_some:
            lines.append(html.P([html.B("Only some hours: "), ", ".join(only_some)]))
        notes.append(_warning(lines))

    table_div = None
    if blocks:
        table_div = _html_table(
            blocks, list(full_table.index), full_table, ref_zone,
            None if nordic_only else highlight_range)
    return notes, table_div, caption


# The table goes to the page as one HTML text (see PAINT_JS below), not as
# Dash components: a full day is ~1,300 cells, and Dash takes seconds to
# draw that many separate components. As plain HTML it is instant.
_HTML_ATTRIBUTES = {"className": "class", "colSpan": "colspan"}


def to_html(node) -> str:
    """HTML text of a tree of dash.html components (or a list of them)."""
    if node is None:
        return ""
    if isinstance(node, (list, tuple)):
        return "".join(to_html(child) for child in node)
    if isinstance(node, (str, int, float)):
        return escape(str(node))
    tag = type(node).__name__.lower()
    attributes = ""
    for prop, name in _HTML_ATTRIBUTES.items():
        value = getattr(node, prop, None)
        if value is not None:
            attributes += f' {name}="{escape(str(value), quote=True)}"'
    style = getattr(node, "style", None)
    if style:
        css = "; ".join(
            re.sub(r"[A-Z]", lambda m: "-" + m.group().lower(), key) + f": {value}"
            for key, value in style.items()
        )
        attributes += f' style="{escape(css, quote=True)}"'
    return f"<{tag}{attributes}>{to_html(getattr(node, 'children', None))}</{tag}>"


def zone_choices(day):
    """Zones offered as reference zone for the picked day (table order)."""
    if not DB_PATH.is_file():
        return []
    df = load_day(day)
    if df.empty:
        return []
    return list(build_tables(df)[1].columns)


def _parse_day(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _parse_range(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number >= 0 and number == number else 0.0


def _check(label, disabled=False):
    return [{"label": label, "value": "on", "disabled": disabled}]


def serve_layout():
    """Built on every page load, so the date always starts on tomorrow."""
    tomorrow = date.today() + timedelta(days=1)
    choices = zone_choices(tomorrow)
    ref = "DE_LU" if "DE_LU" in choices or not choices else choices[0]
    charts = ed.make_layout(title=False) if ed is not None else None
    return _page(tomorrow, choices, ref, charts)


def validation_layout():
    """Every component of both tabs, without reading the database - so Dash
    knows all ids even when the charts tab shows a 'no data' note instead."""
    tomorrow = date.today() + timedelta(days=1)
    charts = ed.full_layout(title=False) if ed is not None else None
    return _page(tomorrow, [], "DE_LU", charts)


def _page(tomorrow, choices, ref, charts):
    table = _table_panel(tomorrow, choices, ref, headline=charts is None)
    if charts is None:
        return table
    return html.Div([
        html.Div(className="dp-head", children=dcc.Tabs(
            id="dp-tabs", value="table", className="dp-tabs",
            children=[dcc.Tab(label=label, value=value) for value, label in TABS])),
        table,
        html.Div(id="dp-panel-charts", style={"display": "none"}, children=charts),
    ])


def _table_panel(tomorrow, choices, ref, headline):
    """The price table. headline=False when the tab bar above already names it."""
    return html.Div(id="dp-panel-table", className="pt-page", children=[
        # Everything except the table sits in two compact rows at the top.
        html.Div(className="pt-top", children=[
            *([html.H1("Hourly prices")] if headline else []),
            html.Div(className="pt-date-box", children=dcc.DatePickerSingle(
                id="pt-date", date=tomorrow.isoformat(), display_format="YYYY-MM-DD",
                first_day_of_week=1, clearable=False)),
            html.Div(id="pt-highlight", className="pt-hl-row", children=[
                html.Span("Highlight prices within"),
                dcc.Dropdown(id="pt-ref", options=choices or ["DE_LU"], value=ref,
                             clearable=False, searchable=True, className="pt-ref"),
                html.Span("+/- (EUR/MWh)"),
                html.Div(className="pt-stepper", children=[
                    html.Button("\u2212", id="pt-minus", title="10 cent less", className="pt-step"),
                    dcc.Input(id="pt-range", type="number", min=0, step=0.01, value=0,
                              debounce=False, className="pt-range"),
                    html.Button("+", id="pt-plus", title="10 cent more", className="pt-step"),
                ]),
            ]),
        ]),
        html.Div(className="pt-checks", children=[
            dcc.Checklist(id="pt-nordic", options=_check(LABEL_NORDIC), value=[],
                          className="pt-check"),
            dcc.Checklist(id="pt-core", options=_check(LABEL_CORE), value=[],
                          className="pt-check"),
            dcc.Checklist(id="pt-switch", options=_check(LABEL_SWITCH), value=["on"],
                          className="pt-check"),
        ]),
        dcc.Loading(type="dot", delay_show=400, children=[
            dcc.Store(id="pt-html"),   # the warnings + table as HTML text
            html.Div(id="pt-body"),    # ... written in here by PAINT_JS
        ]),
        html.Div(id="pt-caption", className="pt-caption"),
    ])


CSS = """
body { margin: 0; background: #ffffff; color: #1f2430;
       font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
       font-size: 15px; }
.pt-page { padding: 18px 24px 40px; }
.pt-top { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 22px; }
.pt-page h1 { font-size: 24px; margin: 0; font-weight: 700; }
.pt-date-box { width: 11rem; }
.pt-checks { display: flex; flex-wrap: wrap; gap: 2px 26px; margin: 8px 0 2px; }
.pt-check label { display: flex !important; align-items: center; gap: 8px; cursor: pointer; }
.pt-check input { width: 16px; height: 16px; margin: 0; flex: 0 0 auto; }
.pt-check input:disabled, .pt-check input:disabled + * { opacity: .45; cursor: not-allowed; }
.pt-ref { width: 9rem; }
.pt-hl-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; }
.pt-stepper { display: flex; align-items: stretch; }
.pt-range { width: 5.5rem; flex: 0 0 auto; box-sizing: border-box; padding: 6px 8px;
       border: 1px solid #c6cad3; border-radius: 0; border-left: 0; border-right: 0;
       font: inherit; text-align: center; -moz-appearance: textfield; }
.pt-range::-webkit-inner-spin-button, .pt-range::-webkit-outer-spin-button {
       -webkit-appearance: none; margin: 0; }
.pt-step { width: 34px; border: 1px solid #c6cad3; background: #f4f5f8; font: inherit;
       font-size: 18px; line-height: 1; cursor: pointer; color: #1f2430; }
.pt-step:first-child { border-radius: 6px 0 0 6px; }
.pt-step:last-child { border-radius: 0 6px 6px 0; }
.pt-step:hover { background: #e6e9f0; }
.pt-warning { background: #fff8db; border: 1px solid #f0dd8a; color: #6b5300;
       border-radius: 8px; padding: 10px 14px; margin: 12px 0; }
.pt-warning p { margin: 0 0 6px; }
.pt-warning p:last-child { margin-bottom: 0; }
.pt-table-wrap { overflow-x: auto; margin-top: 12px; max-width: 100%; }
.pt-table { border-collapse: collapse; width: auto; font-size: 14px;
       font-variant-numeric: tabular-nums; }
.pt-table th, .pt-table td { border: 1px solid #e3e5ea; padding: 4px 8px;
       white-space: nowrap; text-align: right; }
.pt-table thead th { font-weight: 600; color: #555b6a; background: #fafbfc; text-align: center; }
.pt-table th.pt-rowhead, .pt-table th.pt-corner { position: sticky; left: 0; z-index: 1;
       background: #fafbfc; color: #555b6a; font-weight: 500; text-align: left; }
.pt-table tr.pt-summary th.pt-rowhead { font-weight: 700; color: #1f2430; }
.pt-table thead th.pt-group { text-align: left; color: #1f2430; }
.pt-table .pt-group-start { border-left: 3px solid #8a8f9c; }
.pt-caption { color: #6b7180; font-size: 13px; margin-top: 10px; }
.dp-head { padding: 12px 24px 0; }
.dp-tabs { margin: 0; }
#dp-panel-table.pt-page { padding-top: 14px; }
#dp-panel-charts .wrap { max-width: none; margin: 0; padding: 6px 24px 40px; }
"""

DB_PATH = find_db()  # main() replaces it when --db is given

if ed is not None:
    ed.DB_PATH = DB_PATH

app = Dash(__name__, title=APP_TITLE, update_title=None)
# The charts tab's CSS first, so this file's rules (body, fonts) win.
_ALL_CSS = (ed.CSS if ed is not None else "") + CSS
app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + _ALL_CSS + "</style>")
app.layout = serve_layout
app.validation_layout = validation_layout()

if ed is not None:
    @app.callback(
        Output("dp-panel-table", "style"),
        Output("dp-panel-charts", "style"),
        Input("dp-tabs", "value"),
    )
    def show_tab(active):
        hidden = {"display": "none"}
        return ({} if active == "table" else hidden), ({} if active == "charts" else hidden)

    # The charts are only drawn while their tab is open.
    ed.register_callbacks(app, active_when=("dp-tabs", "charts"))


@app.callback(
    Output("pt-core", "options"),
    Output("pt-switch", "options"),
    Output("pt-highlight", "style"),
    Input("pt-nordic", "value"),
    Input("pt-core", "value"),
)
def lock_controls(nordic, core):
    """Short list is not offered together with the Nordic view; the third box
    does nothing in either of those two views, so it is greyed out there."""
    nordic_on = bool(nordic)
    core_on = bool(core) and not nordic_on
    return (
        _check(LABEL_CORE, disabled=nordic_on),
        _check(LABEL_SWITCH, disabled=nordic_on or core_on),
        {"display": "none"} if nordic_on else {},
    )


@app.callback(
    Output("pt-ref", "options"),
    Output("pt-ref", "value"),
    Input("pt-date", "date"),
    State("pt-ref", "value"),
)
def fill_reference_zones(day_text, current):
    """The reference dropdown lists the zones of the picked day. Your choice
    is kept when you change date, if that zone has data that day."""
    day = _parse_day(day_text)
    choices = zone_choices(day) if day else []
    if not choices:
        return [current or "DE_LU"], current or "DE_LU"
    if current in choices:
        return choices, current
    return choices, "DE_LU" if "DE_LU" in choices else choices[0]


@app.callback(
    Output("pt-html", "data"),
    Output("pt-caption", "children"),
    Input("pt-date", "date"),
    Input("pt-nordic", "value"),
    Input("pt-core", "value"),
    Input("pt-switch", "value"),
    Input("pt-ref", "value"),
    Input("pt-range", "value"),
)
def update_table(day_text, nordic, core, switch, ref_zone, range_value):
    day = _parse_day(day_text)
    ref_zone = ref_zone or "DE_LU"
    if day is None:
        return to_html(_warning("Pick a date.")), ""
    notes, table_div, caption = render(
        day, bool(nordic), bool(core), bool(switch), ref_zone, _parse_range(range_value))
    # The nightly price check (check_prices.py): one line in the caption, and
    # a warning under the table when it found something.
    check = price_check_status()
    if check:
        caption = f"{caption} Nightly price check {check['when']:%Y-%m-%d %H:%M}: {check['summary']}."
        if not check.get("ok", True):
            notes = [notes, _warning([
                html.B("The nightly price check found something: "),
                check["summary"].removeprefix("CHECK: "),
                f". Details in {PRICE_CHECK_FILE} next to the database."])]
    # The table first; the warnings (missing zones etc.) go below it.
    return to_html([table_div, notes]), caption


PRICE_CHECK_FILE = "price_check_status.json"  # written by check_prices.py (nightly)


def price_check_status():
    """The nightly price check's result (check_prices.py), or None."""
    try:
        status = json.loads((Path(DB_PATH).parent / PRICE_CHECK_FILE).read_text(encoding="utf-8"))
        status["when"] = datetime.fromisoformat(status["checked_at"])
        return status
    except (OSError, ValueError, KeyError, TypeError):
        return None


RANGE_BUTTON_STEP = 0.10  # EUR/MWh per click on the - / + buttons


@app.callback(
    Output("pt-range", "value"),
    Input("pt-minus", "n_clicks"),
    Input("pt-plus", "n_clicks"),
    State("pt-range", "value"),
    prevent_initial_call=True,
)
def step_range(_minus, _plus, current):
    """The - and + buttons change the range by 10 cent. Never below 0."""
    step = -RANGE_BUTTON_STEP if ctx.triggered_id == "pt-minus" else RANGE_BUTTON_STEP
    return max(round(_parse_range(current) + step, 2), 0)


# Runs in the browser: writes the HTML text from the Store into the page.
PAINT_JS = """
function (html_text) {
    var target = document.getElementById('pt-body');
    if (target && typeof html_text === 'string') { target.innerHTML = html_text; }
    return window.dash_clientside.no_update;
}
"""
app.clientside_callback(PAINT_JS, Output("pt-html", "storage_type"), Input("pt-html", "data"))


def main():
    global DB_PATH
    parser = argparse.ArgumentParser(description="Day-ahead prices: hourly price table + charts (Dash).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from the phone / Tailscale")
    parser.add_argument("--db", default=None, help=f"path to {DB_NAME}")
    parser.add_argument("--open", action="store_true", help="open the browser")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DB_PATH = find_db(args.db)
    if ed is not None:
        ed.DB_PATH = DB_PATH
    url = f"http://127.0.0.1:{args.port}/"
    # use_reloader: restarts the app whenever this file (or entsoe_dash.py)
    # changes, so an update shows up on the next browser refresh. Only the
    # first (watcher) process prints and opens the browser.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        if not DB_PATH.is_file():
            print(f"Warning: database not found at {DB_PATH}", file=sys.stderr, flush=True)
        print(f"{APP_TITLE} (Dash): {url}   database: {DB_PATH}", flush=True)
        if args.open:
            threading.Timer(3.5, webbrowser.open, args=(url,)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
