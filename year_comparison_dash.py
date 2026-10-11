"""
year_comparison_dash.py -- Year comparison, Dash version.

Same 5 tabs and the same numbers as year_comparison.py (the Streamlit one):

  1. Year comparison         - average price by period of year, across years,
                               for two countries: Base, Spread, Margin,
                               capture value, JAO auction price, Peak.
  2. Hours below threshold   - hours per period with a price below a value.
  3. Corridor margins        - every JAO corridor for one year: Margin,
                               Auction price or Settle, with a Winner picker.
  4. Market value            - MW sold x hours x Margin per corridor.
  5. Market value - overview - the same, added up per tenor and year.

Every calculation below the "DATA" line is the code of year_comparison.py,
unchanged. Only two things differ there: results are kept until one of the
two databases changes (instead of for 60 seconds), and both databases are
opened read-only.

Run it:
    python year_comparison_dash.py            -> http://127.0.0.1:8052/
    python year_comparison_dash.py --open     (also opens the browser)
    python year_comparison_dash.py --host 0.0.0.0   (phone / Tailscale)
or double-click start_year_comparison_dash.bat, or start it from Dash Hub.

Reads entsoe_data.db (next to this file) and jao_scraper\\jao_data.db.
"""

import argparse
import calendar
import functools
import io
import sqlite3
import sys
import threading
import webbrowser
from datetime import date
from html import escape
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html, no_update
from matplotlib import colors as mcolors

try:
    # The same zone list the scrapers and the Streamlit dashboards use.
    from fetch_data import ZONES
except Exception:  # noqa: BLE001 - e.g. entsoe-py not installed for this Python
    ZONES = [
        "AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE_LU", "DK_1",
        "DK_2", "EE", "ES", "FI", "FR", "GB", "GE", "GR", "HR", "HU", "IE_SEM",
        "IS", "IT_CALA", "IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI",
        "IT_SUD", "LT", "LU", "LV", "MD", "ME", "MK", "MT", "NL", "NO_1", "NO_2",
        "NO_3", "NO_4", "NO_5", "PL", "PT", "RO", "RS", "RU", "SE_1", "SE_2",
        "SE_3", "SE_4", "SI", "SK", "TR", "UA", "XK",
    ]

DEFAULT_PORT = 8052
DB_NAME = "entsoe_data.db"


# =====================================================================
# Databases (read-only) and the result cache
# =====================================================================
def find_db(cli_path=None):
    """--db if given, else entsoe_data.db next to this script, in
    'Claude outputs', one folder up, or in the folder you started from."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / DB_NAME,
        here / "Claude outputs" / DB_NAME,
        here.parent / DB_NAME,
        Path.cwd() / DB_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


DB_PATH = find_db()  # main() replaces it when --db is given


def _open_read_only(path, allow_plain=False):
    """A read-only connection, so this viewer can never change or lock a
    scraper's database. Raises sqlite3.OperationalError if it cannot be
    opened (the callers treat that as "no data")."""
    if not Path(path).is_file():
        raise sqlite3.OperationalError(f"database file not found: {path}")
    last_error = None
    for uri in (Path(path).as_uri() + "?mode=ro", f"file:{path}?mode=ro"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
            return conn
        except sqlite3.Error as exc:
            last_error = exc
    if allow_plain:
        return sqlite3.connect(str(path), timeout=10)
    raise sqlite3.OperationalError(str(last_error))


def _entsoe_conn():
    return _open_read_only(DB_PATH, allow_plain=True)


def _jao_conn():
    return _open_read_only(JAO_DB_FILE)


def _file_stamp(path):
    """(time, size) of a database file and of its -wal file (where SQLite
    keeps the newest writes). An empty -wal counts as "no -wal": SQLite
    creates and removes that empty file just from opening the database,
    which must not look like new data."""
    stamp = []
    for p in (Path(path), Path(str(path) + "-wal")):
        try:
            st = p.stat()
            stamp.append((st.st_mtime_ns, st.st_size) if st.st_size else None)
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def _data_stamp():
    """Changes when either database is written to, and when a new month
    starts (some tables depend on which month is the running one)."""
    today = date.today()
    return (_file_stamp(DB_PATH), _file_stamp(JAO_DB_FILE), today.year, today.month)


_cache = {}
_cache_stamp = None
_cache_lock = threading.Lock()
_key_locks = {}
CACHE_MAX_ENTRIES = 400


def _clone(value):
    """A private copy, so a caller can change what it got without changing
    what is kept (Streamlit's cache hands out copies the same way)."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return value


def cached(fn):
    """Keep a function's result until one of the two databases changes.
    Two requests for the same thing at the same moment compute it once."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _cache_stamp
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        stamp = _data_stamp()
        with _cache_lock:
            if stamp != _cache_stamp or len(_cache) > CACHE_MAX_ENTRIES:
                _cache.clear()
                _key_locks.clear()
                _cache_stamp = stamp
            key_lock = _key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with _cache_lock:
                found = key in _cache
                value = _cache.get(key)
            if not found:
                value = fn(*args, **kwargs)
                with _cache_lock:
                    if _cache_stamp == stamp:
                        _cache[key] = value
        return _clone(value)
    return wrapper


# =====================================================================
# DATA -- everything from here to the "PAGE" line is year_comparison.py's
# own code (see the note at the top of this file).
# =====================================================================
START_YEAR = 2020

# JAO auction price data lives in a SEPARATE project/database
# (jao_scraper\jao_data.db) -- a sibling folder to this one, both under
# the same parent on Peter's machine (see
# claude/jao-auctions-scraping.md, claude/year-comparison-dashboard.md).
# Resolved relative to THIS file's own location, not the current working
# directory, so it still finds jao_data.db regardless of where
# `streamlit run` was invoked from.
JAO_DB_FILE = Path(__file__).resolve().parent.parent / "jao_scraper" / "jao_data.db"

# A handful of this project's ENTSO-E zone codes don't match JAO's own
# corridor-code country tokens directly -- confirmed against JAO's real
# 103 corridor codes, not guessed: DE_LU is plain "DE" in JAO corridors,
# IE_SEM is plain "IE", the two Danish price zones are "D1"/"D2" (not
# "DK1"/"DK2" -- e.g. JAO's real "D1-D2"/"DE-D1"/"DE-D2" corridors), and
# every Italian sub-zone collapses to plain "IT" (JAO has no separate
# Italian zone corridors). Norway's and Sweden's split zones (NO_1..5,
# SE_1..4) have no JAO equivalent at all -- JAO has zero corridor
# auctions touching Norway or Sweden -- so they're deliberately left
# unmapped; a pair involving one of them just won't find a match below,
# same as any other zone pair JAO doesn't have a corridor for.
ZONE_TO_JAO_TOKEN = {
    "DE_LU": "DE",
    "IE_SEM": "IE",
    "DK_1": "D1",
    "DK_2": "D2",
    "IT_NORD": "IT", "IT_CNOR": "IT", "IT_CSUD": "IT", "IT_SUD": "IT",
    "IT_SARD": "IT", "IT_SICI": "IT", "IT_CALA": "IT",
    # The 2 German TSO-area pseudo-countries -- see GERMAN_TSO_AREAS below.
    "DE_50Hz": "DE(50Hz)",
    "DE_TenneT": "DE(TenneT)",
}

# The 3 GB<->FR interconnectors that each have their own separate JAO
# auction corridor, even though none of them has a separate ENTSO-E spot
# price -- confirmed live against the real jao_data.db (2026-09-18):
# "IF1-FR-GB"/"IF1-GB-FR" (IFA), "IF2-FR-GB"/"IF2-GB-FR" (IFA2),
# "EL1-FR-GB"/"EL1-GB-FR" (ElecLink), all with real Monthly auction data.
# Peter: "IFA, IFA2, ElecLink all have their own auction price so they
# should be treated as countries all using the GB hourly price."
# Selectable as their own "country" below purely so the 2 JAO tables can
# show each interconnector's own auction price -- every ENTSO-E-side
# table (Base/Peak price, Spread, Capture value, Export) uses plain GB's
# hourly price for all 3, since that's the only price any of them
# actually has on that side (see _entsoe_zone() below). GB itself
# stopped getting an ENTSO-E price after 15 June 2021;
# fetch_nordpool_bridge.py now also bridges it from Nord Pool's N2EX
# auction (the same GB price these 3 interconnectors settle against),
# so this isn't just pre-2021 data going forward.
INTERCONNECTOR_JAO_PREFIX = {
    "IFA": "IF1",
    "IFA2": "IF2",
    "ElecLink": "EL1",
}
INTERCONNECTOR_ENTSOE_ZONE = {name: "GB" for name in INTERCONNECTOR_JAO_PREFIX}
# 2 more pseudo-countries, same idea: JAO does not auction "CZ-DE" or
# "PL-DE" as one border, but per German TSO area -- "CZ-DE(TenneT)",
# and "CZ-DE(50Hz)" / "PL-DE(50Hz)" (the 50Hertz ones are sold inside
# the bundled "CEEXD" auction, see claude/jao-ceexd-bundle.md). Picking
# plain DE_LU against CZ or PL therefore finds no JAO corridor; picking
# DE_50Hz or DE_TenneT does. Both use plain DE_LU's hourly price for
# every ENTSO-E-side table -- there is only one German spot price.
GERMAN_TSO_AREAS = ["DE_50Hz", "DE_TenneT"]
INTERCONNECTOR_ENTSOE_ZONE.update({name: "DE_LU" for name in GERMAN_TSO_AREAS})

# What the Country 1 / Country 2 pickers actually offer: every real zone,
# plus the 3 interconnector pseudo-countries tacked on at the end (so
# _default_zone_index()'s ZONES.index(...) results, used as fallback
# indices, stay valid -- it only ever looks up real zone names).
PICKER_ZONES = ZONES + list(INTERCONNECTOR_ENTSOE_ZONE)


def _entsoe_zone(zone: str) -> str:
    """The zone to actually query entsoe_data.db for -- a plain
    passthrough for a real ENTSO-E zone, "GB" for one of the 3
    interconnector pseudo-countries above."""
    return INTERCONNECTOR_ENTSOE_ZONE.get(zone, zone)

# table name + how to turn a date into this resolution's row label.
# (Which column is the date and which is the price is figured out
# automatically -- see detect_columns() -- so it doesn't matter that
# daily/weekly/monthly aggregate tables don't all use the same column
# names.)
# Monthly listed first so it's both the first option AND the default
# selection in the resolution picker below (st.radio defaults to
# whichever key comes first); Weekly sits in the middle, Daily last.
RESOLUTIONS = {
    "Monthly": {
        "table": "monthly_prices",
        "period_label": lambda d: d.strftime("%B"),  # month name only, no number
        "year_label": lambda d: d.year,
        # Fixed row set: always exactly these 12 rows, Jan->Dec, in this
        # order -- regardless of which months happen to have data. Any
        # year (2001, non-leap) works here, it's only used for the
        # month name/number, never compared against real dates.
        "canonical_periods": [
            pd.Timestamp(2001, month, 1).strftime("%B") for month in range(1, 13)
        ],
    },
    "Weekly": {
        "table": "weekly_prices",
        "period_label": lambda d: f"W{d.isocalendar().week:02d}",
        # The ISO week-YEAR, not the plain calendar year: a week's own
        # Monday can fall in a different calendar year than most of that
        # week's days (e.g. the Monday starting ISO week 1 of 2025 is
        # 2024-12-30). Keying the "year" pivot column off plain dt.year
        # instead bucketed that week's row into the SAME cell as the real
        # W01 of the calendar year the Monday matched -- silently
        # combining 2 different weeks and losing the other year's W01
        # entirely. Confirmed live: 4 such collisions for DE_LU alone
        # (W01/2024, W01/2025, W52/2022, W52/2023).
        "year_label": lambda d: d.isocalendar().year,
    },
    "Daily": {
        "table": "daily_prices",
        "period_label": lambda d: d.strftime("%m-%d"),
        "year_label": lambda d: d.year,
    },
}

# Country picker width, in characters. Unlike the table columns above,
# Streamlit selectboxes don't take a pixel/character width directly, so
# this is applied via a small CSS override instead -- see the
# st.markdown(...) block near the bottom of this file.
COUNTRY_PICKER_DIGITS = 20

# Streamlit's own per-row and header height for st.dataframe, in pixels --
# used both to size each table's container (see _show_pivot) and, for
# Monthly, to work out exactly where the December/Q1 divider line needs
# to be drawn (see the injected script at the bottom of this file).
ROW_HEIGHT_PX = 35
HEADER_HEIGHT_PX = 35

# Monthly only: which 3 month rows each quarter row averages together --
# keyed by the same month-name labels RESOLUTIONS["Monthly"]["period_label"]
# produces, so pivot.loc[...] always finds them.
QUARTER_MONTHS = {
    "Q1": ["January", "February", "March"],
    "Q2": ["April", "May", "June"],
    "Q3": ["July", "August", "September"],
    "Q4": ["October", "November", "December"],
}


def _append_quarter_year_rows(pivot: pd.DataFrame) -> pd.DataFrame:
    """Monthly resolution only: adds Q1-Q4 and Year rows below the 12
    month rows, each column's (year's) own average over the relevant
    months. Built from the pivot's already-rounded, on-screen values (not
    some hidden extra precision) so a quarter row matches an average of
    the exact numbers displayed above it; skipna means a still-missing
    month (e.g. the current one) doesn't blank out its whole quarter/year."""
    summary_rows = {label: pivot.loc[months].mean(skipna=True) for label, months in QUARTER_MONTHS.items()}
    summary_rows["Year"] = pivot.mean(skipna=True)
    return pd.concat([pivot, pd.DataFrame(summary_rows).T]).round(2)


def detect_columns(table: str, conn: sqlite3.Connection) -> tuple[str, str]:
    """Figure out which non-'zone' column holds the date and which holds
    the average price, by looking at the actual type of one real row's
    values rather than guessing a column name -- avoids hardcoding a
    name that turns out to be wrong for a particular table (e.g. 'day'
    vs 'date'). Text -> date column, number -> price column. (Trying to
    parse the value as a date instead of checking its type doesn't work
    here: pandas happily parses a plain number like 45.0 as a timestamp
    too, so a numeric price column would get mistaken for the date one.)
    """
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    other_cols = [c for c in cols if c != "zone"]
    row = conn.execute(
        f"SELECT {', '.join(other_cols)} FROM {table} WHERE zone IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError(f"'{table}' has no rows to inspect its columns from.")

    date_col = None
    price_col = None
    for name, value in zip(other_cols, row):
        if isinstance(value, str):
            date_col = name
        elif isinstance(value, (int, float)):
            price_col = name

    if date_col is None or price_col is None:
        raise ValueError(
            f"Couldn't tell which column in '{table}' is the date and which "
            f"is the price -- columns are {cols}, sample row is {row}"
        )
    return date_col, price_col


@cached
def load_pivot(zone: str, resolution: str) -> pd.DataFrame:
    conf = RESOLUTIONS[resolution]
    conn = _entsoe_conn()
    try:
        date_col, price_col = detect_columns(conf["table"], conn)
        df = pd.read_sql(
            f"SELECT {date_col} AS period_date, {price_col} AS avg_price "
            f"FROM {conf['table']} WHERE zone = ?",
            conn,
            params=(zone,),
            parse_dates=["period_date"],
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # year_label (not plain dt.year) so Weekly's "year" column reflects the
    # ISO week-year -- see RESOLUTIONS["Weekly"]["year_label"] above.
    year_series = df["period_date"].apply(conf["year_label"])
    df = df[year_series >= START_YEAR]
    year_series = year_series[df.index]
    if df.empty:
        return pd.DataFrame()

    df["year"] = year_series
    df["period"] = df["period_date"].apply(conf["period_label"])

    pivot = df.pivot_table(
        index="period", columns="year", values="avg_price", aggfunc="mean"
    )

    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        # Monthly: always show exactly these 12 rows in this order, even
        # if a month has no data yet (e.g. the current, still-in-progress
        # month) -- rather than whatever set of months happens to appear
        # in the data.
        pivot = pivot.reindex(canonical_periods)
    else:
        pivot = pivot.sort_index()  # zero-padded labels sort correctly as text

    return pivot.round(2)


@cached
def load_spread(zone1: str, zone2: str, resolution: str) -> pd.DataFrame:
    """zone1's price minus zone2's price, period by period. Built from the
    same two pivots show_country() would show, so it lines up with them
    row-for-row and year-for-year -- pandas' subtract() aligns by index
    and column label automatically, filling NaN wherever either side is
    missing a period, rather than requiring both pivots to already share
    the exact same shape."""
    p1 = load_pivot(zone1, resolution)
    p2 = load_pivot(zone2, resolution)
    if p1.empty or p2.empty:
        return pd.DataFrame()

    spread = p1.subtract(p2)

    # Put rows back in the right order (subtract()'s union of the two
    # indexes isn't guaranteed to come out Jan->Dec / chronological) --
    # same logic load_pivot() itself uses.
    conf = RESOLUTIONS[resolution]
    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        spread = spread.reindex(canonical_periods)
    else:
        spread = spread.sort_index()

    return spread.round(2)


# "Peak" = the standard Mon-Fri, 08:00-20:00 local-time window (12 hours
# a day) -- everything outside it (weekends, nights/early mornings) is
# "off-peak" and excluded from these tables.
PEAK_START_HOUR = 8
PEAK_END_HOUR = 20  # exclusive -- so hours 8..19
PEAK_WEEKDAYS = range(0, 5)  # Monday=0 .. Friday=4


@cached
def load_pivot_peak(zone: str, resolution: str) -> pd.DataFrame:
    """Same shape as load_pivot(), but averaged over Peak hours only. The
    daily/weekly/monthly aggregate tables only store the all-hours
    average, so this reads straight from the hourly day_ahead_prices
    table instead and does the Peak filtering + period averaging here."""
    conf = RESOLUTIONS[resolution]
    conn = _entsoe_conn()
    try:
        df = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS avg_price "
            "FROM day_ahead_prices WHERE zone = ?",
            conn,
            params=(zone,),
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Timestamps are saved as ISO strings with a UTC offset (CET/CEST).
    # Peak hours mean local wall-clock time, so the offset is dropped by
    # slicing to just the "YYYY-MM-DDTHH:MM:SS" part rather than parsing
    # it -- that keeps every row's hour/weekday exactly as it was in
    # local time, instead of pandas folding mixed summer/winter offsets
    # into UTC.
    df["period_date"] = pd.to_datetime(df["timestamp"].str.slice(0, 19))
    # year_label (not plain dt.year) -- see RESOLUTIONS["Weekly"]["year_label"].
    year_series = df["period_date"].apply(conf["year_label"])
    df = df[year_series >= START_YEAR]
    year_series = year_series[df.index]
    if df.empty:
        return pd.DataFrame()

    is_peak = (
        df["period_date"].dt.weekday.isin(PEAK_WEEKDAYS)
        & (df["period_date"].dt.hour >= PEAK_START_HOUR)
        & (df["period_date"].dt.hour < PEAK_END_HOUR)
    )
    df = df[is_peak]
    year_series = year_series[df.index]
    if df.empty:
        return pd.DataFrame()

    df["year"] = year_series
    df["period"] = df["period_date"].apply(conf["period_label"])

    pivot = df.pivot_table(
        index="period", columns="year", values="avg_price", aggfunc="mean"
    )

    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        pivot = pivot.reindex(canonical_periods)
    else:
        pivot = pivot.sort_index()

    return pivot.round(2)


@cached
def load_hours_below(zone: str, resolution: str, threshold: float) -> pd.DataFrame:
    """Same shape as load_pivot()/load_pivot_peak(), but each cell counts
    the hours in that period whose price was below `threshold` (EUR/MWh),
    rather than averaging the price. Built from the hourly
    day_ahead_prices table (like load_pivot_peak()) -- the pre-aggregated
    daily/weekly/monthly tables only store the period average, not a
    per-hour breakdown to count against a threshold."""
    conf = RESOLUTIONS[resolution]
    conn = _entsoe_conn()
    try:
        df = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS price FROM day_ahead_prices WHERE zone = ?",
            conn,
            params=(zone,),
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Same offset-dropping trick as load_pivot_peak(): keep each row's
    # local (CET) wall-clock time rather than a UTC reinterpretation of it.
    df["period_date"] = pd.to_datetime(df["timestamp"].str.slice(0, 19))
    # year_label (not plain dt.year) -- see RESOLUTIONS["Weekly"]["year_label"].
    year_series = df["period_date"].apply(conf["year_label"])
    df = df[year_series >= START_YEAR]
    year_series = year_series[df.index]
    if df.empty:
        return pd.DataFrame()

    df["year"] = year_series
    df["period"] = df["period_date"].apply(conf["period_label"])
    df["below"] = df["price"] < threshold

    pivot = df.pivot_table(index="period", columns="year", values="below", aggfunc="sum")

    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        pivot = pivot.reindex(canonical_periods)
    else:
        pivot = pivot.sort_index()

    return pivot.fillna(0).astype(int)


@cached
def load_spread_peak(zone1: str, zone2: str, resolution: str) -> pd.DataFrame:
    """Peak-hours version of load_spread() -- same subtract-and-realign
    logic, built from load_pivot_peak() instead of load_pivot()."""
    p1 = load_pivot_peak(zone1, resolution)
    p2 = load_pivot_peak(zone2, resolution)
    if p1.empty or p2.empty:
        return pd.DataFrame()

    spread = p1.subtract(p2)

    conf = RESOLUTIONS[resolution]
    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        spread = spread.reindex(canonical_periods)
    else:
        spread = spread.sort_index()

    return spread.round(2)


@cached
def load_capture_values(zone1: str, zone2: str, resolution: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-border capture value, split by which zone was pricier that
    hour. diff = zone1's hourly price minus zone2's (same direction as
    load_spread()'s zone1-minus-zone2). For each period/year cell:
      - "positive" pivot: the positive diffs (hours zone1 > zone2) summed,
        divided by ALL hours in that period -- not just the positive
        ones -- so it's the average per-hour value of flowing
        zone2 -> zone1 (buy zone2, sell zone1) across the whole period.
      - "negative" pivot: same idea for the negative diffs (hours
        zone2 > zone1), shown as a positive magnitude -- the average
        per-hour value of the opposite flow, zone1 -> zone2.
    Built from the hourly day_ahead_prices table (like the Peak tables),
    since this needs real hour-by-hour prices, not a pre-aggregated
    period average."""
    conn = _entsoe_conn()
    try:
        df1 = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS p1 FROM day_ahead_prices WHERE zone = ?",
            conn,
            params=(zone1,),
        )
        df2 = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS p2 FROM day_ahead_prices WHERE zone = ?",
            conn,
            params=(zone2,),
        )
    finally:
        conn.close()

    if df1.empty or df2.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = pd.merge(df1, df2, on="timestamp", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Same offset-dropping trick as load_pivot_peak(): keep each row's
    # local (CET) wall-clock time, so "hours in this period" counts real
    # local hours -- including a 23- or 25-hour DST changeover day --
    # rather than a UTC reinterpretation of it.
    merged["period_date"] = pd.to_datetime(merged["timestamp"].str.slice(0, 19))
    conf = RESOLUTIONS[resolution]
    # year_label (not plain dt.year) -- see RESOLUTIONS["Weekly"]["year_label"].
    year_series = merged["period_date"].apply(conf["year_label"])
    merged = merged[year_series >= START_YEAR]
    year_series = year_series[merged.index]
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["diff"] = merged["p1"] - merged["p2"]
    merged["year"] = year_series
    merged["period"] = merged["period_date"].apply(conf["period_label"])

    def _per_period(sub: pd.DataFrame) -> pd.Series:
        # Divide by every hour actually seen for this period/year (not a
        # theoretical calendar hour count) so a partial or gappy period
        # doesn't silently understate the per-hour average.
        hours = len(sub)
        positive_sum = sub.loc[sub["diff"] > 0, "diff"].sum()
        negative_sum = sub.loc[sub["diff"] < 0, "diff"].sum()
        return pd.Series(
            {
                "positive": positive_sum / hours if hours else float("nan"),
                "negative": abs(negative_sum) / hours if hours else float("nan"),
            }
        )

    grouped = merged.groupby(["period", "year"]).apply(_per_period).reset_index()

    positive_pivot = grouped.pivot_table(index="period", columns="year", values="positive")
    negative_pivot = grouped.pivot_table(index="period", columns="year", values="negative")

    canonical_periods = conf.get("canonical_periods")
    if canonical_periods is not None:
        positive_pivot = positive_pivot.reindex(canonical_periods)
        negative_pivot = negative_pivot.reindex(canonical_periods)
    else:
        positive_pivot = positive_pivot.sort_index()
        negative_pivot = negative_pivot.sort_index()

    return positive_pivot.round(2), negative_pivot.round(2)


@cached
def load_available_export_months(zone1: str, zone2: str) -> list[tuple[int, int]]:
    """Every (year, month) that has at least one hourly row for BOTH
    zones -- what the Year/Month export pickers offer. Sorted oldest
    first, so the most recent month is last (a sensible default index)."""
    conn = _entsoe_conn()
    try:
        months = pd.read_sql(
            "SELECT DISTINCT substr(timestamp, 1, 7) AS ym FROM day_ahead_prices "
            "WHERE zone = ?",
            conn,
            params=(zone1,),
        )["ym"]
        months2 = pd.read_sql(
            "SELECT DISTINCT substr(timestamp, 1, 7) AS ym FROM day_ahead_prices "
            "WHERE zone = ?",
            conn,
            params=(zone2,),
        )["ym"]
    finally:
        conn.close()

    common = sorted(set(months) & set(months2))
    return [(int(ym[:4]), int(ym[5:7])) for ym in common if int(ym[:4]) >= START_YEAR]


@cached
def load_hourly_export(
    zone1: str, zone2: str, year: int, month: int,
    label1: str | None = None, label2: str | None = None,
) -> pd.DataFrame:
    """Hour-by-hour prices for both zones, one row per local hour of the
    given month, ready to hand straight to Excel. Filtered directly in
    SQL (a single month, not the whole table) since this is a one-off
    export rather than a cached full-table read like the tables above.
    label1/label2 name the 2 output columns when given (falling back to
    zone1/zone2) -- needed so an interconnector pseudo-country (e.g.
    "IFA", queried here as plain "GB") shows its own picked name in the
    export rather than "GB", and so 2 columns don't collide when both
    picks resolve to the same underlying zone (e.g. IFA vs IFA2)."""
    d1, d2 = label1 or zone1, label2 or zone2
    month_str = f"{year:04d}-{month:02d}"
    conn = _entsoe_conn()
    try:
        df1 = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS p1 FROM day_ahead_prices "
            "WHERE zone = ? AND substr(timestamp, 1, 7) = ? ORDER BY timestamp",
            conn,
            params=(zone1, month_str),
        )
        df2 = pd.read_sql(
            "SELECT timestamp, price_eur_mwh AS p2 FROM day_ahead_prices "
            "WHERE zone = ? AND substr(timestamp, 1, 7) = ? ORDER BY timestamp",
            conn,
            params=(zone2, month_str),
        )
    finally:
        conn.close()

    if df1.empty or df2.empty:
        return pd.DataFrame()

    merged = pd.merge(df1, df2, on="timestamp", how="outer", sort=True)
    # Same offset-dropping trick used throughout this file: keep each
    # row's local (CET) wall-clock time rather than a UTC reinterpretation
    # of it, so the exported hours read the way a trader actually sees
    # them on the day.
    merged["Timestamp"] = pd.to_datetime(merged["timestamp"].str.slice(0, 19))
    merged = merged.drop(columns=["timestamp"]).rename(
        columns={"p1": d1, "p2": d2}
    )
    return merged[["Timestamp", d1, d2]].round(2)


def _build_export_excel(df: pd.DataFrame, zone1: str, zone2: str) -> bytes:
    """One sheet, Timestamp + both zones' hourly prices side by side --
    autosized columns and a formatted timestamp so it opens ready to use,
    no manual cleanup needed."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Hourly prices", index=False)
        sheet = writer.sheets["Hourly prices"]
        sheet.column_dimensions["A"].width = 20
        sheet.column_dimensions["B"].width = max(12, len(zone1) + 2)
        sheet.column_dimensions["C"].width = max(12, len(zone2) + 2)
        for row in sheet.iter_rows(min_row=2, min_col=1, max_col=1):
            row[0].number_format = "yyyy-mm-dd hh:mm"
    buffer.seek(0)
    return buffer.getvalue()


# Heatmap color range is clipped to this low/high percentile of each
# table's own values, rather than its strict min/max -- with real price
# data spanning 2020-2026, one extreme outlier period (e.g. the 2022
# energy-crisis price spike) would otherwise stretch the color scale so
# far that every normal month ends up looking like a similar pale
# yellow-red, with almost no visible gradient. Clipping the top/bottom
# slice lets a genuine outlier simply saturate at pure red/green instead
# of dominating the scale, so the rest of the cells spread across the
# full red-to-green range and show far more visible color variation.
# Clipping MORE (a bigger percentile) squeezes that "normal" range
# narrower, which stretches it across the same full color spectrum --
# i.e. more contrast between ordinary values, at the cost of more cells
# simply pinned to solid red/green at the extremes.
HEATMAP_CLIP_PERCENTILE = 15


def _truncated_cmap(name: str, low_frac: float, high_frac: float, n: int = 256):
    """A copy of the named colormap that only ever uses its
    [low_frac, high_frac] slice. background_gradient() clips any value
    past vmin/vmax to the colormap's own 0.0/1.0 ends -- with the plain
    "RdYlGn" colormap those ends are a near-black maroon and a near-black
    green, so a genuine outlier (see HEATMAP_CLIP_PERCENTILE above) always
    looked much darker than every other cell no matter how the vmin/vmax
    range itself was chosen. Handing background_gradient this narrower
    slice instead means even a value that lands exactly on 0.0 or 1.0
    still gets a legible medium red/green, never the source colormap's
    own darkest extreme."""
    base = matplotlib.colormaps[name]
    colors = base(np.linspace(low_frac, high_frac, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_trunc", colors)


# DE_LU / FR / capture tables' color scale -- see _truncated_cmap() above.
HEATMAP_CMAP = _truncated_cmap("RdYlGn", 0.25, 0.75)



# The colored part of each Spread cell's color range never goes past this
# fraction of the Greens/Reds colormap -- keeps even the most extreme
# cells a legible medium shade rather than near-black/near-solid.
HEATMAP_MIN_SHADE = 0.2
HEATMAP_MAX_SHADE = 0.6


def _heatmap_sign_bound(pivot: pd.DataFrame) -> float:
    """Outlier-robust magnitude to scale Spread color intensity against --
    see HEATMAP_CLIP_PERCENTILE above. Positive and negative cells share
    this one bound, so a +50 cell and a -50 cell end up equally
    saturated."""
    values = pivot.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    bound = float(np.percentile(finite, 100 - HEATMAP_CLIP_PERCENTILE))
    if bound == 0:
        bound = float(finite.max()) or 1.0
    return bound


def _sign_color(value: float, bound: float) -> str:
    """Every positive value is some shade of green, every negative value
    some shade of red -- never the reverse and never a yellow in-between,
    however small the value. Missing values get a plain white cell
    instead of a colored one."""
    if pd.isna(value):
        return "background-color: white"
    cmap = matplotlib.colormaps["Greens"] if value >= 0 else matplotlib.colormaps["Reds"]
    frac = min(abs(value) / bound, 1.0) if bound else 0.0
    shade = HEATMAP_MIN_SHADE + frac * (HEATMAP_MAX_SHADE - HEATMAP_MIN_SHADE)
    return f"background-color: {mcolors.rgb2hex(cmap(shade))}"



# Fixed column widths (Peter: "use fixed size in all table. size is
# determined by you shall see all values"): every value column of a table
# gets the SAME pixel width, just wide enough for the longest text shown
# anywhere in that table (never narrower than FIXED_COL_MIN_CHARS, so
# ordinary price tables all come out the same size). The row-label column
# is sized the same way from its own labels.
# Measured off Peter's own screenshot: bold 9-11 character values came
# out ~6.7 px per character, so 7 px + 8 px padding each side fits the
# longest value snugly (the first try, 8.5 px + 26, left the columns
# visibly too wide).
FIXED_COL_CHAR_PX = 7.0
FIXED_COL_PAD_PX = 18
FIXED_COL_MIN_CHARS = 6


def _fixed_width_px(texts, min_chars: int = FIXED_COL_MIN_CHARS) -> int:
    """Pixel width that fits the longest of `texts` (any iterable of
    strings)."""
    longest = max((len(str(t)) for t in texts), default=0)
    return int(max(longest, min_chars) * FIXED_COL_CHAR_PX + FIXED_COL_PAD_PX)


@cached
def load_jao_corridors() -> set[str]:
    """Every corridor code JAO has ever saved an auction for, read
    straight off jao_data.db (see JAO_DB_FILE above). Returns an empty
    set -- not an error -- if that database can't be found or opened
    (e.g. this is running somewhere without the JAO project next to it,
    or the file is momentarily locked by jao_scraper.py running a fetch
    -- same read-only-connection pattern jao_dashboard.py itself uses,
    so this behaves the same way that dashboard does when it hits the
    same situation)."""
    if not JAO_DB_FILE.exists():
        return set()
    try:
        conn = _jao_conn()
        try:
            rows = conn.execute("SELECT DISTINCT corridor_code FROM auctions").fetchall()
            # Plus the real borders sold inside the bundled "CEEXD"
            # auction (CZ-DE(50Hz), PL-CZ, PL-SK, ...) -- those only
            # appear on the result rows, never as an auction's own code.
            rows += conn.execute(
                "SELECT DISTINCT corridor_code FROM auction_results "
                "WHERE identification LIKE 'CEEXD-%'"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return set()
    return {r[0] for r in rows} - {"CEEXD"}


def _jao_corridors_for(zone1: str, zone2: str) -> tuple[str | None, str | None]:
    """The plain "A-B" JAO corridor code for these two ENTSO-E zones, and
    its opposite direction -- None for either JAO doesn't have. Only
    handles the plain "A-B" corridor-code shape (covers ~90 of JAO's 103
    real corridors, confirmed against the live list) -- the few with
    parentheses (e.g. "CZ-DE(TenneT)") aren't derivable from just two
    zone codes, so those simply come back None here, same as a corridor
    JAO has no reverse direction saved for (e.g. EE-LV) -- see
    opposite_corridor() in jao_dashboard.py for the full 3-shape logic
    this deliberately doesn't replicate; this only needs to go one
    direction, from 2 known zones to a corridor code, not reverse an
    already-picked one.

    The named-interconnector prefix shape (e.g. GB's "IF1-FR-GB") IS
    handled, but only for the 3 pseudo-countries in
    INTERCONNECTOR_JAO_PREFIX -- see that dict's comment. zone1/zone2
    here are the pseudo names as picked in the UI (e.g. "IFA"), not
    _entsoe_zone()'s resolved "GB" -- this function must see the real
    picks to tell IFA/IFA2/ElecLink apart, since all 3 resolve to the
    same ENTSO-E zone."""
    prefix1 = INTERCONNECTOR_JAO_PREFIX.get(zone1)
    prefix2 = INTERCONNECTOR_JAO_PREFIX.get(zone2)
    if prefix1 or prefix2:
        # Exactly one side is an interconnector pseudo-country -- these 3
        # corridors are specifically FR<->GB physical links, so the other
        # side must be FR for a corridor to exist at all (IFA vs IFA2,
        # IFA vs DE_LU, etc. correctly return None, None below).
        corridors = load_jao_corridors()
        prefix = prefix1 or prefix2
        other = zone2 if prefix1 else zone1
        if ZONE_TO_JAO_TOKEN.get(other, other) != "FR":
            return None, None
        fr_gb, gb_fr = f"{prefix}-FR-GB", f"{prefix}-GB-FR"
        # forward = zone1 -> zone2, same convention as the plain-shape
        # branch below.
        forward, backward = (gb_fr, fr_gb) if prefix1 else (fr_gb, gb_fr)
        return (
            forward if forward in corridors else None,
            backward if backward in corridors else None,
        )

    t1 = ZONE_TO_JAO_TOKEN.get(zone1, zone1)
    t2 = ZONE_TO_JAO_TOKEN.get(zone2, zone2)
    corridors = load_jao_corridors()
    forward, backward = f"{t1}-{t2}", f"{t2}-{t1}"
    return (
        forward if forward in corridors else None,
        backward if backward in corridors else None,
    )


# WHERE-clause piece: "this result row belongs to border ?" (2 params,
# the same corridor twice). Either the auction itself is that corridor,
# or it is the bundled CEEXD auction and the result row is that border
# and something was actually sold (a border with nothing on offer still
# gets a 0 MW / price 0 row -- that is "no auction", so it is skipped).
JAO_BORDER_MATCH_SQL = (
    "(a.corridor_code = ? OR (a.corridor_code = 'CEEXD' AND r.corridor_code = ? "
    "AND NOT (r.allocated_capacity = 0 AND r.auction_price = 0)))"
)


@cached
def load_jao_monthly_price(corridor: str) -> pd.DataFrame:
    """Monthly-horizon JAO auction clearing price (EUR/MW), months down,
    years across -- the exact same shape jao_dashboard.py's Corridor
    explorer tab builds (its render_monthly_pivot()), rebuilt here rather
    than imported across folders since this reads a different project's
    database. Defaults to the "BASE------" product where present (the
    only product a Monthly corridor usually has -- see
    claude/jao-auctions-scraping.md), else whichever product comes
    first. Applies the same +1 day shift jao_scraper.py/jao_dashboard.py
    apply to market_period_start everywhere (JAO stores it as CET/CEST
    local midnight, which lands one calendar day early in UTC)."""
    if not JAO_DB_FILE.exists():
        return pd.DataFrame()
    try:
        conn = _jao_conn()
    except sqlite3.OperationalError:
        return pd.DataFrame()
    try:
        products = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT r.product_identification "
                "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
                f"WHERE {JAO_BORDER_MATCH_SQL} AND a.horizon_name = 'Monthly'",
                (corridor, corridor),
            ).fetchall()
        ]
        if not products:
            return pd.DataFrame()
        product = "BASE------" if "BASE------" in products else products[0]

        df = pd.read_sql_query(
            "SELECT a.market_period_start, r.auction_price "
            "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
            f"WHERE {JAO_BORDER_MATCH_SQL} AND a.horizon_name = 'Monthly' AND r.product_identification = ?",
            conn,
            params=(corridor, corridor, product),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["market_period_start"] = pd.to_datetime(df["market_period_start"], utc=True) + pd.Timedelta(days=1)
    df["year"] = df["market_period_start"].dt.year
    # Full month names ("January", not "Jan") -- MUST match this file's
    # OWN Monthly-resolution convention (RESOLUTIONS["Monthly"]
    # ["period_label"] and QUARTER_MONTHS both use %B), not
    # jao_dashboard.py's abbreviated MONTH_ORDER ("Jan".."Dec") this was
    # first copied from -- _show_pivot()'s Q1-Q4/Year step
    # (_append_quarter_year_rows) does pivot.loc[["January", "February",
    # "March"]] etc. and raises a KeyError against abbreviated names.
    # Real bug, hit by Peter on his actual machine (KeyError: "None of
    # [Index(['January', 'February', 'March']...] are in the [index]"),
    # not caught by the standalone verification here since that only
    # checked the raw pivot, never routed it through _show_pivot's
    # Monthly-only quarter/year step.
    df["month"] = df["market_period_start"].dt.strftime("%B")
    pivot = df.pivot_table(index="month", columns="year", values="auction_price", aggfunc="mean")
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return pivot.reindex(month_order).round(2)


def _margin_pivot(capture_pivot: pd.DataFrame, jao_pivot: pd.DataFrame) -> pd.DataFrame:
    """Capture value minus JAO Monthly auction price, for one flow
    direction (Peter: "deduct table 3 from table 1 and table 4 from
    table 2 in row 2") -- what's left of the captured price spread after
    paying for the transmission capacity needed to actually flow that
    direction. Both pivots already share the same row labels (full month
    names -- see RESOLUTIONS["Monthly"] and load_jao_monthly_price's own
    reindex), so a plain DataFrame subtraction aligns them on both axes
    automatically: a year either side doesn't have data for comes out
    NaN, same as a missing corridor/period does everywhere else on this
    page. Capture value is €/MWh (an energy price spread) and the JAO
    price is €/MW (a capacity price) -- different units -- so this is a
    quick trading comparison, not a like-for-like subtraction; shown as
    asked, not adjusted for that."""
    if capture_pivot.empty or jao_pivot.empty:
        return pd.DataFrame()
    return (capture_pivot - jao_pivot).round(2)


# --- Corridor margins tab ------------------------------------------------
# Peter: "make a new tab. Here is a table. Column is month, row is
# corridors. content margin for each month. toggle between auction price,
# settle and margin. pick year." Same 3 numbers the Year comparison tab
# already shows for ONE picked country pair (JAO auction price, capture
# value, margin = capture minus auction price), but for EVERY JAO
# corridor at once, one year at a time. "Settle" here = that same capture
# value: what the corridor's direction actually settled at in the
# day-ahead market.

# JAO corridor-code token -> the ENTSO-E zone whose hourly price it
# settles against. The reverse of ZONE_TO_JAO_TOKEN above, but that one
# is many-to-one for Italy (7 zones -> "IT"), so it can't just be
# flipped: "IT" defaults to IT_NORD here (where the FR/CH/AT/SI borders
# physically connect), with the 2 cables that land elsewhere overridden
# in CORRIDOR_ZONE_OVERRIDES below. "DK" (DK-NL / NL-DK, the COBRAcable)
# lands in DK_1; "N2" (North Sea Link) is NO_2. Any token not listed is
# used as-is (AT, FR, GB, ...). A corridor whose zones have no hourly
# price saved (CEEXD, the Ukraine ones) simply has no Settle/Margin.
CORRIDOR_TOKEN_TO_ZONE = {
    "DE": "DE_LU",
    "D1": "DK_1",
    "D2": "DK_2",
    "DK": "DK_1",
    "N2": "NO_2",
    "IE": "IE_SEM",
    "IT": "IT_NORD",
}
CORRIDOR_ZONE_OVERRIDES = {
    "GR-IT": ("GR", "IT_SUD"),    # GRITA cable lands in Puglia
    "IT-GR": ("IT_SUD", "GR"),
    "ME-IT": ("ME", "IT_CSUD"),   # MONITA cable lands in Abruzzo
    "IT-ME": ("IT_CSUD", "ME"),
}
CORRIDOR_MONTH_LABELS = [calendar.month_abbr[m] for m in range(1, 13)]
CORRIDOR_VIEWS = ["Margin", "Auction price", "Settle"]
# Tenor picker order (longest first). Only the ones jao_data.db actually
# has are offered -- see load_corridor_tenors().
CORRIDOR_TENOR_ORDER = ["Yearly", "Seasonal", "Quarterly", "Monthly", "Weekly", "Daily"]


def _corridor_zones(corridor: str) -> tuple[str | None, str | None]:
    """(from_zone, to_zone) for a JAO corridor code, as ENTSO-E zones.
    Handles all 3 corridor-code shapes: plain "DE-FR", parenthesised
    "CZ-DE(TenneT)", and named-interconnector "IF1-FR-GB" (the first
    token is the cable, the last 2 are from/to). (None, None) for
    anything else (e.g. "CEEXD")."""
    if corridor in CORRIDOR_ZONE_OVERRIDES:
        return CORRIDOR_ZONE_OVERRIDES[corridor]
    parts = [part.split("(")[0] for part in corridor.split("-")]
    if len(parts) == 3:
        parts = parts[1:]
    if len(parts) != 2:
        return None, None
    return (
        CORRIDOR_TOKEN_TO_ZONE.get(parts[0], parts[0]),
        CORRIDOR_TOKEN_TO_ZONE.get(parts[1], parts[1]),
    )


@cached
def load_corridor_tenors() -> list[str]:
    """Tenors (JAO "horizons") the Corridor margins tab's Tenor picker
    offers: whichever of CORRIDOR_TENOR_ORDER jao_data.db actually has
    auctions for, longest first."""
    if not JAO_DB_FILE.exists():
        return []
    try:
        conn = _jao_conn()
        try:
            found = {r[0] for r in conn.execute("SELECT DISTINCT horizon_name FROM auctions")}
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return []
    return [t for t in CORRIDOR_TENOR_ORDER if t in found]


# ttl=600, not the usual 60: this is the slow query on this tab (Daily
# takes ~10 s) and JAO auction prices only change a few times a day.
@cached
def _load_jao_tenor_prices(tenor: str) -> pd.DataFrame:
    """Every JAO auction price for one tenor, all corridors, all years,
    spread over the calendar months each auction DELIVERS in: columns
    corridor / year / month (1-12) / auction_price. Peter: "If year 2026
    is picked then it shows the value each month where auction price is
    the yearly auction in 2025 for 2026" -- so the Yearly 2026 product's
    price lands on all 12 months of 2026, a Quarterly one on its 3
    months, a Seasonal one on its 6, a Monthly one on its own month, and
    a Weekly/Daily one on the month it starts in (those get averaged per
    month further down the line). Keyed on the delivery period
    (market_period_start/stop, read as CET/CEST local time), NOT on when
    the auction was held.

    Product: "BASE------" where the corridor has one, same as
    load_jao_monthly_price() on the Year comparison tab -- so Monthly
    here matches that tab exactly. Daily has no BASE product, only 24
    hourly ones, so there every hour is used. Several auctions for the
    same delivery period (e.g. IFA sells a year in several rounds) are
    averaged."""
    empty = pd.DataFrame(columns=["corridor", "year", "month", "auction_price"])
    if not JAO_DB_FILE.exists():
        return empty
    try:
        conn = _jao_conn()
    except sqlite3.OperationalError:
        return empty
    try:
        # Averaged per auction + product inside SQLite: Daily alone is
        # ~1.5 million hourly rows, far too many to pull into pandas raw.
        df = pd.read_sql_query(
            # "CEEXD" is not a corridor: it is ONE JAO auction that sells
            # 8 separate borders at once (CZ-DE(50Hz), DE(50Hz)-CZ, PL-CZ,
            # CZ-PL, PL-DE(50Hz), DE(50Hz)-PL, PL-SK, SK-PL), each with its
            # own price -- the real border is on the result row, not the
            # auction. Averaging them into one "CEEXD" number (what this
            # did at first) is meaningless, so they are split out here.
            f"SELECT {CEEXD_SPLIT_SQL} AS corridor, a.market_period_start AS start, "
            "a.market_period_stop AS stop, r.product_identification AS product, "
            "AVG(r.auction_price) AS auction_price "
            "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
            "WHERE a.horizon_name = ? AND r.auction_price IS NOT NULL "
            # Inside the CEEXD bundle a border with nothing on offer that
            # month still gets a result row (0 MW allocated, price 0) --
            # that is "no auction", not "free capacity", so it is dropped
            # rather than shown as a 0.00 price with a fake full Margin.
            "AND NOT (a.corridor_code = 'CEEXD' AND r.allocated_capacity = 0 AND r.auction_price = 0) "
            "GROUP BY a.identification, r.product_identification, corridor",
            conn,
            params=(tenor,),
        )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return empty
    finally:
        conn.close()
    if df.empty:
        return empty

    if tenor != "Daily":
        has_base = df.groupby("corridor")["product"].transform(lambda s: (s == "BASE------").any())
        first_product = df.groupby("corridor")["product"].transform("first")
        df = df[df["product"] == first_product.where(~has_base, "BASE------")]

    # Delivery period in local (CET/CEST) time: JAO stores local midnight
    # as UTC, e.g. "2025-12-31T23:00Z" = 1 Jan 2026 00:00 CET. The stop
    # is exclusive (the next period's start), hence the minus 1 second.
    start = pd.to_datetime(df["start"], utc=True).dt.tz_convert("CET")
    stop = pd.to_datetime(df["stop"], utc=True).dt.tz_convert("CET") - pd.Timedelta(seconds=1)
    first_month = (start.dt.year * 12 + start.dt.month - 1).to_numpy()
    last_month = (stop.dt.year * 12 + stop.dt.month - 1).fillna(-1).astype(int).to_numpy()
    if tenor in ("Weekly", "Daily"):
        # Shorter than a month: counts for the month it starts in only.
        last_month = first_month
    n_months = np.maximum(last_month - first_month + 1, 1)

    # One row per (auction, delivery month).
    out = df.loc[df.index.repeat(n_months), ["corridor", "auction_price"]].reset_index(drop=True)
    month_index = np.repeat(first_month, n_months) + np.concatenate([np.arange(n) for n in n_months])
    out["year"] = month_index // 12
    out["month"] = month_index % 12 + 1
    return out[["corridor", "year", "month", "auction_price"]]


@cached
def load_corridor_years(tenor: str) -> list[int]:
    """Years the Corridor margins tab's Year picker offers -- every year
    from START_YEAR on that the picked tenor has an auction delivering in."""
    df = _load_jao_tenor_prices(tenor)
    return sorted(int(y) for y in df["year"].unique() if y >= START_YEAR)


@cached
def load_corridor_auction_prices(year: int, tenor: str = "Monthly") -> pd.DataFrame:
    """Rows = every corridor with an auction of that tenor delivering in
    that year, columns = month 1..12, values = auction price."""
    df = _load_jao_tenor_prices(tenor)
    df = df[df["year"] == year]
    if df.empty:
        return pd.DataFrame(columns=range(1, 13))
    pivot = df.pivot_table(index="corridor", columns="month", values="auction_price", aggfunc="mean")
    return pivot.reindex(columns=range(1, 13)).sort_index().round(2)


@cached
def load_corridor_settle(year: int, tenor: str = "Monthly") -> pd.DataFrame:
    """Rows = corridor, columns = month 1..12, values = what that
    corridor's direction settled at in the day-ahead market: for corridor
    A-B (capacity to flow A -> B), the hours B was pricier than A, summed
    and divided by ALL hours of the month both zones have a price for.
    Exactly load_capture_values()'s definition (so this matches the
    Year comparison tab's capture-value tables), just computed for every
    corridor in one pass: one query for the year's hourly prices of
    every zone involved, instead of one pair at a time."""
    corridors = load_corridor_auction_prices(year, tenor).index
    zone_pairs = {c: _corridor_zones(c) for c in corridors}
    zones = sorted({z for pair in zone_pairs.values() for z in pair if z})
    empty = pd.DataFrame(columns=range(1, 13))
    if not zones:
        return empty

    conn = _entsoe_conn()
    try:
        placeholders = ",".join("?" * len(zones))
        df = pd.read_sql(
            "SELECT zone, timestamp, price_eur_mwh AS price FROM day_ahead_prices "
            f"WHERE timestamp >= ? AND timestamp < ? AND zone IN ({placeholders})",
            conn,
            params=(f"{year}-01-01", f"{year + 1}-01-01", *zones),
        )
    finally:
        conn.close()
    if df.empty:
        return empty

    # One column per zone, one row per hour. The timestamp strings carry
    # their local (CET/CEST) wall-clock time up front -- same
    # offset-dropping slice as load_capture_values() -- so "month" is the
    # local calendar month, and a 23-/25-hour DST day counts its real hours.
    wide = df.pivot(index="timestamp", columns="zone", values="price")
    month = pd.Series(
        pd.to_datetime(wide.index.str.slice(0, 19)).month, index=wide.index
    )

    rows = {}
    for corridor, (zone_from, zone_to) in zone_pairs.items():
        if zone_from not in wide.columns or zone_to not in wide.columns:
            continue
        diff = (wide[zone_to] - wide[zone_from]).dropna()
        if diff.empty:
            continue
        rows[corridor] = diff.clip(lower=0).groupby(month[diff.index]).mean()
    if not rows:
        return empty
    return pd.DataFrame(rows).T.reindex(columns=range(1, 13)).round(2)


# What the Winner picker shows when nobody is picked (the default).
CORRIDOR_NO_WINNER = "(none)"
CORRIDOR_WINNER_MODES = ["Only winner's", "All, winner marked"]
# SQL for "the real border of this result row" -- see the CEEXD comment
# in _load_jao_tenor_prices().
CEEXD_SPLIT_SQL = "CASE WHEN a.corridor_code = 'CEEXD' THEN r.corridor_code ELSE a.corridor_code END"


@cached
def _load_jao_tenor_winners(tenor: str) -> pd.DataFrame:
    """Who won capacity in each auction of one tenor, spread over the
    months that auction delivers in -- same month logic as
    _load_jao_tenor_prices(), so a cell here lines up with the same cell
    there. Columns corridor / year / month / winner. The winner label is
    the resolved company name where eic_lookup.py has one, else the raw
    EIC code (same labels jao_dashboard.py's buyer tabs use). JAO only
    publishes WHO won, not how many MW each -- so this is "holds some
    capacity", never how much."""
    empty = pd.DataFrame(columns=["corridor", "year", "month", "winner"])
    if not JAO_DB_FILE.exists():
        return empty
    try:
        conn = _jao_conn()
    except sqlite3.OperationalError:
        return empty
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT a.identification, a.corridor_code AS corridor, "
            "a.market_period_start AS start, a.market_period_stop AS stop, "
            "COALESCE(NULLIF(n.party_name, ''), w.eic_code) AS winner "
            "FROM auction_winners w JOIN auctions a ON a.identification = w.identification "
            "LEFT JOIN eic_names n ON n.eic_code = w.eic_code "
            "WHERE a.horizon_name = ?",
            conn,
            params=(tenor,),
        )
        # The CEEXD bundle (see _load_jao_tenor_prices): JAO lists its
        # winners per AUCTION, not per border inside it, so a CEEXD
        # winner is shown on every border of that auction that actually
        # allocated capacity -- the closest the data allows.
        ceexd = pd.read_sql_query(
            "SELECT DISTINCT identification, corridor_code AS sub_corridor "
            "FROM auction_results WHERE identification LIKE 'CEEXD-%' AND allocated_capacity > 0",
            conn,
        )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return empty
    finally:
        conn.close()
    if df.empty:
        return empty

    is_bundle = df["corridor"] == "CEEXD"
    split = df[is_bundle].merge(ceexd, on="identification", how="inner")
    split["corridor"] = split["sub_corridor"]
    df = pd.concat([df[~is_bundle], split[df.columns]], ignore_index=True)

    start = pd.to_datetime(df["start"], utc=True).dt.tz_convert("CET")
    stop = pd.to_datetime(df["stop"], utc=True).dt.tz_convert("CET") - pd.Timedelta(seconds=1)
    first_month = (start.dt.year * 12 + start.dt.month - 1).to_numpy()
    last_month = (stop.dt.year * 12 + stop.dt.month - 1).fillna(-1).astype(int).to_numpy()
    if tenor in ("Weekly", "Daily"):
        last_month = first_month
    n_months = np.maximum(last_month - first_month + 1, 1)

    out = df.loc[df.index.repeat(n_months), ["corridor", "winner"]].reset_index(drop=True)
    month_index = np.repeat(first_month, n_months) + np.concatenate([np.arange(n) for n in n_months])
    out["year"] = month_index // 12
    out["month"] = month_index % 12 + 1
    return out[["corridor", "year", "month", "winner"]].drop_duplicates()


@cached
def load_corridor_winners(year: int, tenor: str) -> list[str]:
    """Winner picker options: everyone who won any auction of this tenor
    delivering in this year, A-Z."""
    df = _load_jao_tenor_winners(tenor)
    names = df.loc[df["year"] == year, "winner"].dropna().unique()
    return sorted(names, key=str.casefold)


@cached
def load_corridor_winner_months(year: int, tenor: str, winner: str) -> list[tuple[str, int]]:
    """(corridor, month 1-12) pairs where this winner holds capacity of
    this tenor in this year."""
    df = _load_jao_tenor_winners(tenor)
    df = df[(df["year"] == year) & (df["winner"] == winner)]
    return sorted(set(zip(df["corridor"], df["month"].astype(int))))


def _corridor_winner_stats(
    settle: pd.DataFrame, auction: pd.DataFrame, held: list[tuple[str, int]]
) -> pd.Series:
    """Per corridor the picked winner holds: their average Margin (Settle
    minus Auction price) over ONLY the months they hold capacity there.
    Always Margin, whatever the Show toggle is on. Months with no Margin
    yet (future months, or no hourly price) are left out; a corridor with
    none at all comes back NaN."""
    margin = settle - auction
    by_corridor: dict[str, list[float]] = {}
    for corridor, month in held:
        if corridor in margin.index:
            by_corridor.setdefault(corridor, []).append(margin.at[corridor, month])
    return pd.Series(
        {c: pd.Series(v, dtype=float).mean() for c, v in by_corridor.items()}, dtype=float
    ).round(2)


CORRIDOR_SUM_LABEL = "Sum"
CORRIDOR_NO_AUCTION_TEXT = "no auction"
CORRIDOR_SORT_BY_NAME = "Corridor"
CORRIDOR_POSITIVE_LABEL = "Positive"


# --- Market value tab ----------------------------------------------------
# Peter: "make a new tab showing total market value. here you multiply
# the MW and hours to the margin. sortable on corridor, month and year."
# Value of a corridor in a month, in EUR, for the WHOLE market (every
# buyer together -- JAO publishes the MW sold per auction, not per buyer):
#     MW sold  x  hours in the month  x  (Settle - Auction price)
# Worked out per auction, so a product sold in several rounds is exact:
# each round's own MW against its own price, no averaging of prices.
# Only tenors that cover whole months; Weekly and Daily products cover
# part of a month (Daily even hour by hour), so MW x month-hours would
# be wrong for them.
MARKET_VALUE_TENORS = ["Yearly", "Seasonal", "Quarterly", "Monthly"]
MARKET_VALUE_VIEWS = ["Market value (k€)", "MW sold"]


@cached
def _load_jao_tenor_allocations(tenor: str) -> pd.DataFrame:
    """One row per (auction, delivery month): corridor / year / month /
    mw (allocated capacity) / cost_rate (mw x auction price, EUR per
    hour). Same product, CEEXD-split and delivery-month rules as
    _load_jao_tenor_prices()."""
    empty = pd.DataFrame(columns=["corridor", "year", "month", "mw", "cost_rate"])
    if not JAO_DB_FILE.exists():
        return empty
    try:
        conn = _jao_conn()
    except sqlite3.OperationalError:
        return empty
    try:
        df = pd.read_sql_query(
            f"SELECT {CEEXD_SPLIT_SQL} AS corridor, a.market_period_start AS start, "
            "a.market_period_stop AS stop, r.product_identification AS product, "
            "r.auction_price AS auction_price, r.allocated_capacity AS mw "
            "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
            "WHERE a.horizon_name = ? AND r.auction_price IS NOT NULL "
            "AND r.allocated_capacity > 0",
            conn,
            params=(tenor,),
        )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return empty
    finally:
        conn.close()
    if df.empty:
        return empty

    has_base = df.groupby("corridor")["product"].transform(lambda s: (s == "BASE------").any())
    first_product = df.groupby("corridor")["product"].transform("first")
    df = df[df["product"] == first_product.where(~has_base, "BASE------")].reset_index(drop=True)

    start = pd.to_datetime(df["start"], utc=True).dt.tz_convert("CET")
    stop = pd.to_datetime(df["stop"], utc=True).dt.tz_convert("CET") - pd.Timedelta(seconds=1)
    first_month = (start.dt.year * 12 + start.dt.month - 1).to_numpy()
    last_month = (stop.dt.year * 12 + stop.dt.month - 1).fillna(-1).astype(int).to_numpy()
    n_months = np.maximum(last_month - first_month + 1, 1)

    out = df.loc[df.index.repeat(n_months), ["corridor", "mw", "auction_price"]].reset_index(drop=True)
    month_index = np.repeat(first_month, n_months) + np.concatenate([np.arange(n) for n in n_months])
    out["year"] = month_index // 12
    out["month"] = month_index % 12 + 1
    out["cost_rate"] = out["mw"] * out["auction_price"]
    return out[["corridor", "year", "month", "mw", "cost_rate"]]


@cached
def load_market_value(year: int, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(value, mw): both corridors down, month 1..12 across.
    value = EUR for the whole market that month:
        mw x SUM over the month's hours of max(price_to - price_from, 0)
        - cost_rate x hours
    which is exactly MW x hours x (Settle - Auction price), added up per
    auction. Hours = the hours both zones actually have a price for (so
    the running month is month-to-date on both sides). Never below
    -(MW x hours x price paid): Settle cannot go under 0.
    mw = total MW sold for that month (all rounds together)."""
    months = range(1, 13)
    empty = pd.DataFrame(columns=months, dtype=float)
    alloc = _load_jao_tenor_allocations(tenor)
    alloc = alloc[alloc["year"] == year]
    if alloc.empty:
        return empty, empty
    grouped = alloc.groupby(["corridor", "month"])[["mw", "cost_rate"]].sum()
    mw = grouped["mw"].unstack("month").reindex(columns=months).sort_index()
    cost_rate = grouped["cost_rate"].unstack("month").reindex(columns=months).sort_index()

    zone_pairs = {c: _corridor_zones(c) for c in mw.index}
    zones = sorted({z for pair in zone_pairs.values() for z in pair if z})
    value = pd.DataFrame(np.nan, index=mw.index, columns=months)
    if not zones:
        return value, mw

    conn = _entsoe_conn()
    try:
        placeholders = ",".join("?" * len(zones))
        df = pd.read_sql(
            "SELECT zone, timestamp, price_eur_mwh AS price FROM day_ahead_prices "
            f"WHERE timestamp >= ? AND timestamp < ? AND zone IN ({placeholders})",
            conn,
            params=(f"{year}-01-01", f"{year + 1}-01-01", *zones),
        )
    finally:
        conn.close()
    if df.empty:
        return value, mw

    wide = df.pivot(index="timestamp", columns="zone", values="price")
    month = pd.Series(pd.to_datetime(wide.index.str.slice(0, 19)).month, index=wide.index)
    for corridor, (zone_from, zone_to) in zone_pairs.items():
        if zone_from not in wide.columns or zone_to not in wide.columns:
            continue
        diff = (wide[zone_to] - wide[zone_from]).dropna()
        if diff.empty:
            continue
        by_month = diff.clip(lower=0).groupby(month[diff.index])
        spread_sum = by_month.sum().reindex(months)   # EUR/MW over the month
        hours = by_month.size().reindex(months)       # hours with a price
        value.loc[corridor] = mw.loc[corridor] * spread_sum - cost_rate.loc[corridor] * hours
    return value, mw


# --- Market value - overview tab ----------------------------------------
# Peter: "make a tab called market value - overview where you can see
# the market value per tenor per year". Same number as the Market value
# tab (MW sold x hours x Margin, whole market), added up over every
# corridor and month: one row per tenor, one column per year.
@cached
def load_market_value_overview(include_running_month: bool = True) -> pd.DataFrame:
    """Rows = MARKET_VALUE_TENORS, columns = years (START_YEAR .. this
    year), values = EUR. Cached for an hour: it reads a full year of
    hourly prices per year (about 10-15 s in total). The hourly prices
    are read ONCE per year and shared by all tenors, rather than calling
    load_market_value() tenor by tenor."""
    today = pd.Timestamp.now()
    this_year = today.year
    years = list(range(START_YEAR, this_year + 1))
    allocations = {t: _load_jao_tenor_allocations(t) for t in MARKET_VALUE_TENORS}
    if not include_running_month:
        # Peter: "toggle current month on/off" -- the running month is
        # only month-to-date, so it can be left out of this year's total.
        allocations = {
            t: a[~((a["year"] == this_year) & (a["month"] == today.month))]
            for t, a in allocations.items()
        }
    result = pd.DataFrame(np.nan, index=MARKET_VALUE_TENORS, columns=years)

    for year in years:
        per_tenor = {
            t: a[a["year"] == year].groupby(["corridor", "month"])[["mw", "cost_rate"]].sum()
            for t, a in allocations.items()
        }
        corridors = sorted({c for g in per_tenor.values() for c in g.index.get_level_values("corridor")})
        zone_pairs = {c: _corridor_zones(c) for c in corridors}
        zones = sorted({z for pair in zone_pairs.values() for z in pair if z})
        if not zones:
            continue
        conn = _entsoe_conn()
        try:
            placeholders = ",".join("?" * len(zones))
            df = pd.read_sql(
                "SELECT zone, timestamp, price_eur_mwh AS price FROM day_ahead_prices "
                f"WHERE timestamp >= ? AND timestamp < ? AND zone IN ({placeholders})",
                conn,
                params=(f"{year}-01-01", f"{year + 1}-01-01", *zones),
            )
        finally:
            conn.close()
        if df.empty:
            continue
        wide = df.pivot(index="timestamp", columns="zone", values="price")
        month = pd.Series(pd.to_datetime(wide.index.str.slice(0, 19)).month, index=wide.index)

        # Per corridor: the month's summed positive spread and its hours.
        spread_sum, hours = {}, {}
        for corridor, (zone_from, zone_to) in zone_pairs.items():
            if zone_from not in wide.columns or zone_to not in wide.columns:
                continue
            diff = (wide[zone_to] - wide[zone_from]).dropna()
            if diff.empty:
                continue
            by_month = diff.clip(lower=0).groupby(month[diff.index])
            spread_sum[corridor], hours[corridor] = by_month.sum(), by_month.size()
        if not spread_sum:
            continue
        spread_sum = pd.DataFrame(spread_sum).T.stack()
        hours = pd.DataFrame(hours).T.stack()
        spread_sum.index.names = hours.index.names = ["corridor", "month"]

        for tenor, grouped in per_tenor.items():
            if grouped.empty:
                continue
            joined = grouped.join(spread_sum.rename("spread_sum"), how="inner").join(hours.rename("hours"))
            if joined.empty:
                continue
            result.at[tenor, year] = float(
                (joined["mw"] * joined["spread_sum"] - joined["cost_rate"] * joined["hours"]).sum()
            )
    return result


# =====================================================================
# PAGE -- tables, layout and callbacks (the Dash part)
# =====================================================================
# Every table is described first as a plain "model" (texts, colours,
# widths) and only then turned into HTML. The models are what the
# comparison test checks cell by cell against the Streamlit version.

def _gradient_backgrounds(values: np.ndarray, scale_values: np.ndarray) -> np.ndarray:
    """Background colour per cell: red = low, green = high, scaled on
    `scale_values` with the top/bottom HEATMAP_CLIP_PERCENTILE clipped.
    Empty cells are white. Same colours as the Streamlit tables."""
    out = np.full(values.shape, "white", dtype=object)
    finite = scale_values[np.isfinite(scale_values)]
    if finite.size == 0:
        return out
    vmin, vmax = np.percentile(finite, [HEATMAP_CLIP_PERCENTILE, 100 - HEATMAP_CLIP_PERCENTILE])
    if vmin == vmax:
        vmin, vmax = float(finite.min()), float(finite.max())
    rgba = HEATMAP_CMAP(mcolors.Normalize(vmin, vmax)(values))
    for idx in np.ndindex(values.shape):
        if np.isfinite(values[idx]):
            out[idx] = mcolors.rgb2hex(rgba[idx])
    return out


def _sign_backgrounds(values: np.ndarray, bound: float) -> np.ndarray:
    """Green for positive, red for negative (see _sign_color)."""
    out = np.empty(values.shape, dtype=object)
    for idx in np.ndindex(values.shape):
        out[idx] = _sign_color(values[idx], bound).split(": ", 1)[1]
    return out


def pivot_model(pivot: pd.DataFrame, resolution: str, spread: bool = False, fmt: str = "%.2f") -> dict:
    """The model of one period x year table (tabs 1 and 2): Q1-Q4 / Year
    rows for Monthly, fixed column widths, heatmap colours, and a height
    cap so a long Daily table scrolls inside its own box."""
    if resolution == "Monthly" and not pivot.empty:
        pivot = _append_quarter_year_rows(pivot)
    values = pivot.to_numpy(dtype=float)
    shown = [fmt % v for v in values.ravel() if np.isfinite(v)]
    columns = [str(c) for c in pivot.columns]
    if spread:
        backgrounds = _sign_backgrounds(values, _heatmap_sign_bound(pivot))
    else:
        backgrounds = _gradient_backgrounds(values, values)
    rows = []
    for r, label in enumerate(pivot.index):
        cells = [
            {"text": fmt % values[r, c] if np.isfinite(values[r, c]) else "",
             "bg": backgrounds[r, c], "css": {}}
            for c in range(values.shape[1])
        ]
        # Monthly: a fat line under December and under Q4.
        divider = resolution == "Monthly" and r in (11, 15)
        rows.append({"label": str(label), "cells": cells, "divider": divider, "css": {}})
    return {
        "index_label": "Period",
        "columns": columns,
        "rows": rows,
        "col_px": _fixed_width_px(shown + columns),
        "index_px": _fixed_width_px(["Period"] + [str(i) for i in pivot.index]),
        "height": min(HEADER_HEIGHT_PX + len(pivot) * ROW_HEIGHT_PX + 3, 650),
    }


# ---- tab 1 / tab 2 tables: (title, model or None, message or None) ----
def country_view(zone: str, resolution: str, peak: bool = False, label: str | None = None):
    display = label or zone
    unit = "€/MWh, Peak" if peak else "€/MWh, Base"
    title = f"{display}  —  {unit}"
    pivot = load_pivot_peak(zone, resolution) if peak else load_pivot(zone, resolution)
    if pivot.empty:
        return title, None, f"No data for {display} from {START_YEAR} onward."
    return title, pivot_model(pivot, resolution), None


def spread_view(zone1, zone2, resolution, peak=False, label1=None, label2=None):
    d1, d2 = label1 or zone1, label2 or zone2
    unit = "€/MWh, Peak" if peak else "€/MWh, Base"
    title = f"Spread  —  {unit}   ({d1} minus {d2})"
    spread = load_spread_peak(zone1, zone2, resolution) if peak else load_spread(zone1, zone2, resolution)
    if spread.empty:
        return title, None, f"No data to compare {d1} and {d2} from {START_YEAR} onward."
    return title, pivot_model(spread, resolution, spread=True), None


def capture_view(zone1, zone2, resolution, positive, label1=None, label2=None):
    positive_pivot, negative_pivot = load_capture_values(zone1, zone2, resolution)
    pivot = positive_pivot if positive else negative_pivot
    d1, d2 = label1 or zone1, label2 or zone2
    title = (f"{d2}-{d1}" if positive else f"{d1}-{d2}") + "  —  €/MWh"
    if pivot.empty:
        return title, None, f"No hourly data to compare {d1} and {d2} from {START_YEAR} onward."
    return title, pivot_model(pivot, resolution), None


def hours_below_view(zone, resolution, threshold, label=None):
    display = label or zone
    title = f"{display}  —  Hours below {threshold:g} €/MWh"
    pivot = load_hours_below(zone, resolution, threshold)
    if pivot.empty:
        return title, None, f"No hourly data for {display} from {START_YEAR} onward."
    return title, pivot_model(pivot, resolution, fmt="%d"), None


def jao_corridor_view(corridor, label):
    """JAO Monthly auction price for one direction. Always Monthly."""
    title = f"{label}  —  JAO auction price, EUR/MW"
    if corridor is None:
        return title, None, "No matching JAO corridor for this pair/direction."
    pivot = load_jao_monthly_price(corridor)
    if pivot.empty:
        return title, None, f"No Monthly JAO auctions saved for {corridor} yet."
    return title, pivot_model(pivot, "Monthly"), None


def margin_view(capture_pivot, corridor, label):
    """Capture value minus the JAO Monthly auction price. Always Monthly."""
    title = f"Margin: {label}  —  €/MWh (capture minus JAO price)"
    if corridor is None:
        return title, None, "No matching JAO corridor for this pair/direction."
    pivot = _margin_pivot(capture_pivot, load_jao_monthly_price(corridor))
    if pivot.empty:
        return title, None, "Not enough capture-value or JAO auction data yet to compute a margin."
    return title, pivot_model(pivot, "Monthly"), None


def year_comparison_views(country_1: str, country_2: str, resolution: str) -> list:
    """The 3 rows of tab 1, each a list of (title, model, message)."""
    zone_1, zone_2 = _entsoe_zone(country_1), _entsoe_zone(country_2)
    corridor_1_2, corridor_2_1 = _jao_corridors_for(country_1, country_2)
    capture_positive, capture_negative = load_capture_values(zone_1, zone_2, "Monthly")
    label_2_1 = corridor_2_1 or f"{country_2}-{country_1}"
    label_1_2 = corridor_1_2 or f"{country_1}-{country_2}"
    row_1 = [
        country_view(zone_1, resolution, label=country_1),
        country_view(zone_2, resolution, label=country_2),
        spread_view(zone_1, zone_2, resolution, label1=country_1, label2=country_2),
        margin_view(capture_positive, corridor_2_1, label_2_1),
        margin_view(capture_negative, corridor_1_2, label_1_2),
    ]
    row_2 = [
        capture_view(zone_1, zone_2, resolution, positive=True, label1=country_1, label2=country_2),
        capture_view(zone_1, zone_2, resolution, positive=False, label1=country_1, label2=country_2),
        jao_corridor_view(corridor_2_1, label_2_1),
        jao_corridor_view(corridor_1_2, label_1_2),
    ]
    row_3 = [
        country_view(zone_1, resolution, peak=True, label=country_1),
        country_view(zone_2, resolution, peak=True, label=country_2),
        spread_view(zone_1, zone_2, resolution, peak=True, label1=country_1, label2=country_2),
    ]
    return [row_1, row_2, row_3]


# ---- tabs 3-5: corridor tables --------------------------------------
def corridor_table_model(
    table: pd.DataFrame, view: str, fmt: str = "{:.2f}",
    no_auction: pd.DataFrame | None = None,
    held: list | None = None, bold_held: bool = False,
) -> dict:
    """Corridors down, Jan..Dec + Year across, with a "Sum" row on top
    (and for Margin a "Positive" row: how many corridors are above 0).
    Colours are scaled on the corridor rows only; the summary rows are
    grey and bold. `no_auction` marks cells that say "no auction";
    `held` = (corridor, month) cells that get a fat black frame."""
    total = table.sum(axis=0, min_count=1).round(2)
    summary = [(CORRIDOR_SUM_LABEL, total)]
    positive_text: dict[str, str] = {}
    if view == "Margin":
        n_positive = (table > 0).sum(axis=0)
        n_total = table.notna().sum(axis=0)
        share = (n_positive / n_total.where(n_total > 0)).astype(float)
        summary.append((CORRIDOR_POSITIVE_LABEL, share))
        positive_text = {
            col: (
                f"{int(n_positive[col])}/{int(n_total[col])} "
                f"({n_positive[col] / n_total[col]:.0%})"
            ) if n_total[col] else ""
            for col in table.columns
        }
    columns = [str(c) for c in table.columns]
    values = table.to_numpy(dtype=float)
    if view == "Margin":
        backgrounds = _sign_backgrounds(values, _heatmap_sign_bound(table))
    else:
        backgrounds = _gradient_backgrounds(values, values)

    grey_cells = set()
    extra_texts = []
    if no_auction is not None:
        for col in no_auction.columns:
            marked = [r for r in no_auction.index[no_auction[col]] if r in table.index]
            if not marked or col not in table.columns:
                continue
            grey_cells.update((r, col) for r in marked)
            extra_texts = [CORRIDOR_NO_AUCTION_TEXT]
    held_cells = set()
    for corridor, month in held or []:
        if corridor in table.index:
            held_cells.add((corridor, CORRIDOR_MONTH_LABELS[month - 1]))

    rows = []
    shown = []
    for i, (label, series) in enumerate(summary):
        last = i == len(summary) - 1
        cells = []
        for col in table.columns:
            value = series[col]
            if np.isfinite(value):
                shown.append(fmt.format(value))
            if label == CORRIDOR_POSITIVE_LABEL:
                text = positive_text[col]
            else:
                text = fmt.format(value) if np.isfinite(value) else ""
            css = {"fontWeight": "bold"}
            if last:
                css["borderBottom"] = "3px solid #000000"
            cells.append({"text": text, "bg": "#e8e8e8", "css": css})
        row_css = {"fontWeight": "bold", "backgroundColor": "#e8e8e8"}
        if last:
            row_css["borderBottom"] = "3px solid #000000"
        rows.append({"label": label, "cells": cells, "divider": False, "css": row_css, "summary": True})
    for r, corridor in enumerate(table.index):
        cells = []
        for c, col in enumerate(table.columns):
            value = values[r, c]
            css = {}
            if np.isfinite(value):
                text = fmt.format(value)
                shown.append(text)
            else:
                text = CORRIDOR_NO_AUCTION_TEXT if (corridor, col) in grey_cells else ""
            if (corridor, col) in grey_cells:
                css.update({"color": "#808080", "fontStyle": "italic"})
            if (corridor, col) in held_cells:
                css["border"] = "3px solid #000000"
                if bold_held:
                    css["fontWeight"] = "bold"
            cells.append({"text": text, "bg": backgrounds[r, c], "css": css})
        rows.append({"label": str(corridor), "cells": cells, "divider": False, "css": {}})
    shown += extra_texts
    return {
        "index_label": "Corridor",
        "columns": columns,
        "rows": rows,
        "col_px": _fixed_width_px(shown + list(positive_text.values()) + columns),
        "index_px": _fixed_width_px(["Corridor"] + [label for label, _ in summary] + [str(i) for i in table.index]),
        "height": None,  # shown in full, the page scrolls
    }


def winner_stats(stats: pd.Series) -> dict:
    """The numbers of the stats block shown when a Winner is picked. A
    border = one corridor direction; winning = average Margin above 0."""
    known = stats.dropna().sort_values(ascending=False)
    winning = int((known > 0).sum())
    losing = int((known < 0).sum())
    pending = int(stats.isna().sum()) + int((known == 0).sum())
    winning_sum = float(known[known > 0].sum())
    losing_sum = float(known[known < 0].sum())
    note = "Margin = average over the months they hold, per corridor."
    if pending:
        note += f" {pending} border(s) not counted (no Margin yet, or exactly 0)."
    return {
        "winning": winning, "losing": losing, "total": winning + losing,
        "winning_sum": winning_sum, "losing_sum": losing_sum,
        "total_sum": winning_sum + losing_sum,
        "highest": [(c, float(v)) for c, v in known.head(3).items()],
        "lowest": [(c, float(v)) for c, v in known.tail(3).iloc[::-1].items()],
        "note": note,
    }


def corridor_margins_view(
    year: int, view: str, tenor: str = "Monthly", winner: str | None = None,
    include_running_month: bool = True, sort_by: str | None = None,
    sort_descending: bool = True, winner_show_all: bool = False,
) -> dict:
    """The Corridor margins tab's table for the picked year and view.
    Returns {"message", "caption", "stats", "model"} (unused ones None)."""
    result = {"message": None, "caption": None, "stats": None, "model": None}
    auction = load_corridor_auction_prices(year, tenor)
    if auction.empty:
        result["message"] = f"No {tenor} JAO auctions saved for {year}."
        return result
    # Same rows in all 3 views, so toggling the view never reshuffles the
    # table -- a corridor with no hourly price just shows blank Settle/Margin.
    # .astype(float): a year with no hourly prices at all yet (e.g. next
    # year's Yearly auction) gives an empty Settle table, which must still
    # be numbers -- the Streamlit version stops with an error there.
    settle = load_corridor_settle(year, tenor).reindex(auction.index).astype(float)
    # Running month off: its Settle (only month-to-date) is dropped, so
    # Settle and Margin go blank and it stays out of the Year average and
    # the winner stats. Auction price is still shown.
    today = pd.Timestamp.now()
    if not include_running_month and year == today.year:
        settle = settle.copy()
        settle[today.month] = np.nan
    if view == "Auction price":
        table = auction
    elif view == "Settle":
        table = settle
    else:
        table = (settle - auction).round(2)

    # Months with no auction price for a corridor -- not counting months
    # still ahead (their auction simply has not been held yet).
    no_auction = auction.isna()
    if year >= today.year:
        last_past = today.month if year == today.year else 0
        no_auction.loc[:, [m for m in no_auction.columns if m > last_past]] = False
    no_auction.columns = CORRIDOR_MONTH_LABELS
    if view == "Settle":
        no_auction = None

    table = table.copy()
    table.columns = CORRIDOR_MONTH_LABELS
    # Plain average of the months shown -- for Margin only months with BOTH numbers.
    table["Year"] = table.mean(axis=1, skipna=True).round(2)

    held = None
    if winner:
        held = load_corridor_winner_months(year, tenor, winner)
    if winner and not winner_show_all:
        # Only the corridors the winner holds, and only the months they won.
        table = table.loc[table.index.isin({c for c, _ in held})].copy()
        held_cells = pd.DataFrame(False, index=table.index, columns=CORRIDOR_MONTH_LABELS)
        for corridor, month in held:
            if corridor in held_cells.index:
                held_cells.at[corridor, CORRIDOR_MONTH_LABELS[month - 1]] = True
        table[CORRIDOR_MONTH_LABELS] = table[CORRIDOR_MONTH_LABELS].where(held_cells)
        no_auction = None
        table["Year"] = table[CORRIDOR_MONTH_LABELS].mean(axis=1, skipna=True).round(2)
        if table.empty:
            result["message"] = f"{winner} holds no {tenor} capacity in {year}."
            return result
    # Sorted on the corridor rows only, so Sum / Positive stay on top.
    # Blank cells always go last.
    if sort_by in table.columns:
        table = table.sort_values(
            sort_by, ascending=not sort_descending, na_position="last", kind="stable"
        )
    if winner:
        corridors_held = len({c for c, _ in held})
        how = (
            "every corridor is shown; bold with a fat frame = the months they won."
            if winner_show_all
            else "only their corridors and the months they won are shown (fat frame)."
        )
        result["caption"] = f"{winner}: {tenor} capacity on {corridors_held} corridor(s) in {year} -- {how}"
        result["stats"] = winner_stats(_corridor_winner_stats(settle, auction, held))
    result["model"] = corridor_table_model(
        table, view, no_auction=no_auction, held=held, bold_held=winner_show_all
    )
    return result


def market_value_view(
    year: int, tenor: str, view: str, include_running_month: bool = True,
    sort_by: str | None = None, sort_descending: bool = True,
) -> dict:
    """The Market value tab's table. Returns {"message", "model"}."""
    value, mw = load_market_value(year, tenor)
    if mw.empty:
        return {"message": f"No {tenor} JAO auctions with MW sold for {year}.", "model": None}
    today = pd.Timestamp.now()
    if not include_running_month and year == today.year:
        value = value.copy()
        value[today.month] = np.nan

    is_value = view == MARKET_VALUE_VIEWS[0]
    table = (value / 1000.0).round(0) if is_value else mw.round(0)
    table = table.copy()
    table.columns = CORRIDOR_MONTH_LABELS
    # Value: the year's total. MW: the average MW over the months sold.
    if is_value:
        table["Year"] = table.sum(axis=1, min_count=1)
    else:
        table["Year"] = table.mean(axis=1, skipna=True).round(0)
    if sort_by in table.columns:
        table = table.sort_values(
            sort_by, ascending=not sort_descending, na_position="last", kind="stable"
        )
    model = corridor_table_model(table, "Margin" if is_value else "MW", fmt="{:,.0f}")
    return {"message": None, "model": model}


def market_value_overview_view(include_running_month: bool = True) -> dict:
    """Tenors down, years across, in million EUR, with a Total row."""
    value = load_market_value_overview(include_running_month)
    if value.isna().all().all():
        return {"message": "No JAO auction data with MW sold found (jao_data.db).", "model": None}
    table = (value / 1e6).round(1)
    table.loc["Total"] = table.sum(axis=0, min_count=1).round(1)
    columns = [str(c) for c in table.columns]
    fmt = "{:,.1f}"
    body = table.drop(index="Total")
    backgrounds = _sign_backgrounds(body.to_numpy(dtype=float), _heatmap_sign_bound(body))
    values = table.to_numpy(dtype=float)
    rows = []
    for r, label in enumerate(table.index):
        is_total = label == "Total"
        cells = []
        for c in range(values.shape[1]):
            text = fmt.format(values[r, c]) if np.isfinite(values[r, c]) else ""
            if is_total:
                cells.append({"text": text, "bg": "#e8e8e8", "css": {"fontWeight": "bold"}})
            else:
                cells.append({"text": text, "bg": backgrounds[r, c], "css": {}})
        row_css = {"fontWeight": "bold", "backgroundColor": "#e8e8e8"} if is_total else {}
        rows.append({"label": str(label), "cells": cells, "divider": False, "css": row_css})
    shown = [fmt.format(v) for v in values.ravel() if np.isfinite(v)]
    model = {
        "index_label": "Tenor",
        "columns": columns,
        "rows": rows,
        "col_px": _fixed_width_px(shown + columns),
        "index_px": _fixed_width_px(["Quarterly", "Tenor"]),
        "height": None,
    }
    return {"message": None, "model": model}


# ---- models -> HTML --------------------------------------------------
# The tables are written as one HTML text and put on the page in one go
# (see PAINT_JS further down), not built from Dash components: a Daily
# view is ~9,000 cells, and Dash needs minutes to draw that many separate
# components. As plain HTML it is instant.
TABLE_ROW_PX = 31  # roughly what one table row takes on screen
_CSS_NAMES = {
    "backgroundColor": "background-color", "fontWeight": "font-weight",
    "fontStyle": "font-style", "borderBottom": "border-bottom",
    "minWidth": "min-width", "maxHeight": "max-height",
}


def _style(css: dict) -> str:
    if not css:
        return ""
    text = "; ".join(f"{_CSS_NAMES.get(k, k)}: {v}" for k, v in css.items())
    return f' style="{escape(text, quote=True)}"'


def table_component(model: dict) -> str:
    """One HTML table from a model. A table taller than its model's height
    scrolls inside its own box (the header stays put); the others are
    shown in full."""
    col_style = _style({"minWidth": f"{model['col_px']}px", "width": f"{model['col_px']}px"})
    index_style = _style({"minWidth": f"{model['index_px']}px", "width": f"{model['index_px']}px"})
    parts = ['<table class="yc-table"><thead><tr>']
    parts.append(f'<th class="yc-rowhead"{index_style}>{escape(model["index_label"])}</th>')
    parts.extend(f"<th{col_style}>{escape(col)}</th>" for col in model["columns"])
    parts.append("</tr></thead><tbody>")
    for row in model["rows"]:
        parts.append('<tr class="yc-divider">' if row["divider"] else "<tr>")
        parts.append(f'<th class="yc-rowhead"{_style(row["css"])}>{escape(row["label"])}</th>')
        for cell in row["cells"]:
            css = {"backgroundColor": cell["bg"]}
            css.update(cell["css"])
            parts.append(f"<td{_style(css)}>{escape(cell['text'])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    table = "".join(parts)
    if model["height"] and (len(model["rows"]) + 1) * TABLE_ROW_PX > model["height"]:
        return f'<div class="yc-scroll"{_style({"maxHeight": str(model["height"]) + "px"})}>{table}</div>'
    return f'<div class="yc-noscroll">{table}</div>'


def info_box(text) -> str:
    return f'<div class="yc-info">{escape(str(text))}</div>'


def caption_box(text) -> str:
    return f'<div class="yc-caption">{escape(str(text))}</div>'


def table_block(view) -> str:
    title, model, message = view
    body = table_component(model) if model else info_box(message)
    return f'<div class="yc-block"><h3>{escape(title)}</h3>{body}</div>'


def tables_row(views) -> str:
    return '<div class="yc-row">' + "".join(table_block(v) for v in views) + "</div>"


def stats_component(stats: dict) -> str:
    def metric(label, count, total):
        return (
            f'<div class="yc-metric"><div class="yc-metric-label">{escape(label)}</div>'
            f'<div class="yc-metric-value">{count}</div>'
            f"<div>Sum of Margin: <b>{total:+.2f}</b></div></div>"
        )

    def ranking(label, items):
        lines = "".join(f"<div>{escape(str(c))}: <b>{v:+.2f}</b></div>" for c, v in items) or "<div>–</div>"
        return f'<div class="yc-metric"><b>{escape(label)}</b>{lines}</div>'

    return (
        '<div class="yc-stats">'
        + metric("Winning borders", stats["winning"], stats["winning_sum"])
        + metric("Losing borders", stats["losing"], stats["losing_sum"])
        + metric("Total borders", stats["total"], stats["total_sum"])
        + ranking("3 highest Margin", stats["highest"])
        + ranking("3 lowest Margin", stats["lowest"])
        + "</div>"
        + caption_box(stats["note"])
    )


# ---- layout ----------------------------------------------------------
TAB_LABELS = [
    ("yc", "Year comparison"),
    ("hb", "Hours below threshold"),
    ("cm", "Corridor margins"),
    ("mv", "Market value"),
    ("ov", "Market value - overview"),
]
SORT_OPTIONS = [CORRIDOR_SORT_BY_NAME] + CORRIDOR_MONTH_LABELS + ["Year"]

CM_CAPTION = (
    "Every JAO corridor, one year at a time. Auction price = the JAO "
    "auction price of the picked tenor, shown on each month that "
    "product delivers in (Yearly 2026 = the Cal-26 auction, on all 12 "
    "months). Settle = what the corridor's direction settled at "
    "day-ahead (hours the importing side was pricier, averaged over all "
    "hours of the month). Margin = Settle minus Auction price. The "
    "current month's Settle is month-to-date. Pick a Winner to put a "
    "fat line around every corridor/month where they won capacity of "
    "that tenor."
)
MV_CAPTION = (
    "What each corridor was worth to the whole market, in thousand EUR: "
    "MW sold at the JAO auction x hours in the month x Margin (Settle "
    "minus Auction price). All buyers together -- JAO does not publish "
    "MW per buyer. Assumes the full MW was available every hour. The "
    "running month is month-to-date. Weekly and Daily tenors are left "
    "out (they do not cover whole months)."
)
OV_CAPTION = (
    "Total market value per tenor and year, in million EUR: MW sold at "
    "the JAO auction x hours x Margin, added up over every corridor and "
    "month (same number as the Market value tab). Whole market, all "
    "buyers together. The current year is year-to-date. The first load "
    "takes a little while; after that it is kept until the data changes."
)
HB_CAPTION = (
    "Count of hours per period where the price was below the threshold. "
    "The countries here are picked separately from the Year comparison tab."
)
NO_JAO_TEXT = "No JAO auction data found (jao_data.db)."
NBSP = "\u00a0"  # an empty label that still takes one line of height


def field(label, control, width=None):
    style = {"width": width} if width else None
    return html.Div(className="yc-field", style=style, children=[html.Label(label), control])


def dropdown(component_id, options, value, width="20ch", **kwargs):
    return dcc.Dropdown(
        id=component_id, options=options, value=value, clearable=False,
        maxHeight=420, style={"width": width}, **kwargs,
    )


def radio(component_id, options, value):
    return dcc.RadioItems(id=component_id, options=options, value=value, inline=True, className="yc-radio")


def check(label, disabled=False):
    return [{"label": label, "value": "on", "disabled": disabled}]


def _default_zone(preferred: str, fallback_index: int) -> str:
    if preferred in ZONES:
        return preferred
    return PICKER_ZONES[min(fallback_index, len(PICKER_ZONES) - 1)]


def loading(name):
    """Where a tab's tables go. The callback puts the HTML text in the
    Store; PAINT_JS then writes it into the empty Div next to it."""
    return dcc.Loading(type="dot", delay_show=300, children=[
        dcc.Store(id=f"{name}-html"),
        html.Div(id=f"{name}-body"),
    ])


def serve_layout():
    resolutions = list(RESOLUTIONS.keys())
    country_1 = _default_zone("DE_LU", 0)
    country_2 = _default_zone("FR", 1)
    tenors = load_corridor_tenors()
    mv_tenors = [t for t in MARKET_VALUE_TENORS if t in tenors]
    pick = lambda options: "Monthly" if "Monthly" in options else (options[0] if options else None)  # noqa: E731

    tab_yc = html.Div(id="panel-yc", children=[
        html.Div(className="yc-controls", children=[
            field("Country 1", dropdown("yc-c1", PICKER_ZONES, country_1)),
            field("Country 2", dropdown("yc-c2", PICKER_ZONES, country_2)),
            field("Resolution", radio("yc-res", resolutions, resolutions[0])),
            field("Export year", dropdown("yc-exp-year", [], None, width="11ch")),
            field("Export month", dropdown("yc-exp-month", [], None, width="15ch")),
            html.Div(className="yc-field", children=[
                html.Label(NBSP),
                html.Button("Download Excel", id="yc-exp-btn", className="yc-button"),
                html.Div(id="yc-exp-note", className="yc-caption"),
            ]),
            dcc.Download(id="yc-download"),
        ]),
        loading("yc"),
    ])

    tab_hb = html.Div(id="panel-hb", style={"display": "none"}, children=[
        html.Div(HB_CAPTION, className="yc-caption"),
        html.Div(className="yc-controls", children=[
            field("Country 1", dropdown("hb-c1", PICKER_ZONES, country_1)),
            field("Country 2", dropdown("hb-c2", PICKER_ZONES, country_2)),
            field("Resolution", radio("hb-res", resolutions, resolutions[0])),
            field("Threshold (EUR/MWh)", dcc.Input(
                id="hb-threshold", type="number", value=0, step=10, className="yc-number")),
        ]),
        loading("hb"),
    ])

    cm_controls = html.Div(className="yc-controls", children=[
        field("Tenor", dropdown("cm-tenor", tenors, pick(tenors), width="13ch")),
        field("Year", dropdown("cm-year", [], None, width="11ch")),
        html.Div(className="yc-field", children=[
            html.Label("Winner"),
            dropdown("cm-winner", [CORRIDOR_NO_WINNER], CORRIDOR_NO_WINNER, width="34ch"),
            radio("cm-winner-mode", CORRIDOR_WINNER_MODES, CORRIDOR_WINNER_MODES[0]),
        ]),
        field("Show", radio("cm-view", CORRIDOR_VIEWS, CORRIDOR_VIEWS[0])),
        html.Div(className="yc-field", children=[
            html.Label(NBSP),
            dcc.Checklist(id="cm-running", options=check("Include running month"),
                          value=["on"], className="yc-check"),
        ]),
        html.Div(className="yc-field", children=[
            html.Label("Sort by"),
            dropdown("cm-sort", SORT_OPTIONS, CORRIDOR_SORT_BY_NAME, width="13ch"),
            dcc.Checklist(id="cm-desc", options=check("High to low", disabled=True),
                          value=["on"], className="yc-check"),
        ]),
    ])
    tab_cm = html.Div(id="panel-cm", style={"display": "none"}, children=[
        html.Div(CM_CAPTION, className="yc-caption"), cm_controls, loading("cm"),
    ])

    mv_controls = html.Div(className="yc-controls", children=[
        field("Tenor", dropdown("mv-tenor", mv_tenors, pick(mv_tenors), width="13ch")),
        field("Year", dropdown("mv-year", [], None, width="11ch")),
        field("Show", radio("mv-view", MARKET_VALUE_VIEWS, MARKET_VALUE_VIEWS[0])),
        html.Div(className="yc-field", children=[
            html.Label(NBSP),
            dcc.Checklist(id="mv-running", options=check("Include running month"),
                          value=["on"], className="yc-check"),
        ]),
        html.Div(className="yc-field", children=[
            html.Label("Sort by"),
            dropdown("mv-sort", SORT_OPTIONS, CORRIDOR_SORT_BY_NAME, width="13ch"),
            dcc.Checklist(id="mv-desc", options=check("High to low", disabled=True),
                          value=["on"], className="yc-check"),
        ]),
    ])
    tab_mv = html.Div(id="panel-mv", style={"display": "none"}, children=[
        html.Div(MV_CAPTION, className="yc-caption"), mv_controls, loading("mv"),
    ])

    tab_ov = html.Div(id="panel-ov", style={"display": "none"}, children=[
        html.Div(OV_CAPTION, className="yc-caption"),
        dcc.Checklist(id="ov-running", options=check("Include running month"),
                      value=["on"], className="yc-check"),
        loading("ov"),
    ])

    return html.Div(className="yc-page", children=[
        html.H1("Average price -- year comparison"),
        html.Div(
            "Average price by period of year, across years, for two countries. "
            "Data from entsoe_data.db.", className="yc-caption"),
        dcc.Tabs(id="tabs", value="yc", className="yc-tabs", children=[
            dcc.Tab(label=label, value=value) for value, label in TAB_LABELS
        ]),
        tab_yc, tab_hb, tab_cm, tab_mv, tab_ov,
    ])


CSS = """
body { margin: 0; background: #ffffff; color: #1f2430;
       font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
       font-size: 15px; }
.yc-page { padding: 18px 24px 60px; }
.yc-page h1 { font-size: 28px; margin: 0 0 4px; font-weight: 700; }
.yc-tabs { margin: 12px 0 16px; }
.yc-caption { color: #6b7180; font-size: 13px; margin: 4px 0 10px; max-width: 150ch; }
.yc-controls { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 12px 22px;
       margin: 6px 0 14px; }
.yc-field label { display: block; font-size: 13px; color: #555b6a; margin-bottom: 4px; }
.yc-radio { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 7px 0; }
.yc-radio label, .yc-check label { display: inline-flex !important; align-items: center;
       gap: 6px; font-size: 15px; color: #1f2430; margin: 0; cursor: pointer; }
.yc-check { padding: 7px 0; }
.yc-radio input:disabled + *, .yc-check input:disabled + *,
.yc-radio input:disabled, .yc-check input:disabled { opacity: .45; cursor: not-allowed; }
.yc-number { width: 9rem; padding: 7px 8px; border: 1px solid #c6cad3; border-radius: 6px;
       font: inherit; box-sizing: border-box; }
.yc-button { padding: 7px 14px; border: 1px solid #c6cad3; border-radius: 6px;
       background: #fff; font: inherit; cursor: pointer; }
.yc-button:hover:not(:disabled) { border-color: #2a78d6; color: #2a78d6; }
.yc-button:disabled { opacity: .45; cursor: not-allowed; }
.yc-info { background: #eaf2fc; border: 1px solid #c5dbf5; color: #1d4f8c;
       border-radius: 8px; padding: 10px 14px; margin: 6px 0; max-width: 60ch; }
.yc-row { display: flex; flex-wrap: nowrap; gap: 0; align-items: flex-start; }
.yc-block { flex: 0 0 auto; padding-right: 14px; }
.yc-block h3 { font-size: 17px; font-weight: 600; margin: 8px 0 8px; white-space: nowrap; }
.yc-block .yc-info { max-width: 32ch; white-space: normal; }
.yc-rule { border: 0; border-top: 1px solid #e3e5ea; margin: 18px 0 10px; }
.yc-scroll { overflow-y: scroll; overflow-x: hidden; display: inline-block; }
.yc-noscroll { display: inline-block; max-width: 100%; }
.yc-table { border-collapse: separate; border-spacing: 0; font-size: 13.5px;
       font-variant-numeric: tabular-nums; }
.yc-table th, .yc-table td { border-right: 1px solid #e6e6e6; border-bottom: 1px solid #e6e6e6;
       padding: 6px 8px; text-align: right; white-space: nowrap; box-sizing: border-box;
       color: #000000; }
.yc-table thead th { position: sticky; top: 0; z-index: 2; background: #fafafa;
       color: #555555; font-weight: 400; border-top: 1px solid #e6e6e6; }
.yc-table th.yc-rowhead { text-align: left; background: #fafafa; color: #555555;
       font-weight: 400; border-left: 1px solid #e6e6e6; }
.yc-table tr.yc-divider > * { border-bottom: 3px solid #1a1a1a; }
.yc-stats { display: flex; flex-wrap: wrap; gap: 10px 36px; margin: 8px 0 4px; }
.yc-metric-label { font-size: 13px; color: #555b6a; }
.yc-metric-value { font-size: 30px; font-weight: 600; line-height: 1.2; }
"""

# Scrolling one table of a row scrolls the others in that row with it, so
# the same period stays lined up (long Daily / Weekly tables).
SCROLL_SYNC_JS = """
document.addEventListener('scroll', function (event) {
    var el = event.target;
    if (!el.classList || !el.classList.contains('yc-scroll')) return;
    var row = el.closest('.yc-row');
    if (!row || row._syncing) return;
    row._syncing = true;
    row.querySelectorAll('.yc-scroll').forEach(function (other) {
        if (other !== el && other.scrollHeight === el.scrollHeight) other.scrollTop = el.scrollTop;
    });
    row._syncing = false;
}, true);
"""

app = Dash(__name__, title="Year comparison", update_title=None)
app.index_string = (
    app.index_string
    .replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    .replace("{%renderer%}", "{%renderer%}<script>" + SCROLL_SYNC_JS + "</script>")
)
app.layout = serve_layout


# ---- callbacks -------------------------------------------------------
# Writes a tab's HTML text (from its Store) into its Div. Runs in the browser.
PAINT_JS = """
function (html_text) {
    var target = document.getElementById('%s-body');
    if (target && typeof html_text === 'string') { target.innerHTML = html_text; }
    return window.dash_clientside.no_update;
}
"""
for _name, _ in TAB_LABELS:
    app.clientside_callback(
        PAINT_JS % _name,
        Output(f"{_name}-html", "storage_type"),
        Input(f"{_name}-html", "data"),
    )


@app.callback(
    [Output(f"panel-{value}", "style") for value, _ in TAB_LABELS],
    Input("tabs", "value"),
)
def show_tab(active):
    return [{} if value == active else {"display": "none"} for value, _ in TAB_LABELS]


def _on(value) -> bool:
    return bool(value)


def _missing_data_box(need_jao: bool = False):
    """A message instead of a table when a database file is not there."""
    if not Path(DB_PATH).is_file():
        return info_box(f"Database not found: {DB_PATH}")
    if need_jao and not Path(JAO_DB_FILE).is_file():
        return info_box(NO_JAO_TEXT)
    return None


@app.callback(
    Output("yc-html", "data"),
    Input("tabs", "value"),
    Input("yc-c1", "value"),
    Input("yc-c2", "value"),
    Input("yc-res", "value"),
)
def render_year_comparison(active, country_1, country_2, resolution):
    if active != "yc" or not country_1 or not country_2 or resolution not in RESOLUTIONS:
        return no_update
    problem = _missing_data_box()
    if problem:
        return problem
    rows = year_comparison_views(country_1, country_2, resolution)
    return '<hr class="yc-rule">'.join(tables_row(row) for row in rows)


@app.callback(
    Output("yc-exp-year", "options"),
    Output("yc-exp-year", "value"),
    Output("yc-exp-month", "options"),
    Output("yc-exp-month", "value"),
    Output("yc-exp-btn", "disabled"),
    Output("yc-exp-note", "children"),
    Input("yc-c1", "value"),
    Input("yc-c2", "value"),
    Input("yc-exp-year", "value"),
    State("yc-exp-month", "value"),
)
def fill_export_pickers(country_1, country_2, year, month):
    """Year / Month offered = months both countries have hourly prices
    for. Default: the latest."""
    if not country_1 or not country_2 or not Path(DB_PATH).is_file():
        return [], None, [], None, True, ""
    available = load_available_export_months(_entsoe_zone(country_1), _entsoe_zone(country_2))
    if not available:
        note = f"No overlapping hourly data for {country_1} and {country_2} yet."
        return [], None, [], None, True, note
    years = sorted({y for y, _ in available})
    if year not in years:
        year = years[-1]
    months = sorted(m for y, m in available if y == year)
    if month not in months:
        month = months[-1]
    month_options = [{"label": calendar.month_name[m], "value": m} for m in months]
    return years, year, month_options, month, False, ""


@app.callback(
    Output("yc-download", "data"),
    Input("yc-exp-btn", "n_clicks"),
    State("yc-c1", "value"),
    State("yc-c2", "value"),
    State("yc-exp-year", "value"),
    State("yc-exp-month", "value"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, country_1, country_2, year, month):
    if not n_clicks or not country_1 or not country_2 or year is None or month is None:
        return no_update
    df = load_hourly_export(
        _entsoe_zone(country_1), _entsoe_zone(country_2), int(year), int(month),
        label1=country_1, label2=country_2,
    )
    if df.empty:
        return no_update
    return dcc.send_bytes(
        _build_export_excel(df, country_1, country_2),
        f"{country_1}_{country_2}_hourly_{int(year)}-{int(month):02d}.xlsx",
    )


def _threshold(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


@app.callback(
    Output("hb-html", "data"),
    Input("tabs", "value"),
    Input("hb-c1", "value"),
    Input("hb-c2", "value"),
    Input("hb-res", "value"),
    Input("hb-threshold", "value"),
)
def render_hours_below(active, country_1, country_2, resolution, threshold):
    if active != "hb" or not country_1 or not country_2 or resolution not in RESOLUTIONS:
        return no_update
    problem = _missing_data_box()
    if problem:
        return problem
    threshold = _threshold(threshold)
    return tables_row([
        hours_below_view(_entsoe_zone(country_1), resolution, threshold, label=country_1),
        hours_below_view(_entsoe_zone(country_2), resolution, threshold, label=country_2),
    ])


def _pick_year(years: list, current):
    """Keep the picked year if the tenor has it, else this year, else the
    latest one."""
    if current in years:
        return current
    this_year = pd.Timestamp.now().year
    if this_year in years:
        return this_year
    return years[-1] if years else None


def _radio_options(labels, disabled):
    return [{"label": label, "value": label, "disabled": disabled} for label in labels]


@app.callback(
    Output("cm-year", "options"),
    Output("cm-year", "value"),
    Output("cm-winner", "options"),
    Output("cm-winner", "value"),
    Output("cm-winner-mode", "options"),
    Output("cm-desc", "options"),
    Output("cm-html", "data"),
    Input("tabs", "value"),
    Input("cm-tenor", "value"),
    Input("cm-year", "value"),
    Input("cm-winner", "value"),
    Input("cm-winner-mode", "value"),
    Input("cm-view", "value"),
    Input("cm-running", "value"),
    Input("cm-sort", "value"),
    Input("cm-desc", "value"),
)
def render_corridor_margins(active, tenor, year, winner, winner_mode, view, running, sort_by, descending):
    if active != "cm":
        return (no_update,) * 7
    problem = _missing_data_box(need_jao=True) if tenor else info_box(NO_JAO_TEXT)
    if problem:
        return no_update, no_update, no_update, no_update, no_update, no_update, problem
    years = load_corridor_years(tenor)
    year = _pick_year(years, year)
    winners = [CORRIDOR_NO_WINNER] + (load_corridor_winners(year, tenor) if year is not None else [])
    if winner not in winners:
        winner = CORRIDOR_NO_WINNER
    no_winner = winner == CORRIDOR_NO_WINNER
    mode_options = _radio_options(CORRIDOR_WINNER_MODES, disabled=no_winner)
    desc_options = check("High to low", disabled=sort_by == CORRIDOR_SORT_BY_NAME)
    if year is None:
        body = info_box(f"No {tenor} JAO auctions from {START_YEAR} onward.")
    else:
        result = corridor_margins_view(
            year, view, tenor,
            winner=None if no_winner else winner,
            include_running_month=_on(running),
            sort_by=sort_by,
            sort_descending=_on(descending),
            winner_show_all=winner_mode == CORRIDOR_WINNER_MODES[1],
        )
        body = ""
        if result["message"]:
            body += info_box(result["message"])
        if result["caption"]:
            body += caption_box(result["caption"])
        if result["stats"]:
            body += stats_component(result["stats"])
        if result["model"]:
            body += table_component(result["model"])
    return years, year, winners, winner, mode_options, desc_options, body


@app.callback(
    Output("mv-year", "options"),
    Output("mv-year", "value"),
    Output("mv-desc", "options"),
    Output("mv-html", "data"),
    Input("tabs", "value"),
    Input("mv-tenor", "value"),
    Input("mv-year", "value"),
    Input("mv-view", "value"),
    Input("mv-running", "value"),
    Input("mv-sort", "value"),
    Input("mv-desc", "value"),
)
def render_market_value(active, tenor, year, view, running, sort_by, descending):
    if active != "mv":
        return (no_update,) * 4
    problem = _missing_data_box(need_jao=True) if tenor else info_box(NO_JAO_TEXT)
    if problem:
        return no_update, no_update, no_update, problem
    years = load_corridor_years(tenor)
    year = _pick_year(years, year)
    desc_options = check("High to low", disabled=sort_by == CORRIDOR_SORT_BY_NAME)
    if year is None:
        body = info_box(f"No {tenor} JAO auctions from {START_YEAR} onward.")
    else:
        result = market_value_view(
            year, tenor, view, include_running_month=_on(running),
            sort_by=sort_by, sort_descending=_on(descending),
        )
        if result["message"]:
            body = info_box(result["message"])
        else:
            body = table_component(result["model"])
    return years, year, desc_options, body


@app.callback(
    Output("ov-html", "data"),
    Input("tabs", "value"),
    Input("ov-running", "value"),
)
def render_overview(active, running):
    if active != "ov":
        return no_update
    problem = _missing_data_box(need_jao=True)
    if problem:
        return problem
    result = market_value_overview_view(_on(running))
    if result["message"]:
        return info_box(result["message"])
    return table_component(result["model"])


def main():
    global DB_PATH, JAO_DB_FILE
    parser = argparse.ArgumentParser(description="Year comparison (Dash).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from the phone / Tailscale")
    parser.add_argument("--db", default=None, help=f"path to {DB_NAME}")
    parser.add_argument("--jao-db", default=None, help="path to jao_data.db")
    parser.add_argument("--open", action="store_true", help="open the browser")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.db:
        DB_PATH = find_db(args.db)
    if args.jao_db:
        JAO_DB_FILE = Path(args.jao_db).expanduser().resolve()
    for label, path in (("entsoe_data.db", DB_PATH), ("jao_data.db", JAO_DB_FILE)):
        if not Path(path).is_file():
            print(f"Warning: {label} not found at {path}", file=sys.stderr, flush=True)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Year comparison (Dash): {url}   databases: {DB_PATH} | {JAO_DB_FILE}", flush=True)
    if args.open:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
