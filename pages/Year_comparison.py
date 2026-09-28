"""
year_comparison.py -- average price by period-of-year, across years, for
two countries side by side.

You pick:
  - 2 countries (same list as dashboard.py's zone picker)
  - a resolution: Daily / Weekly / Monthly

Output: one table per country. Rows = period within the year (day,
ISO week, or month). Columns = years, starting at 2020. Values =
average price (EUR/MWh), 2 decimals -- same formatting rule as the
rest of this project.

Run it the same way as the other dashboards:
    streamlit run year_comparison.py
    (or: python -m streamlit run year_comparison.py, if the bare
    "streamlit" command isn't recognized in your terminal)

Reads only from entsoe_data.db (daily_prices / weekly_prices /
monthly_prices) -- never touches the ENTSO-E API, so it works offline
against whatever you've already fetched.
"""

import calendar
import io
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import colors as mcolors

# This cloud deployment only ever fetches/stores DE_LU and FR (Peter's
# pick for the online year-comparison dashboard's country pickers) plus
# GB (needed only because the 3 interconnector pseudo-countries below all
# price off GB's hourly price -- GB itself is never directly selectable).
# See fetch_recent_years.py/fetch_gb_bridge.py, and
# claude/entsoe-price-table-cloud-deploy.md. Trimmed from fetch_data.py's
# own full ~52-zone ZONES list (imported there via `from fetch_data import
# ZONES`) since fetch_data.py itself -- and the 655MB local database it
# assumes -- isn't part of this cloud deployment at all.
ZONES = ["DE_LU", "FR"]

# Resolved relative to THIS file's own location (cloud_deploy/pages/), not
# the current working directory, same reasoning as JAO_DB_FILE below --
# entsoe_data.db lives one level up, in cloud_deploy/ itself.
DB_FILE = Path(__file__).resolve().parent.parent / "entsoe_data.db"
START_YEAR = 2020

# The small cloud twin of jao_data.db (see fetch_jao_corridors.py) -- the
# real jao_scraper.py's own 183MB jao_data.db lives in a separate sibling
# project folder that isn't part of this deployment at all; this twin
# lives right next to entsoe_data.db in cloud_deploy/ instead. Resolved
# relative to THIS file's own location (cloud_deploy/pages/), not the
# current working directory.
JAO_DB_FILE = Path(__file__).resolve().parent.parent / "jao_data.db"

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


@st.cache_data(ttl=60)
def load_pivot(zone: str, resolution: str) -> pd.DataFrame:
    conf = RESOLUTIONS[resolution]
    conn = sqlite3.connect(DB_FILE)
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


@st.cache_data(ttl=60)
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


@st.cache_data(ttl=60)
def load_pivot_peak(zone: str, resolution: str) -> pd.DataFrame:
    """Same shape as load_pivot(), but averaged over Peak hours only. The
    daily/weekly/monthly aggregate tables only store the all-hours
    average, so this reads straight from the hourly day_ahead_prices
    table instead and does the Peak filtering + period averaging here."""
    conf = RESOLUTIONS[resolution]
    conn = sqlite3.connect(DB_FILE)
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


@st.cache_data(ttl=60)
def load_hours_below(zone: str, resolution: str, threshold: float) -> pd.DataFrame:
    """Same shape as load_pivot()/load_pivot_peak(), but each cell counts
    the hours in that period whose price was below `threshold` (EUR/MWh),
    rather than averaging the price. Built from the hourly
    day_ahead_prices table (like load_pivot_peak()) -- the pre-aggregated
    daily/weekly/monthly tables only store the period average, not a
    per-hour breakdown to count against a threshold."""
    conf = RESOLUTIONS[resolution]
    conn = sqlite3.connect(DB_FILE)
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


@st.cache_data(ttl=60)
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


@st.cache_data(ttl=60)
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
    conn = sqlite3.connect(DB_FILE)
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


@st.cache_data(ttl=60)
def load_available_export_months(zone1: str, zone2: str) -> list[tuple[int, int]]:
    """Every (year, month) that has at least one hourly row for BOTH
    zones -- what the Year/Month export pickers offer. Sorted oldest
    first, so the most recent month is last (a sensible default index)."""
    conn = sqlite3.connect(DB_FILE)
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


@st.cache_data(ttl=60)
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
    conn = sqlite3.connect(DB_FILE)
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


def _heatmap(pivot: pd.DataFrame):
    """DE_LU / FR tables: green = high, red = low relative to the rest of
    THIS table's own values (not zero) -- scaled across the whole table
    (not per-row/per-column) so every cell is comparable to every other
    cell in it. Missing values get a plain white cell instead of
    background_gradient's own default (black), and every number is plain
    black text, readable on every background shade."""
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin = vmax = None
    else:
        vmin, vmax = np.percentile(finite, [HEATMAP_CLIP_PERCENTILE, 100 - HEATMAP_CLIP_PERCENTILE])
        if vmin == vmax:
            # Degenerate case (every value the same, or too few distinct
            # values for percentiles to spread out) -- fall back to the
            # plain min/max so background_gradient still has a real range
            # to work with instead of a single point.
            vmin, vmax = float(finite.min()), float(finite.max())

    return (
        pivot.style.background_gradient(cmap=HEATMAP_CMAP, axis=None, vmin=vmin, vmax=vmax)
        .highlight_null(color="white")
        .set_properties(**{"color": "black"})
    )


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


def _heatmap_spread(pivot: pd.DataFrame):
    """Spread table only: positive = green, negative = red, scaled across
    the whole table -- and every number is plain black text, readable on
    every background shade."""
    bound = _heatmap_sign_bound(pivot)
    return pivot.style.map(lambda v: _sign_color(v, bound)).set_properties(
        **{"color": "black"}
    )


def _show_pivot(pivot: pd.DataFrame, resolution: str, spread: bool = False, fmt: str = "%.2f") -> None:
    """Shared rendering for every table on the page: Q1-Q4/Year summary
    rows (Monthly only) + autosized columns + heatmap coloring + a height
    tall enough to avoid an inner scrollbar. fmt is the per-cell number
    format -- "%.2f" (the default) for every price/spread/margin table,
    "%d" for the Hours-below-threshold tables (a whole number of hours,
    not a price)."""
    if resolution == "Monthly" and not pivot.empty:
        pivot = _append_quarter_year_rows(pivot)

    year_columns = {
        str(year): st.column_config.NumberColumn(str(year), format=fmt)
        for year in pivot.columns
    }
    year_columns["_index"] = st.column_config.Column("Period")
    pivot.columns = [str(c) for c in pivot.columns]
    # Tall enough to show every row without an inner scrollbar -- capped so
    # a huge table (Daily = 366 rows) doesn't take over the whole page;
    # Monthly's 12 months + Q1-Q4 + Year (17 rows) always fit well under
    # the cap.
    height = min(HEADER_HEIGHT_PX + len(pivot) * ROW_HEIGHT_PX + 3, 650)
    heatmap_fn = _heatmap_spread if spread else _heatmap
    # width="content" (not "stretch"): the table only takes as much width
    # as its (now autosized) columns actually need, instead of stretching
    # to fill the whole column and leaving a big blank gap on the right.
    st.dataframe(heatmap_fn(pivot), column_config=year_columns, width="content", height=height)


def show_country(zone: str, resolution: str, peak: bool = False, label: str | None = None) -> None:
    display = label or zone  # the interconnector pseudo-name (e.g. "IFA"), not the "GB" it actually queries
    unit = "€/MWh, Peak" if peak else "€/MWh, Base"  # unit sits next to the country name, not per-cell
    st.subheader(f"{display}  —  {unit}")
    pivot = load_pivot_peak(zone, resolution) if peak else load_pivot(zone, resolution)
    if pivot.empty:
        st.info(f"No data for {display} from {START_YEAR} onward.")
        return
    _show_pivot(pivot, resolution)


def show_spread(
    zone1: str, zone2: str, resolution: str, peak: bool = False,
    label1: str | None = None, label2: str | None = None,
) -> None:
    # "{zone1} minus {zone2}" sits on the SAME line as the title, to the
    # right of it, rather than on its own line below -- a second line
    # here would push this table down relative to the other two (whose
    # headers are single-line), throwing off the row-of-3 alignment.
    d1, d2 = label1 or zone1, label2 or zone2
    unit = "€/MWh, Peak" if peak else "€/MWh, Base"
    st.subheader(f"Spread  —  {unit}   ({d1} minus {d2})")
    spread = load_spread_peak(zone1, zone2, resolution) if peak else load_spread(zone1, zone2, resolution)
    if spread.empty:
        st.info(f"No data to compare {d1} and {d2} from {START_YEAR} onward.")
        return
    _show_pivot(spread, resolution, spread=True)


def show_capture(
    zone1: str, zone2: str, resolution: str, positive: bool,
    label1: str | None = None, label2: str | None = None,
) -> None:
    """The 2 capture-value tables at the bottom. positive=True: the
    "{zone2}-{zone1}" table -- average per-hour value of the hours zone1
    was pricier than zone2 that period. positive=False: the
    "{zone1}-{zone2}" table -- same idea for the hours zone2 was pricier,
    shown as a positive number."""
    positive_pivot, negative_pivot = load_capture_values(zone1, zone2, resolution)
    pivot = positive_pivot if positive else negative_pivot
    d1, d2 = label1 or zone1, label2 or zone2
    title = f"{d2}-{d1}" if positive else f"{d1}-{d2}"
    st.subheader(f"{title}  —  €/MWh")
    if pivot.empty:
        st.info(f"No hourly data to compare {d1} and {d2} from {START_YEAR} onward.")
        return
    _show_pivot(pivot, resolution)


def show_hours_below(zone: str, resolution: str, threshold: float, label: str | None = None) -> None:
    """The 2 tables on the Hours-below-threshold tab: how many hours per
    period the price was below `threshold` -- a count, not an average, so
    rendered with fmt="%d" rather than _show_pivot()'s usual 2-decimal
    price format."""
    display = label or zone
    st.subheader(f"{display}  —  Hours below {threshold:g} €/MWh")
    pivot = load_hours_below(zone, resolution, threshold)
    if pivot.empty:
        st.info(f"No hourly data for {display} from {START_YEAR} onward.")
        return
    _show_pivot(pivot, resolution, fmt="%d")


@st.cache_data(ttl=60)
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
        conn = sqlite3.connect(f"file:{JAO_DB_FILE}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT DISTINCT corridor_code FROM auctions").fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return set()
    return {r[0] for r in rows}


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


@st.cache_data(ttl=60)
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
        conn = sqlite3.connect(f"file:{JAO_DB_FILE}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return pd.DataFrame()
    try:
        products = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT r.product_identification "
                "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
                "WHERE a.corridor_code = ? AND a.horizon_name = 'Monthly'",
                (corridor,),
            ).fetchall()
        ]
        if not products:
            return pd.DataFrame()
        product = "BASE------" if "BASE------" in products else products[0]

        df = pd.read_sql_query(
            "SELECT a.market_period_start, r.auction_price "
            "FROM auction_results r JOIN auctions a ON a.identification = r.identification "
            "WHERE a.corridor_code = ? AND a.horizon_name = 'Monthly' AND r.product_identification = ?",
            conn,
            params=(corridor, product),
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


def show_jao_corridor(corridor: str | None, label: str) -> None:
    """One of the 2 JAO Monthly-auction-price tables appended after the
    Peak row's 3 ENTSO-E tables (Peter: "take the 2 tables from jao
    auction data and add them after the 3 tables in row 2 ... need to be
    the corridors corresponding to country 1 and 2"). Always shows JAO's
    Monthly-horizon price, independent of this page's own Daily/Weekly/
    Monthly resolution picker -- JAO's corridor explorer only has a
    Monthly pivot to begin with, there's no Daily/Weekly equivalent to
    switch to. `corridor` is None when JAO has no matching corridor for
    this zone pair/direction (see _jao_corridors_for) -- handled the same
    way show_country()/show_spread() handle a genuinely empty pivot."""
    st.subheader(f"{label}  —  JAO auction price, EUR/MW")
    if corridor is None:
        st.info("No matching JAO corridor for this pair/direction.")
        return
    pivot = load_jao_monthly_price(corridor)
    if pivot.empty:
        st.info(f"No Monthly JAO auctions saved for {corridor} yet.")
        return
    _show_pivot(pivot, "Monthly")


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


def show_margin(capture_pivot: pd.DataFrame, corridor: str | None, label: str) -> None:
    """One of the 2 new margin tables at the very front of row 2 --
    capture value minus the JAO Monthly auction price for the same flow
    direction. Always Monthly, same reason show_jao_corridor() always is
    -- JAO has no Daily/Weekly equivalent to fall back to."""
    st.subheader(f"Margin: {label}  —  €/MWh (capture minus JAO price)")
    if corridor is None:
        st.info("No matching JAO corridor for this pair/direction.")
        return
    jao_pivot = load_jao_monthly_price(corridor)
    pivot = _margin_pivot(capture_pivot, jao_pivot)
    if pivot.empty:
        st.info("Not enough capture-value or JAO auction data yet to compute a margin.")
        return
    _show_pivot(pivot, "Monthly")


st.set_page_config(page_title="Year comparison", layout="wide")
st.title("Average price -- year comparison")
st.caption(
    "Average price by period of year, across years, for two countries. "
    "Data from entsoe_data.db."
)

# Fixes the two country selectboxes to COUNTRY_PICKER_DIGITS characters
# wide -- Streamlit has no built-in width-in-characters option for
# selectboxes, so this targets them directly with CSS ("ch" is a real CSS
# unit: the width of the "0" character, so this lines up exactly with
# "N digits/characters wide"). Only stSelectbox widgets are matched, so
# the resolution radio buttons below are unaffected.
st.markdown(
    f"""
    <style>
    div[data-testid="stSelectbox"] {{
        max-width: {COUNTRY_PICKER_DIGITS}ch;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# How much space sits between the tables, in characters -- 0 = as close
# as possible (edge to edge, no gap at all). The "tables-row" /
# "tables-row-mid" / "tables-row-peak" keys below give each row's own
# element the CSS class "st-key-<key>" directly (Streamlit puts the key
# class on the row element itself, not on a wrapper around it), so this
# targets those 3 rows by themselves and doesn't affect the picker row
# above.
#
# overflow-x is forced to "visible" (Streamlit's own default for this kind
# of row is "auto") so this row is never its OWN independently-scrollable
# strip -- with "auto", the tables in it sit inside one shared horizontal
# scroll region, so scrolling over any one of them (even a small sideways
# nudge from a trackpad) drags the rest of that row sideways too, which
# looked like "scrolling one table scrolls the others". "visible" removes
# that shared scroll region entirely: each table's own vertical scrolling
# (e.g. Daily, with hundreds of rows) stays independent, and on the rare
# screen too narrow to fit a whole row, the overflow simply falls through
# to Streamlit's own page-level horizontal scrollbar instead.
TABLES_GAP_DIGITS = 0
st.markdown(
    f"""
    <style>
    div[data-testid="stHorizontalBlock"].st-key-tables-row,
    div[data-testid="stHorizontalBlock"].st-key-tables-row-mid,
    div[data-testid="stHorizontalBlock"].st-key-tables-row-peak,
    div[data-testid="stHorizontalBlock"].st-key-tables-row-hours-below {{
        gap: {TABLES_GAP_DIGITS}ch !important;
        flex-wrap: nowrap !important;
        overflow-x: visible !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _default_zone_index(preferred: str, fallback: int) -> int:
    """Index of `preferred` in ZONES, or `fallback` if it's not present
    (e.g. this database's ZONES list doesn't include it)."""
    return ZONES.index(preferred) if preferred in ZONES else fallback


tab1, tab2 = st.tabs(["Year comparison", "Hours below threshold"])

with tab1:
    # Narrow, tightly-spaced columns for the two country pickers (they're
    # already capped to COUNTRY_PICKER_DIGITS wide via the CSS above -- giving
    # them narrow columns too, with gap="small", keeps Country 2 sitting right
    # next to Country 1 instead of drifting across a wide, evenly-split
    # column) -- resolution gets its own share, and the Export hourly data to
    # Excel controls (Year / Month / Download) sit right after it in the same
    # row rather than in a separate section further down the page.
    pick_col1, pick_col2, pick_col3, exp_col1, exp_col2, exp_col3 = st.columns(
        [1, 1, 1.6, 0.9, 1.1, 1.3], gap="small"
    )
    with pick_col1:
        country_1 = st.selectbox("Country 1", PICKER_ZONES, index=_default_zone_index("DE_LU", 0))
    with pick_col2:
        default_2 = 1 if len(ZONES) > 1 else 0
        country_2 = st.selectbox("Country 2", PICKER_ZONES, index=_default_zone_index("FR", default_2))
    with pick_col3:
        resolution = st.radio("Resolution", list(RESOLUTIONS.keys()), horizontal=True)

    # The zone actually queried in entsoe_data.db for each pick -- a plain
    # passthrough unless the pick is one of the 3 interconnector
    # pseudo-countries (see INTERCONNECTOR_ENTSOE_ZONE), which all resolve to
    # GB's price. country_1/country_2 stay as the picked (possibly pseudo)
    # names for display and for _jao_corridors_for(), which needs to tell
    # them apart.
    entsoe_zone_1 = _entsoe_zone(country_1)
    entsoe_zone_2 = _entsoe_zone(country_2)

    available_months = load_available_export_months(entsoe_zone_1, entsoe_zone_2)
    export_year = export_month = None
    with exp_col1:
        if not available_months:
            st.selectbox("Export year", [], disabled=True)
        else:
            available_years = sorted({year for year, _ in available_months})
            export_year = st.selectbox("Export year", available_years, index=len(available_years) - 1)
    with exp_col2:
        if not available_months:
            st.selectbox("Export month", [], disabled=True)
        else:
            months_this_year = sorted(m for y, m in available_months if y == export_year)
            export_month = st.selectbox(
                "Export month",
                months_this_year,
                index=len(months_this_year) - 1,
                format_func=lambda m: calendar.month_name[m],
            )
    with exp_col3:
        st.write("")  # vertical spacer to line the button up with the pickers
        if not available_months:
            st.caption(f"No overlapping hourly data for {country_1} and {country_2} yet.")
        else:
            export_df = load_hourly_export(
                entsoe_zone_1, entsoe_zone_2, export_year, export_month,
                label1=country_1, label2=country_2,
            )
            if export_df.empty:
                st.caption("No hourly data for that month.")
            else:
                st.download_button(
                    "Download Excel",
                    data=_build_export_excel(export_df, country_1, country_2),
                    file_name=(
                        f"{country_1}_{country_2}_hourly_{export_year}-{export_month:02d}.xlsx"
                    ),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # st.columns() always splits the row into equal-width slots regardless of
    # how wide each table actually is, which left visible whitespace between
    # tables no matter how tight the gap was set. A horizontal container of
    # content-width sub-containers instead gives each table only as much
    # space as it actually needs, so they sit truly next to each other.
    # jao_corridor_1_2/2_1 and the Monthly capture pivots are needed by row 1
    # (the 2 margin tables) and reused below by row 2 (the JAO corridor
    # tables themselves) -- computed once, up here, for both.
    jao_corridor_1_2, jao_corridor_2_1 = _jao_corridors_for(country_1, country_2)
    capture_positive_monthly, capture_negative_monthly = load_capture_values(
        entsoe_zone_1, entsoe_zone_2, "Monthly"
    )

    with st.container(key="tables-row", horizontal=True, wrap=False):
        with st.container(width="content"):
            show_country(entsoe_zone_1, resolution, label=country_1)
        with st.container(width="content"):
            show_country(entsoe_zone_2, resolution, label=country_2)
        with st.container(width="content"):
            show_spread(entsoe_zone_1, entsoe_zone_2, resolution, label1=country_1, label2=country_2)
        # Margin: capture value minus JAO auction price, same direction --
        # capture_positive_monthly is the zone2->zone1 flow value (same
        # direction as jao_corridor_2_1), capture_negative_monthly is
        # zone1->zone2 (same direction as jao_corridor_1_2) -- see
        # load_capture_values() and show_jao_corridor(). Moved up here from
        # row 2, per Peter.
        with st.container(width="content"):
            margin_label_2_1 = jao_corridor_2_1 or f"{country_2}-{country_1}"
            show_margin(capture_positive_monthly, jao_corridor_2_1, margin_label_2_1)
        with st.container(width="content"):
            margin_label_1_2 = jao_corridor_1_2 or f"{country_1}-{country_2}"
            show_margin(capture_negative_monthly, jao_corridor_1_2, margin_label_1_2)

    st.divider()

    # Row 2, 4 tables: the capture-value tables (moved down from row 1's
    # original last 2 tables) followed by the 2 JAO corridor auction-price
    # tables -- the corridors corresponding to Country 1 and Country 2, one
    # per direction.
    with st.container(key="tables-row-mid", horizontal=True, wrap=False):
        # Capture value: split the (country_1 minus country_2) hourly spread by
        # sign, and for each period, average the positive hours and the
        # negative hours (as a positive number) separately over ALL hours in
        # that period -- see load_capture_values() for the full definition.
        # Two tables since the two directions don't share a color scale or a
        # "which one's bigger" comparison the way DE_LU/FR/Spread do.
        with st.container(width="content"):
            show_capture(
                entsoe_zone_1, entsoe_zone_2, resolution, positive=True,
                label1=country_1, label2=country_2,
            )
        with st.container(width="content"):
            show_capture(
                entsoe_zone_1, entsoe_zone_2, resolution, positive=False,
                label1=country_1, label2=country_2,
            )
        # The 2 JAO corridor auction-price tables. From a SEPARATE database
        # (jao_data.db), not entsoe_data.db -- see JAO_DB_FILE and
        # show_jao_corridor() above.
        with st.container(width="content"):
            jao_label_2_1 = jao_corridor_2_1 or f"{country_2}-{country_1}"
            show_jao_corridor(jao_corridor_2_1, jao_label_2_1)
        with st.container(width="content"):
            jao_label_1_2 = jao_corridor_1_2 or f"{country_1}-{country_2}"
            show_jao_corridor(jao_corridor_1_2, jao_label_1_2)

    st.divider()

    # Row 3: same first 3 tables again, but Peak hours only (Mon-Fri,
    # 08:00-20:00 local time) instead of the all-hours average above -- moved
    # down to row 3 to make room for row 2 above.
    with st.container(key="tables-row-peak", horizontal=True, wrap=False):
        with st.container(width="content"):
            show_country(entsoe_zone_1, resolution, peak=True, label=country_1)
        with st.container(width="content"):
            show_country(entsoe_zone_2, resolution, peak=True, label=country_2)
        with st.container(width="content"):
            show_spread(
                entsoe_zone_1, entsoe_zone_2, resolution, peak=True,
                label1=country_1, label2=country_2,
            )

with tab2:
    st.caption(
        "Count of hours per period where the price was below the "
        "threshold. Defaults to the countries picked in Year comparison "
        "-- change them here without affecting that tab."
    )
    hb_default_1 = PICKER_ZONES.index(country_1) if country_1 in PICKER_ZONES else 0
    hb_default_2 = PICKER_ZONES.index(country_2) if country_2 in PICKER_ZONES else (1 if len(PICKER_ZONES) > 1 else 0)
    hb_pick_col1, hb_pick_col2, hb_pick_col3, hb_pick_col4 = st.columns(
        [1, 1, 1.6, 1.2], gap="small"
    )
    with hb_pick_col1:
        hb_country_1 = st.selectbox(
            "Country 1", PICKER_ZONES, index=hb_default_1, key="hb_country_1"
        )
    with hb_pick_col2:
        hb_country_2 = st.selectbox(
            "Country 2", PICKER_ZONES, index=hb_default_2, key="hb_country_2"
        )
    with hb_pick_col3:
        hb_resolution = st.radio(
            "Resolution", list(RESOLUTIONS.keys()), horizontal=True, key="hb_resolution"
        )
    with hb_pick_col4:
        hb_threshold = st.number_input(
            "Threshold (EUR/MWh)", value=0.0, step=10.0, key="hb_threshold"
        )

    hb_zone_1 = _entsoe_zone(hb_country_1)
    hb_zone_2 = _entsoe_zone(hb_country_2)

    with st.container(key="tables-row-hours-below", horizontal=True, wrap=False):
        with st.container(width="content"):
            show_hours_below(hb_zone_1, hb_resolution, hb_threshold, label=hb_country_1)
        with st.container(width="content"):
            show_hours_below(hb_zone_2, hb_resolution, hb_threshold, label=hb_country_2)

# Scrolling one table (vertically or horizontally) scrolls the other 2 in
# its row along with it, by the same amount -- so the same period/year
# stays lined up across DE_LU / FR / Spread as you scroll through a long
# Daily or Weekly table. Also draws the fat divider lines under December
# and under Q4 (Monthly only) -- st.dataframe (Streamlit's canvas-rendered
# grid) has no per-row border option, not even through pandas Styler CSS
# (tried it -- background-color/color/font-weight come through, border-*
# is silently dropped), so there's no cell/row to attach a real border to.
# This instead overlays a plain colored div on top of the grid at the pixel
# offset where each boundary falls -- HEADER_HEIGHT_PX + N rows down from
# each table's own top edge (N = 12 after Jan-Dec, 16 after Q1-Q4), which
# is exact because Monthly's row/header heights are fixed and known (same
# constants _show_pivot sizes the table with) and that block of rows above
# each divider never changes length or scrolls.
#
# st.markdown() strips out <script> tags, so both of these are injected via
# components.html() instead (a 0-height iframe); the script reaches into
# the MAIN page via window.parent.document. A MutationObserver re-attaches
# the scroll sync and redraws the divider lines after every rerun
# (switching country/resolution replaces the actual table elements), and a
# per-row "_syncing" flag stops each mirrored scroll from re-triggering the
# other two in an infinite loop.
_is_monthly_js = "true" if resolution == "Monthly" else "false"
_hb_is_monthly_js = "true" if hb_resolution == "Monthly" else "false"
components.html(
    """
    <script>
    (function () {
        var IS_MONTHLY = """ + _is_monthly_js + """;
        var HOURS_BELOW_IS_MONTHLY = """ + _hb_is_monthly_js + """;
        var HEADER_HEIGHT_PX = """ + str(HEADER_HEIGHT_PX) + """;
        var ROW_HEIGHT_PX = """ + str(ROW_HEIGHT_PX) + """;
        // Row counts (from the top of the table) after which a divider is
        // drawn: 12 = after December (Jan..Dec), 16 = after Q4 (the 4
        // quarter rows that follow it), leaving Year on its own below.
        var DIVIDER_AFTER_ROWS = [12, 16];
        var DIVIDER_THICKNESS_PX = 3;
        var DIVIDER_COLOR = "#1a1a1a";

        function setupSync() {
            const doc = window.parent.document;
            const groups = doc.querySelectorAll(".st-key-tables-row, .st-key-tables-row-mid, .st-key-tables-row-peak, .st-key-tables-row-hours-below");
            groups.forEach(function (group) {
                const scrollers = Array.from(group.querySelectorAll(".dvn-scroller"));
                if (scrollers.length < 2) return;
                scrollers.forEach(function (el) {
                    if (el.dataset.scrollSynced) return;
                    el.dataset.scrollSynced = "1";
                    el.addEventListener("scroll", function () {
                        if (group._syncing) return;
                        group._syncing = true;
                        const top = el.scrollTop;
                        const left = el.scrollLeft;
                        scrollers.forEach(function (other) {
                            if (other !== el) {
                                other.scrollTop = top;
                                other.scrollLeft = left;
                            }
                        });
                        group._syncing = false;
                    });
                });
            });
        }

        function setupDividers() {
            const doc = window.parent.document;
            doc.querySelectorAll(".quarter-divider-line").forEach(function (el) {
                el.remove();
            });

            // Anchored to the table's OWN box (as a child of it) rather
            // than document.body at a page-scroll-computed pixel offset:
            // Streamlit's actual scrolling element is an inner
            // [data-testid="stMain"] container, not the page/window, so
            // a body-appended, scroll-position-computed line went stale
            // (stayed put) the moment that inner container scrolled
            // instead of the page. Being a real child of the table means
            // it scrolls with it natively, no tracking needed.
            function drawDividersOn(dataframes) {
                dataframes.forEach(function (df) {
                    if (getComputedStyle(df).position === "static") {
                        df.style.position = "relative";
                    }
                    DIVIDER_AFTER_ROWS.forEach(function (rowsBefore) {
                        const line = doc.createElement("div");
                        line.className = "quarter-divider-line";
                        line.style.position = "absolute";
                        line.style.left = "0";
                        line.style.top = (HEADER_HEIGHT_PX + rowsBefore * ROW_HEIGHT_PX) + "px";
                        line.style.width = "100%";
                        line.style.height = DIVIDER_THICKNESS_PX + "px";
                        line.style.background = DIVIDER_COLOR;
                        line.style.pointerEvents = "none";
                        line.style.zIndex = 1000;
                        df.appendChild(line);
                    });
                });
            }

            // The Hours-below row has its OWN, independent resolution
            // picker (hb_resolution) -- separate from the main
            // "resolution" IS_MONTHLY reflects -- so it's excluded from
            // the blanket pass below and handled on its own via
            // HOURS_BELOW_IS_MONTHLY. Without this split, e.g. main
            // resolution=Daily + hb_resolution=Monthly would leave the
            // Hours-below tables without dividers (IS_MONTHLY false), or
            // the reverse would draw spurious divider lines on Daily's
            // 366-row tables (IS_MONTHLY true, applied blanket).
            if (IS_MONTHLY) {
                const dataframes = Array.from(doc.querySelectorAll('[data-testid="stDataFrame"]')).filter(function (df) {
                    return !df.closest(".st-key-tables-row-hours-below");
                });
                drawDividersOn(dataframes);
            }
            if (HOURS_BELOW_IS_MONTHLY) {
                drawDividersOn(doc.querySelectorAll('.st-key-tables-row-hours-below [data-testid="stDataFrame"]'));
            }
        }

        // setupDividers() mutates the DOM (removes/re-adds lines), and the
        // observer below watches for exactly that kind of mutation -- so
        // without disconnecting first, every redraw would trigger another
        // redraw, forever. Disconnect while we make our own changes, then
        // reconnect so it still catches the next REAL change (a rerun
        // swapping in new table elements).
        var observer = new MutationObserver(setupAll);
        function setupAll() {
            observer.disconnect();
            setupSync();
            setupDividers();
            observer.observe(window.parent.document.body, { childList: true, subtree: true });
            setupTabRerun();
        }

        // The Hours-below tab starts hidden (Year comparison is the
        // default active tab). Streamlit keeps both tabs' elements
        // mounted in the DOM the whole time -- switching tabs is a CSS
        // visibility change, not new elements being added -- so the
        // MutationObserver above never fires just from a tab switch.
        // st.dataframe's canvas-based grid can fail to size its
        // ".dvn-scroller" element correctly while its container sits at
        // display:none, so a scroll-sync attached while Hours-below was
        // still hidden may not actually work once it becomes visible.
        // Re-running setupAll() shortly after a tab click re-attaches
        // sync against the now-visible, now-correctly-sized elements.
        function setupTabRerun() {
            const doc = window.parent.document;
            const tabsRoot = doc.querySelector('[data-testid="stTabs"]');
            if (!tabsRoot || tabsRoot.dataset.tabRerunAttached) return;
            tabsRoot.dataset.tabRerunAttached = "1";
            tabsRoot.addEventListener("click", function (e) {
                if (!e.target.closest("button")) return;
                setTimeout(setupAll, 150);
            });
        }

        setupAll();
        window.parent.addEventListener("resize", setupAll);
    })();
    </script>
    """,
    height=0,
)
