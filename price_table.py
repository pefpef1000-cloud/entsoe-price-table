"""
price_table.py - one day, every country side by side.

Shows a table for a single day:
  - rows: the 24 hours (00:00 .. 23:00), one column per country/zone
  - then three summary rows: Base, Peak, Off-peak

  Base     = average of all 24 hourly prices
  Peak     = average of the 12 "peak" hours, 08:00-20:00 (i.e. the hours
             starting at 08:00 up to and including 19:00)
  Off-peak = 2 x Base - Peak
             (this is exact, not an approximation: since peak and off-peak
             are each exactly 12 hours, Base is just the average of Peak
             and Off-peak, so Off-peak = 2*Base - Peak falls straight out
             of that. It also matches the actual average of the 12
             off-peak hours (00:00-08:00 and 20:00-24:00) - checked
             against synthetic data before delivery.)

There's also a toggle for a Nordic/Baltic-only view: SYS, DK_1, DK_2,
NO_1-4, SE_1-4, FI, EE, LV, LT - shown twice side by side, first as raw
prices, then again as each zone's price minus the SYS (system) price for
that same hour, so you can see each zone's spread to the system price at
a glance.

Date picker defaults to TOMORROW every time the page loads (day-ahead
prices for tomorrow are usually the ones you actually want to look at,
since today's/yesterday's are already old news) - pick any other date
from the dropdown/calendar if you want a different day.

Run it the same way as the other dashboards:
    streamlit run price_table.py
    (or: python -m streamlit run price_table.py, if the bare command
    isn't recognized in your terminal)
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

DB_FILE = "entsoe_data.db"

PEAK_START_HOUR = 8   # inclusive
PEAK_END_HOUR = 20    # exclusive -> peak hours are 08:00 through 19:00 (12 hours)

st.set_page_config(page_title="Hourly price table - all countries", layout="wide")
st.title("Hourly day-ahead prices - all countries")

# ---------------------------------------------------------------------
# Date picker - default is always tomorrow
# ---------------------------------------------------------------------
tomorrow = date.today() + timedelta(days=1)
picked_date = st.date_input("Date", value=tomorrow)


def _find_price_column(conn: sqlite3.Connection) -> str:
    """Work out which column in day_ahead_prices actually holds the price.

    Different builds of this project have used different column names
    here (e.g. "price" vs "price_eur_mwh") - rather than hardcode one and
    break if it doesn't match, this looks at the table's real columns and
    picks whichever one isn't "zone" or "timestamp". If that's not
    obvious, it falls back to a few common names before giving up with a
    clear error instead of a cryptic "no such column" one.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(day_ahead_prices)")]
    known = {"zone", "timestamp"}
    candidates = [c for c in cols if c not in known]
    if len(candidates) == 1:
        return candidates[0]
    for guess in ("price", "price_eur_mwh", "eur_mwh", "value", "price_eur"):
        if guess in cols:
            return guess
    raise RuntimeError(
        "Could not figure out which column in day_ahead_prices holds the "
        f"price. Columns found: {cols}. Open entsoe_data.db (e.g. with "
        "DB Browser for SQLite) and check the real column name."
    )


@st.cache_data(ttl=60)
def load_day(day: date) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    price_col = _find_price_column(conn)
    # Match on the first 10 characters of the stored timestamp (the plain
    # "YYYY-MM-DD" calendar date) rather than a BETWEEN range built from a
    # guessed "YYYY-MM-DD HH:MM:SS" boundary string. A straight BETWEEN
    # comparison is comparing TEXT, not real datetimes - if the timestamps
    # in this database happen to use a "T" separator or a different
    # format than the boundary strings guess, every row silently fails to
    # match and the query comes back empty even on a date that clearly
    # has data (confirmed against a synthetic "T"-separated database).
    # Matching just the date prefix sidesteps that entirely - every
    # variant this project has used still starts with "YYYY-MM-DD".
    query = f"""
        SELECT zone, timestamp, {price_col} AS price
        FROM day_ahead_prices
        WHERE substr(timestamp, 1, 10) = ?
    """
    df = pd.read_sql_query(query, conn, params=(day.isoformat(),))
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")
    df["hour"] = df["timestamp"].dt.strftime("%H:%M")
    df["hour_num"] = df["timestamp"].dt.hour
    return df


def _robust_vmin_vmax(values_table: pd.DataFrame):
    """vmin/vmax for a heatmap color scale, excluding whole OUTLIER
    COLUMNS (e.g. a zone whose prices are simply on a different scale
    from everyone else - 10-50x higher). A simple percentile trim on all
    the flattened numbers doesn't work here: if the outlier column is,
    say, 1/20th of all the columns, its 24 hourly values are comfortably
    more than 2% of the total cells, so a 2nd/98th percentile cut doesn't
    actually exclude it and the scale still gets stretched (confirmed
    with synthetic data - the trim made no visible difference). Instead,
    find outlier COLUMNS first (by comparing each column's median against
    the spread of all columns' medians, using the standard IQR fence),
    then set vmin/vmax from only the remaining normal columns. An outlier
    column's own cells are still shown and still colored - they just clip
    to the strongest color instead of dragging every other column's color
    down with them.
    """
    col_medians = values_table.median()
    if col_medians.empty:
        return None, None
    q1, q3 = col_medians.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        lower_fence = q1 - 3 * iqr
        upper_fence = q3 + 3 * iqr
        normal_cols = col_medians[
            (col_medians >= lower_fence) & (col_medians <= upper_fence)
        ].index
    else:
        normal_cols = col_medians.index

    normal_values = values_table[normal_cols].to_numpy(dtype=float).flatten()
    normal_values = normal_values[~pd.isna(normal_values)]
    if normal_values.size:
        return float(np.nanmin(normal_values)), float(np.nanmax(normal_values))
    return None, None


df = load_day(picked_date)

if df.empty:
    st.warning(
        f"No price data stored for {picked_date.isoformat()} yet. "
        "Day-ahead prices for tomorrow are usually only published in the "
        "early afternoon (CET) - run fetch_data.py again a bit later if "
        "it's still too early."
    )
else:
    # rows = hour, columns = country/zone, values = price
    table = df.pivot_table(index="hour", columns="zone", values="price")

    # Column order per Peter's own layout (screenshot), not alphabetical.
    # LT sits next to LV (Baltic neighbours), not next to FR. Any zone not
    # in this list (e.g. the Italian sub-zones) is appended afterwards,
    # A-Z, so nothing just silently disappears.
    PREFERRED_ZONE_ORDER = [
        "DE_LU", "AT", "CH", "FR", "BE", "NL", "ES", "PT", "PL",
        "CZ", "SK", "HU", "HR", "NORDIC_SYSTEM", "DK_1", "DK_2", "NO_1",
        "NO_2", "NO_3", "NO_4", "NO_5", "SE_1", "SE_2", "SE_3", "SE_4",
        "FI", "LV", "LT", "SI", "EE", "BG", "RS", "RO", "GR", "MK", "XK",
        "ME", "AL", "IE_SEM",
    ]
    available = list(table.columns)
    ordered = [z for z in PREFERRED_ZONE_ORDER if z in available]
    leftover = sorted(z for z in available if z not in PREFERRED_ZONE_ORDER)
    table = table.reindex(ordered + leftover, axis=1)
    table = table.rename(columns={"NORDIC_SYSTEM": "SYS"})  # shorter display label
    table = table.sort_index()  # "00:00" < "01:00" < ... sorts correctly as text

    hour_lookup = df.drop_duplicates("hour").set_index("hour")["hour_num"]
    peak_hours = [
        h for h in table.index
        if PEAK_START_HOUR <= hour_lookup[h] < PEAK_END_HOUR
    ]

    base = table.mean()
    peak = table.loc[peak_hours].mean() if peak_hours else pd.Series(dtype=float)
    off_peak = 2 * base - peak

    summary = pd.DataFrame({"Base": base, "Peak": peak, "Off-peak": off_peak}).T
    full_table = pd.concat([table, summary])

    # ---------------------------------------------------------------
    # Nordic/Baltic-only view: SYS + DK/NO/SE/FI/EE/LV/LT, shown twice -
    # once as raw prices, once as each zone's price minus SYS for that
    # same hour (i.e. its spread to the system price).
    # ---------------------------------------------------------------
    NORDIC_BALTIC_ORDER = [
        "SYS", "DK_1", "DK_2", "NO_1", "NO_2", "NO_3", "NO_4",
        "SE_1", "SE_2", "SE_3", "SE_4", "FI", "EE", "LV", "LT",
    ]

    show_nordic_spread = st.checkbox(
        "Show Nordic/Baltic zones only, with a second block of each "
        "price minus the system price (SYS)"
    )

    highlight_de_lu_matches = False
    if not show_nordic_spread:
        highlight_de_lu_matches = st.checkbox(
            "Highlight prices that match DE_LU for that hour", value=True
        )

    if show_nordic_spread:
        available_nb = [z for z in NORDIC_BALTIC_ORDER if z in full_table.columns]
        if not available_nb:
            st.warning(
                "None of the Nordic/Baltic zones (SYS, DK, NO, SE, FI, "
                "EE, LV, LT) are in the data for this date."
            )
            display_table = None
        else:
            price_block = full_table[available_nb].copy()
            if "SYS" in price_block.columns:
                # Row-wise subtraction: each row's SYS value comes off
                # every column in that same row (hour, or Base/Peak/
                # Off-peak), so this is each zone's spread to the system
                # price for that row.
                spread_block = price_block.sub(price_block["SYS"], axis=0)
            else:
                spread_block = price_block.copy()
                st.caption(
                    "SYS wasn't found in the data for this date, so the "
                    "second block below shows raw prices instead of a "
                    "spread to SYS."
                )
            price_block.columns = pd.MultiIndex.from_product(
                [["Price"], price_block.columns]
            )
            spread_block.columns = pd.MultiIndex.from_product(
                [["vs SYS"], spread_block.columns]
            )
            display_table = pd.concat([price_block, spread_block], axis=1)
    else:
        display_table = full_table

    if display_table is not None:
        # Heatmap: green = high price, red = low price. One shared color
        # scale per block (axis=None), not per column - that way a color
        # is directly comparable between zones, not just relative to
        # each column's own min/max. See _robust_vmin_vmax for why the
        # scale excludes outlier columns rather than using the raw
        # min/max.
        if show_nordic_spread:
            vmin, vmax = _robust_vmin_vmax(price_block)
            styled = display_table.style.background_gradient(
                cmap="RdYlGn", axis=None, low=0.15, high=0.15,
                vmin=vmin, vmax=vmax, subset=price_block.columns,
            )

            # "vs SYS" block: a diverging scale centered on zero, so 0
            # (== the system price itself) sits in the middle color,
            # above-system shades toward green and below-system toward
            # red - the same "green=high, red=low" convention as the
            # price block, just relative to SYS instead of absolute.
            spread_values = spread_block.to_numpy(dtype=float)
            spread_values = spread_values[~np.isnan(spread_values)]
            spread_extreme = float(np.nanmax(np.abs(spread_values))) if spread_values.size else 0.0
            if spread_extreme > 0:
                styled = styled.background_gradient(
                    cmap="RdYlGn", axis=None,
                    vmin=-spread_extreme, vmax=spread_extreme,
                    subset=spread_block.columns,
                )
        else:
            vmin, vmax = _robust_vmin_vmax(full_table)
            styled = display_table.style.background_gradient(
                cmap="RdYlGn", axis=None, low=0.15, high=0.15, vmin=vmin, vmax=vmax
            )

            if highlight_de_lu_matches and "DE_LU" in full_table.columns:
                def _border_matches(row: pd.Series) -> list[str]:
                    ref = row["DE_LU"]
                    out = []
                    for col, val in row.items():
                        if col == "DE_LU":
                            out.append("")  # never frame the reference column itself
                        elif pd.notna(val) and pd.notna(ref) and round(val, 2) == round(ref, 2):
                            out.append("border: 3px solid black")
                        else:
                            out.append("")
                    return out

                styled = styled.apply(_border_matches, axis=1)

        # Fat frame around the summary block (Base/Peak/Off-peak) - a
        # thick rectangle around just those last 3 rows, to set them
        # visually apart from the hourly rows above. Top/bottom edges go
        # on the first and last of the 3 rows; left/right edges go on the
        # first and last column - together that traces a full box.
        SUMMARY_ROWS = {"Base", "Peak", "Off-peak"}

        def _frame_summary_block(row: pd.Series) -> list[str]:
            if row.name not in SUMMARY_ROWS:
                return ["" for _ in row]
            last_col_idx = len(row) - 1
            out = []
            for i, _ in enumerate(row.index):
                edges = []
                if row.name == "Base":
                    edges.append("border-top: 4px solid black")
                if row.name == "Off-peak":
                    edges.append("border-bottom: 4px solid black")
                if i == 0:
                    edges.append("border-left: 4px solid black")
                if i == last_col_idx:
                    edges.append("border-right: 4px solid black")
                out.append("; ".join(edges))
            return out

        styled = styled.apply(_frame_summary_block, axis=1)

        styled = styled.format("{:.2f}")

        # Autosize columns: each column only as wide as its own content
        # needs (e.g. "50.00"), instead of Streamlit's default table CSS
        # stretching every column to fill the page evenly - important
        # once there are a few dozen zone columns side by side.
        #
        # Two things had to be worked around to get this to actually
        # show up:
        # 1. A Styler's own set_table_attributes()/set_table_styles() do
        #    NOT survive st.table - Streamlit rebuilds the table from the
        #    Styler's per-cell values and keeps only per-cell styling
        #    (background-color, the borders above), so table-level
        #    sizing has to be injected as separate CSS instead.
        # 2. A plain class selector for that CSS (e.g. ".stTable table")
        #    LOSES to Streamlit's own internal styling even with
        #    !important, because Streamlit's generated class carries
        #    higher specificity (confirmed by inspecting the live page -
        #    the computed width never actually changed with a
        #    class-only selector). Fixed by pinning the table's id with
        #    set_uuid() and targeting that id instead - an id selector
        #    always outranks any number of classes, !important or not.
        # 3. Even after that, the table kept the container's full width -
        #    Streamlit's own table class sets `min-width: 100%` (not
        #    `width`), which enforces a floor under the rendered size
        #    that a `width: auto` override alone doesn't touch. Needed
        #    its own override too (confirmed by walking the actual
        #    matching CSS rules on the live page rather than guessing).
        styled = styled.set_uuid("pricetable")
        st.markdown(
            """
            <style>
            #T_pricetable { width: auto !important; min-width: 0 !important; table-layout: auto !important; }
            #T_pricetable th, #T_pricetable td { white-space: nowrap !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # st.table, not st.dataframe: st.dataframe's grid widget draws
        # cells on a canvas and only picks up background-color/text-color
        # from a Styler - it silently drops anything else, including the
        # border above (confirmed by inspecting the actual rendered page
        # - the border never showed up with st.dataframe even though the
        # code ran with no error). st.table renders a real HTML table
        # instead, so every style rule actually shows, and it also
        # naturally displays every row with no inner scrollbar - no
        # separate height calculation needed for that anymore either.
        st.table(styled)

        if show_nordic_spread:
            missing = [z for z in available_nb if table[z].isna().all()]
        else:
            missing = [z for z in table.columns if table[z].isna().all()]
        if missing:
            st.caption(f"No data at all for: {', '.join(missing)}")

    st.caption(
        f"Peak = average price {PEAK_START_HOUR:02d}:00-{PEAK_END_HOUR:02d}:00. "
        "Off-peak = 2 x Base - Peak. Prices in EUR/MWh."
        + (
            " 'vs SYS' = that zone's price minus the system price (SYS) "
            "for the same hour/row."
            if show_nordic_spread
            else ""
        )
    )
