"""
fetch_recent_years.py - the cloud-side twin of fetch_data.py, trimmed to
the 3 zones the online year-comparison dashboard (pages/Year_comparison.py)
actually needs.

Your real fetch_data.py keeps 2020-present history for every ENTSO-E zone
in a 655MB local database - way too big for GitHub (100MB per-file limit).
This script keeps the SAME full-history, incremental-backfill approach
(year-by-year chunking, so a rate limit or a failed run only ever costs
the one year-chunk that failed, not the whole backfill - see
fetch_data.py's own module docstring for the full reasoning, copied here
almost verbatim) but only for:

  - DE_LU, FR: the 2 zones Peter picked for the online dashboard's
    Country 1 / Country 2 pickers.
  - GB: not itself selectable, but needed because the dashboard's 3
    GB-interconnector "countries" (IFA/IFA2/ElecLink) all price off plain
    GB's hourly price (see year_comparison.py's INTERCONNECTOR_ENTSOE_ZONE).
    GB stopped getting an ENTSO-E price after 15 June 2021 - same as
    fetch_data.py, this script doesn't hardcode that cutoff, it just keeps
    asking, gets "no data" for every post-cutoff year-chunk forever, and
    that costs nothing extra (a handful of cheap requests each run). GB's
    price from 2021 onward comes from fetch_gb_bridge.py (Nord Pool's
    N2EX auction) instead - see that file.

Writes into entsoe_data.db right here in this folder - the SAME database
fetch_recent.py and price_table.py already use (day_ahead_prices, tagged
source='entsoe') - then rebuilds daily_prices/weekly_prices/monthly_prices
from it, since year_comparison.py reads those 3 aggregate tables (not the
raw hourly one) for its Daily/Weekly/Monthly views.

Safe and cheap to run on every scheduled refresh (see
.github/workflows/fetch.yml): like fetch_data.py, it only asks the API
for what's actually missing since the last run - the full 2020-present
backfill only really happens once, the first time this ever runs.

Needs ENTSOE_API_KEY in the environment - the same GitHub Actions repo
secret fetch_recent.py already uses.

Run it with:
    python fetch_recent_years.py
"""

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

DB_PATH = Path(__file__).parent / "entsoe_data.db"

ZONES = ["DE_LU", "FR", "GB"]
START_DATE = pd.Timestamp("2020-01-01", tz="CET")

MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_SECONDS = 60  # first wait; doubles after each retry

_rate_limited_until = 0.0  # a time.time() value; 0 = no active cooldown


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Same day_ahead_prices/zone_fetch_log/daily_prices/weekly_prices/
    monthly_prices schema as the real fetch_data.py's own ensure_tables()
    (minus the load_forecast_* tables - not used by anything in this cloud
    deployment), including its migration-safe ALTER TABLE for a database
    that predates the 'source' column. This cloud database is always
    created fresh by fetch_recent.py or this script, but sharing the exact
    same migration-safe logic across every script that touches it means
    it's never a problem which one happens to run first."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS day_ahead_prices (
            zone TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price_eur_mwh REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'entsoe',
            PRIMARY KEY (zone, timestamp)
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(day_ahead_prices)")}
    if "source" not in existing_cols:
        try:
            conn.execute(
                "ALTER TABLE day_ahead_prices ADD COLUMN source TEXT NOT NULL DEFAULT 'entsoe'"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS zone_fetch_log (
            zone TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_fetched INTEGER NOT NULL,
            message TEXT,
            PRIMARY KEY (zone, checked_at)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            zone TEXT NOT NULL,
            date TEXT NOT NULL,
            avg_price_eur_mwh REAL NOT NULL,
            PRIMARY KEY (zone, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_prices (
            zone TEXT NOT NULL,
            week_start TEXT NOT NULL,
            avg_price_eur_mwh REAL NOT NULL,
            PRIMARY KEY (zone, week_start)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_prices (
            zone TEXT NOT NULL,
            month_start TEXT NOT NULL,
            avg_price_eur_mwh REAL NOT NULL,
            PRIMARY KEY (zone, month_start)
        )
        """
    )
    conn.commit()


def _wait_out_shared_rate_limit() -> None:
    remaining = _rate_limited_until - time.time()
    if remaining > 0:
        time.sleep(remaining)


def fetch_day_ahead_prices(zone: str, start: pd.Timestamp, end: pd.Timestamp, client) -> pd.Series:
    """Same retry-on-429 behaviour as fetch_data.py's own
    fetch_day_ahead_prices - see that file for the full reasoning."""
    global _rate_limited_until
    wait = RATE_LIMIT_BACKOFF_SECONDS
    prices = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        _wait_out_shared_rate_limit()
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            break
        except requests.exceptions.HTTPError as e:
            rate_limited = e.response is not None and e.response.status_code == 429
            if not rate_limited or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            print(f"  {zone}: rate-limited by ENTSO-E (429), waiting {wait}s...")
            _rate_limited_until = time.time() + wait
            time.sleep(wait)
            wait *= 2
    prices = prices.tz_convert("CET").resample("h").mean().dropna().round(2)
    return prices


def _split_into_year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Identical to fetch_data.py's own _split_into_year_chunks - see that
    file for the full reasoning (a big backfill saves progress year by
    year instead of all-or-nothing)."""
    chunks = []
    chunk_start = start
    while chunk_start < end:
        next_new_year = pd.Timestamp(year=chunk_start.year + 1, month=1, day=1, tz=chunk_start.tz)
        chunk_end = min(next_new_year, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def get_data_bounds(zone: str, conn: sqlite3.Connection) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    row = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM day_ahead_prices WHERE zone = ?", (zone,)
    ).fetchone()
    if row is None or row[0] is None:
        return None, None
    earliest = pd.Timestamp(row[0]).tz_convert("CET")
    latest = pd.Timestamp(row[1]).tz_convert("CET")
    return earliest, latest


def save_prices_to_db(prices: pd.Series, zone: str, conn: sqlite3.Connection) -> int:
    rows = [(zone, ts.isoformat(), float(price)) for ts, price in prices.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO day_ahead_prices (zone, timestamp, price_eur_mwh, source) "
        "VALUES (?, ?, ?, 'entsoe')",
        rows,
    )
    conn.commit()
    return len(rows)


def log_zone_status(conn: sqlite3.Connection, zone: str, status: str, rows_fetched: int, message: str = "") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO zone_fetch_log (zone, checked_at, status, rows_fetched, message) "
        "VALUES (?, ?, ?, ?, ?)",
        (zone, pd.Timestamp.now("UTC").isoformat(), status, rows_fetched, message),
    )
    conn.commit()


def rebuild_aggregate_tables(conn: sqlite3.Connection) -> None:
    """Identical to fetch_data.py's own rebuild_aggregate_tables, including
    the DST week_start fix (converting to a tz-naive date before
    subtracting the weekday offset, so the CET/CEST fall-back day doesn't
    produce a spurious 4th week_start) - see that file for the full
    reasoning."""
    df = pd.read_sql_query("SELECT zone, timestamp, price_eur_mwh FROM day_ahead_prices", conn)
    if df.empty:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("CET")

    def _aggregate(freq: str) -> pd.DataFrame:
        out = (
            df.groupby(["zone", pd.Grouper(key="timestamp", freq=freq)])["price_eur_mwh"]
            .mean().round(2).reset_index()
        )
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")
        return out.rename(columns={"timestamp": "period", "price_eur_mwh": "avg_price_eur_mwh"})

    local_date = pd.to_datetime(df["timestamp"].dt.date)
    week_start = local_date - pd.to_timedelta(local_date.dt.weekday, unit="D")
    weekly = (
        df.assign(period=week_start).groupby(["zone", "period"])["price_eur_mwh"]
        .mean().round(2).reset_index()
    )
    weekly["period"] = weekly["period"].dt.strftime("%Y-%m-%d")
    weekly = weekly.rename(columns={"price_eur_mwh": "avg_price_eur_mwh"})

    tables = {
        "daily_prices": ("date", _aggregate("D")),
        "weekly_prices": ("week_start", weekly),
        "monthly_prices": ("month_start", _aggregate("MS")),
    }
    for table_name, (period_col, data) in tables.items():
        data = data.rename(columns={"period": period_col})
        conn.execute(f"DELETE FROM {table_name}")
        data.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()


def main() -> None:
    api_key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ENTSOE_API_KEY environment variable is not set. In GitHub "
            "Actions this comes from a repo secret - see the workflow "
            "file. Running locally, set it yourself first."
        )

    client = EntsoePandasClient(api_key=api_key)
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    now = pd.Timestamp.now(tz="CET")
    # Same "day after tomorrow" reasoning as fetch_data.py: tomorrow's
    # day-ahead prices are usually already published by early afternoon
    # today, so stopping at tomorrow's midnight would miss them.
    end = now.normalize() + pd.Timedelta(days=2)

    for zone in ZONES:
        earliest, latest = get_data_bounds(zone, conn)
        ranges_needed = []
        if earliest is None:
            ranges_needed.extend(_split_into_year_chunks(START_DATE, end))
        else:
            if earliest > START_DATE:
                ranges_needed.extend(_split_into_year_chunks(START_DATE, earliest))
            if latest < end:
                ranges_needed.extend(_split_into_year_chunks(latest, end))

        if not ranges_needed:
            print(f"{zone}: already up to date")
            log_zone_status(conn, zone, "up_to_date", 0)
            continue

        zone_rows = 0
        zone_errors = []
        for range_start, range_end in ranges_needed:
            try:
                prices = fetch_day_ahead_prices(zone, range_start, range_end, client)
            except NoMatchingDataError:
                continue
            except Exception as e:  # noqa: BLE001 - one bad chunk shouldn't stop the others
                zone_errors.append(f"{range_start.date()}-{range_end.date()}: {e}")
                continue
            if not prices.empty:
                zone_rows += save_prices_to_db(prices, zone, conn)
            time.sleep(0.2)  # be polite to the API between requests

        message = "; ".join(zone_errors)
        if zone_errors and zone_rows == 0:
            print(f"{zone}: no data ({message})")
            log_zone_status(conn, zone, "error", zone_rows, message)
        elif zone_errors:
            print(f"{zone}: saved {zone_rows} new row(s) ({len(zone_errors)} chunk(s) still failed: {message})")
            log_zone_status(conn, zone, "ok", zone_rows, message)
        elif zone_rows == 0:
            print(f"{zone}: no new data")
            log_zone_status(conn, zone, "no_data", 0)
        else:
            print(f"{zone}: saved {zone_rows} new row(s)")
            log_zone_status(conn, zone, "ok", zone_rows)

    rebuild_aggregate_tables(conn)
    conn.close()
    print("Rebuilt daily_prices/weekly_prices/monthly_prices.")


if __name__ == "__main__":
    main()
