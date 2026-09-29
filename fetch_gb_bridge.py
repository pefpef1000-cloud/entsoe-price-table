"""
fetch_gb_bridge.py - the cloud-side twin of fetch_nordpool_bridge.py,
trimmed to just GB's N2EX auction (the online year-comparison dashboard
doesn't need any of the other Nordic/Baltic zones fetch_nordpool_bridge.py
also bridges).

Why this exists: ENTSO-E stopped publishing a GB day-ahead price after
2020-12-31 (confirmed live via zone_fetch_log - "no_data" every check
since; see fetch_recent_years.py). GB's own N2EX auction (run by Nord
Pool) is the same real auction result, published on Nord Pool's own
public Data Portal - this is what keeps the dashboard's 3
GB-interconnector "countries" (IFA/IFA2/ElecLink, see year_comparison.py's
INTERCONNECTOR_ENTSOE_ZONE) fed with the REAL N2EX price for recent days.

Saves into the SAME day_ahead_prices table fetch_recent_years.py uses,
tagged source='nordpool'. Here that means PROVISIONAL: GB's price series is
built from Elexon's APX index (fetch_gb_apx.py) so that every year is
like-for-like, and APX replaces the N2EX price of a day once the day is
complete. So this script's rows matter for today and tomorrow (APX has
nothing for those yet - this is what the price table shows) and as a
fallback for any hour APX has no trades in. It therefore NEVER overwrites
an 'entsoe' or an 'elexon_apx' row.

THIS SCRIPT ONLY COVERS RECENT DAYS - by design. Nord Pool's free Data
Portal API refuses dates older than roughly a month or two with HTTP 401
("Unauthorized"), from any IP and however slowly it is asked. Tested live
2026-09-29: yesterday and 30 days ago -> HTTP 200; 90 days, 180 days,
1 year ago and 2021-01-01 -> HTTP 401. (An earlier version of this file
tried to backfill 2021 -> now from here; every one of those requests was
refused, and the retries cost about an hour per workflow run.) The history
from 2021 up to a few weeks ago is filled by fetch_gb_apx.py instead,
from Elexon's public API.

Nord Pool's API only takes ONE delivery date per request (no range
parameter - see fetch_nordpool_bridge.py's own module docstring). Every
run fetches a trailing TRAILING_DAYS-day window plus tomorrow, day by day,
newest day first - so if Nord Pool ever starts refusing the oldest days of
the window, the days that matter most have already been fetched. Since
TRAILING_DAYS is comfortably wider than the gap between runs (every 4
hours - see .github/workflows/fetch.yml), coverage stays continuous.

Run it with:
    python fetch_gb_bridge.py
"""

import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from fetch_recent_years import ensure_tables, rebuild_aggregate_tables

DB_PATH = Path(__file__).parent / "entsoe_data.db"

API_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
N2EX_MARKET = "N2EX_DayAhead"
N2EX_AREA = "UK"
CURRENCY = "EUR"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# How far back to (re)fetch every run - see module docstring. Wider than
# the 4-hourly run interval, and well inside the ~month-or-two of history
# Nord Pool's free API serves.
TRAILING_DAYS = 21

REQUEST_PAUSE_SECONDS = 0.5
# On HTTP 401/429 wait and retry the same day a couple of times (a genuine
# short block clears quickly) - but not for long: a day Nord Pool simply
# refuses never becomes available, so don't sit and wait for it.
RETRY_BACKOFFS = [20, 60]  # seconds
ABORT_AFTER_CONSECUTIVE_FAILS = 3


def fetch_n2ex_day_ahead(delivery_date: str) -> dict | None:
    """Same idea as fetch_nordpool_bridge.py's own fetch_n2ex_day_ahead,
    plus a short retry on 401/429. None = "not published (yet)" (HTTP 204
    or an empty 200 body, e.g. tomorrow's prices before ~13:00 CET)."""
    params = {
        "date": delivery_date,
        "market": N2EX_MARKET,
        "deliveryArea": N2EX_AREA,
        "currency": CURRENCY,
    }
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)

        if resp.status_code == 204 or (resp.status_code == 200 and not resp.text.strip()):
            return None

        if resp.status_code in (401, 429):
            if attempt < len(RETRY_BACKOFFS):
                wait = RETRY_BACKOFFS[attempt]
                print(f"    HTTP {resp.status_code} on {delivery_date} - waiting {wait}s "
                      f"before retry {attempt + 1}/{len(RETRY_BACKOFFS)}...")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Nord Pool (N2EX) still HTTP {resp.status_code} for {delivery_date} "
                f"after {len(RETRY_BACKOFFS)} retries"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Nord Pool (N2EX) returned HTTP {resp.status_code} for {delivery_date} "
                f"(body: {resp.text[:300]!r})"
            )

        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"Nord Pool (N2EX)'s response for {delivery_date} wasn't valid JSON "
                f"(first 300 chars: {resp.text[:300]!r})"
            ) from e


def parse_hourly(data: dict) -> pd.Series:
    """Same idea as fetch_nordpool_bridge.py's parse_hourly, trimmed to
    just the single GB column (that file builds a whole DataFrame across
    every Nordic/Baltic area; this only ever needs one)."""
    rows = []
    for entry in data["multiAreaEntries"]:
        price = entry["entryPerArea"].get(N2EX_AREA)
        if price is not None:
            rows.append((entry["deliveryStart"], price))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["deliveryStart", "price"])
    df["deliveryStart"] = pd.to_datetime(df["deliveryStart"], utc=True).dt.tz_convert("CET")
    series = df.set_index("deliveryStart")["price"].sort_index()
    return series.resample("h").mean().round(2)


def save_bridge_prices(hourly: pd.Series, conn) -> tuple[int, int]:
    """Like fetch_nordpool_bridge.py's own save_bridge_prices, but also
    protects 'elexon_apx' rows (see module docstring). Returns (written,
    skipped_because_an_official_or_APX_price_was_already_there)."""
    written = skipped = 0
    for ts, price in hourly.items():
        if pd.isna(price):
            continue
        cur = conn.execute(
            """
            INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, source)
            VALUES ('GB', ?, ?, 'nordpool')
            ON CONFLICT(zone, timestamp) DO UPDATE SET
                price_eur_mwh = excluded.price_eur_mwh,
                source = 'nordpool'
            WHERE day_ahead_prices.source NOT IN ('entsoe', 'elexon_apx')
            """,
            (ts.isoformat(), float(price)),
        )
        if cur.rowcount:
            written += 1
        else:
            skipped += 1
    conn.commit()
    return written, skipped


def fetch_and_save_range(dates: list[str], conn) -> tuple[int, int, int, int, int]:
    """Fetches and saves each date in the order given. Returns
    (published, unpublished, total_written, total_skipped, errored)."""
    published = unpublished = total_written = total_skipped = errored = 0
    consecutive_fails = 0
    for d_str in dates:
        try:
            data = fetch_n2ex_day_ahead(d_str)
            consecutive_fails = 0
        except Exception as e:
            print(f"    {d_str}: ERROR - {e}")
            errored += 1
            consecutive_fails += 1
            if consecutive_fails >= ABORT_AFTER_CONSECUTIVE_FAILS:
                print(f"    {consecutive_fails} day(s) in a row failed - stopping here. "
                      "The next scheduled run tries again.")
                break
            time.sleep(REQUEST_PAUSE_SECONDS)
            continue

        if data is not None:
            hourly = parse_hourly(data)
            if not hourly.empty:
                written, skipped = save_bridge_prices(hourly, conn)
                total_written += written
                total_skipped += skipped
                published += 1
            else:
                unpublished += 1
        else:
            unpublished += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    return published, unpublished, total_written, total_skipped, errored


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    # Tomorrow first, then today, yesterday ... back to TRAILING_DAYS ago.
    dates = [
        (date.today() - timedelta(days=i)).isoformat()
        for i in range(-1, TRAILING_DAYS + 1)
    ]
    published, unpublished, written, skipped, errored = fetch_and_save_range(dates, conn)

    rebuild_aggregate_tables(conn)
    conn.close()

    print(
        f"N2EX: {published} day(s) published and saved, "
        f"{unpublished} not published/no data, {errored} errored. "
        f"{written} hour(s) written, {skipped} left untouched "
        "(already had an official ENTSO-E or APX price)."
    )


if __name__ == "__main__":
    main()
