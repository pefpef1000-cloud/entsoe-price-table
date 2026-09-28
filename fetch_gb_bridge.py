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
INTERCONNECTOR_ENTSOE_ZONE) fed.

Saves into the SAME day_ahead_prices table fetch_recent_years.py uses,
tagged source='nordpool' - but only for an hour that doesn't already have
an official 'entsoe' value (never happens for GB post-cutoff in practice,
but this is the exact same safe ON CONFLICT precedence
fetch_nordpool_bridge.py itself uses: an 'entsoe' row always wins if one
ever exists for the same hour).

Nord Pool's API only takes ONE delivery date per request (no range
parameter - see fetch_nordpool_bridge.py's own module docstring). Every
run fetches a trailing TRAILING_DAYS-day window plus tomorrow, day by
day - since TRAILING_DAYS is comfortably wider than the gap between runs
(every 4 hours - see .github/workflows/fetch.yml), coverage stays
continuous once this has run at least once.

GAP_START backfill (added after Peter noticed IFA/IFA2/ElecLink showing
no price for most of their history): on top of the trailing window
above, every run also checks whether GB's earliest saved row (any
source) is later than GAP_START (2021-01-01, the day after ENTSO-E's
last published GB price). If so, that means the historical gap between
GAP_START and the start of the trailing window hasn't been backfilled
yet, so this run also walks through EVERY missing day back to
GAP_START, one request each, before doing the normal trailing window.
That's a one-time ~2,000-request job the first time it runs after this
was added - after that, GB's earliest row is <= GAP_START, so this
branch is skipped on every future run (a single cheap MIN(timestamp)
query) and only the normal trailing window runs. Same idea as Peter's
own one-time local backfill_gb_history.py, just self-triggering here
instead of something Peter has to remember to run once.

Run it with:
    python fetch_gb_bridge.py
"""

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
# the 4-hourly run interval, so coverage never has a gap once this has
# run at least once.
TRAILING_DAYS = 30

# The day after ENTSO-E's last published GB day-ahead price
# (2020-12-31, confirmed live). See module docstring's "GAP_START
# backfill" section.
GAP_START = date(2021, 1, 1)

# Pace requests to stay polite to Nord Pool's public API - matters far
# more during a GAP_START backfill (up to ~2,000 requests in one run)
# than for the normal ~32-day trailing window.
REQUEST_PAUSE_SECONDS = 0.3


def fetch_n2ex_day_ahead(delivery_date: str) -> dict | None:
    """Identical to fetch_nordpool_bridge.py's own fetch_n2ex_day_ahead -
    see that file for the full reasoning on the HTTP 204 / empty-200
    "not published yet" handling."""
    params = {
        "date": delivery_date,
        "market": N2EX_MARKET,
        "deliveryArea": N2EX_AREA,
        "currency": CURRENCY,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)

    if resp.status_code == 204 or (resp.status_code == 200 and not resp.text.strip()):
        return None

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
    """Same ON CONFLICT ... WHERE source != 'entsoe' precedence as
    fetch_nordpool_bridge.py's own save_bridge_prices. Returns (written,
    skipped_because_entsoe_already_had_it)."""
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
            WHERE day_ahead_prices.source != 'entsoe'
            """,
            (ts.isoformat(), float(price)),
        )
        if cur.rowcount:
            written += 1
        else:
            skipped += 1
    conn.commit()
    return written, skipped


def existing_gb_min_date(conn) -> date | None:
    """Earliest GB row on record, any source - used to decide whether
    the GAP_START historical backfill still needs to run. None if GB
    has no rows at all yet."""
    row = conn.execute(
        "SELECT MIN(timestamp) FROM day_ahead_prices WHERE zone = 'GB'"
    ).fetchone()
    if not row or not row[0]:
        return None
    return pd.Timestamp(row[0]).date()


def fetch_and_save_range(dates: list[str], conn) -> tuple[int, int, int, int]:
    """Fetches and saves each date in order. Returns
    (published, unpublished, total_written, total_skipped)."""
    published = unpublished = total_written = total_skipped = 0
    for d in dates:
        data = fetch_n2ex_day_ahead(d)
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
    return published, unpublished, total_written, total_skipped


def main() -> None:
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    backfill_published = backfill_unpublished = 0
    backfill_written = backfill_skipped = 0

    min_date = existing_gb_min_date(conn)
    if min_date is None or min_date > GAP_START:
        # See module docstring's "GAP_START backfill" section. Only
        # true while the historical gap actually exists - skipped on
        # every run after the first one that completes it.
        backfill_end = date.today() - timedelta(days=TRAILING_DAYS + 1)
        if backfill_end >= GAP_START:
            backfill_dates = [
                (GAP_START + timedelta(days=i)).isoformat()
                for i in range((backfill_end - GAP_START).days + 1)
            ]
            print(
                f"GB's earliest saved row is after {GAP_START.isoformat()} "
                f"(or missing entirely) - backfilling {len(backfill_dates)} "
                f"historical day(s) from {GAP_START.isoformat()} through "
                f"{backfill_end.isoformat()}. This is a one-time job and will "
                "take a while (~15-25 min)."
            )
            (
                backfill_published,
                backfill_unpublished,
                backfill_written,
                backfill_skipped,
            ) = fetch_and_save_range(backfill_dates, conn)
            print(
                f"Historical backfill done: {backfill_published} day(s) published, "
                f"{backfill_unpublished} had no data, {backfill_written} hour(s) written."
            )

    # TRAILING_DAYS back through tomorrow - the normal, fast, every-run window.
    trailing_dates = [
        (date.today() - timedelta(days=i)).isoformat()
        for i in range(TRAILING_DAYS, -2, -1)
    ]
    trailing_published, trailing_unpublished, trailing_written, trailing_skipped = (
        fetch_and_save_range(trailing_dates, conn)
    )

    rebuild_aggregate_tables(conn)
    conn.close()

    total_published = backfill_published + trailing_published
    total_unpublished = backfill_unpublished + trailing_unpublished
    total_written = backfill_written + trailing_written
    total_skipped = backfill_skipped + trailing_skipped
    print(
        f"N2EX: {total_published} day(s) published and saved, "
        f"{total_unpublished} not published/no data. "
        f"{total_written} hour(s) written, {total_skipped} left untouched "
        "(already had an official ENTSO-E price)."
    )


if __name__ == "__main__":
    main()
