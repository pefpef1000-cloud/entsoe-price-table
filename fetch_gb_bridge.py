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
Each day already covered (any source) is skipped with a cheap check
rather than re-fetched, so an interrupted run (job timeout, a bad
deploy) resumes cheaply next time instead of starting over. Once fully
backfilled, GB's earliest row is <= GAP_START, so this branch is
skipped on every future run (a single MIN(timestamp) query) and only
the normal trailing window runs.

Rate limiting: Nord Pool's public API returned HTTP 401 Unauthorized
after roughly 50 requests in quick succession during a real backfill
run (2026-09-28) - not documented anywhere, discovered live. Treated
like a temporary block: on 401 or 429, this waits and retries the SAME
day (30s, then 60s, 120s, 240s) before giving up on that one day and
moving on; after 3 straight days fail even after retrying, it also
pauses an extra minute, since that's a sign the block is still active.
A day that still fails after all of that is simply picked up on a
later scheduled run (see "already covered" skip above - everything
already fetched stays fetched).

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
# than for the normal ~32-day trailing window. See "Rate limiting" in
# the module docstring for why this isn't faster.
REQUEST_PAUSE_SECONDS = 1.5
RETRY_BACKOFFS = [30, 60, 120, 240]  # seconds, on 401/429, same day
COOLDOWN_AFTER_CONSECUTIVE_FAILS = 3
COOLDOWN_SECONDS = 60
# If even a cooldown doesn't clear it, stop this run's backfill entirely
# rather than burning the whole job on a sustained block - the next
# scheduled run (every 4 hours) picks up where this left off (see
# day_already_covered), so nothing is lost, just deferred.
ABORT_AFTER_CONSECUTIVE_FAILS = 7


def fetch_n2ex_day_ahead(delivery_date: str) -> dict | None:
    """Same idea as fetch_nordpool_bridge.py's own fetch_n2ex_day_ahead,
    plus retry-with-backoff on 401/429 - see module docstring's "Rate
    limiting" section."""
    params = {
        "date": delivery_date,
        "market": N2EX_MARKET,
        "deliveryArea": N2EX_AREA,
        "currency": CURRENCY,
    }
    attempts = len(RETRY_BACKOFFS) + 1
    for attempt in range(attempts):
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)

        if resp.status_code == 204 or (resp.status_code == 200 and not resp.text.strip()):
            return None

        if resp.status_code in (401, 429):
            if attempt < len(RETRY_BACKOFFS):
                wait = RETRY_BACKOFFS[attempt]
                print(f"    rate-limited (HTTP {resp.status_code}) on {delivery_date} - "
                      f"waiting {wait}s before retry {attempt + 1}/{len(RETRY_BACKOFFS)}...")
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


def missing_days_in_range(conn, start: date, end: date) -> int:
    """How many calendar days in [start, end] have no GB row at all
    (any source). This is the correct way to detect whether the
    GAP_START backfill still has work to do - checking only
    MIN(timestamp) is a trap: 2020 has real ENTSO-E data, so
    MIN(timestamp) is always 2020-01-01 regardless of whether
    2021-2025 (the actual gap) has ever been backfilled, which meant
    an earlier version of this check never ran the backfill at all
    (confirmed live, run #77, 2026-09-28 - only did the normal 32-day
    trailing window, 0 historical days attempted)."""
    total_days = (end - start).days + 1
    covered = conn.execute(
        """
        SELECT COUNT(DISTINCT substr(timestamp, 1, 10))
        FROM day_ahead_prices
        WHERE zone = 'GB' AND timestamp >= ? AND timestamp < ?
        """,
        (start.isoformat(), (end + timedelta(days=1)).isoformat()),
    ).fetchone()[0]
    return max(total_days - covered, 0)


def day_already_covered(conn, d: date) -> bool:
    row = conn.execute(
        "SELECT 1 FROM day_ahead_prices WHERE zone='GB' AND timestamp LIKE ? LIMIT 1",
        (d.isoformat() + "%",),
    ).fetchone()
    return row is not None


def fetch_and_save_range(dates: list[str], conn, skip_covered: bool = False) -> tuple[int, int, int, int, int]:
    """Fetches and saves each date in order. Returns
    (published, unpublished, total_written, total_skipped, errored)."""
    published = unpublished = total_written = total_skipped = errored = 0
    consecutive_fails = 0
    for d_str in dates:
        d = date.fromisoformat(d_str)
        if skip_covered and day_already_covered(conn, d):
            continue

        try:
            data = fetch_n2ex_day_ahead(d_str)
            consecutive_fails = 0
        except Exception as e:
            print(f"    {d_str}: ERROR - {e}")
            errored += 1
            consecutive_fails += 1
            if consecutive_fails >= ABORT_AFTER_CONSECUTIVE_FAILS:
                print(f"    {consecutive_fails} day(s) in a row failed even after "
                      "cooldowns - stopping this run's backfill here. The next "
                      "scheduled run picks up from here (already-covered days are "
                      "skipped).")
                break
            if consecutive_fails >= COOLDOWN_AFTER_CONSECUTIVE_FAILS:
                print(f"    {consecutive_fails} day(s) in a row failed - cooling down "
                      f"{COOLDOWN_SECONDS}s before continuing...")
                time.sleep(COOLDOWN_SECONDS)
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
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    backfill_published = backfill_unpublished = backfill_errored = 0
    backfill_written = backfill_skipped = 0

    backfill_end = date.today() - timedelta(days=TRAILING_DAYS + 1)
    if backfill_end >= GAP_START:
        missing = missing_days_in_range(conn, GAP_START, backfill_end)
        if missing > 0:
            # See module docstring's "GAP_START backfill" section.
            # missing_days_in_range only counts actually-empty days, so
            # this correctly stays 0 (and this branch is skipped) once
            # the gap is fully filled, however early 2020's real
            # ENTSO-E data makes MIN(timestamp) look.
            backfill_dates = [
                (GAP_START + timedelta(days=i)).isoformat()
                for i in range((backfill_end - GAP_START).days + 1)
            ]
            print(
                f"{missing} day(s) missing between {GAP_START.isoformat()} and "
                f"{backfill_end.isoformat()} - backfilling (already-covered days "
                "are skipped). This may take a while."
            )
            (
                backfill_published,
                backfill_unpublished,
                backfill_written,
                backfill_skipped,
                backfill_errored,
            ) = fetch_and_save_range(backfill_dates, conn, skip_covered=True)
            print(
                f"Historical backfill this run: {backfill_published} day(s) published, "
                f"{backfill_unpublished} had no data, {backfill_errored} errored, "
                f"{backfill_written} hour(s) written. Any errored days are picked up "
                "on a future run."
            )

    # TRAILING_DAYS back through tomorrow - the normal, fast, every-run window.
    trailing_dates = [
        (date.today() - timedelta(days=i)).isoformat()
        for i in range(TRAILING_DAYS, -2, -1)
    ]
    (
        trailing_published,
        trailing_unpublished,
        trailing_written,
        trailing_skipped,
        trailing_errored,
    ) = fetch_and_save_range(trailing_dates, conn, skip_covered=False)

    rebuild_aggregate_tables(conn)
    conn.close()

    total_published = backfill_published + trailing_published
    total_unpublished = backfill_unpublished + trailing_unpublished
    total_written = backfill_written + trailing_written
    total_skipped = backfill_skipped + trailing_skipped
    total_errored = backfill_errored + trailing_errored
    print(
        f"N2EX: {total_published} day(s) published and saved, "
        f"{total_unpublished} not published/no data, {total_errored} errored. "
        f"{total_written} hour(s) written, {total_skipped} left untouched "
        "(already had an official ENTSO-E price)."
    )


if __name__ == "__main__":
    main()
