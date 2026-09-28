"""
fetch_gb_bridge.py - the cloud-side twin of fetch_nordpool_bridge.py,
trimmed to just GB's N2EX auction (the online year-comparison dashboard
doesn't need any of the other Nordic/Baltic zones fetch_nordpool_bridge.py
also bridges).

Why this exists: ENTSO-E stopped publishing a GB day-ahead price on
15 June 2021 (see fetch_recent_years.py). GB's own N2EX auction (run by
Nord Pool) is the same real auction result, published on Nord Pool's own
public Data Portal - this is what keeps the dashboard's 3
GB-interconnector "countries" (IFA/IFA2/ElecLink, see year_comparison.py's
INTERCONNECTOR_ENTSOE_ZONE) fed going forward.

Saves into the SAME day_ahead_prices table fetch_recent_years.py uses,
tagged source='nordpool' - but only for an hour that doesn't already have
an official 'entsoe' value (never happens for GB post-cutoff in practice,
but this is the exact same safe ON CONFLICT precedence
fetch_nordpool_bridge.py itself uses: an 'entsoe' row always wins if one
ever exists for the same hour).

Nord Pool's API only takes ONE delivery date per request (no range
parameter - see fetch_nordpool_bridge.py's own module docstring). Rather
than an expensive one-time backfill of GB's entire 2021-2026 history
(thousands of individual date requests - Peter's own local system doesn't
do this either, it only ever bridges a trailing window forward from
whenever it started running), this fetches a trailing TRAILING_DAYS-day
window plus tomorrow, day by day, every time it runs. Since it runs on
the same schedule as the rest of this cloud deployment (every 4 hours -
see .github/workflows/fetch.yml) and TRAILING_DAYS is comfortably wider
than the gap between runs, coverage is continuous from whenever this
script first ran onward - it just never reaches back further than that
(same real-world gap Peter's own local database has, between the
2021-06-15 ENTSO-E cutoff and whenever fetch_nordpool_bridge.py started
running there - see claude/entsoe-price-table-cloud-deploy.md).

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


def main() -> None:
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    # TRAILING_DAYS back through tomorrow.
    dates = [
        (date.today() - timedelta(days=i)).isoformat()
        for i in range(TRAILING_DAYS, -2, -1)
    ]

    total_written = total_skipped = 0
    published = unpublished = 0
    for d in dates:
        data = fetch_n2ex_day_ahead(d)
        if data is None:
            unpublished += 1
            time.sleep(0.3)
            continue
        hourly = parse_hourly(data)
        if not hourly.empty:
            written, skipped = save_bridge_prices(hourly, conn)
            total_written += written
            total_skipped += skipped
            published += 1
        time.sleep(0.3)

    rebuild_aggregate_tables(conn)
    conn.close()
    print(
        f"N2EX: {published} day(s) published and saved, {unpublished} not published/no data. "
        f"{total_written} hour(s) written, {total_skipped} left untouched "
        "(already had an official ENTSO-E price)."
    )


if __name__ == "__main__":
    main()
