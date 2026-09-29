"""
fetch_nordpool_zones.py - the cloud twin of the local fetch_nordpool_bridge.py
(every Nord Pool zone EXCEPT GB, which fetch_gb_bridge.py handles - GB is a
different Nord Pool market, N2EX).

Why this exists: the online price table is the same code as the local one
(price_table.py), but its data only came from ENTSO-E. That left two gaps:

  * SYS and TEL. SYS is Nord Pool's Nordic system reference price (stored as
    zone "NORDIC_SYSTEM"), TEL is another Nord-Pool-only code. Neither has
    an ENTSO-E equivalent, so without this step the online table never has
    those two columns and its "price minus the system price (SYS)" view has
    nothing to subtract.
  * Tomorrow's prices. Nord Pool publishes the day-ahead result on its own
    public Data Portal a little BEFORE ENTSO-E's Transparency Platform shows
    it. This step saves Nord Pool's numbers straight away as a stand-in.

How the stand-ins are replaced: rows are saved tagged source='nordpool', and
NEVER over a row that already has an official 'entsoe' price. The next run of
fetch_recent.py (ENTSO-E, all zones - it re-fetches the last few days and
tomorrow every time, with INSERT OR REPLACE) then overwrites them with the
official price. SYS and TEL stay 'nordpool' for good - nothing will ever
replace them.

What it fetches: tomorrow, today and the last DAYS_BACK days, one request per
date (Nord Pool's API takes a single delivery date per request; the zones for
one date all come in that one request). Newest date first, so if Nord Pool
ever refuses the older ones the important days are already saved. Dates
Nord Pool has not published yet (tomorrow, before about 12:45) answer with an
empty body and are simply skipped - the next run picks them up.

Run it with:
    python fetch_nordpool_zones.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

from fetch_recent_years import ensure_tables, rebuild_aggregate_tables

DB_PATH = Path(__file__).parent / "entsoe_data.db"

API_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
MARKET = "DayAhead"
CURRENCY = "EUR"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Nord Pool's area code -> the zone name the rest of this project uses
# (Nord Pool: "GER"/"DK1"; entsoe-py: "DE_LU"/"DK_1"). Same mapping as the
# local fetch_nordpool_bridge.py, minus "UK" (GB - see the module docstring).
NORDPOOL_TO_ZONE = {
    "EE": "EE", "LT": "LT", "LV": "LV", "AT": "AT", "BE": "BE", "FR": "FR",
    "GER": "DE_LU", "NL": "NL", "PL": "PL",
    "DK1": "DK_1", "DK2": "DK_2",
    "FI": "FI",
    "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3", "NO4": "NO_4", "NO5": "NO_5",
    "SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4",
    "BG": "BG",
    "SYS": "NORDIC_SYSTEM",  # Nord Pool's own reference price, not an ENTSO-E zone
    "TEL": "TEL",            # Nord-Pool-only code, no ENTSO-E equivalent
}

# Same look-back as fetch_recent.py's DAYS_BACK (what the price table's date
# picker can show).
DAYS_BACK = 4

REQUEST_PAUSE_SECONDS = 0.5
# On HTTP 401/429 wait and retry the same date a couple of times, but not for
# long - a date Nord Pool refuses does not become available by waiting.
RETRY_BACKOFFS = [20, 60]  # seconds
ABORT_AFTER_CONSECUTIVE_FAILS = 3


def fetch_day_ahead(delivery_date: str) -> dict | None:
    """The parsed JSON for this date, or None if Nord Pool has not published
    it yet (HTTP 204, or HTTP 200 with an empty body - both are normal, e.g.
    tomorrow before ~12:45). A real HTTP error or a body that is not JSON
    raises RuntimeError."""
    params = {
        "date": delivery_date,
        "market": MARKET,
        "deliveryArea": ",".join(NORDPOOL_TO_ZONE),
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
                f"Nord Pool still HTTP {resp.status_code} for {delivery_date} "
                f"after {len(RETRY_BACKOFFS)} retries"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Nord Pool returned HTTP {resp.status_code} for {delivery_date} "
                f"(body: {resp.text[:300]!r})"
            )

        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"Nord Pool's response for {delivery_date} wasn't valid JSON "
                f"(first 300 chars: {resp.text[:300]!r})"
            ) from e


def parse_hourly(data: dict) -> pd.DataFrame:
    """One column per zone (our names), one row per hour, indexed in CET/CEST.
    Nord Pool's delivery intervals are 15 minutes: they are averaged to hours,
    the same convention every ENTSO-E row uses. The averaging is done in UTC
    (no clock-change ambiguity) and converted to CET afterwards."""
    rows = []
    for entry in data.get("multiAreaEntries", []):
        row = {"deliveryStart": entry["deliveryStart"]}
        for area, price in entry["entryPerArea"].items():
            zone = NORDPOOL_TO_ZONE.get(area)
            if zone and price is not None:
                row[zone] = price
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["deliveryStart"] = pd.to_datetime(df["deliveryStart"], utc=True)
    df = df.set_index("deliveryStart").sort_index()
    hourly = df.resample("h").mean().round(2)
    hourly.index = hourly.index.tz_convert("CET")
    return hourly


def save_bridge_prices(hourly: pd.DataFrame, conn: sqlite3.Connection) -> tuple[int, int]:
    """Save every zone/hour as source='nordpool' - but only where there is not
    already an official 'entsoe' value for that exact zone+hour. Returns
    (written, skipped_because_entsoe_already_had_it)."""
    written = skipped = 0
    for zone in hourly.columns:
        for ts, price in hourly[zone].items():
            if pd.isna(price):
                continue
            cur = conn.execute(
                """
                INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, source)
                VALUES (?, ?, ?, 'nordpool')
                ON CONFLICT(zone, timestamp) DO UPDATE SET
                    price_eur_mwh = excluded.price_eur_mwh,
                    source = 'nordpool'
                WHERE day_ahead_prices.source != 'entsoe'
                """,
                (zone, ts.isoformat(), float(price)),
            )
            if cur.rowcount:
                written += 1
            else:
                skipped += 1
    conn.commit()
    return written, skipped


def fetch_and_save(dates: list[str], conn: sqlite3.Connection) -> dict:
    """Fetch and save each date in the order given."""
    stats = {"published": 0, "unpublished": 0, "written": 0, "skipped": 0, "errored": 0}
    consecutive_fails = 0
    for d_str in dates:
        try:
            data = fetch_day_ahead(d_str)
            consecutive_fails = 0
        except Exception as e:  # noqa: BLE001 - report and carry on with the other dates
            print(f"    {d_str}: ERROR - {e}")
            stats["errored"] += 1
            consecutive_fails += 1
            if consecutive_fails >= ABORT_AFTER_CONSECUTIVE_FAILS:
                print(f"    {consecutive_fails} date(s) in a row failed - stopping here. "
                      "The next scheduled run tries again.")
                break
            time.sleep(REQUEST_PAUSE_SECONDS)
            continue

        hourly = parse_hourly(data) if data is not None else pd.DataFrame()
        if hourly.empty:
            stats["unpublished"] += 1
        else:
            written, skipped = save_bridge_prices(hourly, conn)
            stats["written"] += written
            stats["skipped"] += skipped
            stats["published"] += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    return stats


def main() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ensure_tables(conn)

    # "Today" in market time (CET/CEST), like fetch_recent.py - the runner's
    # own clock is UTC. Tomorrow first, then today, then back DAYS_BACK days.
    today = pd.Timestamp.now(tz="CET").date()
    dates = [(today - pd.Timedelta(days=i)).isoformat() for i in range(-1, DAYS_BACK + 1)]
    stats = fetch_and_save(dates, conn)

    if stats["written"]:
        rebuild_aggregate_tables(conn)
    conn.close()

    print(
        f"Nord Pool: {stats['published']} date(s) published and saved, "
        f"{stats['unpublished']} not published yet, {stats['errored']} errored. "
        f"{stats['written']} zone-hour(s) written as provisional 'nordpool', "
        f"{stats['skipped']} left untouched (already had an official ENTSO-E price)."
    )


if __name__ == "__main__":
    main()
