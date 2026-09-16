"""
fetch_recent.py - the cloud-side twin of fetch_data.py.

Your real fetch_data.py keeps years of history in a 480MB database on
your own PC - way too big to store in a GitHub repo (GitHub blocks any
single file over 100MB). This script is what actually keeps the ONLINE
price table dashboard fresh instead:

  - It only asks ENTSO-E for a small recent window (a few days back,
    a couple of days ahead) - never the full history - so the database
    it writes stays tiny (a few hundred KB) forever, no matter how long
    this keeps running.
  - It writes into entsoe_data.db right here in this same folder - the
    one price_table.py in this folder reads from.
  - It's meant to be run on a schedule by GitHub Actions (see
    .github/workflows/fetch.yml), not by you by hand - though you can
    run it yourself the same way as fetch_data.py if you ever want to.

Same underlying ENTSO-E library and zone list as fetch_data.py, just
pointed at a short date window and a fresh small database instead of
your big local one.
"""

import sqlite3
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import Area

DB_PATH = Path(__file__).parent / "entsoe_data.db"

# How far back / forward to fetch. Wide enough that the date picker in
# price_table.py has a few real days to choose from even if a scheduled
# run gets skipped or fails once; small enough that the database this
# writes never grows beyond a few hundred KB.
DAYS_BACK = 4
DAYS_FORWARD = 2

# Same zone list logic as fetch_data.py: plain 2-letter country codes,
# except the handful of countries that actually trade as several separate
# price zones (or, for Germany/Ireland, under a differently-named zone)
# rather than one national price.
_COUNTRY_CODES = sorted(a.name for a in Area if len(a.name) == 2)
_COUNTRY_CODES = [c for c in _COUNTRY_CODES if c != "UK"]  # "UK" never has data - see fetch_data.py
_SPLIT_COUNTRY_ZONES = {
    "DE": ["DE_LU"],
    "IE": ["IE_SEM"],
    "DK": ["DK_1", "DK_2"],
    "NO": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"],
    "SE": ["SE_1", "SE_2", "SE_3", "SE_4"],
    "IT": ["IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD", "IT_SARD", "IT_SICI", "IT_CALA"],
}
ZONES = sorted(
    z
    for code in _COUNTRY_CODES
    for z in _SPLIT_COUNTRY_ZONES.get(code, [code])
)


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS day_ahead_prices (
            zone TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price_eur_mwh REAL NOT NULL,
            PRIMARY KEY (zone, timestamp)
        )
        """
    )


def main() -> None:
    import os

    api_key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ENTSOE_API_KEY environment variable is not set. In GitHub "
            "Actions this comes from a repo secret - see the workflow "
            "file. Running locally, set it yourself first."
        )

    now = pd.Timestamp.now(tz="CET")
    start = (now - pd.Timedelta(days=DAYS_BACK)).normalize()
    end = (now + pd.Timedelta(days=DAYS_FORWARD)).normalize()

    client = EntsoePandasClient(api_key=api_key)

    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)

    ok, empty, failed = [], [], []
    for i, zone in enumerate(ZONES, start=1):
        print(f"[{i}/{len(ZONES)}] {zone}...", end=" ", flush=True)
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
        except Exception as e:  # noqa: BLE001 - one zone failing shouldn't stop the rest
            name = type(e).__name__
            if name == "NoMatchingDataError":
                empty.append(zone)
                print("no data")
            else:
                failed.append(zone)
                print(f"failed ({name})")
            continue

        prices = prices.tz_convert("CET").resample("h").mean().dropna().round(2)
        rows = [(zone, ts.isoformat(), float(p)) for ts, p in prices.items()]
        conn.executemany(
            "INSERT OR REPLACE INTO day_ahead_prices (zone, timestamp, price_eur_mwh) "
            "VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        ok.append(zone)
        print(f"{len(rows)} rows")

    conn.close()
    print()
    print(f"Done. {len(ok)} zones fetched, {len(empty)} had no data, {len(failed)} failed.")
    if failed:
        print(f"Failed zones: {', '.join(failed)}")


if __name__ == "__main__":
    main()
