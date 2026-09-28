"""
fetch_jao_corridors.py - the cloud-side twin of jao_scraper.py, trimmed to
the 8 corridors the online year-comparison dashboard
(pages/Year_comparison.py) actually reads, Monthly horizon only.

Your real jao_scraper.py discovers and backfills all ~120 JAO corridors
across every horizon into a 183MB local database - way too broad (and too
big for GitHub's 100MB file limit) for this cloud deployment. This script
only fetches:

  - DE-FR / FR-DE           (the DE_LU <-> FR comparison itself)
  - IF1-FR-GB / IF1-GB-FR   (IFA)
  - IF2-FR-GB / IF2-GB-FR   (IFA2)
  - EL1-FR-GB / EL1-GB-FR   (ElecLink)

- the exact 8 codes year_comparison.py's _jao_corridors_for() can ever
ask for with ZONES trimmed to DE_LU/FR (see that function, and
INTERCONNECTOR_JAO_PREFIX). All 8 are already confirmed (live, on
Peter's real jao_data.db) to have real Monthly auction data, so this
skips jao_scraper.py's discovery/binary-search machinery entirely and
just fetches these 8 directly.

Same JAO API, same auth (AUTH_API_KEY header), same date-shift quirk
(_month_bounds - JAO's marketPeriodStart is CET/CEST local midnight,
which lands on the previous UTC day, so both ends of each month's request
window are shifted back a day), same "one calendar month per request,
HTTP 400 = no data for this request" rules as jao_scraper.py - see that
script's own module docstring for the full reasoning behind each of
these.

Incremental like fetch_recent_years.py, just simpler: a month already
checked and older than STABLE_AFTER_DAYS is skipped on later runs (JAO
auctions settle for good within days of gate closure - same reasoning,
same value, as jao_scraper.py's own STABLE_AFTER_DAYS); the most recent
RECENT_RECHECK_MONTHS months are always re-checked, in case of a late
revision; anything else missing (including months before a corridor's
real start, or before ElecLink went live) is fetched once and then
remembered as stable too, so a first full backfill (2020-01 through a few
months ahead, all 8 corridors) only ever happens once.

Needs a JAO API token in the JAO_API_KEY environment variable - get one
free at https://www.jao.eu/get-token (same token Peter already has saved
in jao_scraper/api_key.txt for the real jao_scraper.py - add it as a
second GitHub Actions repo secret, alongside ENTSOE_API_KEY).

Run it with:
    python fetch_jao_corridors.py
"""

import calendar
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

DB_PATH = Path(__file__).parent / "jao_data.db"
BASE_URL = "https://api.jao.eu/OWSMP"

CORRIDORS = [
    "DE-FR", "FR-DE",
    "IF1-FR-GB", "IF1-GB-FR",
    "IF2-FR-GB", "IF2-GB-FR",
    "EL1-FR-GB", "EL1-GB-FR",
]
HORIZON = "Monthly"

EARLIEST_YEAR, EARLIEST_MONTH = 2020, 1
# Monthly auctions were measured (jao_scraper.py, 2026-09-16) publishing
# up to ~3 months ahead of today; a little headroom on top of that.
FORWARD_MONTHS = 4

# A month this old is settled for good and never re-checked again - same
# value jao_scraper.py uses for the same reason.
STABLE_AFTER_DAYS = 45
# Always re-check the most recent couple of months even if already saved,
# in case a late dispute revises a result.
RECENT_RECHECK_MONTHS = 2

REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 30


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _index_to_month(index: int) -> tuple[int, int]:
    return index // 12, index % 12 + 1


def _month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    """Identical to jao_scraper.py's own _month_bounds - see that file for
    the full reasoning (both dates shifted back one day so the window
    correctly brackets JAO's CET/CEST-midnight marketPeriodStart, under
    both CET and CEST)."""
    first = date(year, month, 1) - timedelta(days=1)
    last_day = calendar.monthrange(year, month)[1]
    last = date(year, month, last_day) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _months_to_check() -> list[tuple[int, int]]:
    start_idx = _month_index(EARLIEST_YEAR, EARLIEST_MONTH)
    today = date.today()
    end_idx = _month_index(today.year, today.month) + FORWARD_MONTHS
    return [_index_to_month(i) for i in range(start_idx, end_idx + 1)]


def load_api_key() -> str:
    key = os.environ.get("JAO_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "JAO_API_KEY environment variable is not set. In GitHub "
            "Actions this comes from a repo secret - see the workflow "
            "file. Running locally, set it yourself first (the same "
            "token saved in jao_scraper/api_key.txt works)."
        )
    return key


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Same auctions/auction_results columns as jao_scraper.py's own
    ensure_tables() - trimmed to just the 2 tables year_comparison.py
    actually reads (skips auction_winners/eic_names/corridor_horizon_status,
    none of which this cloud twin needs), plus chunk_status for the
    incremental stable-month tracking above."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auctions (
            identification TEXT PRIMARY KEY,
            corridor_code TEXT,
            horizon_name TEXT,
            cancelled INTEGER,
            last_data_update TEXT,
            market_period_start TEXT,
            market_period_stop TEXT,
            atc_gate_opening TEXT,
            atc_gate_closure TEXT,
            bid_gate_opening TEXT,
            bid_gate_closure TEXT,
            is_bid_gate_open INTEGER,
            provisional_auctionresult TEXT,
            xn_rule TEXT,
            operational_message TEXT,
            ftroption TEXT,
            fetched_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auction_results (
            identification TEXT,
            corridor_code TEXT,
            product_identification TEXT,
            product_hour TEXT,
            offered_capacity REAL,
            requested_capacity REAL,
            allocated_capacity REAL,
            auction_price REAL,
            comment TEXT,
            additional_message TEXT,
            FOREIGN KEY (identification) REFERENCES auctions (identification)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_status (
            corridor_code TEXT,
            horizon_name TEXT,
            year_month TEXT,
            status TEXT,
            rows_saved INTEGER,
            checked_at TEXT,
            PRIMARY KEY (corridor_code, horizon_name, year_month)
        )
        """
    )
    conn.commit()


def _request_with_retry(session: requests.Session, url: str, params: dict) -> requests.Response:
    """Same retry-on-429 behaviour as jao_scraper.py's own
    _request_with_retry, without the multi-thread pacing lock (this script
    is single-threaded - 8 corridors x ~85 months is small enough not to
    need it)."""
    wait = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    network error ({e}), retrying in {wait}s...")
            time.sleep(wait)
            wait *= 2
            continue

        if response.status_code == 429:
            if attempt == MAX_RETRIES:
                response.raise_for_status()
            print(f"    rate-limited by JAO (429), waiting {wait}s before retrying...")
            time.sleep(wait)
            wait *= 2
            continue

        return response

    raise RuntimeError("unreachable")  # pragma: no cover


def get_auctions(session: requests.Session, corridor: str, fromdate: str, todate: str) -> tuple[str, list[dict]]:
    """Same idea as jao_scraper.py's own get_auctions - a plain HTTP 400
    means "no data for this request" (a normal, expected outcome, not an
    error), not a real failure."""
    params = {"corridor": corridor, "horizon": HORIZON, "fromdate": fromdate, "todate": todate}
    response = _request_with_retry(session, f"{BASE_URL}/getauctions", params=params)
    if response.status_code == 400:
        return "no_data", []
    response.raise_for_status()
    return "ok", response.json()


def save_auction(conn: sqlite3.Connection, auction: dict, fetched_at: str) -> None:
    """Identical to jao_scraper.py's own save_auction, minus the
    auction_winners part (that table isn't kept in this cloud twin -
    year_comparison.py never reads winner EIC codes)."""
    identification = auction.get("identification")
    conn.execute(
        """
        INSERT OR REPLACE INTO auctions (
            identification, corridor_code, horizon_name, cancelled,
            last_data_update, market_period_start, market_period_stop,
            atc_gate_opening, atc_gate_closure, bid_gate_opening,
            bid_gate_closure, is_bid_gate_open, provisional_auctionresult,
            xn_rule, operational_message, ftroption, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identification,
            auction.get("corridorCode"),
            auction.get("horizonName"),
            int(bool(auction.get("cancelled"))),
            auction.get("lastDataUpdate"),
            auction.get("marketPeriodStart"),
            auction.get("marketPeriodStop"),
            auction.get("atcGateOpening"),
            auction.get("atcGateClosure"),
            auction.get("bidGateOpening"),
            auction.get("bidGateClosure"),
            int(bool(auction.get("isBidGateOpen"))),
            auction.get("provisionalAuctionresult"),
            auction.get("xnRule"),
            auction.get("operationalMessage"),
            auction.get("ftroption"),
            fetched_at,
        ),
    )
    conn.execute("DELETE FROM auction_results WHERE identification = ?", (identification,))
    for result in auction.get("results", []):
        conn.execute(
            """
            INSERT INTO auction_results (
                identification, corridor_code, product_identification,
                product_hour, offered_capacity, requested_capacity,
                allocated_capacity, auction_price, comment, additional_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identification,
                result.get("corridorCode"),
                result.get("productIdentification"),
                result.get("productHour"),
                result.get("offeredCapacity"),
                result.get("requestedCapacity"),
                result.get("allocatedCapacity"),
                result.get("auctionPrice"),
                result.get("comment"),
                result.get("additionalMessage"),
            ),
        )


def _is_stable(conn: sqlite3.Connection, corridor: str, year: int, month: int) -> bool:
    row = conn.execute(
        "SELECT status FROM chunk_status WHERE corridor_code = ? AND horizon_name = ? AND year_month = ?",
        (corridor, HORIZON, _month_key(year, month)),
    ).fetchone()
    if row is None or row[0] not in ("ok", "no_data"):
        return False  # never checked, or the last check was a real error - always retry
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    return (date.today() - month_end).days > STABLE_AFTER_DAYS


def fetch_month(session: requests.Session, conn: sqlite3.Connection, corridor: str, year: int, month: int) -> tuple[str, int]:
    fromdate, todate = _month_bounds(year, month)
    status, auctions = get_auctions(session, corridor, fromdate, todate)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows_saved = 0
    if status == "ok":
        for auction in auctions:
            save_auction(conn, auction, fetched_at)
            rows_saved += 1
    conn.execute(
        "INSERT OR REPLACE INTO chunk_status "
        "(corridor_code, horizon_name, year_month, status, rows_saved, checked_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (corridor, HORIZON, _month_key(year, month), status, rows_saved, fetched_at),
    )
    conn.commit()
    return status, rows_saved


def main() -> None:
    api_key = load_api_key()
    session = requests.Session()
    session.headers.update({"AUTH_API_KEY": api_key})

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    months = _months_to_check()
    recent = set(months[-RECENT_RECHECK_MONTHS:]) if len(months) >= RECENT_RECHECK_MONTHS else set(months)

    checked = skipped = total_saved = 0
    for corridor in CORRIDORS:
        for year, month in months:
            if (year, month) not in recent and _is_stable(conn, corridor, year, month):
                skipped += 1
                continue
            status, rows_saved = fetch_month(session, conn, corridor, year, month)
            checked += 1
            total_saved += rows_saved
            print(f"{corridor} {_month_key(year, month)}: {status} ({rows_saved} auction(s))")
            time.sleep(REQUEST_DELAY_SECONDS)

    conn.close()
    print(
        f"Done. {checked} month(s) checked, {skipped} already-stable month(s) skipped, "
        f"{total_saved} auction(s) saved/refreshed."
    )


if __name__ == "__main__":
    main()
