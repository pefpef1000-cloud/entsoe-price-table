"""
fetch_gb_apx.py - fills GB's day-ahead price history (2021 -> now) from
Elexon's public API, for every day Nord Pool's own free API won't serve.

Why this exists
---------------
ENTSO-E stopped publishing a GB day-ahead price after 2020-12-31 (confirmed
live via zone_fetch_log). fetch_gb_bridge.py bridges GB from Nord Pool's
N2EX auction, but Nord Pool's free Data Portal API only serves roughly the
LAST MONTH OR TWO of dates - anything older is refused with HTTP 401 from
any IP, however slowly it's asked (tested 2026-09-29: yesterday and 30 days
ago HTTP 200; 90 days, 180 days, 1 year and 2021-01-01 all HTTP 401). So
that script can never fill 2021 -> a few weeks ago.

Elexon (the GB balancing/settlement body) publishes "Market Index Data"
(MID) on a free public API with no key and no date restriction, back to at
least 2021. Two providers report into it: N2EXMIDP (Nord Pool's N2EX) and
APXMIDP (EPEX Spot's UK market, formerly APX). Elexon's N2EXMIDP series is
all zeros (price 0, volume 0 - no data), but APXMIDP is fully populated.
That is what this script uses.

IMPORTANT - it is a PROXY, not the N2EX auction itself:
  * APX and N2EX are two different GB exchanges. Their prices are normally
    close but not identical.
  * MID is a volume-weighted market index per half-hour, not exactly an
    auction clearing price.
Rows are therefore tagged source='elexon_apx' (never 'nordpool' or
'entsoe'), so they can always be told apart. After every fill the script
prints a comparison of APX against the real N2EX prices over the last 28
days, so the size of the difference is visible instead of assumed.

What it does
------------
1. Finds every CET day between GAP_START and yesterday that has no
   (or an incomplete) GB price in day_ahead_prices, from ANY source.
2. Fetches those days from Elexon in chunks (the API rejects requests that
   span more than 7 days).
3. Keeps only APXMIDP, drops half-hours with price 0 AND volume 0 (no
   trades - not a real price of zero), and averages the two half-hours of
   each hour (volume-weighted) into one hourly GBP/MWh price.
4. Converts GBP -> EUR with the ECB's daily reference rate for that day
   (forward-filled over weekends/holidays), since the rest of the database
   is EUR/MWh. Rates come from Frankfurter (api.frankfurter.dev, which
   republishes the ECB's reference rates), with the ECB's own data API as a
   backup. NOT the ECB's old eurofxref-hist.csv file: on 2026-09-29 that URL
   served a stale snapshot ending in Feb 2010 with dummy-looking values, so
   every source is now checked to actually cover the dates being converted.
5. Saves with source='elexon_apx' - but NEVER over an existing 'entsoe' or
   'nordpool' row (those always win).

Because step 1 only looks for days with no price, a day that Nord Pool's
N2EX bridge later covers keeps its real N2EX price, and this script leaves
every already-covered day alone. Safe to run as often as you like.

Run it with:
    python fetch_gb_apx.py
(For your own local database use backfill_gb_apx.py instead - it points
this same code at the local entsoe_data.db.)
"""

import io
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
APX_PROVIDER = "APXMIDP"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1/{start}..{end}"  # ECB reference rates, JSON
ECB_SDMX_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.GBP.EUR.SP00.A"  # backup
# Frankfurter serves daily rates for ranges of at least ~5 months (checked
# 2026-09-29); longer ranges are thinned out, so ask in windows of this size.
FX_WINDOW_DAYS = 120

SOURCE_TAG = "elexon_apx"

# The day after ENTSO-E's last published GB day-ahead price.
GAP_START = date(2021, 1, 1)

# Elexon rejects a request spanning more than 7 days (HTTP 400). Each
# chunk is padded by one day on either side (a CET day starts at 23:00 UTC
# the evening before and ends at 22:30 UTC, see fetch_elexon_chunk), so
# 5 days + 2 padding days = a 6-day span, safely inside the limit.
CHUNK_DAYS = 5

# A CET calendar day has 23, 24 or 25 hours (clock changes). A day with at
# least this many GB rows already counts as fully covered.
FULL_DAY_MIN_HOURS = 23

REQUEST_PAUSE_SECONDS = 0.3
RETRY_BACKOFFS = [5, 20, 60]  # seconds, for network errors / HTTP 429 / 5xx
ABORT_AFTER_CONSECUTIVE_FAILED_CHUNKS = 4

HEADERS = {
    "User-Agent": "entsoe-price-table personal research project (GB price fill)",
    "Accept": "application/json",
}


class FetchError(RuntimeError):
    """A fetch failed in a way worth reporting (after any retries)."""


class BadRequest(FetchError):
    """HTTP 400 - e.g. Elexon refusing a too-long date span. Not retried."""


class NotFound(FetchError):
    """HTTP 404 - e.g. an FX date range with no business day in it."""


def _get_with_retry(url: str, params: dict, what: str, headers: dict | None = None) -> requests.Response:
    last_problem = "unknown error"
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        try:
            resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=60)
        except requests.RequestException as e:
            last_problem = f"{what}: network error: {e}"
        else:
            if resp.status_code == 200:
                return resp
            last_problem = f"{what}: HTTP {resp.status_code} ({resp.text[:200]!r})"
            if resp.status_code == 400:
                raise BadRequest(last_problem)
            if resp.status_code == 404:
                raise NotFound(last_problem)
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise FetchError(last_problem)
        if attempt < len(RETRY_BACKOFFS):
            time.sleep(RETRY_BACKOFFS[attempt])
    raise FetchError(last_problem)


# ---------------------------------------------------------------------------
# GBP -> EUR
# ---------------------------------------------------------------------------

def _fx_from_frankfurter(start: date, end: date) -> pd.Series:
    """{date: GBP per 1 EUR} for business days in start..end, in windows."""
    rates: dict[pd.Timestamp, float] = {}
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=FX_WINDOW_DAYS - 1), end)
        url = FRANKFURTER_URL.format(start=window_start.isoformat(), end=window_end.isoformat())
        try:
            resp = _get_with_retry(url, {"base": "EUR", "symbols": "GBP"}, "Frankfurter FX rates")
            payload = resp.json().get("rates", {})
        except NotFound:
            payload = {}  # a window with no business day in it - fine, the others cover it
        except ValueError as e:
            raise FetchError("Frankfurter FX rates: response wasn't valid JSON") from e
        for day, values in payload.items():
            if "GBP" in values:
                rates[pd.Timestamp(day)] = float(values["GBP"])
        window_start = window_end + timedelta(days=1)
    return pd.Series(rates, dtype=float).sort_index()


def _fx_from_ecb_api(start: date, end: date) -> pd.Series:
    """Backup: the ECB's own SDMX data API (CSV)."""
    resp = _get_with_retry(
        ECB_SDMX_URL,
        {"startPeriod": start.isoformat(), "endPeriod": end.isoformat(), "format": "csvdata"},
        "ECB data API",
        headers={"User-Agent": HEADERS["User-Agent"], "Accept": "text/csv"},
    )
    df = pd.read_csv(io.StringIO(resp.text))
    if not {"TIME_PERIOD", "OBS_VALUE"} <= set(df.columns):
        raise FetchError(f"ECB data API: unexpected columns {list(df.columns)[:8]}")
    return pd.Series(
        pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["TIME_PERIOD"]),
    ).dropna().sort_index()


def _check_fx_covers(parsed: pd.Series, first: date, last: date) -> None:
    """Refuse rates that don't actually cover first..last with plausible
    values (a stale or garbled source must never be used to convert)."""
    if parsed.empty:
        raise FetchError("no rates returned")
    if not parsed.between(0.5, 1.5).all():
        raise FetchError(f"implausible GBP/EUR values (min {parsed.min()}, max {parsed.max()})")
    if parsed.index.min() > pd.Timestamp(first):
        raise FetchError(f"rates only start {parsed.index.min().date()}, need {first} or earlier")
    if parsed.index.max() < pd.Timestamp(last) - timedelta(days=10):
        raise FetchError(f"rates only run to {parsed.index.max().date()}, need up to about {last}")


def fetch_gbp_per_eur(first: date, last: date) -> pd.Series:
    """Daily GBP-per-1-EUR, one value for every calendar day from 10 days
    before `first` to 3 days after `last` (weekends and holidays carry the
    previous business day's rate). EUR/MWh = GBP/MWh divided by this.
    Tries Frankfurter, then the ECB data API; each is checked to really
    cover the period, and if neither does this raises."""
    start = first - timedelta(days=15)
    end = min(last + timedelta(days=3), date.today())
    problems = []
    for name, fetch in (("Frankfurter", _fx_from_frankfurter), ("ECB data API", _fx_from_ecb_api)):
        try:
            parsed = fetch(start, end)
            _check_fx_covers(parsed, first, last)
        except FetchError as e:
            problems.append(f"{name}: {e}")
            continue
        parsed = parsed[~parsed.index.duplicated(keep="last")]
        calendar = pd.date_range(first - timedelta(days=10), last + timedelta(days=3), freq="D")
        return parsed.reindex(parsed.index.union(calendar)).ffill().bfill().reindex(calendar)
    raise FetchError("no usable GBP/EUR rates - refusing to convert. " + " | ".join(problems))


# ---------------------------------------------------------------------------
# Elexon
# ---------------------------------------------------------------------------

def fetch_elexon_chunk(first: date, last: date) -> list[dict]:
    """Every half-hourly Market Index record that covers the CET days
    first..last (inclusive). Asks Elexon for one day before `first` through
    one day after `last` (the API filters on the UTC start time of each
    half-hour), which is a little more than needed - the caller trims to the
    exact CET days."""
    params = {
        "from": (first - timedelta(days=1)).isoformat(),
        "to": (last + timedelta(days=1)).isoformat(),
    }
    resp = _get_with_retry(ELEXON_URL, params, f"Elexon {first}..{last}")
    try:
        payload = resp.json()
    except ValueError as e:
        raise FetchError(f"Elexon {first}..{last}: response wasn't valid JSON") from e
    records = payload.get("data") if isinstance(payload, dict) else payload
    return records or []


def fetch_chunk_with_fallback(first: date, last: date) -> list[dict]:
    """fetch_elexon_chunk, but if Elexon refuses the multi-day span (HTTP
    400) fall back to one request per day instead of giving up."""
    try:
        return fetch_elexon_chunk(first, last)
    except BadRequest:
        if first == last:
            raise
    records: list[dict] = []
    d = first
    while d <= last:
        records.extend(fetch_elexon_chunk(d, d))
        time.sleep(REQUEST_PAUSE_SECONDS)
        d += timedelta(days=1)
    return records


def records_to_hourly_gbp(records: list[dict]) -> pd.Series:
    """APXMIDP half-hour records -> hourly GBP/MWh, indexed by UTC hour start.
    See module docstring, steps 3 (what is kept/dropped)."""
    rows = [r for r in records if r.get("dataProvider") == APX_PROVIDER]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    missing = {"startTime", "price"} - set(df.columns)
    if missing:
        raise FetchError(f"Elexon records are missing expected field(s) {sorted(missing)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["start"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["price", "start"])
    # price 0 with volume 0 = "nothing traded", not a genuine zero price.
    df = df[~((df["price"] == 0) & (df["volume"] == 0))]
    # Neighbouring chunks overlap by a day, so the same half-hour can arrive twice.
    df = df.drop_duplicates(subset=["start"])
    if df.empty:
        return pd.Series(dtype=float)
    # Floor in UTC: every CET/CEST hour boundary is also a UTC hour boundary,
    # and flooring a tz-aware CET index can raise at the autumn clock change.
    df["hour"] = df["start"].dt.floor("h")
    df["price_x_volume"] = df["price"] * df["volume"]
    grouped = df.groupby("hour")
    volume = grouped["volume"].sum()
    weighted = grouped["price_x_volume"].sum() / volume.where(volume > 0)
    hourly = weighted.fillna(grouped["price"].mean())  # no volume info -> plain mean
    return hourly.sort_index()


def hourly_gbp_to_eur_cet(hourly_gbp: pd.Series, gbp_per_eur: pd.Series) -> pd.Series:
    """UTC-hour GBP/MWh -> EUR/MWh (2 decimals), re-indexed to CET/CEST hour
    starts - the same timestamps ENTSO-E and Nord Pool rows already use."""
    if hourly_gbp.empty:
        return pd.Series(dtype=float)
    cet_index = hourly_gbp.index.tz_convert("CET")
    local_days = pd.DatetimeIndex([ts.date() for ts in cet_index])
    rate = gbp_per_eur.reindex(local_days, method="ffill").to_numpy()
    eur = np.round(hourly_gbp.to_numpy() / rate, 2)
    return pd.Series(eur, index=cet_index)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def missing_days(conn: sqlite3.Connection, first: date, last: date) -> list[date]:
    """CET days in first..last without a full day of GB prices (any source)."""
    rows = conn.execute(
        """
        SELECT substr(timestamp, 1, 10), COUNT(*)
        FROM day_ahead_prices
        WHERE zone = 'GB' AND timestamp >= ? AND timestamp < ?
        GROUP BY 1
        """,
        (first.isoformat(), (last + timedelta(days=1)).isoformat()),
    ).fetchall()
    full = {day for day, n in rows if n >= FULL_DAY_MIN_HOURS}
    all_days = (first + timedelta(days=i) for i in range((last - first).days + 1))
    return [d for d in all_days if d.isoformat() not in full]


def save_apx_prices(conn: sqlite3.Connection, hourly_eur: pd.Series) -> int:
    """Upsert as source='elexon_apx', never over an 'entsoe' or 'nordpool'
    row. Returns how many rows were actually written."""
    rows = [(ts.isoformat(), float(p)) for ts, p in hourly_eur.items() if pd.notna(p)]
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        f"""
        INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, source)
        VALUES ('GB', ?, ?, '{SOURCE_TAG}')
        ON CONFLICT(zone, timestamp) DO UPDATE SET
            price_eur_mwh = excluded.price_eur_mwh,
            source = '{SOURCE_TAG}'
        WHERE day_ahead_prices.source NOT IN ('entsoe', 'nordpool')
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def chunk_days(days: list[date]) -> list[tuple[date, date]]:
    """Group sorted days into runs of consecutive days, each split into
    windows of at most CHUNK_DAYS days: [(first, last), ...]."""
    chunks: list[tuple[date, date]] = []
    run_start = prev = None
    for d in sorted(days):
        if run_start is not None and (d - prev).days == 1 and (d - run_start).days < CHUNK_DAYS:
            prev = d
            continue
        if run_start is not None:
            chunks.append((run_start, prev))
        run_start = prev = d
    if run_start is not None:
        chunks.append((run_start, prev))
    return chunks


# ---------------------------------------------------------------------------
# The fill + the honesty check
# ---------------------------------------------------------------------------

def fill_gaps(conn: sqlite3.Connection, first: date = GAP_START, last: date | None = None) -> int:
    """Fill every missing GB day in first..last (default: GAP_START through
    yesterday) from Elexon APX. Returns how many hourly rows were written."""
    last = last or (date.today() - timedelta(days=1))
    days = missing_days(conn, first, last)
    if not days:
        print(f"GB APX fill: no gaps between {first} and {last} - nothing to do.")
        return 0

    chunks = chunk_days(days)
    print(f"GB APX fill: {len(days)} day(s) without a full GB price between "
          f"{first} and {last} - fetching {len(chunks)} chunk(s) from Elexon.")
    fx = fetch_gbp_per_eur(days[0], days[-1])
    wanted = set(days)

    rows_written = days_filled = failed_chunks = consecutive_failures = 0
    for i, (c_first, c_last) in enumerate(chunks, start=1):
        try:
            records = fetch_chunk_with_fallback(c_first, c_last)
            hourly = hourly_gbp_to_eur_cet(records_to_hourly_gbp(records), fx)
        except FetchError as e:
            print(f"  [{i}/{len(chunks)}] {c_first}..{c_last}: ERROR - {e}")
            failed_chunks += 1
            consecutive_failures += 1
            if consecutive_failures >= ABORT_AFTER_CONSECUTIVE_FAILED_CHUNKS:
                print(f"  {consecutive_failures} chunks in a row failed - stopping here for "
                      "now. Run again later; already-filled days are skipped.")
                break
            continue
        consecutive_failures = 0

        # Only this chunk's own CET days: the one-day padding around the request
        # also returns a stray hour or two of the neighbouring days, and those
        # would be half-averaged (only one of their two half-hours fetched).
        keep = np.array(
            [c_first <= ts.date() <= c_last and ts.date() in wanted for ts in hourly.index],
            dtype=bool,
        )
        hourly = hourly[keep]
        rows_written += save_apx_prices(conn, hourly)
        days_filled += len({ts.date() for ts in hourly.index})
        if i % 20 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] up to {c_last}: {days_filled} day(s) filled, "
                  f"{rows_written} hour(s) written, {failed_chunks} failed chunk(s)")
        time.sleep(REQUEST_PAUSE_SECONDS)

    print(f"GB APX fill done: {days_filled} of {len(days)} missing day(s) filled "
          f"({rows_written} hourly rows, tagged '{SOURCE_TAG}'), {failed_chunks} failed chunk(s).")
    return rows_written


def validate_against_nordpool(conn: sqlite3.Connection, days: int = 28) -> None:
    """Print how far APX is from the real N2EX price over the last `days`
    days (only hours where the database holds a genuine Nord Pool N2EX
    price). Purely informational - writes nothing."""
    last = date.today() - timedelta(days=1)
    first = last - timedelta(days=days - 1)
    rows = conn.execute(
        """
        SELECT timestamp, price_eur_mwh FROM day_ahead_prices
        WHERE zone = 'GB' AND source = 'nordpool' AND timestamp >= ? AND timestamp < ?
        """,
        (first.isoformat(), (last + timedelta(days=1)).isoformat()),
    ).fetchall()
    if len(rows) < 48:
        print(f"APX vs N2EX check: only {len(rows)} genuine N2EX hour(s) in the last "
              f"{days} days - too few to compare.")
        return
    n2ex = pd.Series({ts: price for ts, price in rows})

    period = [first + timedelta(days=i) for i in range(days)]
    records: list[dict] = []
    for c_first, c_last in chunk_days(period):
        records.extend(fetch_chunk_with_fallback(c_first, c_last))
        time.sleep(REQUEST_PAUSE_SECONDS)
    apx = hourly_gbp_to_eur_cet(records_to_hourly_gbp(records), fetch_gbp_per_eur(first, last))
    apx.index = [ts.isoformat() for ts in apx.index]

    both = pd.concat([n2ex.rename("n2ex"), apx.rename("apx")], axis=1, join="inner").dropna()
    if len(both) < 24:
        print(f"APX vs N2EX check: only {len(both)} overlapping hour(s) - too few to compare.")
        return
    diff = both["apx"] - both["n2ex"]
    within_10pct = ((diff.abs() / both["n2ex"].abs().clip(lower=1)) <= 0.10).mean() * 100
    print(f"APX vs N2EX check, {len(both)} overlapping hours over the last {days} days:")
    print(f"  mean N2EX {both['n2ex'].mean():.2f} EUR/MWh, mean APX {both['apx'].mean():.2f} EUR/MWh "
          f"(APX minus N2EX: {diff.mean():+.2f} on average)")
    print(f"  typical hourly gap {diff.abs().mean():.2f} EUR/MWh, worst hour {diff.abs().max():.2f}, "
          f"correlation {both['apx'].corr(both['n2ex']):.3f}, "
          f"{within_10pct:.0f}% of hours within 10%")


def main() -> None:
    # Cloud entry point: fetch_recent_years.py sits next to this file and
    # owns the database schema. (Imported here, not at the top, so the rest
    # of this file can be reused by the local backfill_gb_apx.py without it.)
    from fetch_recent_years import ensure_tables, rebuild_aggregate_tables

    conn = sqlite3.connect(Path(__file__).parent / "entsoe_data.db", timeout=30)
    ensure_tables(conn)
    written = fill_gaps(conn)
    try:
        validate_against_nordpool(conn)
    except Exception as e:  # informational only - never fail the step over it
        print(f"APX vs N2EX check skipped: {e}")
    if written:
        rebuild_aggregate_tables(conn)
    conn.close()


if __name__ == "__main__":
    main()
