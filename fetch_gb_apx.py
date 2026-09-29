"""
fetch_gb_apx.py - GB's day-ahead price, one consistent source from 2021 on:
Elexon's public APX market index, converted to EUR.

Why this exists
---------------
ENTSO-E stopped publishing a GB day-ahead price after 2020-12-31 (confirmed
live via zone_fetch_log). fetch_gb_bridge.py bridges GB from Nord Pool's
N2EX auction, but Nord Pool's free Data Portal API only serves roughly the
LAST MONTH OR TWO of dates - anything older is refused with HTTP 401 from
any IP, however slowly it's asked (tested 2026-09-29: yesterday and 30 days
ago HTTP 200; 90 days, 180 days, 1 year and 2021-01-01 all HTTP 401). So
N2EX can never provide the history.

Elexon (the GB balancing/settlement body) publishes "Market Index Data"
(MID) on a free public API with no key and no date restriction (checked
back to 2020). Two providers report into it: N2EXMIDP (Nord Pool's N2EX)
and APXMIDP (EPEX Spot's UK market, formerly APX). Elexon's N2EXMIDP series
is all zeros (price 0, volume 0 - no data), but APXMIDP is fully populated.
That is what this script uses.

ONE SOURCE, NOT A MIX. The year-comparison dashboard compares the same GB
prices year against year, so they must be like-for-like. APX is therefore
used for EVERY complete day from GAP_START (2021-01-01) to yesterday -
including the recent weeks where N2EX is also available: the N2EX rows
fetch_gb_bridge.py saved for those days are replaced. N2EX stays only as
the provisional price for today and tomorrow (APX has nothing for those
yet), and as a fallback for any single hour APX has no trades in; once a day
is complete APX replaces it. Only 'entsoe' rows are never overwritten.

IMPORTANT - it is a PROXY, not the N2EX auction itself:
  * APX and N2EX are two different GB exchanges. Their prices are normally
    close but not identical.
  * MID is a volume-weighted market index per half-hour, not exactly an
    auction clearing price.
Rows are tagged source='elexon_apx' so they can always be told apart. Two
honesty checks print how far APX is from a real reference: fill_gaps()
compares APX with each N2EX price it replaces, and fix_entsoe_gb_pounds()
compares APX with ENTSO-E's official 2020 GB prices hour by hour.

THE 2020 ROWS WERE POUNDS
-------------------------
Confirmed 2026-09-29 by asking ENTSO-E directly: its GB day-ahead prices are
quoted in GBP (NL, as a control, in EUR). Nothing converted them on the way
in, so the ENTSO-E GB rows (2020) sat in the price_eur_mwh column as pounds -
roughly 11% too low in euros. fix_entsoe_gb_pounds() repairs that ONCE:
  * every hour APX has for those days becomes an APX price in euros (the same
    source as 2021 onward - one consistent GB series);
  * any hour APX has no price for keeps its ENTSO-E price, converted GBP ->
    EUR with the same ECB daily rate, tagged source='entsoe_converted';
  * the original pound values are first copied to the table
    gb_entsoe_gbp_original, so nothing is lost;
  * it writes all-or-nothing (a failed download changes nothing) and is a
    no-op once no raw 'entsoe' GB rows are left.

What it does
------------
0. (Once) converts the 2020 ENTSO-E GB rows from pounds, as described above.
1. Finds every CET day between GAP_START and yesterday that still needs APX:
   no/incomplete GB price, or N2EX stand-in rows (see days_needing_apx).
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
   every source is checked to actually cover the dates being converted.
5. Saves with source='elexon_apx' - over 'nordpool' or older 'elexon_apx'
   rows, but NEVER over an 'entsoe' row.

Safe to run as often as you like: a finished day is not fetched again.

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
# ENTSO-E's own GB price (originally in pounds) after conversion to euros - only
# for hours APX has no price for. See fix_entsoe_gb_pounds().
ENTSOE_CONVERTED_TAG = "entsoe_converted"
BACKUP_TABLE = "gb_entsoe_gbp_original"

# The day after ENTSO-E's last published GB day-ahead price.
GAP_START = date(2021, 1, 1)

# Elexon rejects a request spanning more than 7 days (HTTP 400). Each
# chunk is padded by one day on either side (a CET day starts at 23:00 UTC
# the evening before and ends at 22:30 UTC, see fetch_elexon_chunk), so
# 5 days + 2 padding days = a 6-day span, safely inside the limit.
CHUNK_DAYS = 5

# A CET calendar day has 23, 24 or 25 hours (clock changes). A day with at
# least this many GB rows counts as complete.
FULL_DAY_MIN_HOURS = 23

# A day this recent that is still incomplete (or still has N2EX rows) is
# fetched again on every run - the last half-hours of the index can arrive
# late. An OLDER day that already has APX rows is final: whatever APX lacks
# for it (an hour with no trades) it never had, so it is not asked again.
RECENT_REFRESH_DAYS = 10

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
    See module docstring, step 3 (what is kept/dropped)."""
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

def days_needing_apx(conn: sqlite3.Connection, first: date, last: date,
                     today: date | None = None) -> list[date]:
    """CET days in first..last that should be (re)fetched from APX:
      * a RECENT day (within RECENT_REFRESH_DAYS) that is incomplete or still
        has N2EX rows - the last half-hours can arrive late;
      * an OLDER day that is incomplete or has N2EX rows AND has never had
        an APX row (an older day that already has APX rows is final).
    A day with a full set of ENTSO-E / APX rows is never fetched."""
    today = today or date.today()
    rows = conn.execute(
        """
        SELECT substr(timestamp, 1, 10), COUNT(*),
               SUM(source = 'nordpool'), SUM(source = ?)
        FROM day_ahead_prices
        WHERE zone = 'GB' AND timestamp >= ? AND timestamp < ?
        GROUP BY 1
        """,
        (SOURCE_TAG, first.isoformat(), (last + timedelta(days=1)).isoformat()),
    ).fetchall()
    info = {day: (n, n_nordpool or 0, n_apx or 0) for day, n, n_nordpool, n_apx in rows}

    needed = []
    for i in range((last - first).days + 1):
        d = first + timedelta(days=i)
        n, n_nordpool, n_apx = info.get(d.isoformat(), (0, 0, 0))
        incomplete = n < FULL_DAY_MIN_HOURS or n_nordpool > 0
        if not incomplete:
            continue
        if (today - d).days <= RECENT_REFRESH_DAYS or n_apx == 0:
            needed.append(d)
    return needed


def save_apx_prices(conn: sqlite3.Connection, hourly_eur: pd.Series) -> int:
    """Upsert as source='elexon_apx' - over 'nordpool' or older 'elexon_apx'
    rows, never over an 'entsoe' row. Returns how many rows were written."""
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
        WHERE day_ahead_prices.source != 'entsoe'
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
# The fill + the honesty checks
# ---------------------------------------------------------------------------

def _print_comparison(title: str, reference_name: str, both: pd.DataFrame) -> None:
    """both has columns 'ref' (the reference price) and 'apx', EUR/MWh."""
    diff = both["apx"] - both["ref"]
    within_10pct = ((diff.abs() / both["ref"].abs().clip(lower=1)) <= 0.10).mean() * 100
    print(f"{title} - {len(both)} overlapping hours:")
    print(f"  mean {reference_name} {both['ref'].mean():.2f} EUR/MWh, mean APX {both['apx'].mean():.2f} "
          f"EUR/MWh (APX minus {reference_name}: {diff.mean():+.2f} on average)")
    print(f"  typical hourly gap {diff.abs().mean():.2f} EUR/MWh, worst hour {diff.abs().max():.2f}, "
          f"correlation {both['apx'].corr(both['ref']):.3f}, {within_10pct:.0f}% of hours within 10%")


def _existing_nordpool(conn: sqlite3.Connection, first: date, last: date) -> pd.Series:
    rows = conn.execute(
        "SELECT timestamp, price_eur_mwh FROM day_ahead_prices "
        "WHERE zone = 'GB' AND source = 'nordpool' AND timestamp >= ? AND timestamp < ?",
        (first.isoformat(), (last + timedelta(days=1)).isoformat()),
    ).fetchall()
    return pd.Series({ts: price for ts, price in rows}, dtype=float)


def fill_gaps(conn: sqlite3.Connection, first: date = GAP_START, last: date | None = None) -> int:
    """Bring every GB day in first..last (default: GAP_START through
    yesterday) up to date from Elexon APX - see days_needing_apx for which
    days that means. Returns how many hourly rows were written."""
    last = last or (date.today() - timedelta(days=1))
    days = days_needing_apx(conn, first, last)
    if not days:
        print(f"GB APX fill: every day between {first} and {last} is already on APX "
              "(or ENTSO-E) - nothing to do.")
        return 0

    chunks = chunk_days(days)
    print(f"GB APX fill: {len(days)} day(s) between {first} and {last} still need APX "
          f"(no price yet, or only N2EX stand-ins) - fetching {len(chunks)} chunk(s) from Elexon.")
    fx = fetch_gbp_per_eur(days[0], days[-1])
    wanted = set(days)

    rows_written = days_filled = failed_chunks = consecutive_failures = 0
    replaced_pairs = []  # (N2EX price, APX price) for every N2EX row APX replaces
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
                      "now. Run again later; finished days are not fetched again.")
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

        n2ex = _existing_nordpool(conn, c_first, c_last)
        if not n2ex.empty and not hourly.empty:
            apx_by_ts = pd.Series(hourly.to_numpy(), index=[ts.isoformat() for ts in hourly.index])
            common = n2ex.index.intersection(apx_by_ts.index)
            if len(common):
                replaced_pairs.append(pd.DataFrame({"ref": n2ex[common], "apx": apx_by_ts[common]}))

        rows_written += save_apx_prices(conn, hourly)
        days_filled += len({ts.date() for ts in hourly.index})
        if i % 20 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] up to {c_last}: {days_filled} day(s) done, "
                  f"{rows_written} hour(s) written, {failed_chunks} failed chunk(s)")
        time.sleep(REQUEST_PAUSE_SECONDS)

    print(f"GB APX fill done: {days_filled} of {len(days)} day(s) updated "
          f"({rows_written} hourly rows, tagged '{SOURCE_TAG}'), {failed_chunks} failed chunk(s).")

    both = pd.concat(replaced_pairs) if replaced_pairs else pd.DataFrame()
    if len(both) >= 24:
        _print_comparison("APX vs the real N2EX prices it just replaced", "N2EX", both)
    return rows_written


def fix_entsoe_gb_pounds(conn: sqlite3.Connection) -> int:
    """One-off (then a no-op) repair of the ENTSO-E GB rows, which are in
    POUNDS although they sit in the price_eur_mwh column - see the module
    docstring. Returns how many rows were rewritten: 0 means nothing to do,
    or that a download failed and NOTHING was changed."""
    rows = conn.execute(
        "SELECT timestamp, price_eur_mwh FROM day_ahead_prices "
        "WHERE zone = 'GB' AND source = 'entsoe' ORDER BY timestamp"
    ).fetchall()
    if not rows:
        print("GB pounds fix: no raw ENTSO-E GB rows left - nothing to do.")
        return 0

    days = sorted({date.fromisoformat(ts[:10]) for ts, _ in rows})
    chunks = chunk_days(days)
    wanted = set(days)
    print(f"GB pounds fix: {len(rows)} ENTSO-E GB hour(s), {days[0]} to {days[-1]}, are in pounds - "
          f"fetching APX for those days ({len(chunks)} chunk(s) from Elexon)...")

    # Everything is downloaded first; the database is only touched afterwards,
    # so a failure part-way leaves it exactly as it was.
    try:
        fx = fetch_gbp_per_eur(days[0], days[-1])
        parts = []
        for i, (c_first, c_last) in enumerate(chunks, start=1):
            records = fetch_chunk_with_fallback(c_first, c_last)
            hourly = hourly_gbp_to_eur_cet(records_to_hourly_gbp(records), fx)
            keep = np.array(
                [c_first <= ts.date() <= c_last and ts.date() in wanted for ts in hourly.index],
                dtype=bool,
            )
            parts.append(hourly[keep])
            if i % 20 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}] up to {c_last}")
            time.sleep(REQUEST_PAUSE_SECONDS)
    except FetchError as e:
        print(f"  ERROR - {e}")
        print("GB pounds fix: NOTHING was changed. Run it again later.")
        return 0

    apx = pd.concat(parts) if parts else pd.Series(dtype=float)
    apx = apx[~apx.index.duplicated(keep="last")]
    apx_by_ts = {ts.isoformat(): float(p) for ts, p in apx.items() if pd.notna(p)}

    # ENTSO-E pounds -> euros, day by day, with the same ECB rate the APX
    # conversion uses.
    day_index = pd.DatetimeIndex([pd.Timestamp(ts[:10]) for ts, _ in rows])
    rate = fx.reindex(day_index, method="ffill").to_numpy()
    if not np.isfinite(rate).all():
        print("  ERROR - a GBP/EUR rate is missing for some day")
        print("GB pounds fix: NOTHING was changed. Run it again later.")
        return 0
    converted = np.round(np.array([p for _, p in rows], dtype=float) / rate, 2)

    updates = []  # (price in EUR, new source tag, timestamp)
    n_apx = n_converted = 0
    for (ts, _), eur in zip(rows, converted):
        if ts in apx_by_ts:
            updates.append((apx_by_ts[ts], SOURCE_TAG, ts))
            n_apx += 1
        else:
            updates.append((float(eur), ENTSOE_CONVERTED_TAG, ts))
            n_converted += 1

    both = pd.DataFrame({
        "ref": pd.Series(converted, index=[ts for ts, _ in rows]),
        "apx": pd.Series(apx_by_ts, dtype=float),
    }).dropna()
    if len(both) >= 24:
        _print_comparison("APX vs ENTSO-E (pounds converted to euros)", "ENTSO-E", both)

    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} "
        "(timestamp TEXT PRIMARY KEY, price_gbp_mwh REAL NOT NULL)"
    )
    with conn:  # one transaction: backup + rewrite together, or neither
        conn.executemany(
            f"INSERT OR IGNORE INTO {BACKUP_TABLE} (timestamp, price_gbp_mwh) VALUES (?, ?)", rows
        )
        conn.executemany(
            "UPDATE day_ahead_prices SET price_eur_mwh = ?, source = ? "
            "WHERE zone = 'GB' AND timestamp = ? AND source = 'entsoe'",
            updates,
        )
    print(f"GB pounds fix done: {n_apx} hour(s) replaced by APX (euros); {n_converted} hour(s) APX has "
          f"no price for kept from ENTSO-E, converted pounds -> euros (source '{ENTSOE_CONVERTED_TAG}'). "
          f"The original pound values are saved in the table {BACKUP_TABLE}.")
    return len(updates)


def main() -> None:
    # Cloud entry point: fetch_recent_years.py sits next to this file and
    # owns the database schema. (Imported here, not at the top, so the rest
    # of this file can be reused by the local backfill_gb_apx.py without it.)
    from fetch_recent_years import ensure_tables, rebuild_aggregate_tables

    conn = sqlite3.connect(Path(__file__).parent / "entsoe_data.db", timeout=30)
    ensure_tables(conn)
    changed = fix_entsoe_gb_pounds(conn)
    written = fill_gaps(conn)
    if changed or written:
        rebuild_aggregate_tables(conn)
    conn.close()


if __name__ == "__main__":
    main()
