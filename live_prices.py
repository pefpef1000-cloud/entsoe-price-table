"""
Live (delayed) futures prices, fetched on demand.

Used by the "Update prices" buttons in eex_dash.py (Energy futures). Fetches
ONLY the contracts a page needs, today's trades only, and stores them in
live_prices.db next to this file. Settlements stay in eex_market_data.db -
live prices never overwrite them.

Sources (both checked 2026-10-21):
  - EEX trade feed  api.eex-group.com/pub/market-data/chart/intraday
      every trade of the day, ~10-15 min delayed. Power (all areas, all
      tenors Year..Day), TTF and other gas hubs, EUA.
  - ICE  www.ice.com/marketdata/api/productguide/charting/data/current-day
      10-minute bars, delayed. Used for API2 / API4 coal only (EEX has no
      coal). Until the first trade of the day ICE returns YESTERDAY's bars,
      so bars are filtered on today's date.

A contract fetched less than FRESH_MINUTES ago is skipped (a second page or
a second click costs nothing). Can also be run by hand:
    python live_prices.py DEBY 202701 Year Base
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_DB = HERE / "live_prices.db"
ICE_DB = HERE.parent / "ice_scraper" / "ice_market_data.db"

EEX_URL = "https://api.eex-group.com/pub/market-data/chart/intraday"
ICE_URL = "https://www.ice.com/marketdata/api/productguide/charting/data/current-day"
ICE_COAL_PAGES = {
    "API2": "https://www.ice.com/products/243/Rotterdam-Coal-Futures/data",
    "API4": "https://www.ice.com/products/241/API4-Richards-Bay-Coal-Futures/data",
}

FRESH_MINUTES = 10        # skip a contract fetched less than this ago
EEX_PAUSE = 2.0           # EEX rate-limits hard (~10 quick calls -> blocked)
ICE_PAUSE = 0.5
TIMEOUT = 20
RETRIES = 2
TICK_MINUTES = 5          # intraday line stored as the last price per 5 minutes

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
EEX_HEADERS = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
               "Referer": "https://www.eex.com/", "Origin": "https://www.eex.com"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_quotes (
    source     TEXT NOT NULL,      -- EEX / ICE
    code       TEXT NOT NULL,      -- EEX short code (DEBY) or coal hub (API2)
    delivery   TEXT NOT NULL,      -- EEX YYYYMM, ICE strip (Nov26, Cal 27)
    trade_date TEXT NOT NULL,      -- local trading day YYYY-MM-DD
    last REAL, open REAL, high REAL, low REAL,
    volume REAL, trades INTEGER,
    first_time TEXT, last_time TEXT,   -- local time of first / last trade
    fetched_at TEXT,                   -- local time of the fetch
    PRIMARY KEY (source, code, delivery, trade_date)
);
CREATE TABLE IF NOT EXISTS live_ticks (
    source TEXT NOT NULL, code TEXT NOT NULL, delivery TEXT NOT NULL,
    trade_date TEXT NOT NULL, ts TEXT NOT NULL, price REAL,
    PRIMARY KEY (source, code, delivery, trade_date, ts)
);
CREATE TABLE IF NOT EXISTS live_fetches (   -- last attempt, also when nothing traded
    source TEXT NOT NULL, code TEXT NOT NULL, delivery TEXT NOT NULL,
    fetched_at TEXT, status TEXT,           -- ok / no trades / error: ...
    PRIMARY KEY (source, code, delivery)
);
"""

_db_lock = threading.Lock()


def now_local() -> datetime:
    return datetime.now().replace(microsecond=0)        # PC clock = Danish time


def today() -> str:
    return now_local().date().isoformat()


def utc_to_local(text: str) -> datetime | None:
    """'2026-10-20T15:43:56.998219000Z' -> local naive datetime."""
    try:
        t = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return t.astimezone().replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


class connect:
    """with connect() as con: ... - creates the tables, commits and CLOSES
    (sqlite3's own context manager doesn't close; open handles keep the
    file locked on Windows/OneDrive)."""
    def __init__(self, path: Path | None = None):
        self.path = path or LIVE_DB

    def __enter__(self):
        self.con = sqlite3.connect(self.path, timeout=30)
        self.con.executescript(SCHEMA)
        return self.con

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.con.commit()
        self.con.close()


# ---------------------------------------------------------------------
# Contract specs
# ---------------------------------------------------------------------
def eex_spec(code: str, delivery: str, maturity: str, product: str = "Base") -> dict:
    """One EEX contract. maturity = the dashboard's maturity_type."""
    commodity = "POWER"
    if code == "FEUA":
        commodity, maturity = "ENVIRONMENTALS", "Month"
    elif code.startswith("G") and len(code) == 4 and code[2] == "B" and code[3] in "MQY":
        commodity = "NATGAS"
    api_product = product
    try:                                   # Week/Weekend/Day families
        import power_contracts as pc
        commodity = pc.commodity_of(code) if commodity == "POWER" else commodity
        api_product = pc.api_product_of(code)
    except Exception:
        pass
    if api_product in ("Unknown", None) or code in ("FEUA",) or commodity == "NATGAS":
        api_product = "Base"
    return {"source": "EEX", "code": code, "delivery": str(delivery),
            "maturity": "Month" if maturity == "December" else maturity,
            "commodity": commodity, "product": api_product}


def ice_spec(hub: str, strip: str, market_id: int) -> dict:
    return {"source": "ICE", "code": hub, "delivery": strip, "market_id": int(market_id)}


def ice_market_ids(hub: str) -> dict:
    """strip -> ICE marketId, from ice_scraper's ice_markets table."""
    if not ICE_DB.exists():
        return {}
    try:
        con = sqlite3.connect(f"{ICE_DB.resolve().as_uri()}?mode=ro", uri=True)
        rows = con.execute("SELECT strip, market_id FROM ice_markets WHERE product = ?",
                           (hub,)).fetchall()
        con.close()
        return {s: m for s, m in rows}
    except sqlite3.Error:
        return {}


# ---------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------
def _get_json(url: str, headers: dict) -> dict | list | None:
    last = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return {}
            last = f"HTTP {e.code}"
        except Exception as e:                         # network, timeout, bad JSON
            last = type(e).__name__
        if attempt < RETRIES:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(last or "failed")


def _ticks_5min(points: list[tuple[datetime, float]]) -> list[tuple[str, float]]:
    """Last price per TICK_MINUTES bucket, time stamped at the bucket end."""
    out: dict[str, float] = {}
    for t, v in points:
        m = (t.minute // TICK_MINUTES + 1) * TICK_MINUTES
        h = t.hour + m // 60
        out[f"{h:02d}:{m % 60:02d}"] = v
    return sorted(out.items())


def fetch_eex(spec: dict, day: str) -> dict:
    q = {"shortCode": spec["code"], "commodity": spec["commodity"], "pricing": "F",
         "area": "DE", "product": spec["product"], "maturity": spec["delivery"],
         "maturityType": spec["maturity"], "startDate": day, "endDate": day,
         "isRolling": "true"}
    j = _get_json(f"{EEX_URL}?{urllib.parse.urlencode(q)}", EEX_HEADERS) or {}
    series = {s.get("serieName"): s.get("timeAndValue") or [] for s in j.get("series") or []}
    px = []
    for t, v in series.get("lastPx", []):
        lt = utc_to_local(t)
        if v is not None and lt is not None and lt.date().isoformat() == day:
            px.append((lt, float(v)))
    px.sort()
    vol = [v for t, v in series.get("volume", [])
           if v is not None and (utc_to_local(t) or datetime.min).date().isoformat() == day]
    return _summary(px, sum(vol) if vol else None)


def fetch_ice(spec: dict, day: str) -> dict:
    url = f"{ICE_URL}?{urllib.parse.urlencode({'mode': 2, 'marketId': spec['market_id']})}"
    headers = {"User-Agent": UA, "Accept": "application/json",
               "Referer": ICE_COAL_PAGES.get(spec["code"], "https://www.ice.com/")}
    j = _get_json(url, headers) or {}
    px = []
    for t, v in j.get("bars") or []:
        try:   # "Tue Oct 20 15:30:00 2026", GMT
            lt = (datetime.strptime(t, "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
                  .astimezone().replace(tzinfo=None))
        except (ValueError, TypeError):
            continue
        if v is not None and lt.date().isoformat() == day:
            px.append((lt, float(v)))
    px.sort()
    return _summary(px, None)


def _summary(px, volume) -> dict:
    if not px:
        return {"trades": 0}
    vals = [v for _, v in px]
    return {"last": vals[-1], "open": vals[0], "high": max(vals), "low": min(vals),
            "volume": volume, "trades": len(vals),
            "first_time": px[0][0].strftime("%H:%M"), "last_time": px[-1][0].strftime("%H:%M"),
            "ticks": _ticks_5min(px)}


def is_fresh(con, spec, minutes=FRESH_MINUTES) -> bool:
    row = con.execute("SELECT fetched_at, status FROM live_fetches WHERE source=? AND code=? "
                      "AND delivery=?", (spec["source"], spec["code"], spec["delivery"])).fetchone()
    if not row or not row[0] or str(row[1]).startswith("error"):
        return False
    age = (now_local() - datetime.fromisoformat(row[0])).total_seconds() / 60
    return row[0][:10] == today() and age < minutes


def fetch(specs: list[dict], progress=None, max_age=FRESH_MINUTES) -> dict:
    """Fetch the given contracts (skipping fresh ones). progress(done, total,
    text) is called after each contract. Returns counts."""
    day = today()
    seen, todo = set(), []
    with _db_lock, connect() as con:
        for s in specs:
            key = (s["source"], s["code"], s["delivery"])
            if key in seen:
                continue
            seen.add(key)
            if not is_fresh(con, s, max_age):
                todo.append(s)
    counts = {"asked": len(seen), "fetched": 0, "fresh": len(seen) - len(todo),
              "traded": 0, "errors": 0}
    last_source = None
    for i, s in enumerate(todo, 1):
        if last_source == s["source"]:
            time.sleep(EEX_PAUSE if s["source"] == "EEX" else ICE_PAUSE)
        last_source = s["source"]
        try:
            r = fetch_eex(s, day) if s["source"] == "EEX" else fetch_ice(s, day)
            status = "ok" if r.get("trades") else "no trades"
        except Exception as e:
            r, status = {"trades": 0}, f"error: {e}"
            counts["errors"] += 1
        stamp = now_local().isoformat(sep=" ")
        with _db_lock, connect() as con:
            con.execute("INSERT OR REPLACE INTO live_fetches VALUES (?,?,?,?,?)",
                        (s["source"], s["code"], s["delivery"], stamp, status))
            if r.get("trades"):
                con.execute(
                    "INSERT OR REPLACE INTO live_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s["source"], s["code"], s["delivery"], day, r["last"], r["open"], r["high"],
                     r["low"], r["volume"], r["trades"], r["first_time"], r["last_time"], stamp))
                con.execute("DELETE FROM live_ticks WHERE source=? AND code=? AND delivery=? "
                            "AND trade_date=?", (s["source"], s["code"], s["delivery"], day))
                con.executemany("INSERT INTO live_ticks VALUES (?,?,?,?,?,?)",
                                [(s["source"], s["code"], s["delivery"], day, t, v)
                                 for t, v in r["ticks"]])
                counts["traded"] += 1
        counts["fetched"] += 1
        if progress:
            progress(i, len(todo), f"{s['code']} {s['delivery']}: {status}")
    return counts


# ---------------------------------------------------------------------
# Reading (for the dashboard)
# ---------------------------------------------------------------------
def read(specs: list[dict]) -> dict:
    """{(source, code, delivery): {...quote..., 'fetched_at', 'status'}} for
    today. Contracts never fetched today are left out."""
    if not LIVE_DB.exists() or not specs:
        return {}
    day = today()
    out = {}
    with _db_lock:
        con = sqlite3.connect(LIVE_DB, timeout=30)
        try:
            for s in specs:
                k = (s["source"], s["code"], s["delivery"])
                f = con.execute("SELECT fetched_at, status FROM live_fetches WHERE source=? AND "
                                "code=? AND delivery=?", k).fetchone()
                if not f or not f[0] or f[0][:10] != day:
                    continue
                q = con.execute("SELECT last, open, high, low, volume, trades, first_time, "
                                "last_time FROM live_quotes WHERE source=? AND code=? AND "
                                "delivery=? AND trade_date=?", (*k, day)).fetchone()
                rec = {"fetched_at": f[0], "status": f[1]}
                if q:
                    rec.update(dict(zip(["last", "open", "high", "low", "volume", "trades",
                                         "first_time", "last_time"], q)))
                out[k] = rec
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return out


def ticks(source: str, code: str, delivery: str, day: str | None = None) -> list[tuple[str, float]]:
    if not LIVE_DB.exists():
        return []
    with _db_lock:
        con = sqlite3.connect(LIVE_DB, timeout=30)
        try:
            return con.execute("SELECT ts, price FROM live_ticks WHERE source=? AND code=? AND "
                               "delivery=? AND trade_date=? ORDER BY ts",
                               (source, code, delivery, day or today())).fetchall()
        except sqlite3.Error:
            return []
        finally:
            con.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    code, delivery, maturity = sys.argv[1:4]
    product = sys.argv[4] if len(sys.argv) > 4 else "Base"
    spec = eex_spec(code, delivery, maturity, product)
    print(fetch([spec], progress=lambda i, n, t: print(f"  {i}/{n} {t}"), max_age=0))
    print(read([spec]))
