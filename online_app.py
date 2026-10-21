"""
online_app.py -- the ONLINE version (on Render) of the Dash dashboards:

    /            Day-ahead prices     (price_table_dash.py: hourly price table
                                       + the charts of entsoe_dash.py as a tab)
    /years/      LTCAP tool           (year_comparison_dash.py + jao_dash.py tabs)
    /dayahead/   forwards to /        (merged into Day-ahead prices 2026-10-14)
    /auctions/   forwards to /years/  (merged into the LTCAP tool 2026-10-13)
    /futures/    Power futures        (eex_dash.py) -- behind a password
    /futures-vs-spot/  Futures vs spot (futures_vs_spot_dash.py) -- same password
                 (FUTURES_PASSWORD): EEX data may not be republished
    /balance/    Balance sheet        (balance_sheet_dash.py)
    /weather/    Weather forecasts    (ecmwf_dash.py; its "Check for new data"
                                       button is hidden online - the PC sends
                                       new runs after each weather update)

The dashboard files are the same files as on the PC, unchanged. This file
only does three things around them:

  1. Gets the data:
       price table -- two sources, merged (whichever has a price wins; an
       official ENTSO-E price beats a Nord Pool stand-in):
         entsoe_data.db        from this GitHub repo (GitHub Actions,
                               ~13 runs a day)
         recent_prices.db.gz   from the GitHub release "online-data": the
                               last 14 days from Peter's PC, sent right after
                               each local run fetched the prices
       the other pages: from the same release, uploaded by
       upload_online_data.py on Peter's PC:
         entsoe_history.db.gz   year comparison, day-ahead prices
         load_forecast.db.gz    day-ahead prices (load forecast)
         jao_history.db.gz      year comparison, capacity auctions
         eex_market_data.db.gz.locked  power futures -- locked with the
                                FUTURES_PASSWORD on the PC (the repo is
                                public), unlocked here with the same one
         eex_gas_eua.db.gz.locked     power futures (TTF gas + EUA), locked
         ice_market_data.db.gz.locked power futures (coal), balance sheet
                                (spark spread), locked
       weather + balance sheet (own stamp weather_stamp.json, sent after
       every weather update on the PC):
         ecmwf_data.db.gz       weather forecasts, balance sheet (model
                                wind/solar, temperature)
         balance_data.db.gz     balance sheet (last 40 days of load, load
                                model, wind/solar, prices)
  2. Looks for newer files: prices every minute, history and weather every
     10 minutes
     (and at every start -- Render's free plan stops the app after 15 idle
     minutes).
  3. Serves both dashboards from one web address, with a link bar on top.

Render starts it with (see render.yaml):
    gunicorn online_app:server --workers 1 --threads 4 --timeout 600
Try it on the PC:
    python online_app.py          -> http://127.0.0.1:8060/
"""

import argparse
import base64
import ctypes
import gc
import gzip
import hashlib
import hmac
import importlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = os.environ.get("DATA_REPO", "pefpef1000-cloud/entsoe-price-table")
BRANCH = os.environ.get("DATA_BRANCH", "master")
TAG = os.environ.get("DATA_TAG", "online-data")
RECENT_URL = os.environ.get(
    "RECENT_DB_URL", f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/entsoe_data.db")
RELEASE_URL = os.environ.get(
    "RELEASE_URL", f"https://github.com/{REPO}/releases/download/{TAG}").rstrip("/")
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(tempfile.gettempdir()) / "online_data")).resolve()
LOCAL_TZ = ZoneInfo("Europe/Copenhagen")
REFRESH_SECONDS = max(60, int(float(os.environ.get("REFRESH_MINUTES", "10")) * 60))
PRICE_REFRESH_SECONDS = max(30, int(float(os.environ.get("PRICE_REFRESH_MINUTES", "1")) * 60))

ACTIONS_DB = DATA_DIR / "actions_prices.db"     # GitHub Actions' entsoe_data.db
PC_RECENT_DB = DATA_DIR / "pc_recent_prices.db"  # last 14 days from Peter's PC
PRICE_DB = DATA_DIR / "entsoe_data.db"          # both merged = what the table reads
HISTORY_DB = DATA_DIR / "entsoe_history.db"
JAO_DB = DATA_DIR / "jao_history.db"
LOAD_DB = DATA_DIR / "load_forecast.db"
EEX_DB = DATA_DIR / "eex_market_data.db"
EEX_GAS_DB = DATA_DIR / "eex_gas_eua.db"        # eex_dash finds it next to EEX_DB
ICE_DB = DATA_DIR / "ice_market_data.db"        # eex_dash finds it next to EEX_DB
WEATHER_DB = DATA_DIR / "ecmwf_data.db"
BALANCE_DB = DATA_DIR / "balance_data.db"
LIVE_DB = DATA_DIR / "live_prices.db"            # today's EEX trades (PC's hourly task)
# (file in the release, local copy, required). The optional ones are left
# out until the PC uploads them; their pages wait. Load forecast first, so
# it is new before the history file (whose change refreshes the caches).
HISTORY_FILES = [
    ("load_forecast.db.gz", LOAD_DB, False),
    ("eex_market_data.db.gz.locked", EEX_DB, False),   # never the unlocked one
    ("eex_gas_eua.db.gz.locked", EEX_GAS_DB, False),
    ("ice_market_data.db.gz.locked", ICE_DB, False),
    ("entsoe_history.db.gz", HISTORY_DB, True),
    ("jao_history.db.gz", JAO_DB, True),
]
# Weather + balance sheet: own stamp, uploaded after every weather update.
WEATHER_FILES = [
    ("balance_data.db.gz", BALANCE_DB, False),
    ("ecmwf_data.db.gz", WEATHER_DB, False),
]
WEATHER_STAMP_NAME = "weather_stamp.json"
# Today's live EEX prices (2026-10-21): uploaded locked by the PC after every
# hourly live_snapshot.py run (weekdays 08:30-18:30). Top 50 "today", Board
# "Today (live)" and the live points in Power futures read it.
LIVE_FILES = [("live_prices.db.gz.locked", LIVE_DB, False)]
LIVE_STAMP_NAME = "live_stamp.json"
FUTURES_PASSWORD = os.environ.get("FUTURES_PASSWORD", "")
# Lock format, the same as in upload_online_data.py on the PC:
# MAGIC + salt (16 bytes) + nonce (12 bytes) + AES-256-GCM(gzip file),
# key = PBKDF2-SHA256(password, salt, LOCK_ITERATIONS).
LOCK_MAGIC = b"ONLINE-DATA-LOCK-1\n"
LOCK_ITERATIONS = 200_000
STAMP_NAME = "stamp.json"          # written last by upload_online_data.py
PC_RECENT_NAME = "recent_prices.db.gz"
PC_RECENT_STAMP_NAME = "recent_stamp.json"
STATE_FILE = DATA_DIR / "downloaded.json"

# Fill the GitHub Actions prices with the PC's: a row the GitHub file lacks
# is added; a row it has is only replaced when the PC has an official price
# and GitHub only a Nord Pool stand-in ('nordpool').
MERGE_SQL = """
INSERT OR REPLACE INTO main.day_ahead_prices (zone, timestamp, price_eur_mwh, source)
SELECT p.zone, p.timestamp, p.price_eur_mwh, p.source
FROM pc.day_ahead_prices p
LEFT JOIN main.day_ahead_prices m ON m.zone = p.zone AND m.timestamp = p.timestamp
WHERE p.price_eur_mwh IS NOT NULL AND (
      m.zone IS NULL
   OR (CASE WHEN COALESCE(p.source, 'entsoe') = 'nordpool' THEN 1 ELSE 2 END)
    > (CASE WHEN COALESCE(m.source, 'entsoe') = 'nordpool' THEN 1 ELSE 2 END))
"""
MERGE_COLUMNS = ("zone", "timestamp", "price_eur_mwh", "source")

_http = requests.Session()
_http.headers["User-Agent"] = "entsoe-price-table-online"
_state_lock = threading.Lock()
_merge_lock = threading.Lock()
_status = {"history": "not downloaded yet", "history_uploaded_at": None,
           "weather": "not downloaded yet", "weather_uploaded_at": None,
           "live": "not downloaded yet", "live_uploaded_at": None,
           "pc_prices_uploaded_at": None, "unlock_problem": None}


def log(text):
    print(f"[online_app {datetime.now():%H:%M:%S}] {text}", flush=True)


# ---------------------------------------------------------------------
# 1 + 2: data
# ---------------------------------------------------------------------
def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def _make_readable(path: Path) -> None:
    """Rollback-journal mode (the dashboards open the files read-only, which
    a WAL-mode copy does not allow) and a quick sanity check."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=DELETE").fetchall()
        tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    finally:
        conn.close()
    if not tables:
        raise ValueError(f"{path.name}: no tables in the downloaded file")


class UnlockError(Exception):
    """A locked file could not be opened (no / wrong password, damaged)."""


def unlock_bytes(blob: bytes, password: str) -> bytes:
    if not password:
        raise UnlockError("FUTURES_PASSWORD is not set in Render")
    if not blob.startswith(LOCK_MAGIC) or len(blob) < len(LOCK_MAGIC) + 28:
        raise UnlockError("not a locked file")
    try:
        from cryptography.exceptions import InvalidTag  # noqa: PLC0415
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415
    except ImportError as exc:
        raise UnlockError("the 'cryptography' package is missing "
                          "(requirements-online.txt)") from exc
    rest = blob[len(LOCK_MAGIC):]
    salt, nonce, sealed = rest[:16], rest[16:28], rest[28:]
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, LOCK_ITERATIONS, dklen=32)
    try:
        return AESGCM(key).decrypt(nonce, sealed, LOCK_MAGIC)
    except InvalidTag as exc:
        raise UnlockError("wrong password: futures_password.txt on the PC is not the same "
                          "as FUTURES_PASSWORD in Render") from exc


def _download(url: str, dest: Path, gunzip: bool = False, headers: dict | None = None,
              password: str | None = None):
    """Download to a temporary file, check it, then swap it in. Returns the
    response headers, or None if the server said "not modified". With
    `password`, the file is a locked .gz: unlocked, then unzipped."""
    tmp = dest.with_name(dest.name + ".download")
    try:
        with _http.get(url, stream=True, timeout=(20, 600), headers=headers or {}) as r:
            if r.status_code == 304:
                return None
            r.raise_for_status()
            with open(tmp, "wb") as out:
                if password is not None:
                    source = gzip.GzipFile(fileobj=io.BytesIO(unlock_bytes(r.content, password)))
                else:
                    source = gzip.GzipFile(fileobj=r.raw) if gunzip else r.raw
                shutil.copyfileobj(source, out, 1024 * 1024)
            response_headers = dict(r.headers)
        _make_readable(tmp)
        os.replace(tmp, dest)
        return response_headers
    finally:
        if tmp.exists():
            tmp.unlink()


def refresh_actions() -> bool:
    """GitHub Actions' prices, from the repo (asks "changed since?").
    True if a new file came in."""
    with _state_lock:
        state = _load_state()
    seen = state.get("actions", {})
    headers = {}
    if ACTIONS_DB.is_file():
        if seen.get("etag"):
            headers["If-None-Match"] = seen["etag"]
        if seen.get("last_modified"):
            headers["If-Modified-Since"] = seen["last_modified"]
    got = _download(RECENT_URL, ACTIONS_DB, headers=headers)
    if got is None:
        return False
    with _state_lock:
        state = _load_state()
        state["actions"] = {"etag": got.get("ETag"), "last_modified": got.get("Last-Modified")}
        _save_state(state)
    log(f"GitHub prices updated ({ACTIONS_DB.stat().st_size / 1e6:.0f} MB)")
    return True


def refresh_pc_recent() -> bool:
    """The PC's last 14 days of prices, from the release -- only when the
    small recent_stamp.json there changed. True if a new file came in."""
    r = _http.get(f"{RELEASE_URL}/{PC_RECENT_STAMP_NAME}", timeout=(20, 60))
    if r.status_code == 404:
        return False          # the PC has not sent any yet
    r.raise_for_status()
    stamp_text = r.text
    with _state_lock:
        state = _load_state()
    if PC_RECENT_DB.is_file() and state.get("pc_recent_stamp") == stamp_text:
        _status["pc_prices_uploaded_at"] = _uploaded_at(stamp_text)
        return False
    _download(f"{RELEASE_URL}/{PC_RECENT_NAME}", PC_RECENT_DB, gunzip=True)
    with _state_lock:
        state = _load_state()
        state["pc_recent_stamp"] = stamp_text
        _save_state(state)
    _status["pc_prices_uploaded_at"] = _uploaded_at(stamp_text)
    log(f"PC prices updated (uploaded {_status['pc_prices_uploaded_at'] or '?'})")
    return True


def rebuild_price_db() -> None:
    """PRICE_DB = GitHub Actions' file, filled up with the PC's prices.
    Built in a temporary file and swapped in, so the table never sees a
    half-built file."""
    with _merge_lock:
        base = ACTIONS_DB if ACTIONS_DB.is_file() else PC_RECENT_DB
        if not base.is_file():
            return
        tmp = PRICE_DB.with_name(PRICE_DB.name + ".merge")
        shutil.copyfile(base, tmp)
        added = 0
        try:
            if base is ACTIONS_DB and PC_RECENT_DB.is_file():
                conn = sqlite3.connect(str(tmp))
                try:
                    conn.execute("ATTACH DATABASE ? AS pc", (str(PC_RECENT_DB),))
                    cols = {r[1] for r in conn.execute("PRAGMA main.table_info(day_ahead_prices)")}
                    pc_cols = {r[1] for r in conn.execute("PRAGMA pc.table_info(day_ahead_prices)")}
                    if set(MERGE_COLUMNS) <= cols and set(MERGE_COLUMNS) <= pc_cols:
                        added = conn.execute(MERGE_SQL).rowcount
                        conn.commit()
                    else:
                        log("PC prices not used: the two price tables have different columns")
                    conn.execute("DETACH DATABASE pc")
                finally:
                    conn.close()
            _make_readable(tmp)
            os.replace(tmp, PRICE_DB)
        finally:
            if tmp.exists():
                tmp.unlink()
    log(f"price table data rebuilt ({added} rows from the PC added or improved)")


def refresh_history() -> None:
    """LTCAP tool, charts and futures files, from the release -- only when
    the small stamp.json there is different from the one we have."""
    _refresh_group("history", STAMP_NAME, HISTORY_FILES)


def refresh_weather() -> None:
    """Weather + balance sheet files (own stamp: the PC sends them after
    every weather update, without touching the big history files)."""
    _refresh_group("weather", WEATHER_STAMP_NAME, WEATHER_FILES)


def refresh_live() -> None:
    """Today's live EEX prices (small; new about every hour on weekdays)."""
    _refresh_group("live", LIVE_STAMP_NAME, LIVE_FILES)


def _refresh_group(group: str, stamp_name: str, files: list) -> None:
    r = _http.get(f"{RELEASE_URL}/{stamp_name}", timeout=(20, 60))
    if r.status_code == 404:
        _status[group] = f"no {group} uploaded yet (run upload_online_data.py on the PC)"
        return
    r.raise_for_status()
    stamp_text = r.text
    with _state_lock:
        state = _load_state()
    # Optional files missing last time (not uploaded / not unlockable) do
    # not count, or every check would download all ~100 MB again.
    skipped_before = set(state.get(f"{group}_skipped", []))
    have_files = all(path.is_file() for name, path, _ in files
                     if name not in skipped_before)
    if have_files and state.get(f"{group}_stamp") == stamp_text:
        _set_uploaded_at(stamp_text, group)
        _status[group] = "ok"
        return
    _status[group] = "downloading"
    started = time.time()
    skipped = []
    for name, path, required in files:
        locked = name.endswith(".locked")
        try:
            _download(f"{RELEASE_URL}/{name}", path, gunzip=True,
                      password=FUTURES_PASSWORD if locked else None)
            if locked:
                _status["unlock_problem"] = None
        except requests.HTTPError as exc:
            if required or exc.response is None or exc.response.status_code != 404:
                raise
            skipped.append(name)
            log(f"{name} is not in the release yet - its page waits")
        except UnlockError as exc:
            if required:
                raise
            skipped.append(name)
            _status["unlock_problem"] = str(exc)
            log(f"{name} could not be unlocked ({exc}) - its pages wait")
    with _state_lock:
        state = _load_state()
        state[f"{group}_stamp"] = stamp_text
        state[f"{group}_skipped"] = skipped
        _save_state(state)
    _set_uploaded_at(stamp_text, group)
    _status[group] = "ok"
    log(f"{group} updated in {time.time() - started:.0f} s "
        f"(uploaded {_status[f'{group}_uploaded_at'] or '?'})")


def _uploaded_at(stamp_text: str):
    try:
        when = datetime.fromisoformat(json.loads(stamp_text)["uploaded_at"])
        return when.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except (ValueError, KeyError, TypeError):
        return None


def _set_uploaded_at(stamp_text: str, group: str = "history") -> None:
    _status[f"{group}_uploaded_at"] = _uploaded_at(stamp_text)


def refresh_prices() -> None:
    """Both price sources; rebuild the merged file if either changed."""
    changed = False
    for name, job in (("GitHub prices", refresh_actions), ("PC prices", refresh_pc_recent)):
        try:
            changed = job() or changed
        except Exception as exc:  # noqa: BLE001 - keep the old files, try again later
            log(f"could not refresh {name}: {exc}")
    if changed or not PRICE_DB.is_file():
        rebuild_price_db()


def _refresh_loop() -> None:
    next_history = 0.0
    while True:
        if time.monotonic() >= next_history:
            for group, job in (("history", refresh_history), ("weather", refresh_weather),
                               ("live", refresh_live)):
                try:
                    job()
                except Exception as exc:  # noqa: BLE001
                    log(f"could not refresh {group}: {exc}")
                    if _status[group] == "downloading":
                        _status[group] = "download failed - retrying in a few minutes"
            next_history = time.monotonic() + REFRESH_SECONDS
        time.sleep(PRICE_REFRESH_SECONDS)
        try:
            refresh_prices()
        except Exception as exc:  # noqa: BLE001
            log(f"could not rebuild the price table data: {exc}")


def prepare_data() -> None:
    """Runs once at start: price table data right away (GitHub's file falls
    back to the copy in the repo), history in the background."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"starting - local time {time.strftime('%Y-%m-%d %H:%M %Z')}, data folder {DATA_DIR}")
    try:
        refresh_actions()
    except Exception as exc:  # noqa: BLE001
        log(f"could not download GitHub's prices: {exc}")
        repo_copy = HERE / "entsoe_data.db"
        if not ACTIONS_DB.is_file() and repo_copy.is_file():
            shutil.copyfile(repo_copy, ACTIONS_DB)
            _make_readable(ACTIONS_DB)
            log("using the copy of entsoe_data.db that came with the code")
    try:
        refresh_pc_recent()
    except Exception as exc:  # noqa: BLE001
        log(f"could not download the PC's prices: {exc}")
    try:
        rebuild_price_db()
    except Exception as exc:  # noqa: BLE001
        log(f"could not build the price table data: {exc}")
    threading.Thread(target=_refresh_loop, name="data-refresh", daemon=True).start()


# ---------------------------------------------------------------------
# 3: the dashboards behind one address
# ---------------------------------------------------------------------
prepare_data()

import dash  # noqa: E402
from dash import Input, Output, dcc, html  # noqa: E402
from werkzeug.middleware.dispatcher import DispatcherMiddleware  # noqa: E402


def _import_under(module_name: str, prefix: str):
    """Import a Dash app module so that its pages and requests live under
    `prefix` (Dash reads this setting when the app is created).

    Dash also CALLS a layout function the moment a module assigns it
    (app.layout = layout), as a check. A layout that reads its database
    then breaks the start: on Render nothing is downloaded yet, and the
    balance sheet even found this repo's own entsoe_data.db (no load tables)
    -> "no such table: load_forecast_mw" -> exit 1 (deploy 2976db7 failed,
    2026-10-16). So during the import that check gets an empty page; the
    real pages and validation layouts are set further down in this file."""
    real_layout_value = dash.Dash._layout_value
    dash.Dash._layout_value = lambda self: html.Div()
    if prefix != "/":
        os.environ["DASH_REQUESTS_PATHNAME_PREFIX"] = prefix
    try:
        return importlib.import_module(module_name)
    finally:
        dash.Dash._layout_value = real_layout_value
        os.environ.pop("DASH_REQUESTS_PATHNAME_PREFIX", None)


pt = _import_under("price_table_dash", "/")
yc = _import_under("year_comparison_dash", "/years/")
# entsoe_dash (day-ahead charts) is a tab of price_table_dash since
# 2026-10-14: price_table_dash imports it, so it is loaded already.
# No page of its own any more - /dayahead/ now forwards to /.
ed = sys.modules.get("entsoe_dash") or importlib.import_module("entsoe_dash")
# jao_dash (capacity auctions) is merged into the LTCAP tool since 2026-10-13:
# year_comparison_dash imports it and shows its tabs, so it is loaded already.
# No page of its own any more - /auctions/ now forwards to /years/.
jd = sys.modules.get("jao_dash") or importlib.import_module("jao_dash")
eex = _import_under("eex_dash", "/futures/")
fvs = _import_under("futures_vs_spot_dash", "/futures-vs-spot/")
bs = _import_under("balance_sheet_dash", "/balance/")
wx = _import_under("ecmwf_dash", "/weather/")
# Spread analyzer (2026-10-21, Peter: "push all apps online"): uses the SAME
# eex_dash and futures_vs_spot_dash modules (imported above), so it shares
# their data and the DB paths set below. spread_delivered is imported by it.
sp = _import_under("spread_dash", "/spreads/")
sd = sys.modules.get("spread_delivered")

pt.DB_PATH = PRICE_DB
yc.DB_PATH = HISTORY_DB
yc.JAO_DB_FILE = JAO_DB
ed.DB_PATH = HISTORY_DB
jd.DB_PATH = JAO_DB
eex.DB_PATH = EEX_DB         # gas_db() / ice_db() look next to it: EEX_GAS_DB, ICE_DB
if getattr(eex, "LP", None) is not None:   # live_prices.py: read-only online (no buttons)
    eex.LP.LIVE_DB = LIVE_DB


# Memory (2026-10-21): eex_dash keeps two almost identical frames, "prices"
# (all rows) and "settled" (rows with a settlement) - ~100 MB each with all
# Day futures. Online they become ONE frame (only ~1 % of the rows have no
# settlement), which keeps the Power futures + Spread analyzer pages inside
# the 512 MB of the free plan. Same dict object, so the app's cache holds it.
_eex_load = eex.load


# Text columns that are only compared / shown (never glued to other text)
# are stored as categories: a few distinct values instead of 100,000 strings.
_EEX_CATEGORY_COLS = ("long_name", "currency", "uom", "unit", "group",
                      "maturity_type")


def _eex_load_one_frame():
    data = _eex_load()
    if data.get("prices") is not data.get("settled"):
        frame = data["settled"]
        for col in _EEX_CATEGORY_COLS:
            if col in frame.columns and frame[col].dtype != "category":
                frame[col] = frame[col].astype("category")
        data["settled"] = data["prices"] = frame
    return data


eex.load = _eex_load_one_frame
fvs.EEX_DB = EEX_DB          # futures (EEX)
fvs.SPOT_DB = HISTORY_DB     # day-ahead prices (ENTSO-E)
wx.DB_PATH = WEATHER_DB
wx.ENTSOE_DB = BALANCE_DB    # Model quality tab: ENTSO-E wind/solar, last 14 days


def _if_there(path: Path):
    """The balance sheet's file finders: our files once they are downloaded
    (an explicit path that does not exist would make it fail)."""
    return lambda: path if path.is_file() else None


bs.entsoe_path = _if_there(BALANCE_DB)
bs.ecmwf_path = _if_there(WEATHER_DB)
bs.ice_path = _if_there(ICE_DB)


# Weather: no downloads online (the PC runs them and sends the result).
# The button is hidden (CSS below); the status line says where the data
# comes from.
def _weather_status() -> dict:
    stamp = _status.get("weather_uploaded_at")
    return {"finished": stamp.replace(" ", "T") if stamp else None,
            "summary": "online copy - the PC sends each new ECMWF/GFS run after its "
                       "weather update (07:15, 10:15, 19:15, 22:15)."}


wx.start_update = lambda: False
wx.update_status = _weather_status


def _dayahead_connect():
    """entsoe_dash reads prices from the history file and the load forecast
    from its own file: attached, so its SQL finds both tables unchanged."""
    conn = sqlite3.connect(HISTORY_DB.resolve().as_uri() + "?mode=ro", uri=True)
    if LOAD_DB.is_file():
        conn.execute("ATTACH DATABASE ? AS lf", (LOAD_DB.resolve().as_uri() + "?mode=ro",))
    return conn


ed._connect = _dayahead_connect
ed.NO_DB_TEXT = ("The price history for the charts is not uploaded yet. It comes with the "
                 "next upload from the PC (after its 04:00 and 13:00 runs) - reload later.")

# Tenors left out of the Corridor margins / Market value pickers online.
# Daily needs ~1.1 GB of memory (1.5 million hourly auction rows), more than
# Render's free and Starter plans have (512 MB). render.yaml sets
# HIDE_TENORS=Daily; delete that setting on a plan with 2 GB or more.
HIDE_TENORS = [t.strip() for t in os.environ.get("HIDE_TENORS", "").split(",") if t.strip()]
if HIDE_TENORS:
    _all_tenors = yc.load_corridor_tenors

    def _online_tenors():
        return [t for t in _all_tenors() if t not in HIDE_TENORS]

    yc.load_corridor_tenors = _online_tenors
    yc.CM_CAPTION += (f" Online version: the {', '.join(HIDE_TENORS)} tenor is left out "
                      "(needs more memory than the online server has).")

NAV_CSS = """
.ol-nav { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 22px;
          padding: 10px 24px; border-bottom: 1px solid #e3e5ea; background: #fafbfc;
          font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
          font-size: 15px; }
.ol-nav a { color: #2a78d6; text-decoration: none; }
.ol-nav a:hover { text-decoration: underline; }
.ol-nav .ol-here { color: #1f2430; font-weight: 700; }
.ol-nav .ol-note { color: #6b7180; font-size: 13px; margin-left: auto; }
.ol-wait { margin: 24px; padding: 14px 18px; max-width: 60ch; border-radius: 8px;
           background: #eaf2fc; border: 1px solid #c5dbf5; color: #1d4f8c;
           font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif; }
"""
PAGES = [
    ("pt", "Day-ahead prices", "/"),
    ("bs", "Balance sheet", "/balance/"),
    ("yc", "LTCAP tool", "/years/"),
    ("eex", "Power futures", "/futures/"),
    ("fvs", "Futures vs spot", "/futures-vs-spot/"),
    ("sp", "Spread analyzer", "/spreads/"),
    ("wx", "Weather forecasts", "/weather/"),
]


def nav(here: str, note: str = ""):
    links = [
        html.Span(label, className="ol-here") if key == here else html.A(label, href=href)
        for key, label, href in PAGES
    ]
    if note:
        links.append(html.Span(note, className="ol-note"))
    return html.Nav(links, className="ol-nav")


def history_note() -> str:
    when = _status.get("history_uploaded_at")
    return f"Data from the PC uploaded {when}" if when else ""


def weather_note() -> str:
    when = _status.get("weather_uploaded_at")
    return f"Data from the PC uploaded {when}" if when else ""


def prices_note() -> str:
    when = _status.get("pc_prices_uploaded_at")
    return f"Prices from the PC uploaded {when}, both sources checked every minute" if when else ""


def wait_page(group: str = "history"):
    state = _status[group]
    if state.startswith("no "):
        text = [html.B("No data uploaded yet. "),
                "Run upload_online_data.py once on the PC. "
                "This page checks again every 20 seconds."]
    elif state == "ok" and _status.get("unlock_problem"):
        text = [html.B("The EEX file could not be unlocked: "),
                f"{_status['unlock_problem']}. Fix: run \"Set online password.bat\" on "
                "the PC with the same password as in Render. This page checks again "
                "every 20 seconds."]
    elif state == "ok":
        text = [html.B("The data for this page is not uploaded yet. "),
                "It comes with the next upload from the PC (after its 04:00 and "
                "13:00 runs%s). This page checks again every 20 seconds."
                % (" and its weather updates" if group == "weather" else "")]
    else:
        text = [html.B("The data is being downloaded (about %s). "
                       % ("50 MB" if group == "weather" else "150 MB")),
                f"Status: {state}. This page reloads by itself every 20 seconds."]
    return html.Div([
        html.Div(text, className="ol-wait"),
        dcc.Interval(id="ol-wait-tick", interval=20_000),
    ])


# key -> (module, page builder, files the page needs, note)
APPS = {
    "pt": (pt, pt.serve_layout, [PRICE_DB], prices_note),
    "yc": (yc, yc.serve_layout, [HISTORY_DB, JAO_DB], history_note),
    "eex": (eex, eex.make_layout, [EEX_DB], history_note),
    "fvs": (fvs, fvs.make_layout, [EEX_DB, HISTORY_DB], history_note),
    "sp": (sp, sp.make_layout, [EEX_DB, HISTORY_DB], history_note),
    "bs": (bs, bs.layout, [BALANCE_DB], weather_note),
    "wx": (wx, wx.serve_layout, [WEATHER_DB], weather_note),
}
APP_GROUP = {"bs": "weather", "wx": "weather"}   # whose download status the wait page shows


def _page_for(key):
    module, build, needs, note = APPS[key]

    def page():
        if not all(path.is_file() for path in needs):
            return html.Div([nav(key), wait_page(APP_GROUP.get(key, "history"))])
        return html.Div([nav(key, note()), build()])
    return page


def _balance_validation():
    """The balance sheet's layout lists the countries from its database,
    which may not be downloaded yet at start: built with an empty list."""
    real = bs.countries
    bs.countries = dict
    try:
        return bs.layout()
    finally:
        bs.countries = real


def _years_validation():
    """The LTCAP tool's JAO tabs are only in its page once jao_history.db is
    downloaded. At start it is not there yet (Render starts with an empty
    disk), so their parts are added here - without them the browser refuses
    every JAO callback ("ID not found in layout") and those tabs stay empty."""
    page = yc.serve_layout()
    if jd is not None and not Path(jd.DB_PATH).is_file():
        page = html.Div([page, *jd.panels(first_visible=False), *jd.header_parts()])
    return page


VALIDATION = {"pt": pt.validation_layout, "yc": _years_validation, "eex": eex.full_layout,
              "fvs": fvs.full_layout, "bs": _balance_validation, "wx": wx.serve_layout,
              "sp": sp.make_layout}
# Weather: the "Check for new data" button starts downloads on the PC only.
EXTRA_CSS = {"wx": "#upd-btn { display: none; }"}
for _key, (_module, _build, _needs, _note) in APPS.items():
    _module.app.index_string = _module.app.index_string.replace(
        "{%css%}", "{%css%}<style>" + NAV_CSS + EXTRA_CSS.get(_key, "") + "</style>")
    _module.app.layout = _page_for(_key)
    _module.app.validation_layout = html.Div([nav(_key, "x"), wait_page(), VALIDATION[_key]()])
    _module.app.clientside_callback(
        "function (n) { if (n) { window.location.reload(); } "
        "return window.dash_clientside.no_update; }",
        Output("ol-wait-tick", "disabled"),
        Input("ol-wait-tick", "n_intervals"),
    )


class PasswordGate:
    """HTTP password (the browser asks once, any user name) in front of an
    app. Since 2026-10-15 the whole site sits behind it (same password as
    Power futures); EEX data may not be republished, and this site is public."""

    def __init__(self, app, password: str, realm: str):
        self.app, self.password, self.realm = app, password, realm

    def __call__(self, environ, start_response):
        if not self.password:
            start_response("403 Forbidden", [("Content-Type", "text/html; charset=utf-8")])
            return [b"<p style='font-family:sans-serif;margin:24px'>The dashboards are switched "
                    b"off online until a password is set: in Render, open the service, "
                    b"Environment, add <b>FUTURES_PASSWORD</b>.</p>"]
        header = environ.get("HTTP_AUTHORIZATION", "")
        if header.startswith("Basic "):
            try:
                _user, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except (ValueError, UnicodeDecodeError):
                given = ""
            if hmac.compare_digest(given.encode(), self.password.encode()):
                return self.app(environ, start_response)
        start_response("401 Unauthorized", [
            ("WWW-Authenticate", f'Basic realm="{self.realm}", charset="UTF-8"'),
            ("Content-Type", "text/plain; charset=utf-8")])
        return [b"Password needed."]


def _moved_to(location: str):
    """Old address -> new one (e.g. bookmarks of the old Capacity auctions page)."""
    def app(environ, start_response):
        start_response("301 Moved Permanently", [("Location", location),
                                                 ("Content-Type", "text/plain")])
        return [b"Moved"]
    return app


server = pt.app.server
_dispatch = DispatcherMiddleware(server.wsgi_app, {
    "/years": yc.app.server,
    "/dayahead": _moved_to("/"),
    "/auctions": _moved_to("/years/"),
    "/futures": eex.app.server,
    "/futures-vs-spot": fvs.app.server,
    "/spreads": sp.app.server,
    "/balance": bs.app.server,
    "/weather": wx.app.server,
})
# One password for every page (Peter, 2026-10-15: "all pages same pw").
# /healthz stays open: Render's health check calls it without a password.
_gate = PasswordGate(_dispatch, os.environ.get("FUTURES_PASSWORD", ""), "Dashboards")


def _site(environ, start_response):
    if environ.get("PATH_INFO", "") == "/healthz":
        return _dispatch(environ, start_response)
    return _gate(environ, start_response)


server.wsgi_app = _site


@server.route("/healthz")
def healthz():
    return "ok"


# Memory guard. Render's free plan has 512 MB and restarts the app when it
# needs more. Each dashboard keeps its results in a cache; when the whole
# app passes MEMORY_LIMIT_MB, all caches are emptied and the memory is given
# back. Pages are then simply worked out again on the next click.
MEMORY_LIMIT_MB = int(os.environ.get("MEMORY_LIMIT_MB", "350"))


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0          # not Linux (e.g. on the PC): the guard does nothing


def free_memory() -> None:
    # Caches that are safe to empty whole. The weather app's _cache also
    # holds its loaded tables (it reads _cache["key"] directly), so only its
    # per-run MW frames and the accuracy table are dropped.
    bs._CACHE.clear()
    eex._coal_cache.clear()
    wx._cache.get("mw", {}).clear()
    wx._cache["acc"] = {}
    sp._eua_cache.clear()
    for module in (pt, yc, ed, jd, eex, fvs, sd):
        cache = getattr(module, "_cache", None)
        lock = getattr(module, "_cache_lock", None) or getattr(module, "_lock", None)
        if cache is None:
            continue
        if lock is None:
            cache.clear()
        else:
            with lock:
                cache.clear()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _memory_guard() -> None:
    while True:
        time.sleep(15)
        used = _rss_mb()
        if MEMORY_LIMIT_MB and used > MEMORY_LIMIT_MB:
            free_memory()
            log(f"memory {used} MB > {MEMORY_LIMIT_MB} MB: caches emptied, now {_rss_mb()} MB")


threading.Thread(target=_memory_guard, name="memory-guard", daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="All online dashboards behind one address.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8060)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    log(f"http://{args.host}:{args.port}/   data folder: {DATA_DIR}")
    server.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
