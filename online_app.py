"""
online_app.py -- the ONLINE version (on Render) of the Dash dashboards:

    /            Hourly price table   (price_table_dash.py)
    /years/      Year comparison      (year_comparison_dash.py, all 5 tabs)
    /dayahead/   Day-ahead prices     (entsoe_dash.py)
    /auctions/   Capacity auctions    (jao_dash.py)
    /futures/    Power futures        (eex_dash.py) -- behind a password
                 (FUTURES_PASSWORD): EEX data may not be republished

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
         eex_market_data.db.gz  power futures
  2. Looks for newer files: prices every 2 minutes, history every 10 minutes
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
import hmac
import importlib
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
PRICE_REFRESH_SECONDS = max(30, int(float(os.environ.get("PRICE_REFRESH_MINUTES", "2")) * 60))

ACTIONS_DB = DATA_DIR / "actions_prices.db"     # GitHub Actions' entsoe_data.db
PC_RECENT_DB = DATA_DIR / "pc_recent_prices.db"  # last 14 days from Peter's PC
PRICE_DB = DATA_DIR / "entsoe_data.db"          # both merged = what the table reads
HISTORY_DB = DATA_DIR / "entsoe_history.db"
JAO_DB = DATA_DIR / "jao_history.db"
LOAD_DB = DATA_DIR / "load_forecast.db"
EEX_DB = DATA_DIR / "eex_market_data.db"
# (file in the release, local copy, required). The optional ones are left
# out until the PC uploads them; their pages wait. Load forecast first, so
# it is new before the history file (whose change refreshes the caches).
HISTORY_FILES = [
    ("load_forecast.db.gz", LOAD_DB, False),
    ("eex_market_data.db.gz", EEX_DB, False),
    ("entsoe_history.db.gz", HISTORY_DB, True),
    ("jao_history.db.gz", JAO_DB, True),
]
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
           "pc_prices_uploaded_at": None}


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


def _download(url: str, dest: Path, gunzip: bool = False, headers: dict | None = None):
    """Download to a temporary file, check it, then swap it in. Returns the
    response headers, or None if the server said "not modified"."""
    tmp = dest.with_name(dest.name + ".download")
    try:
        with _http.get(url, stream=True, timeout=(20, 600), headers=headers or {}) as r:
            if r.status_code == 304:
                return None
            r.raise_for_status()
            with open(tmp, "wb") as out:
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
    """The year comparison's two databases, from the release -- only when the
    small stamp.json there is different from the one we have."""
    r = _http.get(f"{RELEASE_URL}/{STAMP_NAME}", timeout=(20, 60))
    if r.status_code == 404:
        _status["history"] = "no history uploaded yet (run upload_online_data.py on the PC)"
        return
    r.raise_for_status()
    stamp_text = r.text
    with _state_lock:
        state = _load_state()
    have_files = all(path.is_file() for _, path, _ in HISTORY_FILES)
    if have_files and state.get("history_stamp") == stamp_text:
        _set_uploaded_at(stamp_text)
        _status["history"] = "ok"
        return
    _status["history"] = "downloading"
    started = time.time()
    for name, path, required in HISTORY_FILES:
        try:
            _download(f"{RELEASE_URL}/{name}", path, gunzip=True)
        except requests.HTTPError as exc:
            if required or exc.response is None or exc.response.status_code != 404:
                raise
            log(f"{name} is not in the release yet - its page waits")
    with _state_lock:
        state = _load_state()
        state["history_stamp"] = stamp_text
        _save_state(state)
    _set_uploaded_at(stamp_text)
    _status["history"] = "ok"
    log(f"history updated in {time.time() - started:.0f} s "
        f"(uploaded {_status['history_uploaded_at'] or '?'})")


def _uploaded_at(stamp_text: str):
    try:
        when = datetime.fromisoformat(json.loads(stamp_text)["uploaded_at"])
        return when.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except (ValueError, KeyError, TypeError):
        return None


def _set_uploaded_at(stamp_text: str) -> None:
    _status["history_uploaded_at"] = _uploaded_at(stamp_text)


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
            try:
                refresh_history()
            except Exception as exc:  # noqa: BLE001
                log(f"could not refresh history: {exc}")
                if _status["history"] == "downloading":
                    _status["history"] = "download failed - retrying in a few minutes"
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

from dash import Input, Output, dcc, html  # noqa: E402
from werkzeug.middleware.dispatcher import DispatcherMiddleware  # noqa: E402


def _import_under(module_name: str, prefix: str):
    """Import a Dash app module so that its pages and requests live under
    `prefix` (Dash reads this setting when the app is created)."""
    if prefix == "/":
        return importlib.import_module(module_name)
    os.environ["DASH_REQUESTS_PATHNAME_PREFIX"] = prefix
    try:
        return importlib.import_module(module_name)
    finally:
        os.environ.pop("DASH_REQUESTS_PATHNAME_PREFIX", None)


pt = _import_under("price_table_dash", "/")
yc = _import_under("year_comparison_dash", "/years/")
ed = _import_under("entsoe_dash", "/dayahead/")
jd = _import_under("jao_dash", "/auctions/")
eex = _import_under("eex_dash", "/futures/")

pt.DB_PATH = PRICE_DB
yc.DB_PATH = HISTORY_DB
yc.JAO_DB_FILE = JAO_DB
ed.DB_PATH = HISTORY_DB
jd.DB_PATH = JAO_DB
eex.DB_PATH = EEX_DB


def _dayahead_connect():
    """entsoe_dash reads prices from the history file and the load forecast
    from its own file: attached, so its SQL finds both tables unchanged."""
    conn = sqlite3.connect(HISTORY_DB.resolve().as_uri() + "?mode=ro", uri=True)
    if LOAD_DB.is_file():
        conn.execute("ATTACH DATABASE ? AS lf", (LOAD_DB.resolve().as_uri() + "?mode=ro",))
    return conn


ed._connect = _dayahead_connect

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
    ("pt", "Hourly price table", "/"),
    ("yc", "Year comparison", "/years/"),
    ("ed", "Day-ahead prices", "/dayahead/"),
    ("jd", "Capacity auctions", "/auctions/"),
    ("eex", "Power futures", "/futures/"),
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


def prices_note() -> str:
    when = _status.get("pc_prices_uploaded_at")
    return f"Prices from the PC uploaded {when}, from GitHub checked every 2 min" if when else ""


def wait_page():
    state = _status["history"]
    if state.startswith("no history"):
        text = [html.B("No data uploaded yet. "),
                "Run upload_online_data.py once on the PC. "
                "This page checks again every 20 seconds."]
    elif state == "ok":
        text = [html.B("The data for this page is not uploaded yet. "),
                "It comes with the next upload from the PC (after its 04:00 and "
                "13:00 runs). This page checks again every 20 seconds."]
    else:
        text = [html.B("The data is being downloaded (about 100 MB). "),
                f"Status: {state}. This page reloads by itself every 20 seconds."]
    return html.Div([
        html.Div(text, className="ol-wait"),
        dcc.Interval(id="ol-wait-tick", interval=20_000),
    ])


# key -> (module, page builder, files the page needs, note)
APPS = {
    "pt": (pt, pt.serve_layout, [PRICE_DB], prices_note),
    "yc": (yc, yc.serve_layout, [HISTORY_DB, JAO_DB], history_note),
    "ed": (ed, ed.make_layout, [HISTORY_DB, LOAD_DB], history_note),
    "jd": (jd, jd.make_layout, [JAO_DB], history_note),
    "eex": (eex, eex.make_layout, [EEX_DB], history_note),
}


def _page_for(key):
    module, build, needs, note = APPS[key]

    def page():
        if not all(path.is_file() for path in needs):
            return html.Div([nav(key), wait_page()])
        return html.Div([nav(key, note()), build()])
    return page


VALIDATION = {"pt": pt.serve_layout, "yc": yc.serve_layout, "ed": ed.full_layout,
              "jd": jd.full_layout, "eex": eex.full_layout}
for _key, (_module, _build, _needs, _note) in APPS.items():
    _module.app.index_string = _module.app.index_string.replace(
        "{%css%}", "{%css%}<style>" + NAV_CSS + "</style>")
    _module.app.layout = _page_for(_key)
    _module.app.validation_layout = html.Div([nav(_key, "x"), wait_page(), VALIDATION[_key]()])
    _module.app.clientside_callback(
        "function (n) { if (n) { window.location.reload(); } "
        "return window.dash_clientside.no_update; }",
        Output("ol-wait-tick", "disabled"),
        Input("ol-wait-tick", "n_intervals"),
    )


class PasswordGate:
    """HTTP password (the browser asks once, any user name) in front of one
    app. EEX data may not be republished, and this site is public."""

    def __init__(self, app, password: str, realm: str):
        self.app, self.password, self.realm = app, password, realm

    def __call__(self, environ, start_response):
        if not self.password:
            start_response("403 Forbidden", [("Content-Type", "text/html; charset=utf-8")])
            return [b"<p style='font-family:sans-serif;margin:24px'>Power futures is switched "
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


server = pt.app.server
server.wsgi_app = DispatcherMiddleware(server.wsgi_app, {
    "/years": yc.app.server,
    "/dayahead": ed.app.server,
    "/auctions": jd.app.server,
    "/futures": PasswordGate(eex.app.server, os.environ.get("FUTURES_PASSWORD", ""),
                             "Power futures"),
})


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
    for module in (pt, yc, ed, jd, eex):
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
