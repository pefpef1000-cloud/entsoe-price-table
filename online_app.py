"""
online_app.py -- the ONLINE version (on Render) of the two Dash dashboards:

    /          Hourly price table   (price_table_dash.py)
    /years/    Year comparison      (year_comparison_dash.py, all 5 tabs)

Both dashboard files are the same files as on the PC, unchanged. This file
only does three things around them:

  1. Gets the data:
       entsoe_data.db        from this GitHub repo. GitHub Actions refreshes
                             it ~13 times a day.            -> price table
       entsoe_history.db.gz  from the GitHub release "online-data".
       jao_history.db.gz     upload_online_data.py on Peter's PC sends
                             them after its runs.           -> year comparison
  2. Looks for newer files every 10 minutes (and at every start -- Render's
     free plan stops the app after 15 idle minutes).
  3. Serves both dashboards from one web address, with a link bar on top.

Render starts it with (see render.yaml):
    gunicorn online_app:server --workers 1 --threads 4 --timeout 600
Try it on the PC:
    python online_app.py          -> http://127.0.0.1:8060/
"""

import argparse
import gzip
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

RECENT_DB = DATA_DIR / "entsoe_data.db"
HISTORY_DB = DATA_DIR / "entsoe_history.db"
JAO_DB = DATA_DIR / "jao_history.db"
HISTORY_FILES = [("entsoe_history.db.gz", HISTORY_DB), ("jao_history.db.gz", JAO_DB)]
STAMP_NAME = "stamp.json"          # written last by upload_online_data.py
STATE_FILE = DATA_DIR / "downloaded.json"

_http = requests.Session()
_http.headers["User-Agent"] = "entsoe-price-table-online"
_state_lock = threading.Lock()
_status = {"history": "not downloaded yet", "history_uploaded_at": None}


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


def refresh_recent() -> None:
    """The price table's database, from the repo (asks "changed since?")."""
    with _state_lock:
        state = _load_state()
    seen = state.get("recent", {})
    headers = {}
    if RECENT_DB.is_file():
        if seen.get("etag"):
            headers["If-None-Match"] = seen["etag"]
        if seen.get("last_modified"):
            headers["If-Modified-Since"] = seen["last_modified"]
    got = _download(RECENT_URL, RECENT_DB, headers=headers)
    if got is None:
        return
    with _state_lock:
        state = _load_state()
        state["recent"] = {"etag": got.get("ETag"), "last_modified": got.get("Last-Modified")}
        _save_state(state)
    log(f"price table data updated ({RECENT_DB.stat().st_size / 1e6:.0f} MB)")


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
    have_files = all(path.is_file() for _, path in HISTORY_FILES)
    if have_files and state.get("history_stamp") == stamp_text:
        _set_uploaded_at(stamp_text)
        _status["history"] = "ok"
        return
    _status["history"] = "downloading"
    started = time.time()
    for name, path in HISTORY_FILES:
        _download(f"{RELEASE_URL}/{name}", path, gunzip=True)
    with _state_lock:
        state = _load_state()
        state["history_stamp"] = stamp_text
        _save_state(state)
    _set_uploaded_at(stamp_text)
    _status["history"] = "ok"
    log(f"history updated in {time.time() - started:.0f} s "
        f"(uploaded {_status['history_uploaded_at'] or '?'})")


def _set_uploaded_at(stamp_text: str) -> None:
    try:
        when = datetime.fromisoformat(json.loads(stamp_text)["uploaded_at"])
        _status["history_uploaded_at"] = when.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except (ValueError, KeyError, TypeError):
        _status["history_uploaded_at"] = None


def _refresh_loop() -> None:
    first = True
    while True:
        for name, job in (("history", refresh_history), ("price table data", refresh_recent)):
            if first and job is refresh_recent:
                continue   # already done at start
            try:
                job()
            except Exception as exc:  # noqa: BLE001 - keep the old files, try again later
                log(f"could not refresh {name}: {exc}")
                if job is refresh_history and _status["history"] == "downloading":
                    _status["history"] = "download failed - retrying in a few minutes"
        first = False
        time.sleep(REFRESH_SECONDS)


def prepare_data() -> None:
    """Runs once at start: price table data right away (falls back to the
    copy in the repo), history in the background."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"starting - local time {time.strftime('%Y-%m-%d %H:%M %Z')}, data folder {DATA_DIR}")
    try:
        refresh_recent()
    except Exception as exc:  # noqa: BLE001
        log(f"could not download the price table data: {exc}")
        repo_copy = HERE / "entsoe_data.db"
        if not RECENT_DB.is_file() and repo_copy.is_file():
            shutil.copyfile(repo_copy, RECENT_DB)
            _make_readable(RECENT_DB)
            log("using the copy of entsoe_data.db that came with the code")
    threading.Thread(target=_refresh_loop, name="data-refresh", daemon=True).start()


# ---------------------------------------------------------------------
# 3: the two dashboards behind one address
# ---------------------------------------------------------------------
prepare_data()

from dash import Input, Output, dcc, html  # noqa: E402
from werkzeug.middleware.dispatcher import DispatcherMiddleware  # noqa: E402
from werkzeug.utils import redirect  # noqa: E402

import price_table_dash as pt  # noqa: E402

os.environ["DASH_REQUESTS_PATHNAME_PREFIX"] = "/years/"
try:
    import year_comparison_dash as yc  # noqa: E402
finally:
    os.environ.pop("DASH_REQUESTS_PATHNAME_PREFIX", None)

pt.DB_PATH = RECENT_DB
yc.DB_PATH = HISTORY_DB
yc.JAO_DB_FILE = JAO_DB

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
PAGES = [("pt", "Hourly price table", "/"), ("yc", "Year comparison", "/years/")]


def nav(here: str, note: str = ""):
    links = [
        html.Span(label, className="ol-here") if key == here else html.A(label, href=href)
        for key, label, href in PAGES
    ]
    if note:
        links.append(html.Span(note, className="ol-note"))
    return html.Nav(links, className="ol-nav")


def history_ready() -> bool:
    return HISTORY_DB.is_file() and JAO_DB.is_file()


def history_note() -> str:
    when = _status.get("history_uploaded_at")
    return f"History updated {when}" if when else ""


def price_table_page():
    return html.Div([nav("pt"), pt.serve_layout()])


def wait_page():
    state = _status["history"]
    if state.startswith("no history"):
        text = [html.B("No history uploaded yet. "),
                "Run upload_online_data.py once on the PC. "
                "This page checks again every 20 seconds."]
    else:
        text = [html.B("The history is being downloaded (about 70 MB). "),
                f"Status: {state}. This page reloads by itself every 20 seconds."]
    return html.Div([
        html.Div(text, className="ol-wait"),
        dcc.Interval(id="ol-wait-tick", interval=20_000),
    ])


def year_comparison_page():
    if not history_ready():
        return html.Div([nav("yc"), wait_page()])
    return html.Div([nav("yc", history_note()), yc.serve_layout()])


for _module in (pt, yc):
    _module.app.index_string = _module.app.index_string.replace(
        "{%css%}", "{%css%}<style>" + NAV_CSS + "</style>")

pt.app.layout = price_table_page
pt.app.validation_layout = html.Div([nav("pt"), pt.serve_layout()])
yc.app.layout = year_comparison_page
yc.app.validation_layout = html.Div([nav("yc", "x"), wait_page(), yc.serve_layout()])
yc.app.clientside_callback(
    "function (n) { if (n) { window.location.reload(); } return window.dash_clientside.no_update; }",
    Output("ol-wait-tick", "disabled"),
    Input("ol-wait-tick", "n_intervals"),
)

server = pt.app.server
server.wsgi_app = DispatcherMiddleware(server.wsgi_app, {"/years": yc.app.server})


@server.route("/healthz")
def healthz():
    return "ok"


@server.route("/years")
def years_without_slash():
    return redirect("/years/")


def main():
    parser = argparse.ArgumentParser(description="Both dashboards behind one address.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8060)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    log(f"http://{args.host}:{args.port}/   data folder: {DATA_DIR}")
    server.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
