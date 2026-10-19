"""Balance sheet - hourly residual balance, back and forward (Dash).

Balance = Load - the ticked sources (Wind, Solar, ... more later).

Pick a country and a number of days N (3-21):
  * Past (N days, today included): per day two columns - Price or Clean
    spark spread (toggle) and Balance.
  * Forward (N days from tomorrow): per day two columns - Balance and
    Delta (balance minus the balance of the day before, same hour).

Data (read-only):
  * entsoe_data.db  (next to this script): load_forecast_mw,
    wind_solar_forecast, day_ahead_prices.  ENTSO-E day-ahead forecasts.
  * entsoe_data.db: load_forecast_model_mw - the MODELLED hourly load
    curve from model_load_forecast.py.  Used only where ENTSO-E has
    published nothing, i.e. beyond D+1.  Not a TSO publication.
  * ..\\ecmwf_scraper\\ecmwf_data.db: mw_hourly - MODELLED hourly wind
    and solar MW from the ECMWF / GFS runs, used the same way.  Without
    these the forward balance stays empty, because load alone minus a
    missing wind figure is not a balance.

  Any day that uses either is marked "(model)" in the header and printed
  in italics.  The "Model forward" tick box turns the whole thing off.
  * ..\\ice_scraper\\ice_market_data.db: TTF front month + EUA Dec
    settlements for the clean spark spread.

Clean spark spread (hourly) = price - TTF / efficiency - EUA * carbon factor
  carbon factor = 0.184 tCO2/MWh gas (gross CV, Platts) / efficiency
  (0.368 at 50 %), same as the Energy futures app (Peter 2026-10-19: "all
  0,368"; was 0.202 / 0.404).  Gas and EUA = last ICE settlement BEFORE the
  delivery day (the day the day-ahead auction ran), TTF front month,
  EUA December of the delivery year.

Run:  python balance_sheet_dash.py --open      (port 8059)
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme

HERE = Path(__file__).resolve().parent
try:  # where the database lives (entsoe_db_location.txt), see entsoe_db.py
    from entsoe_db import entsoe_db_path as _edp
    _ENTSOE_DB = _edp()
except ImportError:
    _ENTSOE_DB = None
TZ = "Europe/Brussels"
DEFAULT_PORT = 8059
DEFAULT_COUNTRY = "DE"
FONT = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"

# --------------------------------------------------------------------------
# Sources that can be subtracted from load.  Add new ones here:
#   key: (label, table, [columns summed], [columns that must be present])
# Wind rule (project standard): onshore missing -> empty, offshore missing -> 0.
# Order = display order (Peter 2026-10-16: solar first, temperature always last).
SOURCES = {
    "solar": ("Solar", "wind_solar_forecast", ["solar_mw"], ["solar_mw"]),
    "wind": ("Wind", "wind_solar_forecast",
             ["wind_onshore_mw", "wind_offshore_mw"], ["wind_onshore_mw"]),
}
DEFAULT_SOURCES = ["solar", "wind"]

# Modelled stand-in for each source beyond what ENTSO-E publishes:
# column in ecmwf_data.db's mw_hourly.  A source with no entry here simply
# has no forward fallback.
MODEL_SOURCES = {"wind": "wind_mw", "solar": "solar_mw"}

# mw_hourly keeps every run.  Pick the same way model_load_forecast.py
# picks its temperature: prefer IFS, fall back to GFS once IFS is more
# than MAX_STALENESS behind the newest run in the table.
ECMWF_PREFERENCE = ("ifs", "gfs")
ECMWF_MAX_STALENESS_HOURS = 18

NAMES = {
    "AL": "Albania", "AT": "Austria", "BA": "Bosnia", "BE": "Belgium",
    "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "Great Britain", "GE": "Georgia",
    "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "UA": "Ukraine", "XK": "Kosovo",
}
# Price zone when the load zone has no price of its own.
PRICE_OVERRIDE = {"IE": "IE_SEM", "LU": "DE_LU"}

PEAK_HOURS = range(8, 20)  # 08:00-20:00 local
OFFPEAK_HOURS = [h for h in range(24) if h not in PEAK_HOURS]
SUMMARY_ROWS = (("Base", range(24)), ("Peak", PEAK_HOURS),
                ("Off-peak", OFFPEAK_HOURS))
TEMP_ROW = "Temp °C"


# --------------------------------------------------------------------------
# Database helpers
def find_db(name: str, extra: list[Path]) -> Path | None:
    for p in [HERE / name, HERE / "Claude outputs" / name, HERE.parent / name,
              Path.cwd() / name, *extra]:
        if p.exists():
            return p
    return None


ARGS = argparse.Namespace(db=None, ice_db=None, ecmwf_db=None)


def entsoe_path() -> Path | None:
    if ARGS.db:
        return Path(ARGS.db)
    if _ENTSOE_DB is not None and _ENTSOE_DB.exists():
        return _ENTSOE_DB
    return find_db("entsoe_data.db", [])


def ecmwf_path() -> Path | None:
    if ARGS.ecmwf_db:
        return Path(ARGS.ecmwf_db)
    return find_db("ecmwf_data.db",
                   [HERE.parent / "ecmwf_scraper" / "ecmwf_data.db"])


def ice_path() -> Path | None:
    if ARGS.ice_db:
        return Path(ARGS.ice_db)
    return find_db("ice_market_data.db",
                   [HERE.parent / "ice_scraper" / "ice_market_data.db"])


def stamp(p: Path | None) -> tuple:
    if not p:
        return ()
    out = []
    for f in (p, Path(str(p) + "-wal")):
        try:
            s = f.stat()
            out.append((s.st_mtime, s.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def connect(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True,
                           timeout=30)


_CACHE: dict = {}


def cached(key, p, fn):
    k = (key, stamp(p))
    if k not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[k] = fn()
    return _CACHE[k]


def tables(p: Path | None) -> set:
    """Table names in a database.

    Everything modelled is optional: if model_load_forecast.py has never
    been run, or ecmwf_data.db is not there, the dashboard has to show the
    published data and a note - not a stack trace.  Every read of an
    optional table goes through this first.
    """
    if not p:
        return set()

    def q():
        with connect(p) as c:
            return {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}

    return cached(("tables", str(p)), p, q)


# --------------------------------------------------------------------------
# Countries
def countries() -> dict:
    """{code: {"label", "load": [zones], "price": [zones], "ws": code}}"""
    p = entsoe_path()
    if not p:
        return {}

    def build():
        with connect(p) as c:
            zones = [r[0] for r in c.execute(
                "SELECT DISTINCT zone FROM load_forecast_mw ORDER BY zone")]
        groups: dict[str, list[str]] = {}
        for z in zones:
            groups.setdefault(z.split("_")[0], []).append(z)
        out = {}
        for code, zs in sorted(groups.items()):
            load = [code] if code in zs else zs
            if len(load) == 1:
                price = [PRICE_OVERRIDE.get(load[0], load[0])]
            else:
                price = load
            name = NAMES.get(code, code)
            if len(load) == 1 and load[0] != code:
                label = f"{name} ({load[0]})"
            elif len(load) > 1:
                label = f"{name} ({len(load)} zones summed)"
            else:
                label = f"{name} ({code})"
            out[code] = {"label": label, "load": load, "price": price,
                         "ws": code}
        return out

    return cached("countries", p, build)


# --------------------------------------------------------------------------
# Loading hourly series (local time, tz-aware)
def _read(table: str, cols: list[str], zones: list[str], start: date,
          end: date) -> pd.DataFrame:
    p = entsoe_path()
    lo = (start - timedelta(days=2)).isoformat()
    hi = (end + timedelta(days=2)).isoformat()

    def q():
        qm = ",".join("?" * len(zones))
        with connect(p) as c:
            df = pd.read_sql_query(
                f"SELECT zone, timestamp, {', '.join(cols)} FROM {table} "
                f"WHERE zone IN ({qm}) AND timestamp >= ? AND timestamp < ?",
                c, params=[*zones, lo, hi])
        df["t"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        return df.drop(columns="timestamp")

    return cached((table, tuple(cols), tuple(zones), lo, hi), p, q)


def _zone_wide(table: str, zones: list[str], start: date, end: date,
               idx: pd.DatetimeIndex, col: str = "forecast_mw"):
    """Wide frame, rows = idx, one column per zone.  None if there is
    nothing (table absent, or no rows in the window)."""
    p = entsoe_path()
    if not p or table not in tables(p):
        return None
    df = _read(table, [col], zones, start, end)
    if not len(df):
        return None
    return (df.pivot_table(index="t", columns="zone", values=col)
            .reindex(columns=zones).reindex(idx))


def fmt_run(ts) -> str:
    """Run stamp -> "14.10 14:00 ifs", local time.

    model_load_forecast.py writes run_time as "<weather model> <stamp>"
    ("ifs 2026-10-14T12:00:00+00:00"); older rows are the bare stamp. Any
    token that parses as a time is the time, the rest is the model name.
    """
    parts = str(ts).split()
    for i, part in enumerate(parts):
        try:
            when = f"{pd.Timestamp(part).tz_convert(TZ):%d.%m %H:%M}"
        except Exception:
            continue
        rest = " ".join(parts[:i] + parts[i + 1:])
        return f"{when} {rest}".strip()
    return str(ts)[:19]


def ecmwf_pick(zone: str):
    """(model, run_time) to use for one country, or None.

    mw_hourly keeps every run of both models, so a plain read would
    average yesterday's GFS into today's IFS.  One run is chosen here and
    used for everything, which is also what makes the note honest about
    which run you are looking at.
    """
    p = ecmwf_path()
    if not p or "mw_hourly" not in tables(p):
        return None

    def q():
        with connect(p) as c:
            runs = dict(c.execute(
                "SELECT model, MAX(run_time) FROM mw_hourly WHERE zone = ? "
                "GROUP BY model", (zone,)).fetchall())
        if not runs:
            return None
        newest = max(runs.values())
        for m in ECMWF_PREFERENCE:
            rt = runs.get(m)
            if rt and (pd.Timestamp(newest) - pd.Timestamp(rt)
                       <= pd.Timedelta(hours=ECMWF_MAX_STALENESS_HOURS)):
                return m, rt
        return next((m, rt) for m, rt in runs.items() if rt == newest)

    return cached(("mw_pick", zone), p, q)


def ecmwf_mw(zone: str, start: date, end: date, idx: pd.DatetimeIndex):
    """Hourly wind_mw / solar_mw for one country from the chosen run."""
    pick = ecmwf_pick(zone)
    if not pick:
        return None
    p = ecmwf_path()
    lo = (start - timedelta(days=2)).isoformat()
    hi = (end + timedelta(days=2)).isoformat()

    def q():
        with connect(p) as c:
            df = pd.read_sql_query(
                "SELECT time_utc, wind_mw, solar_mw FROM mw_hourly WHERE "
                "zone = ? AND model = ? AND run_time = ? AND time_utc >= ? "
                "AND time_utc < ?", c,
                params=[zone, pick[0], pick[1], lo, hi])
        if df.empty:
            return None
        df["t"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(TZ)
        return df.drop(columns="time_utc").groupby("t").mean()

    out = cached(("mw_hourly", zone, pick, lo, hi), p, q)
    return None if out is None else out.reindex(idx)


def temp_daily(zone: str, days: list[date]):
    """Daily mean temperature (local day) for one country, or None.

    ERA5 (actual) where it exists; after that the ECMWF forecast, each
    hour taken from the newest run that covers it, so past days use the
    shortest lead and forward days the latest run.  The model is chosen
    with the same IFS-first rule as the MW (ecmwf_pick).
    Returns (Series date -> deg C, source text).
    """
    p = ecmwf_path()
    if not p or not days:
        return None
    have = tables(p)
    lo = (days[0] - timedelta(days=2)).isoformat()
    hi = (days[-1] + timedelta(days=2)).isoformat()

    def q():
        parts, src = [], []
        with connect(p) as c:
            if "era5_hourly" in have:
                e = pd.read_sql_query(
                    "SELECT time_utc AS t, temp_c FROM era5_hourly WHERE "
                    "zone = ? AND temp_c IS NOT NULL AND time_utc >= ? AND "
                    "time_utc < ?", c, params=[zone, lo, hi])
                if len(e):
                    e["t"] = pd.to_datetime(e["t"], utc=True, format="mixed")
                    e = e.set_index("t")["temp_c"].sort_index()
                    parts.append(e)
                    src.append(f"ERA5 to {e.index.max().tz_convert(TZ):%d.%m}")
            if "ecmwf_forecast" in have:
                runs = dict(c.execute(
                    "SELECT model, MAX(run_time) FROM ecmwf_forecast WHERE "
                    "zone = ? AND temp_c IS NOT NULL GROUP BY model",
                    (zone,)).fetchall())
                runs = {m: r for m, r in runs.items() if r}
                model = None
                if runs:
                    newest = max(runs.values())
                    for m in ECMWF_PREFERENCE:
                        if m in runs and (pd.Timestamp(newest)
                                          - pd.Timestamp(runs[m])
                                          <= pd.Timedelta(
                                              hours=ECMWF_MAX_STALENESS_HOURS)):
                            model = m
                            break
                    model = model or max(runs, key=runs.get)
                # Chosen model first; the other model only fills hours the
                # chosen one has no run for (e.g. days before IFS was
                # stored).
                fc = None
                for m in ([model] + [x for x in runs if x != model]
                          if model else []):
                    f = pd.read_sql_query(
                        "SELECT run_time, valid_time AS t, temp_c FROM "
                        "ecmwf_forecast WHERE zone = ? AND model = ? AND "
                        "temp_c IS NOT NULL AND valid_time >= ? AND "
                        "valid_time < ? AND run_time >= ?", c,
                        params=[zone, m, lo, hi,
                                (days[0] - timedelta(days=20)).isoformat()])
                    if f.empty:
                        continue
                    f = (f.sort_values("run_time")
                         .drop_duplicates("t", keep="last"))
                    f["t"] = pd.to_datetime(f["t"], utc=True, format="mixed")
                    f = f.set_index("t")["temp_c"].sort_index()
                    if parts:
                        f = f[f.index > parts[0].index.max()]
                    if fc is None:
                        fc = f
                        src.append(f"ECMWF {m.upper()} newest run per hour "
                                   f"(latest {fmt_run(runs[m])})")
                    else:
                        # only before the first / after the last hour of fc,
                        # never mixed into its 3-hourly steps
                        add = f[(f.index < fc.index.min())
                                | (f.index > fc.index.max())]
                        if len(add):
                            fc = pd.concat([fc, add]).sort_index()
                            src.append(f"{m.upper()} where "
                                       f"{model.upper()} has no run")
                if fc is not None:
                    parts.append(fc)
        if not parts:
            return None
        ser = pd.concat(parts).sort_index()
        ser = ser[~ser.index.duplicated(keep="first")]
        grid = pd.date_range(ser.index.min().ceil("h"),
                             ser.index.max().floor("h"), freq="h", tz="UTC")
        ser = (ser.reindex(ser.index.union(grid))
               .interpolate(method="time", limit_area="inside",
                            limit=6).reindex(grid))
        loc = ser.tz_convert(TZ)
        g = loc.groupby(loc.index.date)
        daily = g.mean().where(g.count() >= 20)
        return daily, "; ".join(src)

    return cached(("temp", zone, lo, hi), p, q)


def ecmwf_run_mw(zone: str, model: str, run: str, start: date, end: date,
                 idx: pd.DatetimeIndex):
    """Hourly wind_mw / solar_mw for one named run (local time), or None."""
    p = ecmwf_path()
    if not p or "mw_hourly" not in tables(p):
        return None
    lo = (start - timedelta(days=2)).isoformat()
    hi = (end + timedelta(days=2)).isoformat()

    def q():
        with connect(p) as c:
            df = pd.read_sql_query(
                "SELECT time_utc, wind_mw, solar_mw FROM mw_hourly WHERE "
                "zone = ? AND model = ? AND run_time = ? AND time_utc >= ? "
                "AND time_utc < ?", c, params=[zone, model, run, lo, hi])
        if df.empty:
            return None
        df["t"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(TZ)
        return df.drop(columns="time_utc").groupby("t").mean()

    out = cached(("mw_run", zone, model, run, lo, hi), p, q)
    return None if out is None else out.reindex(idx)


def temp_run_daily(zone: str, model: str, run: str) -> pd.Series | None:
    """Daily mean temperature (local day) from ONE ecmwf_forecast run."""
    p = ecmwf_path()
    if not p or "ecmwf_forecast" not in tables(p):
        return None

    def q():
        with connect(p) as c:
            f = pd.read_sql_query(
                "SELECT valid_time AS t, temp_c FROM ecmwf_forecast WHERE "
                "zone = ? AND model = ? AND run_time = ? AND temp_c IS NOT "
                "NULL", c, params=[zone, model, run])
        if f.empty:
            return None
        f["t"] = pd.to_datetime(f["t"], utc=True, format="mixed")
        ser = f.set_index("t")["temp_c"].sort_index()
        grid = pd.date_range(ser.index.min().ceil("h"),
                             ser.index.max().floor("h"), freq="h", tz="UTC")
        ser = (ser.reindex(ser.index.union(grid))
               .interpolate(method="time", limit_area="inside", limit=6)
               .reindex(grid).tz_convert(TZ))
        g = ser.groupby(ser.index.date)
        return g.mean().where(g.count() >= 20)

    return cached(("temp_run", zone, model, run), p, q)


COMPARE_DAYS = 14   # how far back the "Compare with" picker lists runs


def compare_options(zone: str, today: date):
    """(dropdown options, default value) for the "Compare with" picker.

    Every stored weather run of the last COMPARE_DAYS days except the
    current one, newest first.  Value = "model|run_time".  Default =
    yesterday's 00z run of the current model.
    """
    p = ecmwf_path()
    pick = ecmwf_pick(zone)
    if not p or not pick or "mw_hourly" not in tables(p):
        return [], None
    since = (today - timedelta(days=COMPARE_DAYS)).isoformat()

    def q():
        with connect(p) as c:
            return c.execute(
                "SELECT DISTINCT model, run_time FROM mw_hourly WHERE "
                "zone = ? AND run_time >= ? ORDER BY run_time DESC, model",
                (zone, since)).fetchall()

    runs = [r for r in cached(("runs", zone, since), p, q) if tuple(r) != pick]
    opts = [{"label": f"{m.upper()} {pd.Timestamp(rt):%a %d.%m %H}z",
             "value": f"{m}|{rt}"} for m, rt in runs]
    want = f"{pick[0]}|{today - timedelta(days=1):%Y-%m-%d}T00:00:00+00:00"
    vals = [o["value"] for o in opts]
    default = want if want in vals else (vals[0] if vals else None)
    return opts, default


def revision_runs(zone: str, today: date, cmp_value: str | None):
    """(current model, current run, compare model, compare run) or None.

    Current = the run the table shows (ecmwf_pick).  Compare = the run picked
    in "Compare with" (default yesterday's 00z).  Delta = current - compare
    for the same target hour.
    """
    pick = ecmwf_pick(zone)
    if not pick:
        return None
    opts, default = compare_options(zone, today)
    vals = [o["value"] for o in opts]
    v = cmp_value if cmp_value in vals else default
    if not v:
        return None
    rmodel, rrun = v.split("|", 1)
    return pick[0], pick[1], rmodel, rrun


def load_runs_available(zones: list[str]) -> set:
    """(weather_model, weather_run) pairs with a load curve for ALL zones."""
    p = entsoe_path()
    if not p or "load_forecast_model_runs" not in tables(p):
        return set()

    def q():
        qm = ",".join("?" * len(zones))
        with connect(p) as c:
            return {(m, r) for m, r, n in c.execute(
                "SELECT weather_model, weather_run, COUNT(DISTINCT zone) FROM "
                f"load_forecast_model_runs WHERE zone IN ({qm}) "
                "GROUP BY weather_model, weather_run", zones).fetchall()
                if n == len(zones)}

    return cached(("load_runs", tuple(zones)), p, q)


def load_run_series(zones: list[str], wmodel: str, wrun: str, start: date,
                    end: date, idx: pd.DatetimeIndex):
    """Modelled load (zones summed) for one weather run, local hourly."""
    p = entsoe_path()
    lo = (start - timedelta(days=2)).isoformat()
    hi = (end + timedelta(days=2)).isoformat()

    def q():
        qm = ",".join("?" * len(zones))
        with connect(p) as c:
            df = pd.read_sql_query(
                "SELECT zone, timestamp, forecast_mw FROM "
                f"load_forecast_model_runs WHERE zone IN ({qm}) AND "
                "weather_model = ? AND weather_run = ? AND timestamp >= ? "
                "AND timestamp < ?", c, params=[*zones, wmodel, wrun, lo, hi])
        if df.empty:
            return None
        df["t"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        w = df.pivot_table(index="t", columns="zone", values="forecast_mw")
        w = w.reindex(columns=zones)
        return w.sum(axis=1, min_count=len(zones)).where(w.notna().all(axis=1))

    out = cached(("load_run", tuple(zones), wmodel, wrun, lo, hi), p, q)
    return None if out is None else out.reindex(idx)


def model_zones() -> set:
    """Zones the load model has actually produced a forecast for.  It needs
    ERA5 history, which only exists for the countries in ec46_scraper's
    POINTS - so most of ENTSO-E's 54 zones are not in here."""
    p = entsoe_path()
    if not p or "load_forecast_model_mw" not in tables(p):
        return set()

    def q():
        with connect(p) as c:
            return {r[0] for r in c.execute(
                "SELECT DISTINCT zone FROM load_forecast_model_mw")}

    return cached(("model_zones",), p, q)


def load_model_info():
    """(run_time, {zone: holdout MAPE}) of the latest load model fit."""
    p = entsoe_path()
    if not p or "load_forecast_model_mw" not in tables(p):
        return None

    def q():
        with connect(p) as c:
            rt = c.execute("SELECT MAX(run_time) FROM "
                           "load_forecast_model_mw").fetchone()[0]
            mape = {}
            if "load_model_fit_log" in tables(p):
                # Latest row per zone, not the latest fit: running the
                # model for a few zones only must not blank the others.
                mape = dict(c.execute(
                    "SELECT zone, holdout_mape FROM load_model_fit_log WHERE "
                    "rowid IN (SELECT MAX(rowid) FROM load_model_fit_log "
                    "GROUP BY zone)").fetchall())
        return (rt, mape) if rt else None

    return cached(("load_model_info",), p, q)


def hourly_frame(cc: dict, srcs: list[str], start: date, end: date,
                 use_model: bool = True) -> pd.DataFrame:
    """Hourly frame: load, each source, balance, price.  Index = local time.

    Published data always wins.  The modelled series are only ever used to
    fill holes in it, and every hour that was filled that way is flagged in
    the `is_model` column so the table can say so.
    """
    idx = pd.date_range(pd.Timestamp(start, tz=TZ),
                        pd.Timestamp(end + timedelta(days=1), tz=TZ),
                        freq="h", inclusive="left")
    out = pd.DataFrame(index=idx)
    model_used = pd.Series(False, index=idx)

    # Load: published (ENTSO-E, D+1 at most), then the modelled curve
    # wherever published is missing.  Still only summed where every zone
    # of the country has a value, modelled or not.
    w = _zone_wide("load_forecast_mw", cc["load"], start, end, idx)
    if use_model:
        mod_w = _zone_wide("load_forecast_model_mw", cc["load"], start, end,
                           idx)
        if mod_w is not None:
            if w is None:
                w, filled = mod_w, mod_w.notna()
            else:
                filled = w.isna() & mod_w.notna()
                w = w.where(w.notna(), mod_w)
            model_used |= filled.any(axis=1)
    if w is not None:
        out["load"] = w.sum(axis=1, min_count=len(cc["load"])).where(
            w.notna().all(axis=1))
        load_w = w
    else:
        out["load"] = float("nan")
        load_w = None

    # Sources, same two-step: published first, modelled MW in the holes.
    for key in srcs:
        label, table, cols, must = SOURCES[key]
        df = _read(table, cols, [cc["ws"]], start, end)
        if len(df):
            df = df.groupby("t")[cols].mean().reindex(idx)
            val = df[cols].fillna(0).sum(axis=1)
            val = val.where(df[must].notna().all(axis=1))
        else:
            val = pd.Series(float("nan"), index=idx)
        if use_model and key in MODEL_SOURCES:
            m = ecmwf_mw(cc["ws"], start, end, idx)
            if m is not None:
                ms = m[MODEL_SOURCES[key]]
                model_used |= val.isna() & ms.notna()
                val = val.where(val.notna(), ms)
        out[key] = val

    out["balance"] = out["load"]
    for key in srcs:
        out["balance"] = out["balance"] - out[key]
    # Only flag hours that ended up with a balance - a day with nothing in
    # it should not be labelled "model".
    out["is_model"] = model_used & out["balance"].notna()

    # Price (load-weighted average when several zones)
    pr = _read("day_ahead_prices", ["price_eur_mwh"], cc["price"], start, end)
    if len(pr):
        pw = pr.pivot_table(index="t", columns="zone",
                            values="price_eur_mwh").reindex(idx)
        if len(cc["price"]) == 1 or load_w is None:
            out["price"] = pw.mean(axis=1)
        else:
            lw = load_w.reindex(columns=pw.columns)
            ok = pw.notna().all(axis=1) & lw.notna().all(axis=1)
            out["price"] = ((pw * lw).sum(axis=1) / lw.sum(axis=1)).where(ok)
    else:
        out["price"] = float("nan")
    return out


# --------------------------------------------------------------------------
# Gas + carbon for clean spark spread
_MONTH = re.compile(r"^[A-Z][a-z]{2}\d{2}$")


def fuel_by_day(days: list[date]) -> pd.DataFrame:
    """Per delivery day: TTF front month and EUA Dec settlement (last trade
    date before the day, up to 7 days back)."""
    p = ice_path()
    cols = ["ttf", "ttf_strip", "eua", "eua_strip", "trade_date"]
    if not p or not days:
        return pd.DataFrame(index=days, columns=cols)

    def q():
        with connect(p) as c:
            return pd.read_sql_query(
                "SELECT s.product, s.strip, s.trade_date, s.price, m.end_date "
                "FROM ice_settlement s LEFT JOIN ice_markets m "
                "ON m.product = s.product AND m.market_id = s.market_id "
                "WHERE s.product IN ('TTF','EUA') AND s.price IS NOT NULL", c)

    st = cached("ice", p, q)
    rows = []
    for d in days:
        rec = dict.fromkeys(cols)
        for back in range(1, 8):
            td = (d - timedelta(days=back)).isoformat()
            day = st[st["trade_date"] == td]
            if day.empty:
                continue
            # TTF front month: first month contract ending after td's month
            ttf = day[(day["product"] == "TTF")
                      & day["strip"].fillna("").str.match(_MONTH)]
            month_end = (pd.Timestamp(td) + pd.offsets.MonthEnd(0)).date()
            ttf = ttf[pd.to_datetime(ttf["end_date"]).dt.date > month_end]
            eua = day[(day["product"] == "EUA")
                      & (day["strip"] == f"Dec{d.year % 100:02d}")]
            if ttf.empty or eua.empty:
                continue
            f = ttf.sort_values("end_date").iloc[0]
            rec.update(ttf=f["price"], ttf_strip=f["strip"],
                       eua=eua.iloc[0]["price"], eua_strip=f"Dec{d.year % 100:02d}",
                       trade_date=td)
            break
        rows.append(rec)
    return pd.DataFrame(rows, index=days, columns=cols)


# --------------------------------------------------------------------------
# Table
def cid(d: date, kind: str) -> str:
    return f"d{d:%Y%m%d}_{kind}"


def day_label(d: date, today: date, model: bool = False) -> str:
    s = f"{d:%a %d.%m}"
    if d == today:
        s += " (today)"
    if model:
        s += " (model)"
    return s


def model_note(cc: dict, model_days, srcs: list[str]) -> str:
    """One line saying exactly what the modelled days are made of."""
    bits = []
    info = load_model_info()
    if info:
        rt, mape = info
        zm = [mape[z] for z in cc["load"]
              if mape.get(z) is not None]
        if len(zm) > 1:
            acc = f", holdout MAPE {min(zm):.1f}-{max(zm):.1f}%"
        elif zm:
            acc = f", holdout MAPE {zm[0]:.1f}%"
        else:
            acc = ""
        bits.append(f"load from model_load_forecast.py (run {fmt_run(rt)}"
                    f"{acc})")
    if any(k in MODEL_SOURCES for k in srcs):
        pick = ecmwf_pick(cc["ws"])
        if pick:
            bits.append(f"wind/solar from ECMWF {pick[0].upper()} "
                        f"(run {fmt_run(pick[1])})")
    days = ", ".join(f"{d:%d.%m}" for d in sorted(model_days))
    head = (f"Days marked (model), in italics, are MODELLED - not published "
            f"by any TSO: {days}.")
    if not bits:
        return head
    tail = "; ".join(bits)
    return f"{head} {tail[:1].upper()}{tail[1:]}."


def per_hour(series: pd.Series) -> pd.DataFrame:
    """Series (local hourly) -> DataFrame rows=hour 0-23, cols=date."""
    s = series
    df = pd.DataFrame({"v": s.values,
                       "date": s.index.date, "hour": s.index.hour})
    return df.pivot_table(index="hour", columns="date", values="v",
                          aggfunc="mean", dropna=False).reindex(range(24))


def build_table(code: str, n: int, srcs: list[str], mode: str,
                eff: float, today: date | None = None,
                use_model: bool = True, cmp_value: str | None = None):
    cc = countries()[code]
    today = today or datetime.now(tz=pd.Timestamp.now(tz=TZ).tz).date()
    past = [today - timedelta(days=i) for i in range(n - 1, -1, -1)]
    fwd = [today + timedelta(days=i) for i in range(1, n + 1)]
    hf = hourly_frame(cc, srcs, past[0], fwd[-1], use_model)

    bal = per_hour(hf["balance"])
    price = per_hour(hf["price"])
    ismod = per_hour(hf["is_model"].astype(float))
    model_days = {d for d in past + fwd
                  if d in ismod.columns and (ismod[d].fillna(0) > 0).any()}
    notes = []

    if mode == "css":
        fuel = fuel_by_day(past)
        cf = 0.184 / eff        # 0.368 at 50 % (was 0.202 / 0.404)
        css = pd.DataFrame(index=range(24))
        for d in past:
            g, e = fuel.loc[d, "ttf"], fuel.loc[d, "eua"]
            if pd.isna(g) or pd.isna(e) or d not in price.columns:
                css[d] = float("nan")
            else:
                css[d] = price[d] - g / eff - e * cf
        first = past if not fuel["ttf"].isna().all() else []
        if fuel["ttf"].isna().any():
            miss = [f"{d:%d.%m}" for d in past if pd.isna(fuel.loc[d, "ttf"])]
            notes.append("No ICE TTF/EUA settlement for: " + ", ".join(miss))
        if first:
            last = fuel.dropna(subset=["ttf"]).iloc[-1]
            notes.append(
                f"CSS = price - TTF / {eff:.0%} - EUA x {cf:.3f}. "
                f"Latest inputs (traded {last['trade_date']}): TTF "
                f"{last['ttf_strip']} {last['ttf']:.2f} EUR/MWh, EUA "
                f"{last['eua_strip']} {last['eua']:.2f} EUR/t.")
        first_col = css
        first_name = "CSS"
    else:
        first_col = price
        first_name = "Price"

    columns = [{"name": ["", "", "Hour"], "id": "hour", "type": "text"}]
    money = Format(precision=2, scheme=Scheme.fixed)
    mw = Format(precision=0, scheme=Scheme.fixed, group=Group.yes)
    for d in past:
        lab = day_label(d, today, d in model_days)
        columns += [
            {"name": ["Past (today included)", lab, "Balance"],
             "id": cid(d, "b"), "type": "numeric", "format": mw},
            {"name": ["Past (today included)", lab, first_name],
             "id": cid(d, "p"), "type": "numeric", "format": money},
        ]
    for d in fwd:
        lab = day_label(d, today, d in model_days)
        columns += [
            {"name": ["Forward", lab, "Balance"],
             "id": cid(d, "b"), "type": "numeric", "format": mw},
            {"name": ["Forward", lab, "Delta"],
             "id": cid(d, "d"), "type": "numeric", "format": mw},
        ]

    def col(df, d):
        return df[d] if d in df.columns else pd.Series(float("nan"),
                                                       index=range(24))

    # Delta = forecast revision: current run minus the "Compare with" run
    # (default yesterday's 00z), same target hour.  Wind/solar from
    # mw_hourly (every run kept); load from load_forecast_model_runs (the
    # load model applied to every weather run).
    rev_runs = revision_runs(cc["ws"], today, cmp_value)
    rev = {}
    load_note = ""
    if rev_runs:
        cmodel, ccur, rmodel, rref = rev_runs
        idx = hf.index
        now = ecmwf_run_mw(cc["ws"], cmodel, ccur, past[0], fwd[-1], idx)
        old = ecmwf_run_mw(cc["ws"], rmodel, rref, past[0], fwd[-1], idx)
        if now is not None and old is not None:
            for key, mcol in MODEL_SOURCES.items():
                rev[key] = now[mcol] - old[mcol]
        # Load: current = newest load curve of the current model at or
        # before the current weather run (the load model may lag a run).
        avail = load_runs_available(cc["load"])
        lcur = max((r for m, r in avail if m == cmodel and r <= ccur),
                   default=None)
        if "load_forecast_model_runs" not in tables(entsoe_path()):
            load_note = (" Load revision missing: no load_forecast_model_runs "
                         'yet - run "Update load forecast.bat" once.')
        elif not avail:
            load_note = (" Load revision missing: the load model has no curve "
                         f"for {', '.join(cc['load'])} (no weather/ERA5 for "
                         "this country, or not run for it yet).")
        elif lcur is None or (rmodel, rref) not in avail:
            load_note = (" Load revision missing for this pair of runs (load "
                         "model not run on them).")
        else:
            ln = load_run_series(cc["load"], cmodel, lcur, past[0], fwd[-1],
                                 idx)
            lo_ = load_run_series(cc["load"], rmodel, rref, past[0], fwd[-1],
                                  idx)
            if ln is not None and lo_ is not None:
                rev["load"] = ln - lo_
                load_note = (f" Load: modelled load on {cmodel.upper()} "
                             f"{pd.Timestamp(lcur):%d.%m %H}z vs "
                             f"{rmodel.upper()} {pd.Timestamp(rref):%d.%m %H}z "
                             "(temperature-driven change).")
    if rev:
        need = [k for k in srcs if k in rev] + (["load"] if "load" in rev
                                                 else [])
        bal_rev = rev["load"].copy() if "load" in rev else pd.Series(
            0.0, index=hf.index)
        for key in srcs:
            if key in rev:
                bal_rev = bal_rev - rev[key]
        if need:
            bal_rev = bal_rev.where(pd.concat(
                [rev[k].notna() for k in need], axis=1).all(axis=1))
        delta = per_hour(bal_rev)
        notes.append(
            f"Delta = current forecast minus the compare run, same hour: "
            f"weather ECMWF {cmodel.upper()} {pd.Timestamp(ccur):%d.%m %H}z vs "
            f"{rmodel.upper()} {pd.Timestamp(rref):%d.%m %H}z." + load_note)
    else:
        delta = pd.DataFrame(index=range(24))
        notes.append("Delta empty: no compare run stored for "
                     f"{NAMES.get(code, code)}.")

    table = pd.DataFrame(index=range(24))
    for d in past:
        table[cid(d, "p")] = col(first_col, d)
        table[cid(d, "b")] = col(bal, d)
    for d in fwd:
        table[cid(d, "b")] = col(bal, d)
        table[cid(d, "d")] = col(delta, d)

    rows = table.round(2).astype(object).where(table.notna(), None)
    rows.insert(0, "hour", [f"{h:02d}" for h in range(24)])
    records = rows.to_dict("records")
    # Base / Peak / Off-peak rows (means of the hours; delta = mean of deltas)
    for name, hrs in SUMMARY_ROWS:
        sub = table.loc[list(hrs)]
        r = {"hour": name}
        for c in table.columns:
            v = sub[c].mean() if sub[c].notna().all() else None
            r[c] = None if v is None or pd.isna(v) else round(float(v), 2)
        records.append(r)

    # Daily mean temperature row: value in the Balance column (text, so the
    # MW number format and the heat map leave it alone), day-on-day change
    # in the Delta column.
    tmp = temp_daily(cc["ws"], [past[0] - timedelta(days=1)] + past + fwd)
    r = {"hour": TEMP_ROW}
    if tmp:
        daily, tsrc = tmp
        tv = {d: (None if pd.isna(daily.get(d, float("nan")))
                  else float(daily[d])) for d in [past[0] - timedelta(days=1)]
              + past + fwd}
        for d in past + fwd:
            r[cid(d, "b")] = None if tv[d] is None else f"{tv[d]:.1f}"
        if rev_runs:
            tn = temp_run_daily(cc["ws"], rev_runs[0], rev_runs[1])
            to = temp_run_daily(cc["ws"], rev_runs[2], rev_runs[3])
            if tn is not None and to is not None:
                for d in fwd:
                    a, b = tn.get(d), to.get(d)
                    r[cid(d, "d")] = (None if a is None or b is None
                                      or pd.isna(a) or pd.isna(b)
                                      else f"{a - b:+.1f}")
        notes.append(f"Temperature = daily mean, {NAMES.get(code, code)} "
                     f"country average: {tsrc}.")
    else:
        notes.append(f"No temperature for {NAMES.get(code, code)} - "
                     "ecmwf_data.db covers AT, BE, CH, DE, DK, FI, FR, GB, "
                     "NL, NO, PL, SE.")
    temp_record = r   # appended last: temperature row always at the bottom

    # Daily average Load / Wind / Solar (MW), always all sources whatever is
    # ticked.  Text in the Balance column (heat map and MW format ignore
    # it), day-on-day change in the forward Delta column.
    hf_all = (hf if set(srcs) == set(SOURCES)
              else hourly_frame(cc, list(SOURCES), past[0] - timedelta(days=1),
                                fwd[-1], use_model))
    avg_days = [past[0] - timedelta(days=1)] + past + fwd
    for key, label in [(k, f"{v[0]} MW") for k, v in SOURCES.items()] + [
            ("load", "Load MW")]:
        ser = hf_all[key]
        g = ser.groupby(ser.index.date)
        daily = g.mean().where(g.count() >= 20)
        dv = {d: (None if pd.isna(daily.get(d, float("nan")))
                  else float(daily[d])) for d in avg_days}
        r = {"hour": label}
        for d in past + fwd:
            r[cid(d, "b")] = None if dv[d] is None else f"{dv[d]:,.0f}"
        if key in rev:
            rs = rev[key]
            g = rs.groupby(rs.index.date)
            rd = g.mean().where(g.count() >= 20)
            for d in fwd:
                v = rd.get(d)
                r[cid(d, "d")] = (None if v is None or pd.isna(v)
                                  else f"{v:+,.0f}")
        records.append(r)
    records.append(temp_record)

    # Notes on data coverage
    have = [d for d in fwd if d in bal.columns and bal[d].notna().any()]
    if not have:
        notes.append("No forecast for the forward days. ENTSO-E publishes "
                     "load, wind and solar for tomorrow only (around "
                     "midday); the modelled curves are what fill the rest, "
                     "so if this is empty neither of them is there either.")
    elif len(have) < len(fwd):
        notes.append(f"ENTSO-E forecast available up to {have[-1]:%a %d.%m}; "
                     "later days are empty.")
    if model_days:
        notes.append(model_note(cc, model_days, srcs))
        thin = {d: int(bal[d].notna().sum()) for d in sorted(model_days)
                if d in bal.columns and bal[d].notna().sum() < 24}
        if thin:
            notes.append(
                "Modelled days are not complete: "
                + ", ".join(f"{d:%d.%m} ({h}/24 h)" for d, h in thin.items())
                + '. Re-run "Update load forecast.bat" - the current '
                  "model_load_forecast.py interpolates the 3-hourly ECMWF "
                  "temperature to every hour; an older run stored every "
                  "third hour only.")
    elif use_model and "load_forecast_model_mw" not in tables(entsoe_path()):
        notes.append('No modelled load in the database yet - run "Update '
                     'load forecast.bat" to build load_forecast_model_mw, '
                     "and the forward days fill in.")
    elif use_model and not (set(cc["load"]) & model_zones()):
        covered = sorted({z.split("_")[0] for z in model_zones()})
        notes.append(f"No modelled load for {NAMES.get(code, code)}. The "
                     "model needs ERA5 temperature history, which is set up "
                     "for " + ", ".join(covered) + " only.")
    if len(cc["load"]) > 1:
        notes.append(f"{NAMES.get(code, code)}: load = sum of "
                     f"{', '.join(cc['load'])}; price = load-weighted average "
                     "of those zones. Wind/solar are country totals.")
    if cc["load"] == ["DE_LU"]:
        notes.append("Load is DE_LU (incl. Luxembourg); wind/solar are DE.")
    for key in srcs:
        if hf[key].isna().all():
            notes.append(f"No {SOURCES[key][0].lower()} data for "
                         f"{NAMES.get(code, code)} - balance is empty.")
    if hf["load"].isna().all():
        notes.append("No load data in this window.")
    return columns, records, notes, past, fwd, model_days


# Heat map: low = red, middle = pale yellow, high = green.
# HEAT_STEPS colour bins, interpolated between the three anchor colours.
HEAT_STEPS = 21
_ANCHORS = [(0.0, (0xd7, 0x30, 0x27)), (0.5, (0xfd, 0xf3, 0xc4)),
            (1.0, (0x1a, 0x98, 0x50))]


def _heat_colours(n: int) -> list[str]:
    out = []
    for i in range(n):
        t = i / (n - 1)
        (t0, c0), (t1, c1) = ((_ANCHORS[0], _ANCHORS[1]) if t <= 0.5
                              else (_ANCHORS[1], _ANCHORS[2]))
        f = (t - t0) / (t1 - t0)
        rgb = [round(a + (b - a) * f) for a, b in zip(c0, c1)]
        out.append("#%02x%02x%02x" % tuple(rgb))
    return out


HEAT = _heat_colours(HEAT_STEPS)


def heat_rules(cols: list[str], records: list[dict]) -> list[dict]:
    """One colour scale for all given columns, range from the 24 hour rows."""
    vals = [r[c] for r in records[:24] for c in cols if r.get(c) is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / len(HEAT)
    rules = []
    for c in cols:
        for i, colour in enumerate(HEAT):
            a = lo + i * step
            q = f"{{{c}}} is num && " + (f"{{{c}}} >= {a:.6f}" if i
                                          else f"{{{c}}} < {lo + step:.6f}")
            if 0 < i < len(HEAT) - 1:
                q += f" && {{{c}}} < {a + step:.6f}"
            rules.append({"if": {"column_id": c, "filter_query": q},
                          "backgroundColor": colour, "color": "#1d1d1d"})
    return rules


def styles(past, fwd, today, records, model_days=()):
    st = [
        {"if": {"row_index": [24, 25, 26]}, "fontWeight": "600"},
        {"if": {"row_index": 24}, "borderTop": "2px solid #888"},
        {"if": {"row_index": [27, 28, 29, 30]},
         "backgroundColor": "#eef3f8", "fontStyle": "normal"},
        {"if": {"row_index": 27}, "borderTop": "2px solid #888"},
    ]
    # Modelled days in italics.  Set before the heat rules on purpose: those
    # only touch backgroundColor and color, so both survive.
    for d in model_days:
        for kind in ("b", "d", "p"):
            st.append({"if": {"column_id": cid(d, kind)},
                       "fontStyle": "italic"})
    for d in fwd:
        k = cid(d, "d")
        st.append({"if": {"column_id": k}, "backgroundColor": "#f6f6f6"})
        st.append({"if": {"column_id": k, "filter_query": f"{{{k}}} > 0"},
                   "color": "#1e7f4f"})
        st.append({"if": {"column_id": k, "filter_query": f"{{{k}}} < 0"},
                   "color": "#c0392b"})
    # Heat map on balance only (past + forward on one scale).
    st += heat_rules([cid(d, "b") for d in past + fwd], records)
    # vertical line before each day, thick one around today
    for d in past + fwd:
        st.append({"if": {"column_id": cid(d, "b")},
                   "borderLeft": "2px solid #999"})
    st.append({"if": {"column_id": cid(today, "b")},
               "borderLeft": "3px solid #222"})
    st.append({"if": {"column_id": cid(today, "p")},
               "borderRight": "3px solid #222"})
    return st


# --------------------------------------------------------------------------
# Layout
def layout():
    opts = [{"label": v["label"], "value": k} for k, v in countries().items()]
    default = DEFAULT_COUNTRY if any(o["value"] == DEFAULT_COUNTRY
                                     for o in opts) else (
        opts[0]["value"] if opts else None)
    label_style = {"fontWeight": "600", "marginRight": "8px"}
    return html.Div(style={"fontFamily": FONT, "padding": "12px 16px",
                           "background": "#fff", "color": "#222"}, children=[
        html.H2("Balance sheet", style={"margin": "0 0 4px 0"}),
        html.Div("Balance = Load - ticked sources. Hourly, local time "
                 "(CET/CEST). Published ENTSO-E day-ahead first; past that, "
                 "the modelled load curve and ECMWF wind/solar - in italics "
                 "and marked (model).",
                 style={"color": "#666", "marginBottom": "12px"}),
        html.Div(style={"display": "flex", "alignItems": "center",
                        "gap": "12px", "maxWidth": "720px",
                        "marginBottom": "12px"}, children=[
            html.Span("Days back / forward", style=label_style),
            html.Div(dcc.Slider(id="days", min=3, max=21, step=1, value=7,
                                marks={i: str(i) for i in range(3, 22, 3)},
                                updatemode="mouseup"),
                     style={"flex": "1"}),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "24px",
                        "alignItems": "center", "marginBottom": "10px"},
                 children=[
            html.Div([html.Span("Subtract", style=label_style),
                      dcc.Checklist(
                          id="srcs",
                          options=[{"label": v[0], "value": k}
                                   for k, v in SOURCES.items()],
                          value=DEFAULT_SOURCES, inline=True,
                          inputStyle={"marginRight": "4px",
                                      "marginLeft": "10px"})],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([html.Span("Country", style=label_style),
                      dcc.Dropdown(id="country", options=opts, value=default,
                                   clearable=False,
                                   style={"width": "280px"})],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([dcc.Checklist(
                id="model",
                options=[{"label": " Model forward",
                          "value": "on"}],
                value=["on"], inputStyle={"marginRight": "4px"})],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([html.Span("Delta vs", style=label_style),
                      dcc.Dropdown(id="cmp", options=[], value=None,
                                   clearable=False,
                                   style={"width": "190px"})],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([html.Span("Past column", style=label_style),
                      dcc.RadioItems(id="mode",
                                     options=[{"label": "Price", "value": "price"},
                                              {"label": "Clean spark spread",
                                               "value": "css"}],
                                     value="price", inline=True,
                                     inputStyle={"marginRight": "4px",
                                                 "marginLeft": "10px"})],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([html.Span("Gas plant efficiency %", style=label_style),
                      dcc.Input(id="eff", type="number", value=50, min=20,
                                max=70, step=1, debounce=True,
                                style={"width": "64px"})],
                     id="eff-box",
                     style={"display": "none", "alignItems": "center"}),
        ]),
        html.Ul(id="notes", style={"color": "#8a5a00", "margin": "4px 0 10px",
                                   "fontSize": "13px"}),
        dcc.Loading(dash_table.DataTable(
            id="table",
            merge_duplicate_headers=True,
            # Autosized columns. Hour column kept in view with CSS sticky
            # (DataTable's fixed_columns forces equal fixed widths).
            css=[{"selector": "td.dash-cell.column-0, th.dash-header.column-0",
                  "rule": "position: sticky; left: 0; z-index: 2; "
                          "background-color: #f4f4f4;"}],
            export_format="csv",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": FONT, "fontSize": "13px",
                        "padding": "3px 6px", "width": "auto",
                        "textAlign": "center", "fontWeight": "700",
                        "whiteSpace": "nowrap",
                        "border": "1px solid #e3e3e3"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f4f4",
                          "textAlign": "center"},
            style_cell_conditional=[{"if": {"column_id": "hour"},
                                     "textAlign": "center",
                                     "fontWeight": "600"}],
        )),
        html.Div("Delta = forecast revision: current forecast minus the run picked in Delta vs (default yesterday 00z), same date and hour - load, wind and solar (green = up, red = down). "
                 "Balance heat map: green = high, red = low, one scale for all days. "
                 "Base = mean of 24 h, Peak = 08-20 h, Off-peak = the other 12 h. "
                 "Temp, Load, Wind, Solar = daily averages, always all sources (Delta column: revision vs the compare run). "
                 "Italic and (model) = modelled, not published by a TSO.",
                 style={"color": "#666", "fontSize": "12px",
                        "marginTop": "8px"}),
    ])


app = Dash(__name__, title="Balance sheet")
app.layout = layout


@app.callback(Output("eff-box", "style"), Input("mode", "value"))
def show_eff(mode):
    return {"display": "flex" if mode == "css" else "none",
            "alignItems": "center"}


@app.callback(Output("cmp", "options"), Output("cmp", "value"),
              Input("country", "value"))
def cmp_options(code):
    """Runs to compare with, for the chosen country; default = yesterday 00z."""
    cc = countries().get(code)
    if not cc:
        return [], None
    return compare_options(cc["ws"], pd.Timestamp.now(tz=TZ).date())


@app.callback(
    Output("table", "columns"), Output("table", "data"),
    Output("table", "style_data_conditional"), Output("notes", "children"),
    Input("country", "value"), Input("days", "value"),
    Input("srcs", "value"), Input("mode", "value"), Input("eff", "value"),
    Input("model", "value"), Input("cmp", "value"))
def update(code, n, srcs, mode, eff, model, cmp_value=None):
    if not entsoe_path():
        return [], [], [], [html.Li("entsoe_data.db not found.")]
    if not code:
        return [], [], [], [html.Li("No country with load data.")]
    eff = (eff or 50) / 100
    srcs = [s for s in SOURCES if s in (srcs or [])]
    today = pd.Timestamp.now(tz=TZ).date()
    cols, data, notes, past, fwd, model_days = build_table(
        code, int(n), srcs, mode, eff, today, bool(model), cmp_value)
    return (cols, data, styles(past, fwd, today, data, model_days),
            [html.Li(t) for t in notes])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", help="entsoe_data.db (default: next to script)")
    ap.add_argument("--ice-db", help="ice_market_data.db "
                    "(default: ..\\ice_scraper\\)")
    ap.add_argument("--ecmwf-db", help="ecmwf_data.db "
                    "(default: ..\\ecmwf_scraper\\)")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    ARGS.db, ARGS.ice_db, ARGS.ecmwf_db = a.db, a.ice_db, a.ecmwf_db
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        url = f"http://127.0.0.1:{a.port}/"
        print(f"Balance sheet on {url}\n  entsoe: {entsoe_path()}\n"
              f"  ice   : {ice_path()}\n  ecmwf : {ecmwf_path()}")
        if a.open:
            webbrowser.open(url)
    app.run(host=a.host, port=a.port, debug=a.debug, use_reloader=True,
            reloader_type="stat")


if __name__ == "__main__":
    main()
