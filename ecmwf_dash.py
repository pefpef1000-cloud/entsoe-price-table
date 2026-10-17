"""
ECMWF forecast dashboard (Dash) - reads ecmwf_data.db

Tables used
  ecmwf_forecast   IFS 15-day runs (ecmwf_scraper.py)        - all runs kept
  ec46_daily       EC46 46-day, daily (ec46_scraper.py)        - all fetches kept
  ecmwf_mw_hourly  IFS wind/solar as hourly MW (ecmwf_mw.py)
  ec46_mw_daily    EC46 wind/solar as daily MW (ecmwf_mw.py)

Controls
  Country, Run, Compare with (previous run / 1 week earlier / any stored run),
  Wind & solar as weather or MW.

Tabs
  Forecast     - temperature, wind, solar for the run (thick) vs the compare run (dashed)
  Run changes  - last 6 runs up to the selected run (+ the compare run)
  Countries    - daily averages per country, values or change vs the compare run
  EC46         - 46-day outlook, values or change vs the compare EC46 fetch

All times are Danish local time (Europe/Copenhagen).

Run
  py -3.14 ecmwf_dash.py --open
  py -3.14 ecmwf_dash.py --host 0.0.0.0 --port 8058
"""
import argparse
import datetime as dt
import os
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate

HERE = Path(__file__).resolve().parent
DB_NAME = "ecmwf_data.db"
TZ = "Europe/Copenhagen"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"
INK, MUTED, GRID, BG, CARD = "#1f2328", "#59636e", "#e6e8eb", "#f6f7f9", "#ffffff"

# Order = display order everywhere (Peter 2026-10-16: solar first, temperature always last).
VARS = {
    "solar_wm2": {"label": "Solar radiation", "unit": "W/m²", "fmt": ".0f"},
    "wind100_ms": {"label": "Wind 100 m", "unit": "m/s", "fmt": ".1f"},
    "temp_c": {"label": "Temperature", "unit": "°C", "fmt": ".1f"},
}
MW_META = {
    "wind100_ms": {"label": "Wind power", "unit": "MW", "fmt": ",.0f"},
    "solar_wm2": {"label": "Solar power", "unit": "MW", "fmt": ",.0f"},
}
ZONE_NAMES = {
    "DE": "Germany", "DK": "Denmark", "FR": "France", "NL": "Netherlands",
    "BE": "Belgium", "PL": "Poland", "AT": "Austria", "CH": "Switzerland",
    "NO": "Norway", "SE": "Sweden", "FI": "Finland", "GB": "Great Britain",
}
EC_COLS = {"temp_c": ("temp_c", "temp_p10", "temp_p90"),
           "wind100_ms": ("wind100_ms", "wind100_p10", "wind100_p90"),
           "solar_wm2": ("solar_wm2", "solar_p10", "solar_p90")}
CMP_PREV, CMP_WEEK, CMP_XMODEL = "prev", "week", "xmodel"
CMP_P00, CMP_P12 = "prev00", "prev12"  # last earlier 00z / 12z run
MODELS = {"ifs": "ECMWF", "gfs": "GFS"}
ENS_MODELS = {"ecmwf": "ECMWF ENS", "gefs": "GEFS"}
ENS_DET = {"ecmwf": "ifs", "gefs": "gfs"}


# A run is identified by a key "model|ISO time", e.g. "gfs|2026-10-14T12:00:00+00:00".
def rk(model, ts):
    return f"{model}|{pd.Timestamp(ts).isoformat()}"


def rk_model(key):
    return key.split("|", 1)[0]


def rk_ts(key):
    return pd.Timestamp(key.split("|", 1)[1])

DB_PATH = None


def meta(var, units):
    return MW_META[var] if units == "mw" and var in MW_META else VARS[var]


# ---------------------------------------------------------------- data
def find_db(explicit=None):
    if explicit:
        return Path(explicit)
    for d in (HERE, HERE / "Claude outputs", HERE.parent, Path.cwd()):
        p = d / DB_NAME
        if p.exists():
            return p
    return None


_cache = {"key": None}


def _file_key(p: Path):
    parts = []
    for f in (p, Path(str(p) + "-wal")):
        try:
            s = f.stat()
            parts.append((s.st_mtime, s.st_size))
        except OSError:
            parts.append(None)
    return tuple(parts)


def _connect():
    return sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)


def load():
    """Load the small tables (weather runs, EC46, calibration). Cached on db file time+size.
    The big hourly MW table is read per run on demand (get_mw)."""
    global DB_PATH
    if DB_PATH is None:
        DB_PATH = find_db()
    if DB_PATH is None or not Path(DB_PATH).exists():
        _cache.update(key=None, df=pd.DataFrame(), ec46=pd.DataFrame(), ec46_mw=pd.DataFrame(),
                      calib=pd.DataFrame(), mw_runs=set(), mw={})
        return _cache["df"]
    key = _file_key(Path(DB_PATH))
    if key == _cache["key"]:
        return _cache["df"]
    conn = _connect()
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def q(sql, table):
            return pd.read_sql_query(sql, conn) if table in tables else pd.DataFrame()

        df = q("SELECT model, run_time, zone, valid_time, step_h, temp_c, wind100_ms, solar_wm2 "
               "FROM ecmwf_forecast", "ecmwf_forecast")
        ec = q("SELECT * FROM ec46_daily", "ec46_daily")
        ecmw = q("SELECT * FROM ec46_mw_daily", "ec46_mw_daily")
        calib = q("SELECT * FROM ecmwf_mw_calibration", "ecmwf_mw_calibration")
        normals_doy = q("SELECT * FROM normals_doy", "normals_doy")
        normals_month = q("SELECT * FROM normals_month", "normals_month")
        seasonal = q("SELECT * FROM seasonal_monthly", "seasonal_monthly")
        nfit = q("SELECT * FROM normals_mw_fit", "normals_mw_fit")
        calib_hist = q("SELECT * FROM mw_calibration_history", "mw_calibration_history")
        if "mw_hourly" in tables:
            mw_table = "mw_hourly"
            mw_runs = q("SELECT DISTINCT model, run_time FROM mw_hourly", "mw_hourly")
        else:  # before ecmwf_mw.py v4: IFS only, no model column
            mw_table = "ecmwf_mw_hourly"
            mw_runs = q("SELECT DISTINCT 'ifs' AS model, run_time FROM ecmwf_mw_hourly", "ecmwf_mw_hourly")
    finally:
        conn.close()
    if not df.empty:
        df = df[df.model.isin(MODELS)].copy()
        df["run_time"] = [rk(m, t) for m, t in zip(df["model"], pd.to_datetime(df["run_time"], utc=True))]
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_convert(TZ)
        df["day"] = df["valid_time"].dt.date
    ec46_dups = []
    if not ec.empty:
        ec["day"] = pd.to_datetime(ec["day"])
        ec46_dups = duplicate_fetches(ec)
        ec = ec[~ec.fetched_on.isin(ec46_dups)]
    if not ecmw.empty:
        ecmw = ecmw.rename(columns={
            "wind_mw": "wind100_ms", "wind_p10_mw": "wind100_p10", "wind_p90_mw": "wind100_p90",
            "solar_mw": "solar_wm2", "solar_p10_mw": "solar_p10", "solar_p90_mw": "solar_p90"})
        ecmw["day"] = pd.to_datetime(ecmw["day"])
        ecmw = ecmw[~ecmw.fetched_on.isin(ec46_dups)]
        if not ec.empty:
            ecmw = ecmw.merge(ec[["fetched_on", "zone", "day", "members"]],
                              on=["fetched_on", "zone", "day"], how="left")
    mw_set = {rk(m, t) for m, t in zip(mw_runs["model"], pd.to_datetime(mw_runs["run_time"], utc=True))} \
        if not mw_runs.empty else set()
    if not calib.empty and "model" not in calib:
        calib["model"] = "ifs"
    if not seasonal.empty:
        # SEAS5 radiation anomalies from the API grow with lead time (look accumulated): drop the
        # implausible ones (|anomaly| > 50 % of the monthly mean) and the solar MW built on them
        bad = (seasonal.model == "seas5") & (
            seasonal.solar_anom.abs() > 0.5 * seasonal.solar_wm2.abs().clip(lower=1))
        seasonal.loc[bad, ["solar_anom", "solar_mw", "solar_mw_anom"]] = float("nan")
        seasonal["solar_bad"] = bad
    _cache.update(key=key, df=df, ec46=ec, ec46_mw=ecmw, calib=calib, mw_runs=mw_set, mw={}, mw_table=mw_table,
                  normals_doy=normals_doy, normals_month=normals_month, seasonal=seasonal, nfit=nfit,
                  calib_hist=calib_hist, acc={})
    return df


def duplicate_fetches(ec) -> list:
    """EC46 fetches that are the same run as the fetch before (fetched twice before Open-Meteo updated).
    The first fetch is kept: it is when that EC46 run was first seen."""
    dups, prev = [], None
    for f in sorted(ec.fetched_on.unique()):
        if prev is not None:
            a = ec[ec.fetched_on == f][["zone", "day", "temp_c", "wind100_ms", "solar_wm2"]]
            b = ec[ec.fetched_on == prev][["zone", "day", "temp_c", "wind100_ms", "solar_wm2"]]
            j = a.merge(b, on=["zone", "day"], suffixes=("", "_p"))
            if len(j) >= 0.5 * len(a) and all(((j[c] - j[c + "_p"]).abs().fillna(0) < 0.01).all()
                                               for c in ("temp_c", "wind100_ms", "solar_wm2")):
                dups.append(f)
                continue
        prev = f
    return dups


def get_mw(run_list) -> pd.DataFrame:
    """Hourly MW rows for the given runs, in the weather column layout. Cached per run."""
    load()
    have = _cache.get("mw_runs", set())
    want = [r for r in run_list if r is not None and r in have]
    if not want and not run_list:
        return pd.DataFrame(columns=["run_time", "zone", "valid_time", "temp_c", "wind100_ms", "solar_wm2", "day"])
    missing = [r for r in want if r not in _cache["mw"]]
    if missing:
        table = _cache.get("mw_table", "mw_hourly")
        conn = _connect()
        try:
            parts = []
            for r in missing:
                cond = "model = ? AND run_time = ?" if table == "mw_hourly" else "run_time = ?"
                params = [rk_model(r), rk_ts(r).isoformat()] if table == "mw_hourly" else [rk_ts(r).isoformat()]
                parts.append(pd.read_sql_query(f"SELECT * FROM {table} WHERE {cond}", conn, params=params)
                             .assign(rkey=r))
            raw = pd.concat(parts, ignore_index=True)
        finally:
            conn.close()
        for r in missing:
            g = raw[raw.rkey == r]
            vt = pd.to_datetime(g["time_utc"], utc=True).dt.tz_convert(TZ)
            _cache["mw"][r] = pd.DataFrame({
                "run_time": r, "zone": g["zone"].values, "valid_time": vt.values,
                "temp_c": float("nan"), "wind100_ms": g["wind_mw"].values, "solar_wm2": g["solar_mw"].values,
            }).assign(valid_time=lambda d: pd.to_datetime(d["valid_time"], utc=True).dt.tz_convert(TZ),
                      day=lambda d: d["valid_time"].dt.date)
    # runs downloaded after the last ecmwf_mw.py run: convert now with the saved MW model
    for r in run_list:
        if r is not None and r not in have and r not in _cache["mw"]:
            _cache["mw"][r] = mw_on_the_fly(r)
    frames = [_cache["mw"][r] for r in run_list if r is not None and r in _cache["mw"]]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["run_time", "zone", "valid_time", "temp_c", "wind100_ms", "solar_wm2", "day"])


def mw_on_the_fly(r) -> pd.DataFrame:
    """MW for a run that ecmwf_mw.py hasn't converted yet, using its saved fit (same maths)."""
    empty = pd.DataFrame(columns=["run_time", "zone", "valid_time", "temp_c", "wind100_ms", "solar_wm2", "day"])
    df, calib = _cache.get("df"), _cache.get("calib")
    if df is None or df.empty or calib is None or calib.empty:
        return empty
    try:
        from ecmwf_mw import hourly_weather, wind_cf
    except BaseException:  # never let an import problem take a request down
        return empty
    w = df[df.run_time == r]
    if w.empty:
        return empty
    c = calib[calib.model == rk_model(r)]
    out = []
    for zone, g in w.groupby("zone"):
        cz = c[c.zone == zone].set_index("tech")
        if not {"wind", "solar"} & set(cz.index):
            continue
        g2 = g.assign(run_time=rk_ts(r))[["run_time", "step_h", "wind100_ms", "solar_wm2"]]
        h = hourly_weather(g2, zone)
        wind = cz.loc["wind"] if "wind" in cz.index else None
        sol = cz.loc["solar"] if "solar" in cz.index else None
        out.append(pd.DataFrame({
            "run_time": r, "zone": zone,
            "valid_time": pd.to_datetime(h["time_utc"], utc=True).dt.tz_convert(TZ),
            "temp_c": float("nan"),
            "wind100_ms": (wind.scale_mw * wind_cf(h["wind_ms"].values, wind.v50, wind.width)).round(1)
            if wind is not None else float("nan"),
            "solar_wm2": (sol.scale_mw * h["ghi_wm2"].values / 1000).round(1) if sol is not None else float("nan"),
        }))
    if not out:
        return empty
    res = pd.concat(out, ignore_index=True)
    res["day"] = res["valid_time"].dt.date
    return res


def data(units, var, run_list) -> pd.DataFrame:
    """Rows for the given run keys: weather, or MW (same column names) for wind/solar."""
    df = load()
    run_list = [r for r in run_list if r is not None]
    if units == "mw" and var != "temp_c":
        return get_mw(run_list)
    return df[df.run_time.isin(run_list)] if not df.empty else df


def load_ec46(units, var) -> pd.DataFrame:
    load()
    d = _cache.get("ec46_mw" if (units == "mw" and var != "temp_c") else "ec46")
    return d if d is not None else pd.DataFrame()


def mw_note(zone, model="ifs") -> str:
    calib = _cache.get("calib")
    name = MODELS.get(model, model)
    if calib is None or calib.empty:
        return "No MW conversion yet - run ecmwf_mw.py."
    c = calib[(calib.zone == zone) & (calib.model == model) & calib.tech.isin(["wind", "solar"])]
    if c.empty:
        if model != "ifs" and calib[calib.model == model].empty:
            return f"No {name} MW yet - run ecmwf_mw.py (version that converts {name})."
        return f"No {name} MW for {zone}: no ENTSO-E wind/solar data to calibrate against."
    parts = [f"{r.tech} R² {r.r2:.2f} over {int(r.hours)} h" for r in c.itertuples() if pd.notna(r.hours)]
    return (f"MW = {name} weather converted with a model fitted to ENTSO-E day-ahead forecasts "
            f"(last 60 days; {', '.join(parts)}; R² 1.0 = perfect). Gets better as history builds up.")


# ---------------------------------------------------------------- runs & comparison
def runs(df, model=None) -> list:
    """Run keys, newest first (optionally one model only)."""
    if df.empty:
        return []
    keys = df["run_time"].unique()
    if model:
        keys = [k for k in keys if rk_model(k) == model]
    return sorted(keys, key=rk_ts, reverse=True)


def run_label(key) -> str:
    return f"{MODELS.get(rk_model(key), rk_model(key))} {rk_ts(key):%a %d %b %Hz}"


def run_colour(key) -> str:
    """Fixed colour per run (by 12-hour slot, GFS shifted), so a run keeps its colour as new runs arrive."""
    slot = int(rk_ts(key).timestamp() // (12 * 3600)) + (3 if rk_model(key) == "gfs" else 0)
    return PALETTE[slot % len(PALETTE)]


def resolve_cmp(cmp, rt, all_runs):
    """Compare choice -> run key (or None) and a short explanation if missing."""
    model, t = rk_model(rt), rk_ts(rt)
    same = [r for r in all_runs if rk_model(r) == model]
    older = sorted((r for r in same if rk_ts(r) < t), key=rk_ts, reverse=True)
    if cmp == CMP_PREV or not cmp:
        return (older[0], "") if older else (None, "No older run stored.")
    if cmp in (CMP_P00, CMP_P12):
        h = 0 if cmp == CMP_P00 else 12
        hit = [r for r in older if rk_ts(r).hour == h]
        return (hit[0], "") if hit else (None, f"No earlier {h:02d}z run stored.")
    if cmp == CMP_WEEK:
        target = t - pd.Timedelta(days=7)
        if not older:
            return None, "No older run stored."
        best = min(older, key=lambda r: abs(rk_ts(r) - target))
        if abs(rk_ts(best) - target) > pd.Timedelta(hours=12):
            return None, (f"No run from about a week before {run_label(rt)} yet - oldest stored is "
                          f"{run_label(older[-1])}.")
        return best, ""
    if cmp == CMP_XMODEL:
        other = [r for r in all_runs if rk_model(r) != model and rk_ts(r) <= t]
        if not other:
            return None, "No run of the other model at or before this time."
        best = max(other, key=rk_ts)
        if t - rk_ts(best) > pd.Timedelta(hours=12):
            return None, f"No other-model run near {run_label(rt)} (latest is {run_label(best)})."
        return best, ""
    if cmp == rt:
        return None, "Compare run is the same as the selected run."
    if cmp not in all_runs:
        return None, "Compare run not stored."
    return cmp, ""


def ec_fetch_for(ts, fetches):
    """EC46 fetch that belongs with an IFS run: latest fetch on or before the run's day + 1."""
    if ts is None or not fetches:
        return None
    ts = rk_ts(ts) if isinstance(ts, str) else ts
    limit = (pd.Timestamp(ts).tz_convert(TZ).date() + pd.Timedelta(days=1)).isoformat()
    ok = [f for f in fetches if f <= limit]
    return max(ok) if ok else None


DAY_MIN_SPAN = pd.Timedelta(hours=18)   # a run must cover >= 18 h of a day to give a daily average


def daily(df, var):
    """Daily mean per run/zone/day - only days the run covers (nearly) fully.

    The first and last day of a run are partial (e.g. a 12z run ends at 12 UTC); their mean
    is not a daily average and made "change vs compare" show almost the full value on the
    last day (2026-10-16). 18 h span also passes 3-hourly data (00..21 h).
    """
    g = df.groupby(["run_time", "zone", "day"])
    out = g[var].mean().reset_index()
    span = (g["valid_time"].max() - g["valid_time"].min()).reset_index(drop=True)
    return out[(span >= DAY_MIN_SPAN).to_numpy()].reset_index(drop=True)


# ---------------------------------------------------------------- figures & helpers
def base_fig(title, unit, height=260):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"{title} ({unit})", x=0, font=dict(size=14, color=INK)),
        height=height, margin=dict(l=50, r=110, t=40, b=30),
        paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(family=FONT, color=INK, size=12),
        showlegend=False, hovermode="x unified",
        xaxis=dict(gridcolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def as_var(d, col, var):
    """Column `col` renamed to `var`. A frame that already has a `var` column
    (ensemble statistics next to the plain value) would otherwise end up with
    two columns of that name, which plotly refuses (DuplicateError)."""
    if col == var:
        return d
    return d.drop(columns=[var], errors="ignore").rename(columns={col: var})


def add_series(fig, d, var, name, colour, dash=None, width=2, m=None, x="valid_time", label_shift=0):
    m = m or VARS[var]
    if d is None or d.empty or var not in d:
        return
    d = d.dropna(subset=[var]).sort_values(x)
    if d.empty:
        return
    # weather solar value = mean over the interval ending at the point; MW is already hourly
    shape = "vh" if (var == "solar_wm2" and m["unit"] != "MW" and x == "valid_time") else "linear"
    fig.add_trace(go.Scatter(
        x=d[x], y=d[var], name=name, mode="lines", line_shape=shape,
        line=dict(color=colour, width=width, dash=dash),
        hovertemplate=f"{name}: %{{y:{m['fmt']}}} {m['unit']}<extra></extra>"))
    last = d.iloc[-1]
    fig.add_annotation(x=last[x], y=last[var], text=name, showarrow=False,
                       xanchor="left", xshift=6, yshift=label_shift, font=dict(color=colour, size=11))


def finish_fig(fig, zone, m, now_line=True, diff=False):
    if not fig.data:
        fig.add_annotation(text=f"No {m['label'].lower()} data for {zone}", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(color=MUTED))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    if now_line:
        fig.add_vline(x=pd.Timestamp.now(tz=TZ), line=dict(color=MUTED, width=1, dash="dot"))
    if m["unit"] == "MW" and not diff:
        fig.update_yaxes(rangemode="tozero")
    return fig


def _six_hourly_solar(d):
    """Weather solar is a mean over the interval ending at each point (3 h early, 6 h later).
    Put a run on a common 6-hour grid so two runs can be subtracted fairly."""
    d = d.sort_values("valid_time")
    vt = d["valid_time"].dt.tz_convert("UTC")
    vals = d["solar_wm2"].values
    out_t, out_v = [], []
    for i in range(len(d)):
        if vt.iloc[i].hour % 6:
            continue
        v = vals[i]
        if i > 0 and (vt.iloc[i] - vt.iloc[i - 1]) == pd.Timedelta(hours=3):
            v = (vals[i] + vals[i - 1]) / 2
        out_t.append(d["valid_time"].iloc[i])
        out_v.append(v)
    return pd.DataFrame({"valid_time": out_t, "solar_wm2": out_v})


def diff_series(d, var, rt, ct, m):
    """Selected run minus compare run on the times both cover."""
    a, b = d[d.run_time == rt], d[d.run_time == ct]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["valid_time", var])
    if var == "solar_wm2" and m["unit"] != "MW":
        a, b = _six_hourly_solar(a), _six_hourly_solar(b)
    j = a[["valid_time", var]].merge(b[["valid_time", var]], on="valid_time", suffixes=("", "_c"))
    j[var] = j[var] - j[var + "_c"]
    return j[["valid_time", var]].dropna()


def add_diff(fig, d, var, m, name, x="valid_time", bars=False):
    """Change as orange (higher) / blue (lower) area, or bars for daily data."""
    if d is None or d.empty:
        return
    d = d.sort_values(x)
    y = d[var].astype(float)
    hover = f"{name}: %{{y:+{m['fmt']}}} {m['unit']}<extra></extra>"
    if bars:
        fig.add_trace(go.Bar(x=d[x], y=y, marker_color=[PALETTE[1] if v >= 0 else PALETTE[0] for v in y],
                             hovertemplate=hover))
    else:
        fig.add_trace(go.Scatter(x=d[x], y=y.clip(lower=0), mode="lines", line=dict(width=0.8, color=PALETTE[1]),
                                 fill="tozeroy", fillcolor="rgba(235,104,52,0.35)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d[x], y=y.clip(upper=0), mode="lines", line=dict(width=0.8, color=PALETTE[0]),
                                 fill="tozeroy", fillcolor="rgba(42,120,214,0.35)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d[x], y=y, mode="lines", line=dict(width=1.2, color=INK), hovertemplate=hover))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))


NORMAL_COL = {("temp_c", "weather"): "temp_c", ("temp_c", "mw"): "temp_c",
              ("wind100_ms", "mw"): "wind_mw", ("solar_wm2", "mw"): "solar_mw"}


def add_normal(fig, zone, var, units, t0, t1):
    """Grey dotted ERA5 normal (1996-2025) for the days t0..t1. Wind/solar only in MW: the weather
    normals are land-point averages, the forecasts area averages, so m/s and W/m2 don't compare."""
    col = NORMAL_COL.get((var, units))
    nd = _cache.get("normals_doy")
    if col is None or nd is None or nd.empty or t0 is None:
        return
    n = nd[nd.zone == zone].set_index("doy")[col]
    if n.dropna().empty:
        return
    days = pd.date_range(pd.Timestamp(t0).floor("D"), pd.Timestamp(t1).ceil("D"), freq="D", tz=TZ)
    doy = [min(d.dayofyear - (1 if d.is_leap_year and d.month > 2 else 0), 365) for d in days]
    y = [n.get(k) for k in doy]
    fig.add_trace(go.Scatter(x=days + pd.Timedelta(hours=12), y=y, mode="lines", line_shape="spline",
                             line=dict(color="#8c959f", width=1.5, dash="dot"), name="Normal",
                             hovertemplate="Normal 1996-2025: %{y:,.1f}<extra></extra>"))
    fig.add_annotation(x=days[-1] + pd.Timedelta(hours=12), y=y[-1], text="Normal", showarrow=False,
                       xanchor="left", xshift=6, yshift=-22, font=dict(color="#8c959f", size=11))


def tile(label, value, sub=""):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": MUTED}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": 600, "color": INK}),
        html.Div(sub, style={"fontSize": "12px", "color": MUTED}),
    ], style={"background": CARD, "padding": "10px 14px", "borderRadius": "8px",
              "border": f"1px solid {GRID}", "minWidth": "150px", "flex": "1"})


def radio(id_, value):
    return dcc.RadioItems(id=id_, value=value, inline=True,
                          options=[{"label": v["label"], "value": k} for k, v in VARS.items()],
                          inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "14px"})


def _shade(v, lo, hi, diverging):
    if v is None or pd.isna(v):
        return None
    if diverging:
        m = max(abs(lo), abs(hi)) or 1
        t = max(-1.0, min(1.0, v / m))
        base = (42, 120, 214) if t < 0 else (235, 104, 52)
        a = abs(t) * 0.55
    else:
        t = 0 if hi == lo else (v - lo) / (hi - lo)
        base, a = (42, 120, 214), 0.08 + t * 0.5
    return f"rgba({base[0]},{base[1]},{base[2]},{a:.2f})"


def heat_table(wide, label_fn, digits, diverging):
    """wide: index=zone, columns=day/week -> (rows, columns, style) for a DataTable."""
    wide = wide.dropna(how="all").sort_index(axis=1)
    ids = {c: "c" + pd.Timestamp(c).strftime("%Y%m%d") for c in wide.columns}
    rows = [{"zone": f"{z} {ZONE_NAMES.get(z, '')}",
             **{ids[c]: (None if pd.isna(v) else round(float(v), digits)) for c, v in r.items()}}
            for z, r in wide.iterrows()]
    cols = [{"name": "Country", "id": "zone"}] + [
        {"name": label_fn(c), "id": ids[c], "type": "numeric"} for c in wide.columns]
    vals = wide.values.flatten()
    vals = vals[~pd.isna(vals)]
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0, 1)
    style = [{"if": {"column_id": "zone"}, "textAlign": "left", "fontWeight": 600}]
    for i, r in enumerate(rows):
        for c in wide.columns:
            col = _shade(r[ids[c]], lo, hi, diverging)
            if col:
                style.append({"if": {"row_index": i, "column_id": ids[c]}, "backgroundColor": col})
    return rows, cols, style


TABLE_STYLE = dict(style_table={"overflowX": "auto"},
                   style_cell={"fontFamily": FONT, "fontSize": "13px", "padding": "4px 8px",
                               "textAlign": "right"},
                   style_header={"fontWeight": 600, "background": BG})
NOTE = {"color": MUTED, "fontSize": "12px", "margin": "4px 0 8px"}

# ---------------------------------------------------------------- layout
TABS = [("forecast", "Forecast"), ("runs", "Run changes"), ("countries", "Countries"),
        ("ecgfs", "EC vs GFS"), ("ens", "Ensembles"), ("ec46", "EC46 (46 days)"), ("seasonal", "Seasonal"), ("quality", "Model quality")]


def model_options(df):
    have = sorted({rk_model(k) for k in runs(df)}, key=list(MODELS).index)
    return [{"label": MODELS[m], "value": m} for m in have]


def run_options(df, model):
    return [{"label": run_label(r), "value": r} for r in runs(df, model)]


def cmp_options(df, rt=None):
    """The 3 shortcuts show which run they resolve to for the selected run, e.g. 'Previous run (Wed 14 Oct 12z)'."""
    def lab(text, kind):
        if rt is None:
            return text
        ct, _ = resolve_cmp(kind, rt, runs(df))
        if ct is None:
            return f"{text} (none stored)"
        name = f"{rk_ts(ct):%a %d %b %Hz}" if rk_model(ct) == rk_model(rt) else run_label(ct)
        return f"{text} ({name})"
    return ([{"label": lab("Previous run", CMP_PREV), "value": CMP_PREV},
             {"label": lab("Previous 00z run", CMP_P00), "value": CMP_P00},
             {"label": lab("Previous 12z run", CMP_P12), "value": CMP_P12},
             {"label": lab("1 week earlier", CMP_WEEK), "value": CMP_WEEK},
             {"label": lab("Other model, same time", CMP_XMODEL), "value": CMP_XMODEL}]
            + [{"label": run_label(r), "value": r} for r in runs(df)])


def serve_layout():
    df = load()
    zones = sorted(df["zone"].unique()) if not df.empty else []
    m_opts = model_options(df)
    model0 = m_opts[0]["value"] if m_opts else "ifs"
    r_opts = run_options(df, model0)
    zone_opts = [{"label": f"{z} - {ZONE_NAMES.get(z, z)}", "value": z} for z in zones]
    if df.empty:
        head = "No data yet - run ecmwf_scraper.py first."
    else:
        bits = []
        for m in MODELS:
            rs = runs(df, m)
            if rs:
                bits.append(f"{MODELS[m]}: latest {rk_ts(rs[0]):%a %d %b %Hz}, {len(rs)} runs")
        head = " · ".join(bits) + " · Danish local time"
    return html.Div([
        # Title left, update button + status top right (moved here 2026-10-16)
        html.Div([
            html.Div([html.H2("Weather forecasts", style={"margin": "0 0 2px 0"}),
                      html.Div(head, style={"color": MUTED})]),
            html.Div([
                html.Button("Check for new data", id="upd-btn", n_clicks=0,
                            style={"padding": "6px 14px", "borderRadius": "6px", "border": f"1px solid {PALETTE[0]}",
                                   "background": CARD, "color": PALETTE[0], "fontWeight": 600, "cursor": "pointer",
                                   "whiteSpace": "nowrap"}),
                html.Span(id="upd-status", style={"color": MUTED, "fontSize": "13px", "marginLeft": "12px",
                                                  "maxWidth": "420px"}),
                dcc.Interval(id="upd-poll", interval=3000, disabled=True),
                dcc.Store(id="upd-done"),
            ], style={"display": "flex", "alignItems": "center", "marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "16px", "marginBottom": "12px",
                  "flexWrap": "wrap"}),
        html.Div([
            html.Div([html.Label("Country"), dcc.Dropdown(
                id="zone", options=zone_opts, value="DE" if "DE" in zones else (zones[0] if zones else None),
                clearable=False)], style={"width": "200px"}),
            html.Div([html.Label("Model"), dcc.Dropdown(
                id="model", options=m_opts, value=model0, clearable=False)], style={"width": "120px"}),
            html.Div([html.Label("Run"), dcc.Dropdown(
                id="run", options=r_opts, value=r_opts[0]["value"] if r_opts else None,
                clearable=False)], style={"width": "250px"}),
            html.Div([html.Label("Compare with"), dcc.Dropdown(
                id="cmp", options=cmp_options(df), value=CMP_PREV, clearable=False)], style={"width": "330px"}),
            html.Div([html.Label("Wind & solar as"), dcc.RadioItems(
                id="units", value="mw", inline=True,
                options=[{"label": "Weather (m/s, W/m²)", "value": "weather"},
                         {"label": "Power (MW)", "value": "mw"}],
                inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "14px"},
                style={"paddingTop": "8px"})]),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "6px", "flexWrap": "wrap"}),
        html.Div([html.Label("Show", style={"marginRight": "10px", "fontWeight": 600}), dcc.RadioItems(
            id="view", value="values", inline=True,
            options=[{"label": "Values", "value": "values"},
                     {"label": "Change vs compare", "value": "diff"}],
            inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "14px"})],
            style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        html.Div(id="cmp-note", style={"color": MUTED, "fontSize": "13px", "marginBottom": "6px"}),
        dcc.Tabs(id="tab", value="forecast", children=[dcc.Tab(label=l, value=v) for v, l in TABS]),
        html.Div(id="panel-forecast", children=[
            html.Div(id="f-help", style=NOTE),
            html.Div(id="f-note", style=NOTE),
            dcc.Graph(id="g-solar", config={"displaylogo": False}),
            dcc.Graph(id="g-wind", config={"displaylogo": False}),
            dcc.Graph(id="g-temp", config={"displaylogo": False}),
            # Same three charts for the other model (GFS when ECMWF is picked), on the same
            # run times as picked above (added 2026-10-16).
            html.H3(id="f2-head", style={"margin": "18px 0 2px", "fontSize": "18px"}),
            html.Div(id="f2-note", style=NOTE),
            dcc.Graph(id="g2-solar", config={"displaylogo": False}),
            dcc.Graph(id="g2-wind", config={"displaylogo": False}),
            dcc.Graph(id="g2-temp", config={"displaylogo": False}),
        ]),
        html.Div(id="panel-runs", children=[
            # Solar, wind, temperature charts (2026-10-16: "make the 3 usual graphs instead of just 1");
            # the variable picker below now only chooses the table.
            dcc.Graph(id="g-runs-solar", config={"displaylogo": False}),
            dcc.Graph(id="g-runs-wind", config={"displaylogo": False}),
            dcc.Graph(id="g-runs-temp", config={"displaylogo": False}),
            html.Div([html.Span("Table:", style={"fontWeight": 600, "marginRight": "10px"}),
                      radio("runs-var", "solar_wm2")],
                     style={"display": "flex", "alignItems": "center", "margin": "12px 0 0"}),
            html.H4("Daily average by run", style={"margin": "8px 0 6px"}),
            dash_table.DataTable(id="t-runs", **TABLE_STYLE),
        ]),
        # Solar, wind, temperature tables (2026-10-16: "same for countries the 3 usual tabels")
        html.Div(id="panel-countries", children=[
            *[x for v in ("solar", "wind", "temp") for x in (
                html.H4(id=f"c-title-{v}", style={"margin": "14px 0 2px"}),
                html.Div(id=f"c-note-{v}", style=NOTE),
                dash_table.DataTable(id=f"t-countries-{v}", **TABLE_STYLE))],
        ]),
        html.Div(id="panel-ecgfs", children=[
            html.Div(radio("x-var", "solar_wm2"), style={"margin": "10px 0"}),
            html.Div(id="x-note", style=NOTE),
            html.Div(id="x-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "8px"}),
            dcc.Graph(id="g-x", config={"displaylogo": False}),
            html.H4(id="x-table-title", style={"margin": "12px 0 6px"}),
            dash_table.DataTable(id="t-x", **TABLE_STYLE),
        ]),
        html.Div(id="panel-seasonal", children=[
            html.Div([radio("s-var", "solar_wm2"),
                      dcc.RadioItems(id="s-model", value="seas5", inline=True,
                                     options=[{"label": "Table: SEAS5 (ECMWF)", "value": "seas5"},
                                              {"label": "Table: CFSv2 (NOAA)", "value": "cfs"}],
                                     inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "14px"})],
                     style={"display": "flex", "gap": "30px", "margin": "10px 0", "flexWrap": "wrap"}),
            html.Div(id="s-note", style=NOTE),
            dcc.Graph(id="g-s", config={"displaylogo": False}),
            html.H4(id="s-table-title", style={"margin": "12px 0 6px"}),
            dash_table.DataTable(id="t-s", **TABLE_STYLE),
        ]),
        html.Div(id="panel-quality", children=[
            html.Div(id="q-note", style=NOTE),
            dcc.Graph(id="g-q", config={"displaylogo": False}),
            html.H4("Current MW fits, all countries", style={"margin": "12px 0 6px"}),
            dash_table.DataTable(id="t-qfit", **TABLE_STYLE),
            html.H4(id="q-acc-title", style={"margin": "16px 0 6px"}),
            html.Div(id="q-acc-note", style=NOTE),
            dash_table.DataTable(id="t-qacc", **TABLE_STYLE),
        ]),
        html.Div(id="panel-ens", children=[
            html.Div([
                html.Div([html.Label("Ensemble"), dcc.Dropdown(
                    id="n-model", value="ecmwf", clearable=False,
                    options=[{"label": v, "value": k} for k, v in ENS_MODELS.items()])], style={"width": "190px"}),
                html.Div([html.Label("Run"), dcc.Dropdown(id="n-run", clearable=False)], style={"width": "280px"}),
                html.Div([html.Label("Compare with"), dcc.Dropdown(
                    id="n-cmp", value=CMP_PREV, clearable=False,  # static shortcuts so the value is valid at start
                    options=[{"label": "Previous run", "value": CMP_PREV}, {"label": "Previous 00z run", "value": CMP_P00},
                             {"label": "Previous 12z run", "value": CMP_P12}, {"label": "1 day earlier", "value": "day"},
                             {"label": "Nothing", "value": "none"}])],
                         style={"width": "330px"}),
                html.Div(radio("n-var", "solar_wm2"), style={"paddingTop": "24px"}),
            ], style={"display": "flex", "gap": "16px", "margin": "10px 0", "flexWrap": "wrap"}),
            html.Div(id="n-showing", style={"display": "flex", "alignItems": "center", "gap": "8px",
                                            "flexWrap": "wrap", "margin": "4px 0 8px"}),
            html.Div(id="n-note", style=NOTE),
            dcc.Graph(id="g-ens", config={"displaylogo": False}),
            html.H4(id="n-t1-title", style={"margin": "12px 0 6px"}),
            dash_table.DataTable(id="t-ens", **TABLE_STYLE),
            html.H4(id="n-t2-title", style={"margin": "16px 0 6px"}),
            dash_table.DataTable(id="t-ens-c", **TABLE_STYLE),
        ]),
        html.Div(id="panel-ec46", children=[
            html.Div([
                html.Div([html.Label("EC46 run"), dcc.Dropdown(id="e-run", clearable=False)], style={"width": "220px"}),
                html.Div([html.Label("Compare with"), dcc.Dropdown(id="e-cmp", value=CMP_PREV, clearable=False, options=[
                    {"label": "Previous EC46 run", "value": CMP_PREV}, {"label": "1 week earlier", "value": CMP_WEEK}])],
                         style={"width": "240px"}),
                html.Div(radio("e-var", "solar_wm2"), style={"paddingTop": "24px"}),
            ], style={"display": "flex", "gap": "16px", "margin": "10px 0", "flexWrap": "wrap"}),
            html.Div(id="e-note", style=NOTE),
            dcc.Graph(id="g-ec46", config={"displaylogo": False}),
            html.H4("Weekly averages, all countries", style={"margin": "12px 0 6px"}),
            dash_table.DataTable(id="t-ec46", **TABLE_STYLE),
        ]),
    ], style={"fontFamily": FONT, "background": BG, "padding": "16px 20px", "minHeight": "100vh", "color": INK})


app = Dash(__name__, title="ECMWF forecast")
app.layout = serve_layout
app.validation_layout = serve_layout()


# ---------------------------------------------------------------- callbacks
@app.callback([Output(f"panel-{v}", "style") for v, _ in TABS], Input("tab", "value"))
def show_tab(tab):
    return [{"display": "block"} if v == tab else {"display": "none"} for v, _ in TABS]


# ---------------------------------------------------------------- update button
# The update runs as its own process (update_now.py) and writes logs/update_status.json, so a
# dashboard restart (e.g. after a code update) neither stops it nor forgets it, and only one runs.


def scraper_python():
    """The scrapers need eccodes -> Python 3.14 (3.15 has no Windows eccodes library)."""
    if sys.version_info < (3, 15):
        return [sys.executable]
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Python" / "pythoncore-3.14-64" / "python.exe"
    return [str(cand)] if cand.exists() else ["py", "-3.14"]


def update_status() -> dict:
    try:
        import update_now
        s = update_now.read_status()
        if s.get("running") and not update_now.pid_alive(s.get("pid")):
            s["running"] = False
            s["summary"] = s.get("summary") or f"The last update stopped unexpectedly - see {s.get('log', 'the log')}."
            s["finished"] = s.get("finished") or s.get("step_started") or s.get("started")
        return s
    except Exception as e:
        return {"error": str(e)}


def start_update():
    s = update_status()
    if s.get("running"):
        return False
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(scraper_python() + ["update_now.py"], cwd=HERE, creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return True


@app.callback(Output("upd-status", "children"), Output("upd-poll", "disabled"), Output("upd-btn", "disabled"),
              Output("upd-done", "data"),
              Input("upd-btn", "n_clicks"), Input("upd-poll", "n_intervals"), State("upd-done", "data"))
def update_button(clicks, _n, done_stamp):
    if ctx.triggered_id == "upd-btn" and clicks:
        if start_update():
            return "Updating: starting ...", False, True, done_stamp
    s = update_status()
    if s.get("running"):
        try:
            mins = (dt.datetime.now() - dt.datetime.fromisoformat(s.get("started"))).total_seconds() / 60
            step_mins = (dt.datetime.now() - dt.datetime.fromisoformat(s.get("step_started") or s["started"])
                         ).total_seconds() / 60
        except Exception:
            mins = step_mins = 0
        return (f"Updating: {s.get('step', '')} ... ({step_mins:.0f} min in this step, {mins:.0f} min total; "
                f"a new ECMWF or GFS run takes a few minutes each)", False, True, done_stamp)
    if s.get("finished"):
        try:
            when = f"{dt.datetime.fromisoformat(s['finished']):%d %b %H:%M}"
        except Exception:
            when = s["finished"]
        return f"Last update {when}: {s.get('summary', '')}", True, False, s["finished"]
    return "Downloads new ECMWF/GFS runs and converts them to MW.", True, False, done_stamp


@app.callback(Output("run", "options"), Output("run", "value"), Input("model", "value"), Input("run", "value"),
              Input("upd-done", "data"))
def model_runs(model, run, _stamp):
    df = load()
    opts = run_options(df, model)
    if ctx.triggered_id == "upd-done" and opts:  # new data arrived: jump to the newest run
        return opts, opts[0]["value"]
    keep = run if run in [o["value"] for o in opts] else None
    if keep is None and run and opts:  # switching model: same run time if the other model has it
        t = rk_ts(run)
        near = [o["value"] for o in opts if rk_ts(o["value"]) <= t]
        keep = near[0] if near else opts[0]["value"]
    return opts, keep or (opts[0]["value"] if opts else None)


def _context(run, cmp):
    df = load()
    if df.empty or not run or run not in set(df["run_time"].unique()):
        return None, None, None, ""
    ct, why = resolve_cmp(cmp, run, runs(df))
    return df, run, ct, why


def pill(text, colour, solid=True):
    return html.Span(text, style={"padding": "3px 10px", "borderRadius": "12px", "fontWeight": 700,
                                  "fontSize": "14px", "border": f"2px {'solid' if solid else 'dashed'} {colour}",
                                  "color": "white" if solid else colour, "background": colour if solid else CARD})


def showing_banner(df, rt, ct):
    """Always visible: which model + run (00z/12z) is shown, and what it is compared with."""
    lab = {"fontSize": "13px", "color": MUTED}
    out = [html.Span("Showing", style={**lab, "fontWeight": 600}), pill(run_label(rt), PALETTE[0])]
    if ct is not None:
        out += [html.Span("compared with", style=lab), pill(run_label(ct), PALETTE[1], solid=False)]
    newest = runs(df, rk_model(rt))
    if newest and newest[0] != rt:
        out.append(html.Span(f"(not the newest {MODELS.get(rk_model(rt))} run - newest is "
                             f"{rk_ts(newest[0]):%a %d %b %Hz})", style={**lab, "color": "#b45309"}))
    return out


@app.callback(Output("cmp", "options"), Input("run", "value"), Input("upd-done", "data"))
def cmp_opts(run, _stamp):
    df = load()
    return cmp_options(df, run if (not df.empty and run in set(df["run_time"].unique())) else None)


@app.callback(Output("cmp-note", "children"),
              Input("run", "value"), Input("cmp", "value"))
def cmp_note(run, cmp):
    df, rt, ct, why = _context(run, cmp)
    if rt is None:
        return ""
    if ct is None:
        return why
    days = (rk_ts(rt) - rk_ts(ct)) / pd.Timedelta(days=1)
    when = f"{days:.1f} days older" if days > 0.01 else ("same run time" if abs(days) <= 0.01 else
                                                           f"{-days:.1f} days newer")
    return f"Comparing {run_label(rt)} with {run_label(ct)} ({when})."


def forecast_figs(df, zone, rt, ct, units, diff):
    """Solar, wind, temperature charts for run rt (thick) vs ct (dashed)."""
    figs = []
    for var in VARS:
        m = meta(var, units)
        runs_txt = "" if rt is None else (f" · {run_label(rt)}" + (f" vs {run_label(ct)}" if ct else ""))
        fig = base_fig((("Change in " + m["label"].lower()) if diff else m["label"]) + runs_txt, m["unit"])
        if df is not None and zone and rt is not None:
            d = data(units, var, [rt, ct])
            d = d[d.zone == zone] if not d.empty else d
            if diff:
                if ct is not None and not d.empty:
                    add_diff(fig, diff_series(d, var, rt, ct, m), var, m, "Change")
            else:
                # fixed colours here so the two lines always differ: selected blue, compare orange
                if ct is not None:
                    add_series(fig, d[d.run_time == ct], var, run_label(ct), PALETTE[1], dash="dash",
                               width=1.5, m=m, label_shift=-10)
                add_series(fig, d[d.run_time == rt], var, run_label(rt), PALETTE[0], width=2.5, m=m,
                           label_shift=8 if ct is not None else 0)
                cur = d[d.run_time == rt]
                if fig.data and not cur.empty:
                    add_normal(fig, zone, var, units, cur.valid_time.min(), cur.valid_time.max())
        figs.append(finish_fig(fig, zone, m, diff=diff))
    return figs


def match_run(df, model, key):
    """Run of `model` at the same run time as `key`, else the newest one before it (max 12 h)."""
    if key is None:
        return None, ""
    t = rk_ts(key)
    cands = [r for r in runs(df, model) if rk_ts(r) <= t]
    if not cands:
        return None, f"no {MODELS.get(model, model)} run at or before {t:%a %d %b %Hz}"
    best = max(cands, key=rk_ts)
    if t - rk_ts(best) > pd.Timedelta(hours=12):
        return None, f"no {MODELS.get(model, model)} run within 12 h before {t:%a %d %b %Hz}"
    note = "" if rk_ts(best) == t else f"{run_label(best)} used for {t:%a %d %b %Hz} (no run at that time)"
    return best, note


@app.callback(Output("g-solar", "figure"), Output("g-wind", "figure"), Output("g-temp", "figure"),
              Output("f-note", "children"), Output("f-help", "children"),
              Output("f2-head", "children"), Output("f2-note", "children"),
              Output("g2-solar", "figure"), Output("g2-wind", "figure"), Output("g2-temp", "figure"),
              Input("zone", "value"), Input("run", "value"), Input("cmp", "value"),
              Input("units", "value"), Input("view", "value"), Input("tab", "value"))
def forecast(zone, run, cmp, units, view, tab):
    if tab != "forecast":
        raise PreventUpdate
    df, rt, ct, why = _context(run, cmp)
    diff = view == "diff"
    figs = forecast_figs(df, zone, rt, ct, units, diff)
    if diff:
        help_ = (why if ct is None else
                 f"Run {run_label(rt)} minus {run_label(ct)}, on the hours both cover. Orange = higher, "
                 f"blue = lower. Dotted line = now."
                 + (" Weather solar compared as 6-hour means." if units != "mw" else ""))
    else:
        help_ = ("Selected run = thick line; compare run = dashed. Dotted line = now. "
                 "Grey dotted = normal (ERA5 1996-2025; wind/solar only in MW).")
    # Second block: the other model on the same run times (GFS under ECMWF, ECMWF under GFS).
    other = None
    if rt is not None:
        other = next((m for m in MODELS if m != rk_model(rt)), None)
    if other is None:
        head, note2, figs2 = "", "", forecast_figs(None, zone, None, None, units, diff)
    else:
        rt2, n1 = match_run(df, other, rt)
        ct2, n2 = match_run(df, other, ct) if ct is not None else (None, "")
        head = f"{MODELS[other]} - same run times as above"
        bits = [b for b in (n1, n2) if b]
        if rt2 is not None:
            bits.insert(0, f"{run_label(rt2)}" + (f" vs {run_label(ct2)}" if ct2 else "") + ".")
        if units == "mw" and rt2 is not None:
            bits.append(mw_note(zone, other))
        note2 = " ".join(bits)
        figs2 = forecast_figs(df, zone, rt2, ct2, units, diff)
    return (*figs, mw_note(zone, rk_model(rt)) if (units == "mw" and rt) else "", help_,
            head, note2, *figs2)


def runs_fig(df, zone, rt, ct, var, units, diff):
    """One run-changes chart (last 6 runs of the model up to rt, compare run included)."""
    m = meta(var, units)
    title = (f"Change in {m['label'].lower()} vs {run_label(ct)}" if diff else m["label"]) + \
        f" - runs up to {run_label(rt) if rt is not None else ''}, {zone}"
    fig = base_fig(title, m["unit"], height=420)
    if df is None or not zone:
        return finish_fig(fig, zone, m), pd.DataFrame(), []
    show = [r for r in runs(df, rk_model(rt)) if rk_ts(r) <= rk_ts(rt)][:6]
    if ct is not None and ct not in show:
        show = show[:5] + [ct]  # always include the compare run
    d = data(units, var, show)
    d = d[d.zone == zone] if not d.empty else d
    for r in sorted(show, key=rk_ts):  # oldest first, selected drawn on top
        if diff:
            if r == ct:
                continue
            ds = diff_series(d, var, r, ct, m)
            add_series(fig, ds, var, run_label(r), run_colour(r), width=2.5 if r == rt else 1.5, m=m)
        else:
            add_series(fig, d[d.run_time == r], var, run_label(r), run_colour(r),
                       width=2.5 if r == rt else 1.5, dash="dash" if r == ct else None, m=m)
    if diff and fig.data:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    finish_fig(fig, zone, m, diff=diff)
    return fig, d, show


@app.callback(Output("g-runs-solar", "figure"), Output("g-runs-wind", "figure"),
              Output("g-runs-temp", "figure"),
              Output("t-runs", "data"), Output("t-runs", "columns"),
              Output("t-runs", "style_data_conditional"),
              Input("zone", "value"), Input("run", "value"), Input("cmp", "value"),
              Input("runs-var", "value"), Input("units", "value"), Input("view", "value"), Input("tab", "value"))
def run_changes(zone, run, cmp, var, units, view, tab):
    if tab != "runs":
        raise PreventUpdate
    m = meta(var, units)
    digits = 0 if m["unit"] == "MW" else 1
    df, rt, ct, _ = _context(run, cmp)
    diff = view == "diff" and ct is not None
    figs, d, show = [], pd.DataFrame(), []
    for v in VARS:   # solar, wind, temperature
        fig_v, d_v, show_v = runs_fig(df, zone, rt, ct, v, units, diff)
        figs.append(fig_v)
        if v == var:
            d, show = d_v, show_v
    if d.empty:
        return (*figs, [], [], [])

    wide = daily(d, var).pivot(index="day", columns="run_time", values=var).sort_index()
    if diff and ct in wide.columns:
        wide = wide.sub(wide[ct], axis=0).drop(columns=[ct]).dropna(how="all")
    wide = wide[[r for r in sorted(show, key=rk_ts, reverse=True) if r in wide.columns]]
    rows = [{"day": pd.Timestamp(day).strftime("%a %d %b"),
             **{run_label(c): (None if pd.isna(v) else round(float(v), digits)) for c, v in r.items()}}
            for day, r in wide.iterrows()]
    cols = [{"name": "Day", "id": "day"}] + [
        {"name": run_label(c) + (" (compare)" if c == ct else "") + (f" − {run_label(ct)}" if diff else ""),
         "id": run_label(c), "type": "numeric"}
        for c in wide.columns]
    style = [{"if": {"column_id": run_label(c)}, "color": run_colour(c), "fontWeight": 600}
             for c in wide.columns if c in (rt, ct)]
    if diff:
        for c in wide.columns:
            cid = run_label(c)
            style += [{"if": {"column_id": cid, "filter_query": f"{{{cid}}} > 0"}, "backgroundColor": "rgba(235,104,52,0.18)"},
                      {"if": {"column_id": cid, "filter_query": f"{{{cid}}} < 0"}, "backgroundColor": "rgba(42,120,214,0.18)"}]
    return (*figs, rows, cols, style)


C_VARS = {"solar": "solar_wm2", "wind": "wind100_ms", "temp": "temp_c"}   # display order


def country_table(df, rt, ct, why, var, mode, units):
    """Countries x days table for one variable -> rows, cols, style, title, note."""
    m = meta(var, units)
    title = f"{m['label']} ({m['unit']})"
    if df is None:
        return [], [], [], title, ""
    digits = 0 if m["unit"] == "MW" else 1
    d = data(units, var, [rt, ct])
    if d.empty:
        return [], [], [], title, ("No MW data yet - run ecmwf_mw.py." if units == "mw" else "")
    cur = daily(d[d.run_time == rt], var).pivot(index="zone", columns="day", values=var)
    note = f"Daily average {m['label'].lower()} ({m['unit']}), run {run_label(rt)}."
    diverging = False
    if mode == "diff":
        if ct is None:
            return [], [], [], title, why
        prev = daily(d[d.run_time == ct], var).pivot(index="zone", columns="day", values=var)
        cur = (cur - prev).dropna(axis=1, how="all")
        title = f"Change in {m['label'].lower()} ({m['unit']})"
        note = (f"Change in daily average {m['label'].lower()} ({m['unit']}): run {run_label(rt)} minus "
                f"{run_label(ct)}. Orange = higher, blue = lower. Only days both runs cover.")
        diverging = True
    if cur.dropna(how="all").empty:
        return [], [], [], title, f"No {m['label'].lower()} values for {run_label(rt)}" + (
            " - no MW model for this model yet (runs ecmwf_mw.py tonight)." if units == "mw" else ".")
    rows, cols, style = heat_table(cur, lambda c: pd.Timestamp(c).strftime("%a %d"), digits, diverging)
    return rows, cols, style, title, note


@app.callback([o for v in C_VARS for o in (
                  Output(f"t-countries-{v}", "data"), Output(f"t-countries-{v}", "columns"),
                  Output(f"t-countries-{v}", "style_data_conditional"),
                  Output(f"c-title-{v}", "children"), Output(f"c-note-{v}", "children"))],
              Input("run", "value"), Input("cmp", "value"),
              Input("view", "value"), Input("units", "value"), Input("tab", "value"))
def countries(run, cmp, mode, units, tab):
    if tab != "countries":
        raise PreventUpdate
    df, rt, ct, why = _context(run, cmp)
    out = []
    for v, var in C_VARS.items():
        out += list(country_table(df, rt, ct, why, var, mode, units))
    return out


def ec_gfs_pair(df, run):
    """Latest run time at or before the selected run that both ECMWF and GFS have."""
    t = rk_ts(run)
    ifs = {rk_ts(r) for r in runs(df, "ifs")}
    gfs = {rk_ts(r) for r in runs(df, "gfs")}
    common = sorted(x for x in ifs & gfs if x <= t)
    if not common:
        return None, None
    c = common[-1]
    return rk("ifs", c), rk("gfs", c)


@app.callback(Output("g-x", "figure"), Output("t-x", "data"), Output("t-x", "columns"),
              Output("t-x", "style_data_conditional"), Output("x-note", "children"),
              Output("x-kpis", "children"), Output("x-table-title", "children"),
              Input("zone", "value"), Input("run", "value"), Input("x-var", "value"),
              Input("units", "value"), Input("view", "value"), Input("tab", "value"))
def ec_vs_gfs(zone, run, var, units, view, tab):
    if tab != "ecgfs":
        raise PreventUpdate
    m = meta(var, units)
    digits = 0 if m["unit"] == "MW" else 1
    diff = view == "diff"
    fig = base_fig((f"ECMWF minus GFS, {m['label'].lower()}" if diff else f"{m['label']}: ECMWF vs GFS")
                   + f", {zone}", m["unit"], height=420)
    df = load()
    if df.empty or not run:
        return finish_fig(fig, zone, m), [], [], [], "No data.", [], ""
    ri, rg = ec_gfs_pair(df, run)
    if ri is None:
        return (finish_fig(fig, zone, m), [], [], [],
                "No run time that both ECMWF and GFS have (at or before the selected run) - run gfs_scraper.py.",
                [], "")
    fig.layout.title.text += f" · both {rk_ts(ri):%a %d %b %Hz}"
    d = data(units, var, [ri, rg])
    if d.empty:
        return finish_fig(fig, zone, m), [], [], [], "No MW data yet - run ecmwf_mw.py.", [], ""
    dz = d[d.zone == zone]
    if diff:
        add_diff(fig, diff_series(dz, var, ri, rg, m), var, m, "ECMWF − GFS")
    else:
        add_series(fig, dz[dz.run_time == rg], var, f"GFS {rk_ts(rg):%Hz}", PALETTE[1], width=2, m=m, label_shift=-10)
        add_series(fig, dz[dz.run_time == ri], var, f"ECMWF {rk_ts(ri):%Hz}", PALETTE[0], width=2.5, m=m, label_shift=8)
    finish_fig(fig, zone, m, diff=diff)

    # tiles: next 24 h / 7 days, both models and the gap
    now = pd.Timestamp.now(tz=TZ)
    start = max(now, dz["valid_time"].min()) if not dz.empty else now

    def avg(key, hours):
        w = dz[(dz.run_time == key) & (dz.valid_time > start) & (dz.valid_time <= start + pd.Timedelta(hours=hours))]
        return w[var].mean() if not w.empty else float("nan")

    tiles = []
    for hours, lab in ((24, "next 24 h"), (168, "next 7 days")):
        a, b = avg(ri, hours), avg(rg, hours)
        f = lambda v: "–" if pd.isna(v) else f"{v:{m['fmt']}} {m['unit']}"
        gap = "" if (pd.isna(a) or pd.isna(b)) else f"ECMWF − GFS: {a - b:+{m['fmt']}} {m['unit']}"
        tiles += [tile(f"ECMWF {lab}", f(a), gap), tile(f"GFS {lab}", f(b), "")]

    # table: daily ECMWF - GFS for all countries (only days both cover)
    da = daily(d[d.run_time == ri], var).pivot(index="zone", columns="day", values=var)
    dg = daily(d[d.run_time == rg], var).pivot(index="zone", columns="day", values=var)
    if diff:
        wide, title, div = (da - dg).dropna(axis=1, how="all"), \
            f"Daily average ECMWF − GFS ({m['unit']}), all countries. Orange = ECMWF higher, blue = GFS higher.", True
    else:
        wide, title, div = da, f"Daily average {m['label'].lower()} ({m['unit']}), ECMWF - switch Show to " \
            f"'Change vs compare' for ECMWF − GFS.", False
    rows, cols, style = heat_table(wide, lambda c: pd.Timestamp(c).strftime("%a %d"), digits, div)
    note = (f"ECMWF and GFS runs from {rk_ts(ri):%a %d %b %Hz} (the latest run time both have at or before "
            f"the selected run). ECMWF = blue, GFS = orange.")
    for key, name in ((ri, "ECMWF"), (rg, "GFS")):
        if dz[dz.run_time == key][var].notna().sum() == 0:
            note += f" No {name} {m['label'].lower()} values for {zone} in this run."
    if units == "mw":
        note += " " + mw_note(zone, "gfs")
    return fig, rows, cols, style, note, tiles, title


SEAS_COL = {("temp_c", "weather"): ("temp_c", "temp_anom", "temp_c"),
            ("temp_c", "mw"): ("temp_c", "temp_anom", "temp_c"),
            ("wind100_ms", "weather"): ("wind100_ms", "wind100_anom", "wind100_ms"),
            ("wind100_ms", "mw"): ("wind_mw", "wind_mw_anom", "wind_mw"),
            ("solar_wm2", "weather"): ("solar_wm2", "solar_anom", "solar_wm2"),
            ("solar_wm2", "mw"): ("solar_mw", "solar_mw_anom", "solar_mw")}
SEAS_NAMES = {"seas5": "SEAS5 (ECMWF)", "cfs": "CFSv2 (NOAA)"}
CFS_LAGGED = 4  # CFSv2: average of the last 4 daily runs (lagged ensemble)


def seasonal_latest(seas, model, upto):
    """Latest SEAS5 run, or the mean of the last CFS_LAGGED CFSv2 runs, on/before 'upto' (date string)."""
    s = seas[(seas.model == model) & (seas.run_date <= upto)]
    if s.empty:
        s = seas[seas.model == model]
    if s.empty:
        return s, []
    runs_ = sorted(s.run_date.unique(), reverse=True)[:(CFS_LAGGED if model == "cfs" else 1)]
    s = s[s.run_date.isin(runs_)]
    num = [c for c in s.columns if c not in ("model", "run_date", "zone", "month", "solar_bad")]
    return s.groupby(["zone", "month"])[num].mean().reset_index(), runs_


@app.callback(Output("g-s", "figure"), Output("t-s", "data"), Output("t-s", "columns"),
              Output("t-s", "style_data_conditional"), Output("s-note", "children"),
              Output("s-table-title", "children"),
              Input("zone", "value"), Input("run", "value"), Input("s-var", "value"), Input("s-model", "value"),
              Input("units", "value"), Input("view", "value"), Input("tab", "value"))
def seasonal(zone, run, var, tmodel, units, view, tab):
    if tab != "seasonal":
        raise PreventUpdate
    load()
    m = meta(var, units)
    digits = 0 if m["unit"] == "MW" else 1
    diff = view == "diff"
    vcol, acol, ncol = SEAS_COL[(var, units)]
    fig = base_fig((f"{m['label']} vs normal" if diff else f"{m['label']}, monthly mean") + f", {zone}",
                   m["unit"], height=420)
    fig.update_layout(showlegend=True, hovermode="x unified", margin=dict(l=50, r=30, t=80, b=30),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12)))
    seas, nm = _cache.get("seasonal"), _cache.get("normals_month")
    if seas is None or seas.empty:
        return finish_fig(fig, zone, m, now_line=False), [], [], [], \
            "No seasonal data yet - run seas5_scraper.py and cfs_scraper.py.", ""
    upto = (rk_ts(run).tz_convert(TZ).date() + pd.Timedelta(days=1)).isoformat() if run else "9999"
    notes, frames = [], {}
    for model, colour in (("seas5", PALETTE[0]), ("cfs", PALETTE[1])):
        s, runs_ = seasonal_latest(seas, model, upto)
        if s.empty:
            notes.append(f"No {SEAS_NAMES[model]} data yet.")
            continue
        frames[model] = s
        z = s[s.zone == zone].sort_values("month")
        y = z[acol] if diff else z[vcol]
        if y.notna().any():
            x = pd.to_datetime(z["month"]).dt.strftime("%b %Y")
            fig.add_trace(go.Scatter(x=x, y=y, name=SEAS_NAMES[model], mode="lines+markers",
                                     line=dict(color=colour, width=2.5), marker=dict(size=7),
                                     hovertemplate=f"{SEAS_NAMES[model]}: %{{y:{('+' if diff else '') + m['fmt']}}} "
                                                   f"{m['unit']}<extra></extra>"))
        label = runs_[0] if (model == "seas5" or len(runs_) == 1) else f"average of {len(runs_)} runs {runs_[-1]}..{runs_[0]}"
        notes.append(f"{SEAS_NAMES[model]}: {label}.")
    # normal bars (values) / zero line (change)
    months = sorted(set().union(*[set(f["month"]) for f in frames.values()])) if frames else []
    if not diff and nm is not None and not nm.empty and months:
        n = nm[nm.zone == zone].set_index("month")[ncol]
        y = [n.get(pd.Timestamp(mo).month) for mo in months]
        if any(v is not None and not pd.isna(v) for v in y):
            fig.add_trace(go.Scatter(x=pd.to_datetime(months).strftime("%b %Y"), y=y, name="Normal (ERA5 1996-2025)",
                                     mode="lines+markers", line=dict(color="#8c959f", width=2, dash="dash"),
                                     marker=dict(size=5),
                                     hovertemplate=f"Normal: %{{y:{m['fmt']}}} {m['unit']}<extra></extra>"))
    if diff and fig.data:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    if months:
        fig.update_xaxes(categoryorder="array", categoryarray=list(pd.to_datetime(months).strftime("%b %Y")))
    finish_fig(fig, zone, m, now_line=False, diff=diff)
    fig.update_layout(showlegend=bool(fig.data))
    if nm is None or nm.empty:
        notes.append("No ERA5 normals yet - run era5_normals.py (fills over ~3 nights).")
    if var == "solar_wm2" and "seas5" in frames and "solar_bad" in seas:
        sb = seas[(seas.model == "seas5") & (seas.zone == zone) & seas.solar_bad]
        if not sb.empty and (diff or units == "mw"):
            notes.append("SEAS5 solar left empty for " + ", ".join(pd.to_datetime(sorted(sb.month.unique()))
                                                                 .strftime("%b %Y")) +
                         ": the radiation anomaly delivered by the API is implausible there (larger than half "
                         "the monthly mean, growing with lead time).")
    notes.append("SEAS5 anomalies are vs ECMWF's own model climate. CFSv2 anomalies are vs the ERA5 normals "
                 "and include CFSv2's own bias - its run-to-run changes are more reliable.")
    if units == "mw":
        notes.append("MW: normal MW + MW per m/s × wind anomaly (wind), normal MW × radiation ratio (solar), today's fleet.")
    # table: one model, all countries
    t = frames.get(tmodel)
    if t is None:
        return fig, [], [], [], " ".join(notes), f"No {SEAS_NAMES[tmodel]} data."
    col = acol if diff else vcol
    wide = t.pivot(index="zone", columns="month", values=col)
    wide.columns = pd.to_datetime(wide.columns)
    rows, cols, style = heat_table(wide, lambda c: pd.Timestamp(c).strftime("%b %Y"), digits, diff)
    title = (f"{SEAS_NAMES[tmodel]}: monthly {'anomaly' if diff else 'mean'} {m['label'].lower()} ({m['unit']}), "
             f"all countries" + (". Orange = above normal, blue = below." if diff else ""))
    return fig, rows, cols, style, " ".join(notes), title




def _entsoe_db_default():
    """entsoe_data.db wherever it lives (see entsoe_scraper/entsoe_db.py)."""
    import sys
    folder = str(HERE.parent / "entsoe_scraper")
    if folder not in sys.path:
        sys.path.append(folder)
    try:
        from entsoe_db import entsoe_db_path
        return entsoe_db_path()
    except ImportError:
        return HERE.parent / "entsoe_scraper" / "entsoe_data.db"


ENTSOE_DB = _entsoe_db_default()
ACC_DAYS = 14


def entsoe_mw(since):
    """ENTSO-E day-ahead wind (on+offshore) and solar MW per zone and hour since 'since' (read-only)."""
    if not ENTSOE_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(ENTSOE_DB.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        zones = list(ZONE_NAMES)
        e = pd.read_sql_query(
            f"SELECT zone, timestamp, solar_mw, wind_onshore_mw, wind_offshore_mw FROM wind_solar_forecast "
            f"WHERE zone IN ({','.join('?' * len(zones))}) AND timestamp >= ?", conn,
            params=[*zones, since.strftime("%Y-%m-%d")])
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    e["valid_time"] = pd.to_datetime(e["timestamp"], utc=True).dt.tz_convert(TZ)
    e["wind_e"] = e[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
    return e[["zone", "valid_time", "wind_e", "solar_mw"]].rename(columns={"solar_mw": "solar_e"})


def day_ahead_accuracy():
    """D+1 accuracy of our MW (00z runs) vs the ENTSO-E day-ahead MW, last ACC_DAYS days. Cached per db state."""
    if "table" in _cache.get("acc", {}):
        return _cache["acc"]["table"], _cache["acc"]["span"]
    df = load()
    today = pd.Timestamp.now(tz=TZ).normalize()
    since = today - pd.Timedelta(days=ACC_DAYS + 1)
    ent = entsoe_mw(since)
    rows, span = [], None
    if not df.empty and not ent.empty:
        for model in MODELS:
            keys = [r for r in runs(df, model) if rk_ts(r).hour == 0 and rk_ts(r) >= since.tz_convert("UTC")]
            if not keys:
                continue
            mw = get_mw(keys)
            if mw.empty:
                continue
            mw = mw.copy()
            mw["d1"] = [(rk_ts(k).tz_convert(TZ).normalize() + pd.Timedelta(days=1)).date() for k in mw["run_time"]]
            mw = mw[mw["valid_time"].dt.date == mw["d1"]]  # only the next day: what a day-ahead trader sees
            j = mw.merge(ent, on=["zone", "valid_time"])
            if j.empty:
                continue
            span = (j.valid_time.min().date(), j.valid_time.max().date()) if span is None else \
                (min(span[0], j.valid_time.min().date()), max(span[1], j.valid_time.max().date()))
            for zone, g in j.groupby("zone"):
                row = {"zone": zone, "model": MODELS[model], "days": g.valid_time.dt.date.nunique()}
                for tech, a, b in (("wind", "wind100_ms", "wind_e"), ("solar", "solar_wm2", "solar_e")):
                    gg = g.dropna(subset=[a, b])
                    if gg.empty or gg[b].mean() <= 0:
                        continue
                    err = gg[a] - gg[b]
                    row[f"{tech}_mae"] = float(err.abs().mean())
                    row[f"{tech}_bias"] = float(err.mean())
                    row[f"{tech}_pct"] = 100 * float(err.abs().mean() / gg[b].mean())
                rows.append(row)
    table = pd.DataFrame(rows)
    _cache["acc"] = {"table": table, "span": span}
    return table, span


@app.callback(Output("g-q", "figure"), Output("q-note", "children"), Output("t-qfit", "data"),
              Output("t-qfit", "columns"), Output("t-qfit", "style_data_conditional"),
              Output("t-qacc", "data"), Output("t-qacc", "columns"), Output("t-qacc", "style_data_conditional"),
              Output("q-acc-title", "children"), Output("q-acc-note", "children"),
              Input("zone", "value"), Input("tab", "value"))
def quality(zone, tab):
    if tab != "quality":
        raise PreventUpdate
    load()
    fig = base_fig(f"Fit quality of the MW model over time, {zone} (R²; 1 = perfect, 0 = no skill)", "R²", height=360)
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), margin=dict(l=50, r=30, t=80, b=30))
    hist = _cache.get("calib_hist")
    styles = {("ifs", "wind"): (PALETTE[0], None), ("ifs", "solar"): (PALETTE[0], "dot"),
              ("gfs", "wind"): (PALETTE[1], None), ("gfs", "solar"): (PALETTE[1], "dot")}
    if hist is not None and not hist.empty:
        h = hist[(hist.zone == zone) & hist.tech.isin(["wind", "solar"])].copy()
        h["t"] = pd.to_datetime(h["fitted_at"], utc=True).dt.tz_convert(TZ)
        for (model, tech), g in h.groupby(["model", "tech"]):
            g = g.sort_values("t")
            col, dash_ = styles.get((model, tech), (MUTED, None))
            fig.add_trace(go.Scatter(x=g.t, y=g.r2, mode="lines+markers", name=f"{MODELS.get(model, model)} {tech}",
                                     line=dict(color=col, dash=dash_, width=2), marker=dict(size=5),
                                     customdata=g.hours,
                                     hovertemplate="%{y:.2f} (%{customdata:.0f} h of data)<extra></extra>"))
        fig.update_yaxes(range=[min(-0.1, float(h.r2.min()) - 0.05) if not h.empty else -0.1, 1.0])
        if not h.empty:
            t0, t1 = h.t.min(), h.t.max()
            if t1 - t0 < pd.Timedelta(days=3):  # few fits so far: show a sensible date window
                t0, t1 = t0 - pd.Timedelta(days=2), t1 + pd.Timedelta(days=2)
            fig.update_xaxes(range=[t0, t1], tickformat="%d %b", dtick=86400000 if t1 - t0 < pd.Timedelta(days=20) else None)
    if not fig.data:
        fig.add_annotation(text="No fit history yet - builds up with every ecmwf_mw.py run", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(color=MUTED))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    fig.add_hline(y=0.8, line=dict(color=GRID, width=1, dash="dash"))
    note = ("Every night ecmwf_mw.py refits the weather → MW model per country and model against ENTSO-E's day-ahead "
            "MW (last 60 days). From ~14 days of history it also fits the power-curve shape. Dashed line = 0.8 (good).")

    # current fits table
    calib = _cache.get("calib")
    rows_f, cols_f, sty_f = [], [], []
    if calib is not None and not calib.empty:
        c = calib[calib.tech.isin(["wind", "solar"])]
        for zone_, g in c.groupby("zone"):
            row = {"zone": f"{zone_} {ZONE_NAMES.get(zone_, '')}"}
            for r in g.itertuples():
                row[f"{r.model}_{r.tech}"] = None if pd.isna(r.r2) else round(float(r.r2), 2)
            row["hours"] = int(g.hours.max()) if g.hours.notna().any() else None
            rows_f.append(row)
        cols_f = [{"name": "Country", "id": "zone"}] + [
            {"name": f"{MODELS[m]} {t} R²", "id": f"{m}_{t}", "type": "numeric"}
            for m in MODELS for t in ("wind", "solar")] + [{"name": "Hours fitted", "id": "hours", "type": "numeric"}]
        sty_f = [{"if": {"column_id": "zone"}, "textAlign": "left", "fontWeight": 600}]
        for m in MODELS:
            for t in ("wind", "solar"):
                cid = f"{m}_{t}"
                sty_f += [{"if": {"column_id": cid, "filter_query": f"{{{cid}}} is num && {{{cid}}} >= 0.8"}, "backgroundColor": "rgba(27,175,122,0.18)"},
                          {"if": {"column_id": cid, "filter_query": f"{{{cid}}} is num && {{{cid}}} < 0.5"}, "backgroundColor": "rgba(235,104,52,0.22)"}]

    # day-ahead accuracy
    acc, span = day_ahead_accuracy()
    title = f"Day-ahead accuracy, last {ACC_DAYS} days: our MW (00z run, next day) vs ENTSO-E day-ahead MW"
    if acc.empty:
        anote = ("Not enough overlap yet (needs 00z runs whose next day ENTSO-E already published), or the ENTSO-E "
                 f"database was not found at {ENTSOE_DB}.")
        return fig, note, rows_f, cols_f, sty_f, [], [], [], title, anote
    acc = acc.sort_values(["zone", "model"])
    rows_a = []
    for r in acc.to_dict("records"):
        row = {"zone": f"{r['zone']} {ZONE_NAMES.get(r['zone'], '')}", "model": r["model"], "days": r["days"]}
        for t in ("wind", "solar"):
            for k in ("mae", "bias", "pct"):
                v = r.get(f"{t}_{k}")
                row[f"{t}_{k}"] = None if v is None or pd.isna(v) else round(v, 0 if k != "pct" else 1)
        rows_a.append(row)
    cols_a = [{"name": "Country", "id": "zone"}, {"name": "Model", "id": "model"}, {"name": "Days", "id": "days", "type": "numeric"},
              {"name": "Wind avg error MW", "id": "wind_mae", "type": "numeric"},
              {"name": "Wind error %", "id": "wind_pct", "type": "numeric"},
              {"name": "Wind bias MW", "id": "wind_bias", "type": "numeric"},
              {"name": "Solar avg error MW", "id": "solar_mae", "type": "numeric"},
              {"name": "Solar error %", "id": "solar_pct", "type": "numeric"},
              {"name": "Solar bias MW", "id": "solar_bias", "type": "numeric"}]
    sty_a = [{"if": {"column_id": c}, "textAlign": "left", "fontWeight": 600} for c in ("zone", "model")]
    for t in ("wind", "solar"):
        cid = f"{t}_pct"
        sty_a += [{"if": {"column_id": cid, "filter_query": f"{{{cid}}} is num && {{{cid}}} <= 15"}, "backgroundColor": "rgba(27,175,122,0.18)"},
                  {"if": {"column_id": cid, "filter_query": f"{{{cid}}} is num && {{{cid}}} > 35"}, "backgroundColor": "rgba(235,104,52,0.22)"}]
    anote = (f"Hours {span[0]:%d %b}–{span[1]:%d %b}. Avg error = mean absolute hourly gap; % = of the average ENTSO-E MW; "
             "bias > 0 = we forecast too high. Partly in-sample: the same ENTSO-E data is used to fit the model, "
             "so these numbers flatter the model until longer history exists.")
    return fig, note, rows_f, cols_f, sty_f, rows_a, cols_a, sty_a, title, anote


def _week_label(d):
    d = pd.Timestamp(d)
    return f"W{d.isocalendar().week:02d} ({d.strftime('%d %b')})"


def ec46_label(f):
    return f"EC46 of {pd.Timestamp(f):%a %d %b}"


@app.callback(Output("e-run", "options"), Output("e-run", "value"), Output("e-cmp", "options"),
              Input("tab", "value"), Input("upd-done", "data"), State("e-run", "value"))
def ec46_options(tab, _done, cur):
    if tab != "ec46":
        raise PreventUpdate
    load()
    ec = _cache.get("ec46")
    fetches = sorted(ec["fetched_on"].unique(), reverse=True) if ec is not None and not ec.empty else []
    opts = [{"label": ec46_label(f), "value": f} for f in fetches]
    cmp_opts = [{"label": "Previous EC46 run", "value": CMP_PREV},
                {"label": "1 week earlier", "value": CMP_WEEK}] + opts
    return opts, (cur if cur in fetches else (fetches[0] if fetches else None)), cmp_opts


@app.callback(Output("g-ec46", "figure"), Output("t-ec46", "data"), Output("t-ec46", "columns"),
              Output("t-ec46", "style_data_conditional"), Output("e-note", "children"),
              Input("zone", "value"), Input("e-run", "value"), Input("e-cmp", "value"),
              Input("e-var", "value"), Input("view", "value"), Input("units", "value"), Input("tab", "value"))
def ec46(zone, erun, cmp, var, mode, units, tab):
    if tab != "ec46":
        raise PreventUpdate
    m = meta(var, units)
    is_mw = m["unit"] == "MW"
    digits = 0 if is_mw else 1
    mean_c, p10_c, p90_c = EC_COLS[var]
    fig = base_fig(f"EC46 {m['label'].lower()}, {zone} - ensemble mean with 10-90% range", m["unit"], height=420)
    ec = load_ec46(units, var)
    if ec.empty:
        msg = "No EC46 MW yet - run ecmwf_mw.py." if is_mw else "No EC46 data yet - run ec46_scraper.py first."
        return finish_fig(fig, zone, m), [], [], [], msg
    fetches = sorted(ec["fetched_on"].unique())
    main_f = erun if erun in fetches else fetches[-1]
    older = [f for f in fetches if f < main_f]
    if cmp in (CMP_PREV, None):
        cmp_f = older[-1] if older else None
    elif cmp == CMP_WEEK:
        target = (pd.Timestamp(main_f) - pd.Timedelta(days=7)).date().isoformat()
        cand = [f for f in older if abs((pd.Timestamp(f) - pd.Timestamp(target)).days) <= 2]
        cmp_f = min(cand, key=lambda f: abs((pd.Timestamp(f) - pd.Timestamp(target)).days)) if cand else None
    else:
        cmp_f = cmp if (cmp in fetches and cmp != main_f) else None

    cur = ec[(ec.fetched_on == main_f) & (ec.zone == zone)].sort_values("day")
    if mode == "diff":
        fig = base_fig(f"EC46 change in {m['label'].lower()}, {zone} - {ec46_label(main_f)} minus "
                       f"{ec46_label(cmp_f) if cmp_f else '(nothing to compare)'}",
                       m["unit"], height=420)
        if cmp_f is not None:
            other = ec[(ec.fetched_on == cmp_f) & (ec.zone == zone)][["day", mean_c]]
            j = cur[["day", mean_c]].merge(other, on="day", suffixes=("", "_c"))
            j[mean_c] = j[mean_c] - j[mean_c + "_c"]
            add_diff(fig, j.dropna(subset=[mean_c]), mean_c, m, "Change", x="day", bars=True)
    elif cur[mean_c].notna().any():
        fig.add_trace(go.Scatter(x=cur["day"], y=cur[p90_c], mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=cur["day"], y=cur[p10_c], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                 hovertemplate=f"10%: %{{y:{m['fmt']}}}<extra></extra>"))
    if mode != "diff":
        if cmp_f is not None:
            add_series(fig, ec[(ec.fetched_on == cmp_f) & (ec.zone == zone)], mean_c, ec46_label(cmp_f),
                       PALETTE[1], dash="dash", width=2, m=m, x="day", label_shift=-10)
        add_series(fig, cur, mean_c, ec46_label(main_f), PALETTE[0], width=2.5, m=m, x="day",
                   label_shift=8 if cmp_f else 0)
    finish_fig(fig, zone, m, now_line=False, diff=mode == "diff")

    def weekly(fetch):
        d = ec[ec.fetched_on == fetch].copy()
        d["week"] = d["day"] - pd.to_timedelta(d["day"].dt.weekday, unit="D")
        return d.groupby(["zone", "week"])[mean_c].mean().unstack("week")

    wk = weekly(main_f)
    nmem = ec[ec.fetched_on == main_f]["members"].max() if "members" in ec else None
    nmem = f"{int(nmem)} " if nmem is not None and pd.notna(nmem) else ""
    note = (f"{ec46_label(main_f)} (first seen that day). Daily ensemble mean of {nmem}members; table = "
            f"ISO-week averages (Mon-Sun; first and last week may be partial).")
    if cmp_f:
        note += f" Orange dashed = {ec46_label(cmp_f)}; table rows 'Δ' = change vs it (orange higher, blue lower)."
    else:
        want = "about a week earlier" if cmp == CMP_WEEK else "earlier"
        note += f" No EC46 run from {want} than {pd.Timestamp(main_f):%d %b} stored yet - nothing to compare."
    if is_mw:
        note += " MW = daily average power, same model as the hourly MW."
    if mode == "diff":
        if cmp_f is None:
            return fig, [], [], [], note
        wk = (wk - weekly(cmp_f)).dropna(axis=1, how="all")
        rows, cols, style = heat_table(wk, _week_label, digits, True)
        return fig, rows, cols, style, note
    # values + a change row under each country
    wk = wk.dropna(how="all").sort_index(axis=1)
    rows, cols, style = heat_table(wk, _week_label, digits, False)
    if cmp_f is None:
        return fig, rows, cols, style, note
    dk = (wk - weekly(cmp_f).reindex(index=wk.index, columns=wk.columns))
    ids = [c["id"] for c in cols[1:]]
    out, out_style = [], [s_ for s_ in style if "row_index" not in s_.get("if", {})]
    for i, (zone_, r) in enumerate(wk.iterrows()):
        base = rows[i]
        out.append(base)
        for s_ in style:  # move the value-row shading to the new row index
            if s_.get("if", {}).get("row_index") == i:
                out_style.append({**s_, "if": {**s_["if"], "row_index": len(out) - 1}})
        drow = {"zone": f"{zone_} Δ vs {pd.Timestamp(cmp_f):%d %b}"}
        for cid, w in zip(ids, wk.columns):
            v = dk.loc[zone_, w]
            drow[cid] = None if pd.isna(v) else round(float(v), digits)
            if pd.notna(v) and abs(v) > 0:
                out_style.append({"if": {"row_index": len(out), "column_id": cid},
                                  "backgroundColor": "rgba(235,104,52,0.18)" if v > 0 else "rgba(42,120,214,0.18)"})
        out.append(drow)
        out_style.append({"if": {"row_index": len(out) - 1}, "fontStyle": "italic", "color": MUTED})
    return fig, out, cols, out_style, note


# ---------------------------------------------------------------- ensembles (ens_scraper.py)
ENS_COL = {("temp_c", "weather"): "temp_c", ("temp_c", "mw"): "temp_c",
           ("wind100_ms", "weather"): "wind_ms", ("wind100_ms", "mw"): "wind_mw",
           ("solar_wm2", "weather"): "solar_wm2", ("solar_wm2", "mw"): "solar_mw"}


def ens_label(model, ts) -> str:
    return f"{ENS_MODELS.get(model, model)} {pd.Timestamp(ts):%a %d %b %Hz}"


def _ens_cache():
    load()
    c = _cache.setdefault("ens", {})
    if c.get("key") != _cache.get("key"):
        c.clear()
        c["key"] = _cache.get("key")
    return c


def ens_runs(model) -> list:
    """Stored ensemble run times (UTC Timestamps) for one model, newest first."""
    c = _ens_cache()
    if "runs" not in c:
        conn = _connect()
        try:
            has = conn.execute("SELECT 1 FROM sqlite_master WHERE name='ens_fetch_log'").fetchone()
            r = pd.read_sql_query("SELECT model, run_time, zones, status FROM ens_fetch_log WHERE zones > 0",
                                  conn) if has else pd.DataFrame(columns=["model", "run_time", "zones", "status"])
        finally:
            conn.close()
        r["run_time"] = pd.to_datetime(r["run_time"], utc=True)
        c["runs"] = r
    r = c["runs"]
    return sorted(r[r.model == model].run_time, reverse=True)


def ens_hourly(model, ts) -> pd.DataFrame:
    c = _ens_cache()
    k = ("h", model, pd.Timestamp(ts).isoformat())
    if k not in c:
        conn = _connect()
        try:
            d = pd.read_sql_query("SELECT * FROM ens_hourly WHERE model=? AND run_time=?", conn,
                                  params=[model, pd.Timestamp(ts).isoformat()])
        finally:
            conn.close()
        d["valid_time"] = pd.to_datetime(d["time_utc"], utc=True).dt.tz_convert(TZ)
        c[k] = d
    return c[k]


def ens_daily(model, ts) -> pd.DataFrame:
    """Daily statistics across members (exact: from the per-member daily means)."""
    c = _ens_cache()
    k = ("d", model, pd.Timestamp(ts).isoformat())
    if k not in c:
        conn = _connect()
        try:
            d = pd.read_sql_query("SELECT * FROM ens_member_daily WHERE model=? AND run_time=?", conn,
                                  params=[model, pd.Timestamp(ts).isoformat()])
        finally:
            conn.close()
        out = []
        for q in ("temp_c", "wind_ms", "solar_wm2", "wind_mw", "solar_mw"):
            g = d.groupby(["zone", "day"])[q]
            out.append(pd.DataFrame({q: g.mean(), q + "_p10": g.quantile(0.1), q + "_p90": g.quantile(0.9),
                                     q + "_min": g.min(), q + "_max": g.max()}))
        res = pd.concat(out, axis=1).reset_index() if out and not d.empty else pd.DataFrame()
        if not res.empty:
            res["day"] = pd.to_datetime(res["day"])
        c[k] = res
    return c[k]


def ens_resolve(cmp, ts, all_runs):
    older = [r for r in all_runs if r < ts]
    if cmp in (CMP_PREV, None):
        return older[0] if older else None
    if cmp in (CMP_P00, CMP_P12):
        h = 0 if cmp == CMP_P00 else 12
        hit = [r for r in older if r.hour == h]
        return hit[0] if hit else None
    if cmp == "day":
        hit = [r for r in older if abs(r - (ts - pd.Timedelta(days=1))) <= pd.Timedelta(hours=3)]
        return hit[0] if hit else None
    if cmp == "none":
        return None
    try:
        t = pd.Timestamp(cmp)
        return t if t in older or t in all_runs and t != ts else None
    except Exception:
        return None


@app.callback(Output("n-run", "options"), Output("n-run", "value"), Output("n-cmp", "options"),
              Input("tab", "value"), Input("n-model", "value"), Input("n-run", "value"), Input("upd-done", "data"))
def ens_options(tab, model, cur, _done):
    if tab != "ens":
        raise PreventUpdate
    rs = ens_runs(model)
    opts = [{"label": ens_label(model, r), "value": r.isoformat()} for r in rs]
    vals = [o["value"] for o in opts]
    sel = cur if (cur in vals and ctx.triggered_id not in ("n-model", "upd-done")) else (vals[0] if vals else None)
    ts = pd.Timestamp(sel) if sel else None

    def lab(text, kind):
        if ts is None:
            return text
        c = ens_resolve(kind, ts, rs)
        return f"{text} ({c:%a %d %b %Hz})" if c is not None else f"{text} (none stored)"
    cmp_opts = ([{"label": lab("Previous run", CMP_PREV), "value": CMP_PREV},
                 {"label": lab("Previous 00z run", CMP_P00), "value": CMP_P00},
                 {"label": lab("Previous 12z run", CMP_P12), "value": CMP_P12},
                 {"label": lab("1 day earlier", "day"), "value": "day"},
                 {"label": "Nothing", "value": "none"}] + opts)
    return opts, sel, cmp_opts


def _ens_day_table(dd, cd, zone, col, digits):
    """Selected country: rows Mean/P10/P90/Min/Max (+ change rows) x days."""
    a = dd[dd.zone == zone].set_index("day").sort_index()
    if a.empty or col not in a:
        return [], [], []
    days = list(a.index)
    ids = {d: "c" + pd.Timestamp(d).strftime("%Y%m%d") for d in days}
    cols = [{"name": "", "id": "stat"}] + [{"name": pd.Timestamp(d).strftime("%a %d"), "id": ids[d],
                                            "type": "numeric"} for d in days]
    rows, style = [], [{"if": {"column_id": "stat"}, "textAlign": "left", "fontWeight": 600}]
    stats = [("Mean", ""), ("P10 (low case)", "_p10"), ("P90 (high case)", "_p90"), ("Min", "_min"), ("Max", "_max")]
    for name, suf in stats:
        rows.append({"stat": name, **{ids[d]: (None if pd.isna(a.loc[d, col + suf]) else
                                               round(float(a.loc[d, col + suf]), digits)) for d in days}})
    if cd is not None and not cd.empty:
        b = cd[cd.zone == zone].set_index("day")
        for name, suf in stats[:3]:
            r = {"stat": f"Δ {name.split(' ')[0]} vs compare"}
            for d in days:
                v = (a.loc[d, col + suf] - b.loc[d, col + suf]) if d in b.index else float("nan")
                r[ids[d]] = None if pd.isna(v) else round(float(v), digits)
                if pd.notna(v) and v != 0:
                    style.append({"if": {"row_index": len(rows), "column_id": ids[d]},
                                  "backgroundColor": "rgba(235,104,52,0.18)" if v > 0 else "rgba(42,120,214,0.18)"})
            style.append({"if": {"row_index": len(rows)}, "fontStyle": "italic", "color": MUTED})
            rows.append(r)
    spread = (a[col + "_p90"] - a[col + "_p10"])
    hi = float(spread.max()) if spread.notna().any() else 1
    for d in days:  # shade the P10/P90 rows by the spread: darker = more uncertain
        v = spread.get(d)
        if pd.notna(v) and hi > 0:
            for ri in (1, 2):
                style.append({"if": {"row_index": ri, "column_id": ids[d]},
                              "backgroundColor": f"rgba(42,120,214,{0.05 + 0.4 * v / hi:.2f})"})
    return rows, cols, style


@app.callback(Output("g-ens", "figure"), Output("t-ens", "data"), Output("t-ens", "columns"),
              Output("t-ens", "style_data_conditional"), Output("t-ens-c", "data"), Output("t-ens-c", "columns"),
              Output("t-ens-c", "style_data_conditional"), Output("n-note", "children"),
              Output("n-showing", "children"), Output("n-t1-title", "children"), Output("n-t2-title", "children"),
              Input("zone", "value"), Input("n-model", "value"), Input("n-run", "value"), Input("n-cmp", "value"),
              Input("n-var", "value"), Input("units", "value"), Input("view", "value"), Input("tab", "value"))
def ensembles(zone, model, run, cmp, var, units, view, tab):
    if tab != "ens":
        raise PreventUpdate
    m = meta(var, units)
    is_mw = m["unit"] == "MW"
    digits = 0 if is_mw else 1
    col = ENS_COL[(var, "mw" if is_mw else "weather")]
    diff = view == "diff"
    name = ENS_MODELS.get(model, model)
    fig = base_fig(f"{name} {m['label'].lower()}, {zone}", m["unit"], height=440)
    empty = (finish_fig(fig, zone, m), [], [], [], [], [], [])
    rs = ens_runs(model)
    if not rs or not run:
        return (*empty, f"No {name} runs stored yet - run ens_scraper.py (or Check for new data).", [], "", "")
    ts = pd.Timestamp(run)
    cts = ens_resolve(cmp, ts, rs)
    h = ens_hourly(model, ts)
    hz = h[h.zone == zone].sort_values("valid_time")
    hc = ens_hourly(model, cts) if cts is not None else None
    hcz = hc[hc.zone == zone].sort_values("valid_time") if hc is not None else None
    nmem = int(hz["members"].max()) if not hz.empty else 0

    # deterministic run of the same model and run time (or the latest before)
    df = load()
    det_model = ENS_DET[model]
    det_runs = [r for r in runs(df, det_model) if rk_ts(r) <= ts] if not df.empty else []
    det = det_runs[0] if det_runs else None

    banner = [html.Span("Showing", style={"fontSize": "13px", "color": MUTED, "fontWeight": 600}),
              pill(ens_label(model, ts), PALETTE[0])]
    if cts is not None:
        banner += [html.Span("compared with", style={"fontSize": "13px", "color": MUTED}),
                   pill(ens_label(model, cts), PALETTE[1], solid=False)]
    if det is not None:
        banner += [html.Span("· line:", style={"fontSize": "13px", "color": MUTED}),
                   html.Span(run_label(det), style={"fontSize": "13px", "fontWeight": 600})]
    if rs[0] != ts:
        banner.append(html.Span(f"(not the newest {name} run - newest is {rs[0]:%a %d %b %Hz})",
                                style={"fontSize": "13px", "color": "#b45309"}))

    title = f"{name} {m['label'].lower()}, {zone} · {ens_label(model, ts)}"
    if diff:
        title = f"{name} change in {m['label'].lower()}, {zone} · {ens_label(model, ts)} minus " + \
            (ens_label(model, cts) if cts is not None else "(nothing to compare)")
    fig = base_fig(title, m["unit"], height=440)
    if hz.empty or hz[col].isna().all():
        msg = f"No {m['label'].lower()} for {zone} in this run" + (
            " - no MW fit for this country." if is_mw else ".")
        return (finish_fig(fig, zone, m), [], [], [], [], [], [], msg, banner, "", "")

    x = hz["valid_time"]
    if diff:
        if hcz is not None and not hcz.empty:
            j = hz[["valid_time", col, col + "_p10", col + "_p90"]].merge(
                hcz[["valid_time", col, col + "_p10", col + "_p90"]], on="valid_time", suffixes=("", "_c"))
            for c_ in (col, col + "_p10", col + "_p90"):
                j[c_] = j[c_] - j[c_ + "_c"]
            add_diff(fig, j, col, m, "Change in mean")
            for suf, nm, dash in (("_p10", "P10", "dot"), ("_p90", "P90", "dash")):
                fig.add_trace(go.Scatter(x=j["valid_time"], y=j[col + suf], mode="lines", name=f"Change {nm}",
                                         line=dict(color=MUTED, width=1, dash=dash),
                                         hovertemplate=f"Change {nm}: %{{y:+{m['fmt']}}} {m['unit']}<extra></extra>"))
    else:
        for lo, hi, alpha, nm in ((col + "_min", col + "_max", 0.10, "Min-max"),
                                  (col + "_p10", col + "_p90", 0.25, "P10-P90")):
            fig.add_trace(go.Scatter(x=x, y=hz[hi], mode="lines", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=hz[lo], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=f"rgba(42,120,214,{alpha})", name=nm,
                                     customdata=hz[hi],
                                     hovertemplate=f"{nm}: %{{y:{m['fmt']}}} - %{{customdata:{m['fmt']}}} "
                                                   f"{m['unit']}<extra></extra>"))
        if hcz is not None and not hcz.empty:
            add_series(fig, as_var(hcz, col, var), var, f"Mean {cts:%d %b %Hz}", PALETTE[1],
                       dash="dash", width=1.8, m=m, label_shift=-10)
        if det is not None:
            dd_ = data(units, var, [det])
            dd_ = dd_[dd_.zone == zone] if not dd_.empty else dd_
            if not dd_.empty:
                add_series(fig, dd_[dd_.valid_time <= x.max()], var, run_label(det), INK, dash="dot",
                           width=1.4, m=m, label_shift=22)
        add_series(fig, as_var(hz, col, var), var, f"Mean {ts:%d %b %Hz}", PALETTE[0], width=2.6, m=m,
                   label_shift=8 if hcz is not None else 0)
        if units == "mw" or var == "temp_c":
            add_normal(fig, zone, var, units, x.min(), x.max())
    finish_fig(fig, zone, m, diff=diff)

    # tables
    dd = ens_daily(model, ts)
    cd = ens_daily(model, cts) if cts is not None else None
    rows1, cols1, sty1 = _ens_day_table(dd, cd, zone, col, digits) if not dd.empty else ([], [], [])
    t1 = (f"{zone}: daily averages across the {nmem} members ({m['unit']}) - "
          f"P10/P90 shaded darker where the members disagree more")
    rows2, cols2, sty2, t2 = [], [], [], ""
    if not dd.empty and col in dd:
        wide = dd.pivot(index="zone", columns="day", values=col)
        lab = lambda c: pd.Timestamp(c).strftime("%a %d")
        if diff:
            if cd is not None and not cd.empty:
                wide = (wide - cd.pivot(index="zone", columns="day", values=col)).dropna(axis=1, how="all")
                rows2, cols2, sty2 = heat_table(wide, lab, digits, True)
                t2 = f"All countries: change in daily ensemble mean vs {ens_label(model, cts)} ({m['unit']})"
        else:
            rows2, cols2, sty2 = heat_table(wide, lab, digits, False)
            t2 = f"All countries: daily ensemble mean ({m['unit']})"
    note = (f"Blue band dark = 80 % of members (P10-P90), light = all members (min-max); thick blue = ensemble "
            f"mean; dotted black = the deterministic {MODELS[det_model]} run; orange dashed = compare run's mean. "
            f"{nmem} members, hourly. Weather is the average of 3 land points per country, scaled to the "
            f"{MODELS[det_model]} level; MW per member with the {MODELS[det_model]} MW fit.")
    if cmp not in ("none",) and cts is None:
        note += " Nothing stored to compare with yet."
    return fig, rows1, cols1, sty1, rows2, cols2, sty2, note, banner, t1, t2


# ---------------------------------------------------------------- main
def main():
    global DB_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8058)
    ap.add_argument("--db", default=None)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    DB_PATH = find_db(args.db)
    _cache["key"] = None
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print(f"ECMWF dashboard: http://127.0.0.1:{args.port}/   (database: {DB_PATH})")
        if args.open:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=True, reloader_type="stat")


if __name__ == "__main__":
    main()
