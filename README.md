# Online dashboards (cloud copy)

This is the small, cloud-only twin of 2 of the dashboards from the main
`entsoe_scraper` project - deployed on Streamlit Community Cloud from
this one repo, so they're reachable from any browser (including your
phone), no VPN or Tailscale needed.

- `price_table.py` - the hourly day-ahead price table (same app you
  already use locally). This is the deployment's home page.
- `pages/Year_comparison.py` - the year-over-year price comparison
  dashboard, added to this SAME deployment as a second page (Streamlit's
  `pages/` convention - it shows up in the sidebar, same URL). Trimmed to
  DE_LU and FR (Peter's pick), plus the GB-priced interconnectors
  (IFA/IFA2/ElecLink) and the JAO corridor auction tables - see that
  file's own top-of-file comment for exactly what was trimmed from the
  real `year_comparison.py` and why.

## The fetchers (all run on a schedule by `.github/workflows/fetch.yml`)

- `fetch_recent.py` - a few days of ALL zones' prices, for `price_table.py`.
- `fetch_recent_years.py` - full 2020-present history for just DE_LU, FR
  and GB, for `pages/Year_comparison.py`. Incremental (like the real
  project's `fetch_data.py`) - only the very first run does the full
  backfill.
- `fetch_gb_bridge.py` - bridges GB's price from Nord Pool's N2EX auction
  for the period ENTSO-E doesn't cover (ENTSO-E has published no GB price
  since 31 Dec 2020). Recent days only - a trailing 21-day window,
  refreshed every run. Nord Pool's free API refuses anything older than
  roughly a month or two (HTTP 401), so it can't do the history.
- `fetch_gb_apx.py` - fills GB's history from 1 Jan 2021 up to the start of
  the Nord Pool window, from Elexon's free public API (APX market index,
  converted GBP -> EUR with ECB daily rates). Stored as `source='elexon_apx'`
  - a close proxy for N2EX, not the N2EX auction itself; every run prints
  how far APX is from the real N2EX price over the last 28 days. Only
  fetches days that have no GB price yet, so most runs do nothing.
- `fetch_jao_corridors.py` - JAO corridor auction prices for the 8
  corridors the interconnector tables need (DE-FR/FR-DE plus the 3
  GB-interconnector corridor pairs), Monthly horizon only.

Both databases (`entsoe_data.db`, `jao_data.db`) are committed back to
the repo after every run - see the workflow file. They're deliberately
tiny (a handful of zones/corridors, not the real project's full scope),
comfortably under GitHub's 100MB per-file limit.

## If you ever need to fetch manually

    pip install -r requirements.txt
    set ENTSOE_API_KEY=your-key-here      (Windows)
    set JAO_API_KEY=your-key-here         (Windows - same token as jao_scraper/api_key.txt)
    export ENTSOE_API_KEY=your-key-here   (Mac/Linux)
    export JAO_API_KEY=your-key-here      (Mac/Linux)
    python fetch_recent.py
    python fetch_recent_years.py
    python fetch_gb_bridge.py
    python fetch_gb_apx.py
    python fetch_jao_corridors.py

## GitHub Actions secrets this workflow needs

- `ENTSOE_API_KEY` - already set up (used by `fetch_recent.py`).
- `JAO_API_KEY` - new, needed for `fetch_jao_corridors.py`. Repo Settings
  -> Secrets and variables -> Actions -> New repository secret. Paste the
  same token that's saved in `jao_scraper/api_key.txt`.

## Deployed at

https://entsoe-price-table.streamlit.app
