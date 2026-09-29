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
- `fetch_nordpool_zones.py` - Nord Pool's own day-ahead prices for tomorrow,
  today and the last 4 days, for every zone except GB (cloud twin of the
  local `fetch_nordpool_bridge.py`). Two reasons: SYS (Nordic system price)
  and TEL exist only at Nord Pool, so the price table can only show those
  columns (and its "minus SYS" view) with this step; and Nord Pool publishes
  tomorrow's prices a little before ENTSO-E, so they show up here first.
  Saved as `source='nordpool'`, never over an official ENTSO-E price - the
  next ENTSO-E snapshot replaces the stand-ins (SYS and TEL stay).
- `fetch_gb_apx.py` - GB's price from 1 Jan 2021 onwards (ENTSO-E has
  published no GB price since 31 Dec 2020). Uses Elexon's free public API
  (APX market index), converted GBP -> EUR with ECB daily rates, for EVERY
  complete day - one consistent source, so years are like-for-like on the
  year-comparison page. Stored as `source='elexon_apx'`: a close proxy for
  the N2EX auction, not the N2EX auction itself. Finished days are not
  fetched again, so most runs do almost nothing. Also fixes 2020: ENTSO-E
  quotes GB in POUNDS, and those rows had been stored as if they were euros.
  Once (then it does nothing), every 2020 hour APX has a price for becomes
  the APX price in euros, and any other hour is converted pounds -> euros
  (`source='entsoe_converted'`). The original pound values are kept in the
  table `gb_entsoe_gbp_original`.
- `fetch_gb_bridge.py` - Nord Pool's N2EX auction for the last ~3 weeks plus
  tomorrow. Only a provisional price (what the price table shows for
  today/tomorrow, and a fallback for an hour APX has no trades in): once a
  day is complete `fetch_gb_apx.py` replaces it. Nord Pool's free API
  refuses anything older than about a month or two (HTTP 401), so it can't
  provide history.
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
