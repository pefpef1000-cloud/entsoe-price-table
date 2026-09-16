# Hourly day-ahead price table (online copy)

This is the small, cloud-only twin of the price table dashboard from the
main `entsoe_scraper` project.

- `price_table.py` - the Streamlit app (same one you already use locally).
- `fetch_recent.py` - fetches a few days of prices from ENTSO-E into
  `entsoe_data.db`, right here (not your big local database).
- `.github/workflows/fetch.yml` - runs `fetch_recent.py` on a schedule
  and commits the refreshed database, so the online dashboard stays
  current without needing your PC to be on.

## If you ever need to fetch manually

    pip install -r requirements.txt
    set ENTSOE_API_KEY=your-key-here      (Windows)
    export ENTSOE_API_KEY=your-key-here   (Mac/Linux)
    python fetch_recent.py

## Deployed at

https://entsoe-price-table.streamlit.app
