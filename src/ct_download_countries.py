"""
Climate TRACE country-level data downloader v2
BU7159 ESG Analytics | TCD MSc Business Analytics
------------------------------------------------------
Correct API structure from probe:
  GET /v6/country/emissions?countries=USA&sectors=oil-and-gas-production&since=2015&to=2020
  Returns list of country objects:
  [{"country":"USA", "emissions":{"co2e_100yr": 7017414701}, ...}]

The "since/to" params filter the year. We call once per country+sector+year.
Emissions value is in "emissions.co2e_100yr" (not a nested list).

Usage:
    python ct_download_countries.py
Output:
    ct_country_emissions_2015_2020.csv
"""

import requests, time, json
import pandas as pd
from pathlib import Path

HERE     = Path(__file__).parent
OUT_FILE = HERE / "ct_country_emissions_2015_2020.csv"
BASE_URL = "https://api.climatetrace.org/v6"
SLEEP    = 0.4

KEY_COUNTRIES = [
    "USA","NOR","GBR","SAU","IRQ","NGA","AGO","AUS","CAN",
    "RUS","QAT","DZA","LBY","EGY","ARE","NLD","DEU","ITA",
    "ESP","FRA","BRA","KAZ","MYS","IDN","COL","ECU","PER",
    "OMN","KWT","ZAF","MNG",
]

SECTORS = [
    "oil-and-gas-production",
    "oil-and-gas-refining",
    "oil-and-gas-transport",
    "coal-mining",
    "iron-and-steel",
    "aluminum",
    "copper-mining",
    "chemicals",
]

TARGET_YEARS = list(range(2015, 2021))


def probe_structure():
    """Print raw response to confirm field names."""
    print("--- Probe: USA oil-and-gas-production 2018 ---")
    r = requests.get(f"{BASE_URL}/country/emissions",
                     params={"countries":"USA",
                             "sectors":"oil-and-gas-production",
                             "since":2018, "to":2018},
                     timeout=30)
    print(f"Status: {r.status_code}")
    try:
        data = r.json()
        # Show first item
        item = data[0] if isinstance(data, list) else data
        print(json.dumps(item, indent=2)[:800])
        # Extract the co2e value
        if isinstance(data, list) and len(data) > 0:
            co2e = data[0].get("emissions",{}).get("co2e_100yr")
            print(f"\nExtracted co2e_100yr: {co2e:,.0f} tonnes" if co2e else "co2e not found")
    except Exception as e:
        print(f"Error: {e}\n{r.text[:200]}")
    print()


def fetch_one(country, sector, year):
    """Fetch co2e_100yr for one country+sector+year combination."""
    try:
        r = requests.get(f"{BASE_URL}/country/emissions",
                         params={"countries": country,
                                 "sectors":   sector,
                                 "since":     year,
                                 "to":        year},
                         timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        # Find the matching country in response
        for item in data:
            if item.get("country") == country:
                co2e = item.get("emissions", {}).get("co2e_100yr")
                return float(co2e) if co2e else None
        # If no country match, take first item
        co2e = data[0].get("emissions", {}).get("co2e_100yr")
        return float(co2e) if co2e else None
    except Exception:
        return None


def main():
    probe_structure()

    total  = len(KEY_COUNTRIES) * len(SECTORS) * len(TARGET_YEARS)
    done   = 0
    records = []

    print(f"Fetching {total} combinations (~{total*SLEEP/60:.0f} min)...\n")

    for country in KEY_COUNTRIES:
        for sector in SECTORS:
            for year in TARGET_YEARS:
                done += 1
                co2e = fetch_one(country, sector, year)
                time.sleep(SLEEP)

                if co2e and co2e > 0:
                    records.append({
                        "iso3_country": country,
                        "sector":       sector,
                        "year":         year,
                        "co2e_tonnes":  co2e,
                    })

                if done % 100 == 0:
                    print(f"  [{done}/{total}] {country} {sector} {year} "
                          f"-> {len(records)} non-zero records so far")

    df = pd.DataFrame(records)
    print(f"\nTotal non-zero records: {len(df)}")

    if df.empty:
        print("No data -- check probe output above.")
        return

    pivot = (df.groupby(["iso3_country","year"])["co2e_tonnes"]
             .sum().unstack(fill_value=0) / 1e6)
    print("\nCountry totals Mt CO2e (non-zero only):")
    pivot = pivot.loc[(pivot > 0).any(axis=1)]
    print(pivot.round(1).to_string())

    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}  ({len(df)} rows)")
    print("\nNext: run ct_country_backfill.py to apply firm shares to these totals")


if __name__ == "__main__":
    main()