"""
Climate TRACE country-share backfill for 2015-2020 -- v2
BU7159 ESG Analytics | TCD MSc Business Analytics
------------------------------------------------------
Correct denominator: use the country CSV 2021 total as the base,
not the sum of our 11 firms (which severely undercounts the country).

Method:
  1. Fetch 2021 country totals from API (same endpoint, year=2021)
     -- but we already have these implicitly since the country CSV
     starts at 2015 and we fetched 2015-2020. Need 2021 separately.
  
  Alternative -- use 2021 from the facility CSV:
     firm_co2e_2021(country) = from ct_api_emissions but year 2021
     BUT ct_api_emissions.csv only has 2024.

  Correct approach:
     The ct_emissions_full.csv has 2021 firm totals (44 rows).
     The country CSV has country totals for 2015-2020.
     We need the 2021 country total to compute firm share.
     
     Use: request 2021 country data from the same API that gave us 2015-2020.
     Since the script runs on the user's machine (not here), add it to the 
     ct_download_countries.py output OR compute share differently:
     
     SIMPLEST CORRECT METHOD:
       share(firm, year) = firm_co2e(year) / country_co2e_2021
       where both numerator AND denominator come from the SAME 2021 data
       from ct_emissions_full (facility-level, confirmed)
       and the country 2021 total is fetched fresh.

Actually the cleanest approach given what we have:
  - We have ct_country_emissions_2015_2020.csv with real country totals
  - We need the 2021 country total as denominator
  - Fetch it with one more API call: same script, year=2021 only

"""
import requests, time
import pandas as pd
from pathlib import Path
import sys

HERE = Path(__file__).parent

def find_root():
    for p in [HERE] + [HERE.parents[i] for i in range(4)]:
        if (p / "outputs" / "csv" / "ct_emissions.csv").exists():
            return p
    for p in [
        Path("E:/Trinity College Dublin/MSc_Business_Analytics/ESG ANALYTICS/Group_Assignment"),
        Path.cwd(), Path.cwd().parent,
    ]:
        if (p / "outputs" / "csv" / "ct_emissions.csv").exists():
            return p
    print("ERROR: Cannot find project root"); sys.exit(1)

ROOT         = find_root()
FACILITY_CSV = HERE / "ct_emissions_full.csv"       # firm totals 2021-2024
COUNTRY_CSV  = HERE / "ct_country_emissions_2015_2020.csv"  # country 2015-2020
OUT_FILE     = HERE / "ct_emissions_2015_2024.csv"
OUT_PANEL    = ROOT / "outputs" / "csv" / "ct_emissions_2015_2024.csv"

BASE_URL = "https://api.climatetrace.org/v6"
SLEEP    = 0.4

KEY_COUNTRIES = [
    "USA","NOR","GBR","SAU","IRQ","NGA","AGO","AUS","CAN",
    "RUS","QAT","DZA","LBY","EGY","ARE","NLD","DEU","ITA",
    "ESP","FRA","BRA","KAZ","MYS","IDN","COL","ECU","PER",
    "OMN","KWT","ZAF","MNG",
]
SECTORS = [
    "oil-and-gas-production","oil-and-gas-refining","oil-and-gas-transport",
    "coal-mining","iron-and-steel","aluminum","copper-mining","chemicals",
]

FIRM_API_CSV = HERE / "ct_api_emissions.csv"  # firm x country x 2024


def fetch_country_2021():
    """Fetch 2021 country totals -- these are the correct denominators."""
    print("Fetching 2021 country totals (base year for shares)...")
    records = []
    total = len(KEY_COUNTRIES) * len(SECTORS)
    done  = 0
    for country in KEY_COUNTRIES:
        for sector in SECTORS:
            done += 1
            try:
                r = requests.get(f"{BASE_URL}/country/emissions",
                                 params={"countries": country,
                                         "sectors":   sector,
                                         "since": 2021, "to": 2021},
                                 timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data and isinstance(data, list):
                        for item in data:
                            if item.get("country") == country:
                                co2e = item.get("emissions",{}).get("co2e_100yr")
                                if co2e and float(co2e) > 0:
                                    records.append({
                                        "iso3_country": country,
                                        "sector":       sector,
                                        "co2e_tonnes":  float(co2e),
                                    })
                time.sleep(SLEEP)
            except Exception as e:
                print(f"  [WARN] {country}/{sector}/2021: {e}")

        if done % 80 == 0:
            print(f"  [{done}/{total}] {country}")

    df = pd.DataFrame(records)
    print(f"2021 country records: {len(df)}")
    return df


def main():
    print("=" * 60)
    print("CT Country-Share Backfill v2 -- correct denominators")
    print("=" * 60)

    for f in [FACILITY_CSV, COUNTRY_CSV, FIRM_API_CSV]:
        if not f.exists():
            print(f"ERROR: {f} not found"); sys.exit(1)

    facility_df = pd.read_csv(FACILITY_CSV)
    country_df  = pd.read_csv(COUNTRY_CSV)
    api_df      = pd.read_csv(FIRM_API_CSV)   # firm x country x 2024

    # Step 1: Get 2021 country totals (correct denominators)
    country_2021_path = HERE / "ct_country_2021.csv"
    if country_2021_path.exists():
        print("Loading cached 2021 country totals...")
        country_2021_df = pd.read_csv(country_2021_path)
    else:
        country_2021_df = fetch_country_2021()
        country_2021_df.to_csv(country_2021_path, index=False)
        print(f"Saved 2021 country totals: {country_2021_path}")

    if country_2021_df.empty:
        print("ERROR: Could not fetch 2021 country totals"); sys.exit(1)

    # Country total 2021 (sum across sectors per country)
    country_total_2021 = (country_2021_df.groupby("iso3_country")["co2e_tonnes"]
                          .sum().reset_index()
                          .rename(columns={"co2e_tonnes": "country_total_2021"}))
    print(f"\nCountry totals 2021 (sample):")
    print(country_total_2021.nlargest(5, "country_total_2021").to_string(index=False))

    # Step 2: Firm emissions by country from 2024 API data
    # (best country breakdown we have -- ownership is stable)
    firm_country = (api_df.groupby(["firm_name","country"])["co2e_tonnes"]
                    .sum().reset_index()
                    .rename(columns={"country": "iso3_country",
                                     "co2e_tonnes": "firm_co2e_2024"}))

    # Step 3: Firm share = firm_2024(country) / country_2021(country)
    # Use 2021 country total as the stable denominator
    shares = firm_country.merge(country_total_2021, on="iso3_country", how="left")
    shares["share"] = (shares["firm_co2e_2024"] /
                       shares["country_total_2021"].clip(lower=1))

    # Sanity check: cap share at 50% per country (no single firm > 50% of a country)
    shares["share"] = shares["share"].clip(upper=0.5)

    # Remove negligible shares
    shares = shares[shares["share"] > 0.0001]

    print(f"\nFirm-country shares after capping at 50%:")
    for firm in sorted(shares["firm_name"].unique()):
        top = (shares[shares["firm_name"]==firm]
               .nlargest(3,"share")[["iso3_country","share"]])
        print(f"  {firm}: " +
              ", ".join(f"{r['iso3_country']}={r['share']:.2%}"
                        for _, r in top.iterrows()))

    # Step 4: Apply shares to pre-2021 country totals
    country_totals = (country_df.groupby(["iso3_country","year"])["co2e_tonnes"]
                      .sum().reset_index())

    records = []
    for _, srow in shares.iterrows():
        firm    = srow["firm_name"]
        country = srow["iso3_country"]
        share   = srow["share"]

        c_rows = country_totals[country_totals["iso3_country"] == country]
        for _, crow in c_rows.iterrows():
            est = crow["co2e_tonnes"] * share
            if est > 0:
                records.append({
                    "firm_name":         firm,
                    "year":              int(crow["year"]),
                    "co2e_tonnes":       est,
                    "estimation_method": "country_share_proxy",
                })

    if not records:
        print("ERROR: No pre-2021 records generated"); sys.exit(1)

    pre_df = (pd.DataFrame(records)
              .groupby(["firm_name","year"])["co2e_tonnes"]
              .sum().reset_index())
    pre_df["estimation_method"] = "country_share_proxy"

    print(f"\nPre-2021 proxy records: {len(pre_df)}")
    pivot_pre = pre_df.pivot(index="firm_name", columns="year",
                             values="co2e_tonnes") / 1e6
    print("Pre-2021 CO2e (Mt):")
    print(pivot_pre.round(1).to_string())

    # Step 5: Combine
    fac = facility_df[["firm_name","year","co2e_tonnes"]].copy()
    fac["estimation_method"] = "facility_direct"
    combined = (pd.concat([fac, pre_df], ignore_index=True)
                .sort_values("estimation_method")
                .drop_duplicates(subset=["firm_name","year"], keep="first")
                .sort_values(["firm_name","year"])
                .reset_index(drop=True))
    combined["emissions_delta_pct"] = (
        combined.groupby("firm_name")["co2e_tonnes"]
        .pct_change().round(4)
    )

    pivot = (combined.groupby(["firm_name","year"])["co2e_tonnes"]
             .sum().unstack(fill_value=0) / 1e6)
    print(f"\nFull panel (Mt CO2e) [2015-2020 proxy | 2021-2024 direct]:")
    print(pivot.round(1).to_string())
    print(f"\nPanel rows: {len(combined)}, years: {combined['year'].min()}-{combined['year'].max()}")

    combined.to_csv(OUT_FILE, index=False)
    combined.to_csv(OUT_PANEL, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(f"Saved: {OUT_PANEL}")


if __name__ == "__main__":
    main()