"""Raw downloads -> processed hourly prices, FCR block prices, FCR price curves, data summary."""
import _setup  # noqa: F401

import pandas as pd

from bess import config
from bess.data import daily_spread, read_energy_charts, read_fcr
from bess.fcr_impact import price_curves, read_bids, read_french_demand

config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
config.RESULTS.mkdir(parents=True, exist_ok=True)

fcr_parts, summary = [], []
for year in config.YEARS:
    prices = read_energy_charts(config.DATA_RAW / config.RAW_PRICE_FILE.format(year=year))
    out = prices.to_frame("price")
    out.index = out.index.strftime("%Y-%m-%d %H:%M")
    out.index.name = "date_utc"
    out.to_csv(config.DATA_PROCESSED / config.PRICE_FILE.format(year=year))

    fcr = read_fcr(config.DATA_RAW / config.RAW_FCR_FILE.format(year=year))
    fcr_parts.append(fcr)
    summary.append({
        "year": year,
        "hours": len(prices),
        "mean_price": prices.mean(),
        "std_price": prices.std(),
        "negative_hours": int((prices < 0).sum()),
        "median_daily_spread": daily_spread(prices).median(),
        "fcr_mean_eur_mw_h": fcr.mean(),
        "fcr_keur_per_mw_year": fcr.sum() * 4 / 1000,
    })
    print(f"{year}: {len(prices)} hours, {len(fcr)} FCR blocks")

# Crisis replay years: prices only (no FCR)
for year in config.CRISIS_YEARS:
    prices = read_energy_charts(config.DATA_RAW / config.RAW_PRICE_FILE.format(year=year))
    out = prices.to_frame("price")
    out.index = out.index.strftime("%Y-%m-%d %H:%M")
    out.index.name = "date_utc"
    out.to_csv(config.DATA_PROCESSED / config.PRICE_FILE.format(year=year))
    print(f"{year}: {len(prices)} hours (crisis replay)")

pd.concat(fcr_parts).to_frame().reset_index().to_csv(
    config.DATA_PROCESSED / config.FCR_FILE, index=False)
summary = pd.DataFrame(summary).set_index("year")
summary.to_csv(config.RESULTS / "data_summary.csv")
print(summary.round(1).to_string())

# FCR price curves: French price as a function of the MW added by the battery (bid replay).
# The bid lists are large and not redistributed: without them, keep the curves already in
# data/processed/.
bid_paths = [config.DATA_RAW / config.RAW_BIDS_FILE.format(year=y) for y in config.BID_YEARS]
if not all(p.exists() for p in bid_paths):
    print("FCR bid lists missing in data/raw/: keeping the existing", config.FCR_CURVES_FILE)
    raise SystemExit(0)
curves = []
for year in config.BID_YEARS:
    bids = read_bids(config.DATA_RAW / config.RAW_BIDS_FILE.format(year=year))
    demand = read_french_demand(config.DATA_RAW / config.RAW_FCR_FILE.format(year=year))
    curves.append(price_curves(bids, demand))
    print(f"{year}: FCR price curves rebuilt from {len(bids)} allocated bids")
curves = pd.concat(curves)
# Without the battery, the replay must give back the published French price
published = pd.concat(fcr_parts).reindex(curves.index)
gap = (curves["q0"] - published).abs()
print(f"Replay vs published price: {(gap < 0.01).mean():.1%} of blocks exact, max gap {gap.max():.2f} EUR/MW/h")
assert (gap < 0.01).mean() > 0.99 and gap.max() < 2, "bid replay does not match the published price"
curves.reset_index().to_csv(config.DATA_PROCESSED / config.FCR_CURVES_FILE, index=False)
