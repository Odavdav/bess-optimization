"""Bloc 1: perfect-foresight benchmark, one optimization per day vs one per year."""
import _setup  # noqa: F401

import pandas as pd

from bess import config
from bess.data import load_prices
from bess.optimization import optimize_dispatch

prices = load_prices()
rows = []
for year in (2024, 2025):
    p = prices[prices.index.tz_convert(config.MARKET_TZ).year == year]
    for horizon in ("day", "year"):
        r = optimize_dispatch(p, config.BATTERY, config.DEGRADATION, horizon=horizon)
        rows.append({"year": year, "horizon": horizon, "profit_keur": r.total_profit_eur / 1000,
                     "cycles": r.total_cycles})
        if horizon == "year" and year == 2025:  # week shown in the notebook
            r.schedule.to_csv(config.RESULTS / f"pf_schedule_{year}.csv")
        print(rows[-1])
pd.DataFrame(rows).to_csv(config.RESULTS / "perfect_foresight.csv", index=False)
