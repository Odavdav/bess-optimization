"""Bloc 2.1: forecast quality, bootstrap vs Gaussian scenarios, convergence study (on 2024)."""
import _setup  # noqa: F401

import numpy as np
import pandas as pd

from bess import config
from bess.backtest import load_market_data
from bess.convergence import run_study, summarize
from bess.scenarios import evaluate, generate_scenarios

md = load_market_data(with_fcr=False)
cal = config.CALIBRATION

# 1. Forecast and scenario quality, both Monte Carlo methods, on the calibration year
rows = [evaluate(md.forecasts, *cal, method=m) for m in ("bootstrap", "gaussian")]
quality = pd.DataFrame(rows).set_index("method")
quality.to_csv(config.RESULTS / "scenario_quality.csv")
print(quality.to_string())

# 2. Correlation of forecast errors between hours (why whole days are simulated)
errors = md.errors.loc[cal[0]:cal[1]]
errors.corr().to_csv(config.RESULTS / "error_correlation.csv")

# 3. Example day: scenarios of both methods for the same day
day = pd.Timestamp("2024-06-12")
example = {"actual": md.prices.reindex(
    generate_scenarios(md.forecasts, day, 1).scenarios.index)}
for method in ("bootstrap", "gaussian"):
    ds = generate_scenarios(md.forecasts, day, 30, method=method, rng=np.random.default_rng(1))
    example["forecast"] = ds.forecast
    for col in ds.scenarios:
        example[f"{method}_{col}"] = ds.scenarios[col]
pd.DataFrame(example).to_csv(config.RESULTS / "example_scenarios.csv")

# 4. Convergence: how many scenarios, which alpha
days = sorted(np.random.default_rng(config.SEED).choice(
    pd.date_range(*cal, freq="D"), 12, replace=False))
study = run_study(md, days, k_values=(10, 20, 30, 50, 100, 200, 500))
study.to_csv(config.RESULTS / "convergence.csv", index=False)
print(summarize(study).round(2).to_string())
