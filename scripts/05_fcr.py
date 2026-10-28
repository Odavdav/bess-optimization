"""Extension: revenue stacking, day-ahead arbitrage + FCR capacity (2024 and 2025)."""
import _setup  # noqa: F401

import json
from multiprocessing import Pool

import pandas as pd

from bess import config
from bess.backtest import central_setup, load_market_data, run_backtest

CAL = json.loads((config.RESULTS / "calibration.json").read_text())
_MD = None


def task(args):
    global _MD
    if _MD is None:
        _MD = load_market_data(with_fcr=True)
    kind, start, end = args
    setup = central_setup(fcr=True)
    if kind == "stateless":
        return run_backtest(_MD, setup, start, end, lambdas=sorted({0.0, CAL["lambda"]}),
                            fcr_only=True, progress=True)
    return run_backtest(_MD, setup, start, end, perfect_foresight=False, deterministic=False,
                        dd_pcts=[CAL["drawdown_pct"]], progress=True)


if __name__ == "__main__":
    cal, test = config.CALIBRATION, config.TEST
    jobs = [("stateless", *cal), ("stateless", *test), ("drawdown", cal[0], test[1])]
    with Pool(2) as pool:
        results = pd.concat(pool.map(task, jobs))
    results.to_csv(config.RESULTS / "fcr.csv", index=False)
    print(results.groupby(["strategy"])[["realized_eur", "fcr_revenue_eur"]].sum().div(1000).round(0))
