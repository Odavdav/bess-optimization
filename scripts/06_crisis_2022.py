"""Crisis replay: the same strategies on 2021-2022, when losses piled up.

Same engine, same settings as the main study (lambda chosen on 2024 is NOT reused: the
lambda rule and the drawdown rule are re-applied on May-Dec 2021, then tested on 2022).
Shows whether the drawdown budget protects during repeated losses.
"""
import _setup  # noqa: F401

import json
from multiprocessing import Pool

import pandas as pd

from bess import config
from bess.analysis import choose_drawdown, choose_lambda, drawdown_diagnostics, metrics, split_period
from bess.backtest import MarketData, central_setup, run_backtest
from bess.data import load_prices
from bess.scenarios import run_forecasts, to_daily_matrix

_MD = None


def task(args):
    global _MD
    if _MD is None:
        prices = load_prices(config.CRISIS_YEARS)
        _MD = MarketData(prices, run_forecasts(to_daily_matrix(prices)))
    kind, start, end = args
    if kind == "stateless":
        return run_backtest(_MD, central_setup(), start, end, lambdas=config.LAMBDAS, progress=True)
    return run_backtest(_MD, central_setup(), start, end, perfect_foresight=False,
                        deterministic=False, dd_pcts=config.DD_PCTS, progress=True)


if __name__ == "__main__":
    cal, test = config.CRISIS_CALIBRATION, config.CRISIS_TEST
    jobs = [("stateless", *cal), ("stateless", *test), ("drawdown", cal[0], test[1])]
    with Pool(2) as pool:
        results = pd.concat(pool.map(task, jobs))
    results.to_csv(config.RESULTS / "crisis_backtest.csv", index=False)

    m_cal = metrics(split_period(results, *cal))
    calibration = {"lambda": choose_lambda(m_cal),
                   "drawdown_pct": choose_drawdown(drawdown_diagnostics(split_period(results, *cal)))}
    (config.RESULTS / "crisis_calibration.json").write_text(json.dumps(calibration, indent=2))
    metrics(split_period(results, *test)).to_csv(config.RESULTS / "crisis_metrics_2022.csv")
    drawdown_diagnostics(split_period(results, *test)).to_csv(config.RESULTS / "crisis_drawdown_2022.csv")
    print(calibration)
    print(pd.read_csv(config.RESULTS / "crisis_metrics_2022.csv", index_col=0).round(1).to_string())
