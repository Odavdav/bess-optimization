"""Bloc 3: daily backtest. Calibrate on 2024 (lambda, drawdown budget), test on 2025."""
import _setup  # noqa: F401

import json
from multiprocessing import Pool

import pandas as pd

from bess import config
from bess.analysis import choose_drawdown, choose_lambda, drawdown_diagnostics, metrics, split_period
from bess.backtest import central_setup, load_market_data, run_backtest

_MD = None


def task(args):
    global _MD
    if _MD is None:
        _MD = load_market_data(with_fcr=False)
    kind, start, end = args
    if kind == "stateless":  # independent days: can be split by period
        return run_backtest(_MD, central_setup(), start, end, lambdas=config.LAMBDAS, progress=True)
    return run_backtest(_MD, central_setup(), start, end, perfect_foresight=False,
                        deterministic=False, dd_pcts=config.DD_PCTS, progress=True)


if __name__ == "__main__":
    cal, test = config.CALIBRATION, config.TEST
    jobs = [("stateless", *cal), ("stateless", *test), ("drawdown", cal[0], test[1])]
    with Pool(2) as pool:
        results = pd.concat(pool.map(task, jobs))
    results.to_csv(config.RESULTS / "backtest.csv", index=False)

    # Calibration on 2024 only
    m_cal = metrics(split_period(results, *cal))
    lam = choose_lambda(m_cal)
    diag_cal = drawdown_diagnostics(split_period(results, *cal))
    dd = choose_drawdown(diag_cal)
    calibration = {"lambda": lam, "drawdown_pct": dd,
                   "rule_lambda": "most profitable with at most 1 loss day out of 7 (2024)",
                   "rule_drawdown": "most profitable budget never breached (2024)"}
    (config.RESULTS / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(calibration)

    # Test on 2025, nothing tuned any more
    m_test = metrics(split_period(results, *test))
    m_cal.to_csv(config.RESULTS / "metrics_2024.csv")
    m_test.to_csv(config.RESULTS / "metrics_2025.csv")
    diag_cal.to_csv(config.RESULTS / "drawdown_2024.csv")
    drawdown_diagnostics(split_period(results, *test)).to_csv(config.RESULTS / "drawdown_2025.csv")
    pd.set_option("display.width", 220)
    print(m_cal.round(2).to_string())
    print(m_test.round(2).to_string())
