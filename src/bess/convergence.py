"""How many scenarios are needed? Convergence study of the mean-CVaR dispatch.

For each test day: optimize on K scenarios (several random draws per K), freeze the
schedule, evaluate it on a large independent set of draws. When K is large enough, the
out-of-sample objective stops improving and the gap with the in-sample objective
(what the optimizer believes) closes.
"""
import time

import numpy as np
import pandas as pd

from . import config
from .scenarios import sample_errors, to_hourly
from .stochastic import cvar, optimize_day_cvar, scenario_profits


def objective(profits: np.ndarray, lam: float, alpha: float) -> float:
    return (1 - lam) * profits.mean() + lam * cvar(profits, alpha)


def _hourly(central: np.ndarray, errors: np.ndarray, day) -> pd.DataFrame:
    scen = central + errors
    return pd.DataFrame({f"s{k}": to_hourly(scen[k], day) for k in range(len(scen))})


def run_study(md, days, k_values=(10, 20, 30, 50, 100, 200), alphas=(0.9, 0.95),
              lam=0.5, n_reps=5, n_eval=5000, method="gaussian", seed=config.SEED):
    bat, deg = config.BATTERY, config.DEGRADATION
    c_deg = deg.cost_per_mwh_discharged(bat)
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        day = pd.Timestamp(day)
        central = md.forecasts.loc[day, "arx"].to_numpy()
        pool = md.errors[md.errors.index < day].tail(config.ERROR_POOL_DAYS).to_numpy()
        evaluation = _hourly(central, sample_errors(pool, n_eval, method, rng), day)
        for k in k_values:
            for rep in range(n_reps):
                scen = _hourly(central, sample_errors(pool, k, method, rng), day)
                for alpha in alphas:
                    t0 = time.perf_counter()
                    res = optimize_day_cvar(scen, bat, deg, lam=lam, alpha=alpha)
                    solve_s = time.perf_counter() - t0
                    out = scenario_profits(res.schedule, evaluation, c_deg)
                    rows.append({"day": day, "K": k, "rep": rep, "alpha": alpha,
                                 "in_obj": objective(res.profits, lam, alpha),
                                 "out_obj": objective(out, lam, alpha),
                                 "solve_s": solve_s})
    return pd.DataFrame(rows)


def summarize(study: pd.DataFrame) -> pd.DataFrame:
    """Sum over days for each draw, then mean and spread over the draws."""
    per_rep = study.groupby(["alpha", "K", "rep"])[["in_obj", "out_obj"]].sum()
    per_rep["solve_s"] = study.groupby(["alpha", "K", "rep"])["solve_s"].mean()
    g = per_rep.groupby(["alpha", "K"])
    out = g.mean()
    out["out_obj_std"] = g["out_obj"].std().fillna(0)
    out["gap_pct"] = (out["in_obj"] - out["out_obj"]) / out["out_obj"].abs() * 100
    return out
