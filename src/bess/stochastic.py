"""Risk-averse day-ahead dispatch: one schedule for many price scenarios, mean-CVaR objective.

The schedule (charge, discharge) is decided before prices are known, so it is the
same for every scenario (non-anticipativity). Each scenario gives a different profit.

Objective:  maximize (1 - lam) * E[profit] + lam * CVaR_alpha(profit)
  - lam = 0: risk-neutral (expected profit only)
  - lam = 1: fully risk-averse (only the worst scenarios matter)
  - CVaR_alpha = average profit over the worst (1 - alpha) share of the scenarios

Requirements: pip install "pulp>=4" highspy
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pulp import HiGHS, LpMaximize, LpProblem, LpSolveStatus, lpSum

from .battery import Battery, DegradationModel
from .config import ALPHA
from .optimization import add_fcr


# ---------------------------------------------------------------------------
# Profit and risk measures for a FIXED schedule
# ---------------------------------------------------------------------------

def scenario_profits(
    schedule: pd.DataFrame, prices: pd.DataFrame, c_deg: float, dt_h: float = 1.0,
    fcr_price=None,
) -> np.ndarray:
    """Net profit of a fixed schedule in each price scenario (one value per column).

    fcr_price: hourly FCR price (EUR/MW/h) used to pay the reserved capacity, if any.
    """
    net_mw = (schedule["discharge_mw"] - schedule["charge_mw"]).to_numpy()
    revenue = prices.to_numpy().T @ net_mw * dt_h                # one revenue per scenario
    degradation = c_deg * schedule["discharge_mw"].sum() * dt_h  # same in every scenario
    fcr = 0.0
    if fcr_price is not None and "fcr_mw" in schedule:
        fcr = float(np.asarray(fcr_price) @ schedule["fcr_mw"].to_numpy() * dt_h)
    return revenue - degradation + fcr


def cvar(profits: np.ndarray, alpha: float = ALPHA) -> float:
    """Average profit over the worst (1 - alpha) share of the scenarios."""
    worst = np.sort(profits)[: max(1, int(round((1 - alpha) * len(profits))))]
    return float(worst.mean())


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StochasticResult:
    schedule: pd.DataFrame   # hourly: charge_mw, discharge_mw, energy_mwh, soc
    profits: np.ndarray      # profit in each optimization scenario
    lam: float
    alpha: float

    @property
    def expected_profit(self) -> float:
        return float(self.profits.mean())

    @property
    def cvar(self) -> float:
        return cvar(self.profits, self.alpha)


def optimize_day_cvar(
    scenarios: pd.DataFrame,
    battery: Battery,
    degradation: DegradationModel,
    lam: float = 0.0,
    alpha: float = ALPHA,
    dt_h: float = 1.0,
    time_limit_s: float = 60,
    cvar_floor: float | None = None,
    fcr_price=None,
) -> StochasticResult:
    """One schedule for the day, optimized over all scenarios with a mean-CVaR objective.

    scenarios:  hourly prices, one row per hour of the day, one column per scenario.
    cvar_floor: optional risk limit in EUR. If given, the schedule must satisfy
                CVaR_alpha(profit) >= cvar_floor (e.g. -2000 = "on the 10% worst
                scenarios, lose at most 2,000 EUR on average"). Doing nothing always
                satisfies a floor <= 0, so the problem stays feasible.
    fcr_price:  optional hourly FCR price (EUR/MW/h) used for the decision (a forecast).
                FCR revenue is the same in every scenario: it hedges the arbitrage risk.
    """
    if not 0 <= lam <= 1:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    prices = scenarios.to_numpy()          # hours x scenarios
    n_hours, n_scen = prices.shape
    hours, scens = range(n_hours), range(n_scen)
    p_max = battery.power_mw
    c_deg = degradation.cost_per_mwh_discharged(battery)

    prob = LpProblem("bess_dispatch_cvar", LpMaximize)

    # ---- Decision variables: the SAME for every scenario (decided before prices are known)
    charge = prob.add_variable_dicts("charge", hours, lowBound=0, upBound=p_max)
    discharge = prob.add_variable_dicts("discharge", hours, lowBound=0, upBound=p_max)
    energy = prob.add_variable_dicts(
        "energy", range(n_hours + 1),
        lowBound=battery.energy_min_mwh, upBound=battery.energy_max_mwh,
    )
    is_charging = prob.add_variable_dicts("is_charging", hours, cat="Binary")

    # ---- Optional FCR reservation (same constraints as the deterministic model)
    fcr_h, fcr_revenue = None, 0
    if fcr_price is not None:
        fcr_h, fcr_revenue = add_fcr(prob, scenarios.index, fcr_price, charge, discharge,
                                     energy, battery, p_max, dt_h)

    # ---- Profit in each scenario (a linear expression, not a variable)
    profit = {
        s: lpSum(prices[t, s] * (discharge[t] - charge[t]) * dt_h for t in hours)
           - c_deg * lpSum(discharge[t] for t in hours) * dt_h
           + fcr_revenue
        for s in scens
    }
    expected_profit = lpSum(profit[s] for s in scens) * (1 / n_scen)

    # ---- CVaR, linear formulation (Rockafellar & Uryasev):
    #   CVaR = max over eta of  eta - 1/((1-alpha) K) * sum_s max(0, eta - profit_s)
    #   eta ends up at the VaR (the profit threshold of the worst (1-alpha) share),
    #   shortfall_s = how far scenario s falls below eta (0 if it is above)
    eta = prob.add_variable("eta")  # free variable, in EUR
    shortfall = prob.add_variable_dicts("shortfall", scens, lowBound=0)
    for s in scens:
        prob += shortfall[s] - eta + profit[s] >= 0  # shortfall_s >= eta - profit_s
    cvar_expr = eta - lpSum(shortfall[s] for s in scens) * (1 / ((1 - alpha) * n_scen))

    # ---- Objective
    prob += (1 - lam) * expected_profit + lam * cvar_expr

    # ---- Optional risk limit (CVaR used as a constraint instead of a weight)
    if cvar_floor is not None:
        prob += cvar_expr >= min(cvar_floor, 0.0)

    # ---- Physical constraints (same as the deterministic model)
    prob += energy[0] == battery.initial_energy_mwh
    prob += energy[n_hours] == battery.initial_energy_mwh
    for t in hours:
        prob += energy[t + 1] == (
            energy[t]
            + battery.charge_efficiency * charge[t] * dt_h
            - discharge[t] * dt_h / battery.discharge_efficiency
        )
        prob += charge[t] <= p_max * is_charging[t]
        prob += discharge[t] <= p_max * (1 - is_charging[t])

    stats = prob.solve(HiGHS(msg=False, timeLimit=time_limit_s, gapRel=1e-4))
    if stats.status != LpSolveStatus.Optimal:
        raise RuntimeError(f"Solver status: {stats.status.name}")

    schedule = pd.DataFrame(
        {
            "charge_mw": [charge[t].value() for t in hours],
            "discharge_mw": [discharge[t].value() for t in hours],
            "energy_mwh": [energy[t + 1].value() for t in hours],
        },
        index=scenarios.index,
    )
    schedule["fcr_mw"] = [r.value() for r in fcr_h] if fcr_h is not None else 0.0
    schedule["soc"] = schedule["energy_mwh"] / battery.capacity_mwh
    profits = scenario_profits(schedule, scenarios, c_deg, dt_h, fcr_price=fcr_price)
    return StochasticResult(schedule, profits, lam, alpha)
