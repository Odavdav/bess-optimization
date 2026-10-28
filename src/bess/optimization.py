"""Deterministic battery dispatch, formulated as a MILP (PuLP 4 + HiGHS).

Used for perfect foresight (real prices) and for the deterministic strategy (forecast).
Optional FCR capacity reservation (revenue stacking) through `add_fcr`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pulp import HiGHS, LpMaximize, LpProblem, LpSolveStatus, lpSum

from .battery import Battery, DegradationModel
from .config import MARKET_TZ
from .data import check_hourly


@dataclass(frozen=True)
class DispatchResult:
    schedule: pd.DataFrame  # hourly: price, charge, discharge, energy, soc, profit
    daily: pd.DataFrame     # one row per local day: revenue, degradation cost, profit, cycles

    @property
    def total_profit_eur(self) -> float:
        return float(self.daily["net_profit_eur"].sum())

    @property
    def total_cycles(self) -> float:
        return float(self.daily["equivalent_cycles"].sum())


FCR_HOLD_H = 0.25  # FCR: hold full activation for 15 minutes in both directions
FCR_BLOCK_H = 4    # FCR is sold in 4-hour blocks of LOCAL time


def add_fcr(prob, index, fcr_price, charge, discharge, energy, battery, p_max, dt_h):
    """Add FCR capacity reservation to a dispatch model (one variable per 4-hour block).

    fcr_price: hourly FCR price (EUR/MW/h) aligned on `index`; each hour takes the price
               of its block. Returns (hourly reserved MW expressions, FCR revenue expression).

    Constraints for every hour t of block b, with fcr[b] MW reserved:
      - power:  charge[t] + fcr[b] <= Pmax  and  discharge[t] + fcr[b] <= Pmax
                (room for +fcr and -fcr on top of the arbitrage schedule)
      - energy: Emin + 0.25 h x fcr[b] <= energy <= Emax - 0.25 h x fcr[b],
                at the start and at the end of the hour (15 min of full activation)
    The energy exchanged during activations is neglected (average drift close to zero).
    """
    local = index.tz_convert(MARKET_TZ)
    keys = list(zip(local.date, local.hour // FCR_BLOCK_H))  # (local day, block) of each hour
    blocks = list(dict.fromkeys(keys))
    fcr = prob.add_variable_dicts("fcr", range(len(blocks)), lowBound=0, upBound=p_max)
    block_of = {key: i for i, key in enumerate(blocks)}
    fcr_h = [fcr[block_of[key]] for key in keys]  # reserved MW in each hour

    price = np.asarray(fcr_price, dtype=float)
    for t, r in enumerate(fcr_h):
        prob += charge[t] + r <= p_max
        prob += discharge[t] + r <= p_max
        for e in (energy[t], energy[t + 1]):
            prob += e - FCR_HOLD_H * r >= battery.energy_min_mwh
            prob += e + FCR_HOLD_H * r <= battery.energy_max_mwh
    revenue = lpSum(price[t] * fcr_h[t] * dt_h for t in range(len(fcr_h)))
    return fcr_h, revenue


def optimize_window(
    prices_window: pd.Series,
    battery: Battery,
    degradation: DegradationModel,
    dt_h: float = 1.0,
    time_limit_s: float = 600,
    fcr_price=None,
) -> pd.DataFrame:
    """Solve one window (one day or one whole year) with perfect foresight.

    The window starts and ends at the battery's initial energy level.
    fcr_price: optional hourly FCR price (EUR/MW/h) aligned on the window; if given,
               the battery can also reserve capacity for FCR (revenue stacking).
    """
    n = len(prices_window)
    prices = prices_window.to_numpy()
    hours = range(n)
    p_max = battery.power_mw
    c_deg = degradation.cost_per_mwh_discharged(battery)

    prob = LpProblem("bess_dispatch", LpMaximize)

    # ---- Decision variables ----
    charge = prob.add_variable_dicts("charge", hours, lowBound=0, upBound=p_max)        # MW
    discharge = prob.add_variable_dicts("discharge", hours, lowBound=0, upBound=p_max)  # MW
    energy = prob.add_variable_dicts(                                                     # MWh
        "energy", range(n + 1),
        lowBound=battery.energy_min_mwh, upBound=battery.energy_max_mwh,
    )
    is_charging = prob.add_variable_dicts("is_charging", hours, cat="Binary")  # 1 = charging

    # ---- Objective: arbitrage revenue minus degradation cost (+ FCR revenue) ----
    objective = lpSum(
        prices[t] * (discharge[t] - charge[t]) * dt_h - c_deg * discharge[t] * dt_h
        for t in hours
    )
    fcr_h = None
    if fcr_price is not None:
        fcr_h, fcr_revenue = add_fcr(prob, prices_window.index, fcr_price, charge, discharge,
                                     energy, battery, p_max, dt_h)
        objective = objective + fcr_revenue
    prob += objective

    # ---- Constraints ----
    # Start and end of the window at the initial energy level
    prob += energy[0] == battery.initial_energy_mwh
    prob += energy[n] == battery.initial_energy_mwh

    for t in hours:
        # Energy balance (losses on both charge and discharge)
        prob += energy[t + 1] == (
            energy[t]
            + battery.charge_efficiency * charge[t] * dt_h
            - discharge[t] * dt_h / battery.discharge_efficiency
        )
        # No simultaneous charge and discharge
        prob += charge[t] <= p_max * is_charging[t]
        prob += discharge[t] <= p_max * (1 - is_charging[t])

    # ---- Solve ----
    stats = prob.solve(HiGHS(msg=False, timeLimit=time_limit_s, gapRel=1e-4))
    if stats.status != LpSolveStatus.Optimal:
        raise RuntimeError(
            f"Window starting {prices_window.index[0]}: solver status {stats.status.name}"
        )

    sched = pd.DataFrame(
        {
            "price": prices,
            "charge_mw": [charge[t].value() for t in hours],
            "discharge_mw": [discharge[t].value() for t in hours],
            "energy_mwh": [energy[t + 1].value() for t in hours],  # energy at end of hour
        },
        index=prices_window.index,
    )
    sched["fcr_mw"] = [r.value() for r in fcr_h] if fcr_h is not None else 0.0
    sched["revenue_eur"] = sched["price"] * (sched["discharge_mw"] - sched["charge_mw"]) * dt_h
    sched["fcr_revenue_eur"] = (np.asarray(fcr_price) * sched["fcr_mw"] * dt_h
                                if fcr_price is not None else 0.0)
    sched["degradation_cost_eur"] = c_deg * sched["discharge_mw"] * dt_h
    sched["net_profit_eur"] = (sched["revenue_eur"] + sched["fcr_revenue_eur"]
                               - sched["degradation_cost_eur"])
    return sched


def optimize_dispatch(
    prices: pd.Series,
    battery: Battery,
    degradation: DegradationModel,
    horizon: str = "year",
    dt_h: float = 1.0,
) -> DispatchResult:
    """Perfect-foresight dispatch over the whole price series.

    horizon="year": one optimization over the full series (energy can be carried
                    from one day to the next: true theoretical ceiling).
    horizon="day":  one optimization per local market day, each day starting and
                    ending at the initial SoC (consistent with day-ahead bidding).
    """
    check_hourly(prices)
    local_days = prices.index.tz_convert(MARKET_TZ).date

    if horizon == "year":
        schedule = optimize_window(prices, battery, degradation, dt_h, time_limit_s=1800)
    elif horizon == "day":
        schedule = pd.concat(
            optimize_window(p_day, battery, degradation, dt_h)
            for _, p_day in prices.groupby(local_days)
        )
    else:
        raise ValueError(f"horizon must be 'year' or 'day', got {horizon!r}")

    schedule["soc"] = schedule["energy_mwh"] / battery.capacity_mwh

    # One row per local market day
    daily = schedule.groupby(local_days)[
        ["revenue_eur", "degradation_cost_eur", "net_profit_eur", "discharge_mw"]
    ].sum()
    daily = daily.rename(columns={"discharge_mw": "discharged_mwh"})
    daily["discharged_mwh"] *= dt_h
    daily["equivalent_cycles"] = daily["discharged_mwh"] / battery.usable_energy_mwh
    daily.index.name = "day"

    return DispatchResult(schedule=schedule, daily=daily)
