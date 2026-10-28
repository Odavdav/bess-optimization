"""One daily backtest engine for every strategy and battery configuration.

Every day J, for the delivery day D = J+1, each strategy builds its schedule with the
information available before the auctions (FCR at 8:00, day-ahead at 12:00), then the
schedule is paid at the REAL prices of D. Each day starts and ends at the initial SoC.

Strategy names:
  perfect_foresight   optimized on the real prices (ceiling, not achievable)
  deterministic       optimized on the central ARX forecast
  stochastic_lam{x}   mean-CVaR on Monte Carlo scenarios, CVaR weight x
  drawdown_{p}        expected profit under a daily CVaR limit derived from a drawdown
                      budget of p x accumulated gains (path-dependent: run sequentially)
  fcr_only            whole power reserved for FCR, no arbitrage (reference, FCR runs only)

FCR price impact (FCR runs): the battery lowers the French FCR price. Decisions use the naive
FCR forecast times the factor f of month M-2 (last bids published); the reserved MW are then
paid the price of the replayed auction (see fcr_impact.py). Perfect foresight decides with the
replayed price at full power.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .battery import Battery, DegradationModel
from .data import fcr_hourly, fcr_hourly_forecast, load_fcr, load_fcr_curves, load_prices
from .fcr_impact import decision_factor, monthly_factors, realized_price
from .optimization import optimize_window
from .scenarios import run_forecasts, sample_errors, to_daily_matrix, to_hourly
from .stochastic import optimize_day_cvar, scenario_profits


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setup:
    """A battery configuration, with or without FCR."""
    name: str
    battery: Battery = config.BATTERY
    degradation: DegradationModel = config.DEGRADATION
    fcr: bool = False


def central_setup(fcr: bool = False) -> Setup:
    return Setup("central + FCR" if fcr else "central", fcr=fcr)


@dataclass
class MarketData:
    prices: pd.Series
    forecasts: pd.DataFrame
    errors: pd.DataFrame = field(init=False)
    fcr_blocks: pd.Series | None = None
    fcr_curves: pd.DataFrame | None = None
    fcr_factors: pd.Series = field(init=False, default=None)

    def __post_init__(self):
        self.errors = (self.forecasts["actual"] - self.forecasts["arx"]).unstack("hour")
        if self.fcr_curves is not None:
            self.fcr_factors = monthly_factors(self.fcr_curves)


def load_market_data(with_fcr: bool = True) -> MarketData:
    prices = load_prices()
    forecasts = run_forecasts(to_daily_matrix(prices))
    if not with_fcr:
        return MarketData(prices, forecasts)
    return MarketData(prices, forecasts, load_fcr(), load_fcr_curves())


def day_inputs(md: MarketData, day, n_scenarios=config.N_SCENARIOS,
               method=config.SCENARIO_METHOD, seed=config.SEED):
    """Central forecast, scenarios (known before the auction) and real prices of `day`.

    The scenarios are reproducible: the same day always gets the same draws.
    """
    day = pd.Timestamp(day)
    central = md.forecasts.loc[day, "arx"].to_numpy()
    pool = md.errors[md.errors.index < day].tail(config.ERROR_POOL_DAYS).to_numpy()
    rng = np.random.default_rng([seed, day.toordinal()])
    scen_24 = central + sample_errors(pool, n_scenarios, method, rng)
    scenarios = pd.DataFrame({f"s{k}": to_hourly(scen_24[k], day) for k in range(n_scenarios)})
    actual = md.prices.reindex(scenarios.index)
    return to_hourly(central, day), scenarios, actual


# ---------------------------------------------------------------------------
# Daily loop
# ---------------------------------------------------------------------------

def run_backtest(md: MarketData, setup: Setup, start: str, end: str, *,
                 perfect_foresight=True, deterministic=True, lambdas=(), dd_pcts=(),
                 fcr_only=False, progress=False) -> pd.DataFrame:
    """One row per (day, strategy). Drawdown strategies carry their state from `start`."""
    bat, deg = setup.battery, setup.degradation
    c_deg = deg.cost_per_mwh_discharged(bat)
    dd_floor = config.DD_FLOOR_EUR_PER_MWH * bat.capacity_mwh
    dd_state = {p: [0.0, 0.0] for p in dd_pcts}  # cumulative profit, peak
    rows = []
    days = pd.date_range(start, end, freq="D")

    for i, day in enumerate(days):
        central, scenarios, actual = day_inputs(md, day)
        fcr_observed = fcr_fc = fcr_pf = None
        if setup.fcr:
            fcr_observed = fcr_hourly(md.fcr_blocks, actual.index)  # published, without us
            f = decision_factor(md.fcr_factors, day, config.FCR_FACTOR_LAG_MONTHS)
            fcr_fc = fcr_hourly_forecast(md.fcr_blocks, actual.index, day) * f
            full = np.full(len(actual), bat.power_mw)
            fcr_pf = pd.Series(realized_price(md.fcr_curves, actual.index, full, config.MARKET_TZ),
                               index=actual.index)

        plans = {}  # name -> (schedule, expected profit, extra diagnostics)
        if perfect_foresight:
            plans["perfect_foresight"] = (optimize_window(actual, bat, deg, fcr_price=fcr_pf),
                                          np.nan, {})
        if deterministic:
            plans["deterministic"] = (optimize_window(central, bat, deg, fcr_price=fcr_fc),
                                      np.nan, {})
        for lam in lambdas:
            res = optimize_day_cvar(scenarios, bat, deg, lam=lam, fcr_price=fcr_fc)
            plans[f"stochastic_lam{lam:g}"] = (res.schedule, res.expected_profit, {})
        for pct in dd_pcts:
            cum, peak = dd_state[pct]
            budget = max(pct * peak, dd_floor)
            allowed = max(budget - (peak - cum), 0.0)
            res = optimize_day_cvar(scenarios, bat, deg, lam=0.0, cvar_floor=-allowed,
                                    fcr_price=fcr_fc)
            plans[f"drawdown_{pct:g}"] = (res.schedule, res.expected_profit, {
                "dd_budget_eur": budget, "dd_allowed_loss_eur": allowed,
                "dd_planned_cvar_eur": res.cvar})
        if fcr_only:
            only = pd.DataFrame({"charge_mw": 0.0, "discharge_mw": 0.0, "fcr_mw": bat.power_mw},
                                index=actual.index)
            plans["fcr_only"] = (only, np.nan, {})

        for name, (sched, expected, extra) in plans.items():
            fcr_mw = sched["fcr_mw"].to_numpy() if "fcr_mw" in sched else np.zeros(len(sched))
            fcr_paid = (realized_price(md.fcr_curves, actual.index, fcr_mw, config.MARKET_TZ)
                        if setup.fcr else None)
            realized = scenario_profits(sched, actual.to_frame(), c_deg, fcr_price=fcr_paid)[0]
            if name.startswith("drawdown_"):
                state = dd_state[float(name.split("_")[1])]
                state[0] += realized
                state[1] = max(state[1], state[0])
                extra = {**extra, "dd_cum_eur": state[0], "dd_peak_eur": state[1]}
            rows.append({
                "setup": setup.name, "day": day, "strategy": name,
                "realized_eur": realized,
                "expected_eur": expected,
                "fcr_revenue_eur": float(fcr_paid @ fcr_mw) if setup.fcr else 0.0,
                # what the same MW would earn if the battery did not move the price
                "fcr_revenue_price_taker_eur": (float(fcr_observed.to_numpy() @ fcr_mw)
                                                if setup.fcr else 0.0),
                "fcr_mw_mean": float(fcr_mw.mean()),
                "discharged_mwh": float(sched["discharge_mw"].sum()),
                "usable_mwh": bat.usable_energy_mwh,
                **extra,
            })
        if progress and ((i + 1) % 60 == 0 or i + 1 == len(days)):
            print(f"  [{setup.name}] {day:%Y-%m-%d} ({i + 1}/{len(days)})", flush=True)

    return pd.DataFrame(rows)
