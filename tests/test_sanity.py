"""Consistency checks run during the project, frozen as tests.  Run: pytest tests/

They need the processed data (scripts/01_prepare_data.py) and take about one minute.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bess import config  # noqa: E402
from bess.backtest import MarketData, day_inputs, load_market_data  # noqa: E402
from bess.data import fcr_hourly  # noqa: E402
from bess.optimization import optimize_window  # noqa: E402
from bess.scenarios import run_forecasts, sample_errors, to_daily_matrix  # noqa: E402
from bess.stochastic import cvar, optimize_day_cvar, scenario_profits  # noqa: E402

BAT, DEG = config.BATTERY, config.DEGRADATION
C_DEG = DEG.cost_per_mwh_discharged(BAT)
DAYS = ["2024-03-31", "2024-07-15", "2024-10-27"]  # includes both daylight-saving days


@pytest.fixture(scope="module")
def md() -> MarketData:
    return load_market_data(with_fcr=True)


def test_daylight_saving_days_have_23_and_25_hours(md):
    for day, n in (("2024-03-31", 23), ("2024-10-27", 25), ("2024-07-15", 24)):
        _, scenarios, actual = day_inputs(md, day)
        assert len(scenarios) == len(actual) == n
        assert not actual.isna().any()


def test_risk_neutral_equals_optimizing_on_the_mean_scenario(md):
    """Profit is linear in prices: maximizing the expectation = optimizing on the mean."""
    for day in DAYS:
        _, scenarios, _ = day_inputs(md, day)
        stochastic = optimize_day_cvar(scenarios, BAT, DEG, lam=0.0)
        on_mean = optimize_window(scenarios.mean(axis=1), BAT, DEG)
        assert stochastic.expected_profit == pytest.approx(on_mean["net_profit_eur"].sum(), abs=1.0)


def test_perfect_foresight_dominates(md):
    for day in DAYS:
        central, scenarios, actual = day_inputs(md, day)
        pf = scenario_profits(optimize_window(actual, BAT, DEG), actual.to_frame(), C_DEG)[0]
        others = [optimize_window(central, BAT, DEG)] + [
            optimize_day_cvar(scenarios, BAT, DEG, lam=lam).schedule for lam in (0, 0.5, 1)]
        for sched in others:
            assert scenario_profits(sched, actual.to_frame(), C_DEG)[0] <= pf + 1e-6


def test_no_simultaneous_charge_and_discharge(md):
    _, _, actual = day_inputs(md, DAYS[1])
    s = optimize_window(actual, BAT, DEG)
    assert not ((s["charge_mw"] > 1e-6) & (s["discharge_mw"] > 1e-6)).any()


def test_risk_aversion_trades_expectation_for_cvar(md):
    _, scenarios, _ = day_inputs(md, DAYS[1])
    neutral = optimize_day_cvar(scenarios, BAT, DEG, lam=0.0)
    averse = optimize_day_cvar(scenarios, BAT, DEG, lam=1.0)
    assert neutral.expected_profit >= averse.expected_profit - 1e-6
    assert averse.cvar >= neutral.cvar - 1e-6


def test_cvar_floor_is_respected(md):
    _, scenarios, _ = day_inputs(md, DAYS[1])
    res = optimize_day_cvar(scenarios, BAT, DEG, lam=0.0, cvar_floor=-500)
    assert res.cvar >= -500 - 1e-3


def test_fcr_constraints(md):
    for day in DAYS:
        _, _, actual = day_inputs(md, day)
        s = optimize_window(actual, BAT, DEG, fcr_price=fcr_hourly(md.fcr_blocks, actual.index))
        assert ((s["charge_mw"] + s["fcr_mw"]) <= BAT.power_mw + 1e-6).all()
        assert ((s["discharge_mw"] + s["fcr_mw"]) <= BAT.power_mw + 1e-6).all()
        assert ((s["energy_mwh"] - 0.25 * s["fcr_mw"]) >= BAT.energy_min_mwh - 1e-6).all()
        assert ((s["energy_mwh"] + 0.25 * s["fcr_mw"]) <= BAT.energy_max_mwh + 1e-6).all()
        block = s.index.tz_convert(config.MARKET_TZ).hour // 4
        assert (s.groupby(block)["fcr_mw"].nunique() == 1).all()  # constant within a block


def test_no_information_leakage(md):
    """Changing prices from day D onwards must not change the forecast and scenarios of D."""
    day = pd.Timestamp("2024-07-15")
    _, before, _ = day_inputs(md, day)
    tampered = md.prices.copy()
    tampered[tampered.index >= day.tz_localize(config.MARKET_TZ).tz_convert("UTC")] *= 10
    md2 = MarketData(tampered, run_forecasts(to_daily_matrix(tampered)))
    _, after, _ = day_inputs(md2, day)
    assert np.allclose(before.to_numpy(), after.to_numpy())


def test_gaussian_sampler_reproduces_mean_and_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(24, 24))
    past = rng.multivariate_normal(rng.normal(size=24) * 5, a @ a.T, size=60)
    draws = sample_errors(past, 100_000, "gaussian", rng)
    assert np.abs(draws.mean(axis=0) - past.mean(axis=0)).max() < 0.1
    target = np.cov(past, rowvar=False)
    assert np.abs(np.cov(draws, rowvar=False) - target).max() / target.max() < 0.02


def test_cvar_is_the_mean_of_the_worst_share():
    profits = np.arange(100.0)
    assert cvar(profits, 0.9) == pytest.approx(4.5)  # mean of 0..9


def test_fcr_price_curves_replay_the_auction(md):
    """Without the battery the replay gives the published price; adding MW never raises it."""
    curves = md.fcr_curves
    published = md.fcr_blocks.reindex(curves.index)
    assert ((curves["q0"] - published).abs() < 0.01).mean() > 0.99
    q_cols = [c for c in curves.columns if c.startswith("q")]
    assert (curves[q_cols].diff(axis=1).iloc[:, 1:] <= 1e-9).all().all()


def test_fcr_decision_factor_uses_only_published_bids(md):
    """Decisions in month M use the factor of month M-2; no factor before the first bids."""
    from bess.fcr_impact import decision_factor
    f = md.fcr_factors
    assert decision_factor(f, "2025-03-15") == pytest.approx(f[pd.Period("2025-01", "M")])
    assert decision_factor(f, "2024-02-10") == pytest.approx(f[pd.Period("2023-12", "M")])
    assert decision_factor(f, "2023-02-10") == 1.0  # no bid list before 2023


def test_fcr_curves_cover_the_battery_power(md):
    """The replay grid follows the battery power; reserving more MW than the grid is refused."""
    from bess.fcr_impact import Q_GRID, realized_price
    assert Q_GRID[0] == 0 and Q_GRID[-1] == config.BATTERY.power_mw
    assert all(f"q{q}" in md.fcr_curves.columns for q in Q_GRID)
    hours = pd.date_range("2025-03-01", periods=4, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        realized_price(md.fcr_curves, hours, np.full(4, Q_GRID[-1] + 5.0), config.MARKET_TZ)
