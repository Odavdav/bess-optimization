"""Day-ahead price scenarios: hourly ARX forecast + Monte Carlo simulation of daily errors.

Each day J (before the noon auction), for the target day D = J+1:
  1. fit 24 linear regressions (one per hour) on the last `train_days` days,
  2. forecast the 24 prices of D (central forecast),
  3. scenario k = central forecast + a simulated 24-hour error vector.

Two ways to simulate the error vector (`method`):
  - "bootstrap": the error vector of a random past day (non-parametric Monte Carlo)
  - "gaussian":  a draw from a multivariate normal fitted on past errors
                 (parametric Monte Carlo, correlated hours via Cholesky)

Only information known before noon on J is used: prices up to the end of J,
and forecast errors of days up to J.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ERROR_POOL_DAYS, MARKET_TZ, N_SCENARIOS, TRAIN_DAYS

HOURS = 24


# ---------------------------------------------------------------------------
# 1. From an hourly series to a (days x 24 hours) matrix, and back
# ---------------------------------------------------------------------------

def to_daily_matrix(prices: pd.Series) -> pd.DataFrame:
    """One row per local market day, one column per local hour (0..23).

    Daylight saving: the 23-hour day gets its missing hour filled with the
    previous hour, the 25-hour day gets its doubled hour averaged.
    """
    local = prices.tz_convert(MARKET_TZ)
    df = pd.DataFrame({"price": local.to_numpy(),
                       "day": local.index.date,
                       "hour": local.index.hour})
    matrix = df.pivot_table(index="day", columns="hour", values="price", aggfunc="mean")
    matrix = matrix.reindex(columns=range(HOURS)).ffill(axis=1)
    matrix.index = pd.to_datetime(matrix.index)
    # keep only complete days (the series may start or end mid-day in local time)
    return matrix.dropna()


def to_hourly(values_24: np.ndarray, day: pd.Timestamp) -> pd.Series:
    """Map 24 local-hour values onto the real hourly UTC timestamps of a market day."""
    day = pd.Timestamp(day).normalize()
    start = day.tz_localize(MARKET_TZ)
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)  # local midnight: 23/24/25 h later
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    local_hours = index.tz_convert(MARKET_TZ).hour
    return pd.Series(np.asarray(values_24)[local_hours], index=index)


# ---------------------------------------------------------------------------
# 2. Forecasts
# ---------------------------------------------------------------------------

def _features(P: np.ndarray, weekday: np.ndarray, t: int, h: int) -> np.ndarray:
    """Regression inputs to forecast day t, hour h (all known before noon on day t-1)."""
    return np.array([
        1.0,                       # constant
        P[t - 1, h],               # same hour, day before (J)
        P[t - 2, h],               # same hour, two days before (J-1)
        P[t - 7, h],               # same hour, one week before (same weekday as D)
        P[t - 1].min(),            # lowest price of J
        P[t - 1].max(),            # highest price of J
        float(weekday[t] == 0),    # D is a Monday
        float(weekday[t] == 5),    # D is a Saturday
        float(weekday[t] == 6),    # D is a Sunday
    ])


def forecast_arx(P: np.ndarray, weekday: np.ndarray, t: int, train_days: int) -> np.ndarray:
    """24 forecasts for day t, from regressions fitted on days t-train_days .. t-1."""
    train = range(max(7, t - train_days), t)
    forecast = np.empty(HOURS)
    for h in range(HOURS):
        X = np.array([_features(P, weekday, d, h) for d in train])
        y = P[list(train), h]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        forecast[h] = _features(P, weekday, t, h) @ beta
    return forecast


def run_forecasts(matrix: pd.DataFrame, train_days: int = TRAIN_DAYS) -> pd.DataFrame:
    """Rolling forecasts for every day that has enough history.

    Returns a long DataFrame indexed by (day, hour) with columns:
    actual, arx, naive.
    """
    P = matrix.to_numpy()
    weekday = matrix.index.day_of_week.to_numpy()
    first = train_days + 7  # need train_days training rows, each with a 7-day lag

    arx = np.full_like(P, np.nan)
    for t in range(first, len(P)):
        arx[t] = forecast_arx(P, weekday, t, train_days)

    naive = np.full_like(P, np.nan)
    naive[1:] = P[:-1]  # "tomorrow = today" at the same hour

    out = pd.DataFrame({
        "actual": P.ravel(),
        "arx": arx.ravel(),
        "naive": naive.ravel(),
    }, index=pd.MultiIndex.from_product([matrix.index, range(HOURS)], names=["day", "hour"]))
    return out.dropna()


# ---------------------------------------------------------------------------
# 3. Scenarios
# ---------------------------------------------------------------------------

METHODS = ("bootstrap", "gaussian")


def sample_errors(
    past: np.ndarray, n: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    """Draw n simulated error vectors (n x 24) from the past errors (days x 24)."""
    if method == "bootstrap":
        # Non-parametric: reuse whole past days, so the shape of a day is kept as is
        return past[rng.integers(0, len(past), size=n)]

    if method != "gaussian":
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    # Parametric: fit a multivariate normal N(mu, Sigma) on the past errors
    mu = past.mean(axis=0)               # 24 average errors (bias of the forecast per hour)
    sigma = np.cov(past, rowvar=False)   # 24 x 24 covariance: how hours move together

    # Cholesky: Sigma = L @ L.T. If z ~ N(0, I) then L @ z ~ N(0, Sigma).
    # The tiny diagonal term guarantees Sigma is numerically positive definite.
    jitter = 1e-6 * np.trace(sigma) / HOURS * np.eye(HOURS)
    L = np.linalg.cholesky(sigma + jitter)
    z = rng.standard_normal((n, HOURS))  # standard normal draws, no correlation
    return mu + z @ L.T


@dataclass(frozen=True)
class DayScenarios:
    day: pd.Timestamp
    method: str
    forecast: pd.Series      # hourly UTC, central forecast
    scenarios: pd.DataFrame  # hourly UTC x scenarios (columns s0, s1, ...)
    n_errors: int            # size of the error pool used (reliability indicator)


def generate_scenarios(
    forecasts: pd.DataFrame,
    day: pd.Timestamp,
    n_scenarios: int = N_SCENARIOS,
    pool_days: int = ERROR_POOL_DAYS,
    method: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> DayScenarios:
    """Scenarios for `day` = ARX forecast + simulated error vectors.

    Only errors of days strictly before `day` are used (they are known before
    the auction for `day`).
    """
    rng = rng or np.random.default_rng()
    day = pd.Timestamp(day)

    errors = (forecasts["actual"] - forecasts["arx"]).unstack("hour")  # days x 24
    past = errors[errors.index < day].tail(pool_days)
    if past.empty:
        raise ValueError(f"No past forecast errors available before {day:%Y-%m-%d}")

    central = forecasts.loc[day, "arx"].to_numpy()                          # 24 values
    scen_24 = central + sample_errors(past.to_numpy(), n_scenarios, method, rng)  # n x 24

    forecast_hourly = to_hourly(central, day)
    scen_hourly = pd.DataFrame(
        {f"s{k}": to_hourly(scen_24[k], day) for k in range(n_scenarios)}
    )
    return DayScenarios(day, method, forecast_hourly, scen_hourly, n_errors=len(past))


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    forecasts: pd.DataFrame,
    start: str,
    end: str,
    n_scenarios: int = 200,
    pool_days: int = ERROR_POOL_DAYS,
    method: str = "gaussian",
    seed: int = 0,
) -> dict:
    """Forecast and scenario quality on [start, end].

    - MAE of the ARX forecast vs the naive one,
    - coverage of the 90% scenario interval (should be close to 90%),
    - average daily spread (max - min over the day) of the scenarios vs the real one.
      This is what a battery earns money on: scenarios must reproduce it.
    """
    rng = np.random.default_rng(seed)
    days = forecasts.index.get_level_values("day").unique()
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]

    sub = forecasts.loc[days]
    mae_arx = (sub["actual"] - sub["arx"]).abs().mean()
    mae_naive = (sub["actual"] - sub["naive"]).abs().mean()

    errors = (forecasts["actual"] - forecasts["arx"]).unstack("hour")
    inside, total = 0, 0
    spread_real, spread_scen = [], []
    for day in days:
        past = errors[errors.index < day].tail(pool_days).to_numpy()
        central = forecasts.loc[day, "arx"].to_numpy()
        actual = forecasts.loc[day, "actual"].to_numpy()
        scen = central + sample_errors(past, n_scenarios, method, rng)
        low, high = np.percentile(scen, [5, 95], axis=0)
        inside += ((actual >= low) & (actual <= high)).sum()
        total += HOURS
        spread_real.append(actual.max() - actual.min())
        spread_scen.append((scen.max(axis=1) - scen.min(axis=1)).mean())

    return {
        "period": f"{start} -> {end}",
        "method": method,
        "days": len(days),
        "MAE naive (EUR/MWh)": round(float(mae_naive), 2),
        "MAE ARX (EUR/MWh)": round(float(mae_arx), 2),
        "ARX gain vs naive": f"{1 - mae_arx / mae_naive:.1%}",
        "90% interval coverage": f"{inside / total:.1%}",
        "daily spread real (EUR/MWh)": round(float(np.mean(spread_real)), 1),
        "daily spread scenarios (EUR/MWh)": round(float(np.mean(spread_scen)), 1),
    }
