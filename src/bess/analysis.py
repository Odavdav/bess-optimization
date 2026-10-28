"""Metrics on backtest results, calibration rules and volatility analysis."""
import numpy as np
import pandas as pd

from . import config
from .stochastic import cvar


def daily_profits(results: pd.DataFrame) -> pd.DataFrame:
    return results.pivot(index="day", columns="strategy", values="realized_eur").sort_index()


def max_drawdown(daily: pd.Series) -> float:
    """Largest fall of the cumulative profit below its previous peak (EUR)."""
    cum = daily.cumsum()
    return float((np.maximum(cum.cummax(), 0) - cum).max())


def metrics(results: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy, computed on the daily realized profits."""
    daily = daily_profits(results)
    pf = daily["perfect_foresight"] if "perfect_foresight" in daily else None
    by = results.groupby("strategy")
    out = {}
    for name in daily.columns:
        d = daily[name]
        loss = d < config.LOSS_THRESHOLD_EUR
        weekly_losses = loss.groupby(d.index.to_period("W")).sum()
        out[name] = {
            "profit_keur": d.sum() / 1000,
            "capture_vs_pf": d.sum() / pf.sum() if pf is not None else np.nan,
            "evpi_keur": (pf - d).sum() / 1000 if pf is not None else np.nan,
            "mean_daily_eur": d.mean(),
            "sharpe_annualized": d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan,
            "cvar90_daily_eur": cvar(d.to_numpy(), 0.9),
            "worst_day_eur": d.min(),
            "loss_days_pct": loss.mean(),
            "weeks_over_1_loss_pct": (weekly_losses > 1).mean(),
            "max_drawdown_keur": max_drawdown(d) / 1000,
            "cycles": by.get_group(name)["discharged_mwh"].sum()
                      / by.get_group(name)["usable_mwh"].iloc[0],
            "fcr_share": (by.get_group(name)["fcr_revenue_eur"].sum() / d.sum()
                          if d.sum() else np.nan),
            "avg_fcr_mw": by.get_group(name)["fcr_mw_mean"].mean(),
        }
    return pd.DataFrame(out).T


def split_period(results: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return results[(results["day"] >= start) & (results["day"] <= end)]


# ---------------------------------------------------------------------------
# Calibration rules (applied on the calibration year only)
# ---------------------------------------------------------------------------

def choose_lambda(m: pd.DataFrame, max_loss_days: float = config.MAX_LOSS_DAYS) -> float:
    """Most profitable lambda with at most `max_loss_days` share of loss days.

    Falls back on the lambda with the fewest loss days if none respects the budget.
    """
    rows = m[m.index.str.startswith("stochastic_lam")].copy()
    rows.index = [float(i.split("lam")[1]) for i in rows.index]
    ok = rows[rows["loss_days_pct"] <= max_loss_days]
    if ok.empty:
        return float(rows["loss_days_pct"].idxmin())
    return float(ok["profit_keur"].idxmax())


def drawdown_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    """For each drawdown strategy: did the realized drawdown stay within its budget?"""
    rows = {}
    for name, g in results[results["strategy"].str.startswith("drawdown_")].groupby("strategy"):
        g = g.sort_values("day")
        dd_after = g["dd_peak_eur"] - g["dd_cum_eur"]
        rows[name] = {
            "pct": float(name.split("_")[1]),
            "profit_keur": g["realized_eur"].sum() / 1000,
            "breach_days": int((dd_after > g["dd_budget_eur"] + 1).sum()),
            "worst_dd_vs_budget": float((dd_after / g["dd_budget_eur"]).max()),
            "binding_days_pct": float((g["dd_planned_cvar_eur"]
                                       <= -g["dd_allowed_loss_eur"] + 50).mean()),
        }
    return pd.DataFrame(rows).T


def choose_drawdown(diag: pd.DataFrame) -> float:
    """Most profitable drawdown budget never breached; else the least breached."""
    ok = diag[diag["breach_days"] == 0]
    if ok.empty:
        return float(diag.sort_values(["breach_days", "profit_keur"],
                                      ascending=[True, False])["pct"].iloc[0])
    return float(ok.sort_values("profit_keur")["pct"].iloc[-1])


# ---------------------------------------------------------------------------
# Volatility regimes
# ---------------------------------------------------------------------------

def volatility_quintiles(results: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    """Per volatility quintile (daily price spread): mean profit, loss days, share of total."""
    daily = daily_profits(results)
    s = spread.reindex(daily.index)
    q = pd.qcut(s, 5, labels=[f"Q{i}" for i in range(1, 6)])
    rows = []
    for name in daily.columns:
        d = daily[name]
        g = d.groupby(q, observed=True)
        rows.append(pd.DataFrame({
            "strategy": name,
            "median_spread": s.groupby(q, observed=True).median(),
            "mean_daily_eur": g.mean(),
            "loss_days_pct": g.apply(lambda x: (x < config.LOSS_THRESHOLD_EUR).mean()),
            "share_of_total": g.sum() / d.sum(),
        }))
    return pd.concat(rows).rename_axis("quintile").reset_index()


# ---------------------------------------------------------------------------
# Losing streaks and drawdowns
# ---------------------------------------------------------------------------

def drawdown_path(daily: pd.Series) -> pd.Series:
    """Gap between the running peak of cumulative profit and its current level (EUR)."""
    cum = daily.sort_index().cumsum()
    return np.maximum(cum.cummax(), 0) - cum


def longest_losing_streak(daily: pd.Series) -> int:
    """Longest run of consecutive loss days."""
    loss = (daily.sort_index() < config.LOSS_THRESHOLD_EUR).astype(int)
    runs = loss.groupby((loss == 0).cumsum()).cumsum()
    return int(runs.max())


def longest_underwater(daily: pd.Series) -> int:
    """Longest number of consecutive days spent below the previous peak."""
    under = (drawdown_path(daily) > 1).astype(int)
    runs = under.groupby((under == 0).cumsum()).cumsum()
    return int(runs.max())


def streak_metrics(results: pd.DataFrame) -> pd.DataFrame:
    daily = daily_profits(results)
    return pd.DataFrame({
        "profit_keur": daily.sum() / 1000,
        "max_drawdown_keur": daily.apply(lambda d: drawdown_path(d).max() / 1000),
        "longest_underwater_days": daily.apply(longest_underwater),
        "longest_losing_streak": daily.apply(longest_losing_streak),
        "loss_days_pct": (daily < config.LOSS_THRESHOLD_EUR).mean(),
    })


# ---------------------------------------------------------------------------
# Significance: moving-block bootstrap of daily P&L
# ---------------------------------------------------------------------------

def block_bootstrap(a: pd.Series, b: pd.Series, stat, block: int = 7, n: int = 5000,
                    seed: int = config.SEED) -> dict:
    """Confidence interval of stat(a) - stat(b), resampling the SAME days for both series.

    Blocks of consecutive days keep the autocorrelation of daily profits (calm or volatile
    weeks), which a day-by-day bootstrap would break.
    """
    a, b = a.sort_index().to_numpy(), b.sort_index().to_numpy()
    n_days = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block))
    diffs = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, n_days - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n_days]
        diffs[k] = stat(a[idx]) - stat(b[idx])
    observed = stat(a) - stat(b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"observed": observed, "ci_low": low, "ci_high": high,
            "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}
