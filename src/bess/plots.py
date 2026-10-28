"""Figures of the notebook. Each function returns a matplotlib figure."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

GREY, BLUE, GREEN, ORANGE, YELLOW, VIOLET, RED = (
    "#8a8984", "#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#6250d6", "#e34948")
GRID = "#e6e5e1"


def style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def strategy_style(name: str, lam_star: float | None = None, dd_star: float | None = None):
    """Label and color of a strategy, consistent across all figures."""
    if name == "perfect_foresight":
        return "Perfect foresight (ceiling)", GREY
    if name == "deterministic":
        return "Deterministic", YELLOW
    if name == "fcr_only":
        return "FCR only", VIOLET
    if name.startswith("stochastic_lam"):
        lam = float(name.split("lam")[1])
        chosen = lam_star is not None and lam == lam_star
        return f"Stochastic λ = {lam:g}" + (" (chosen)" if chosen else ""), GREEN if chosen else BLUE
    if name.startswith("drawdown_"):
        pct = float(name.split("_")[1])
        return f"Drawdown budget {pct:.1%}", ORANGE
    return name, GREY


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def plot_data_overview(prices: pd.Series, spread: pd.Series, fcr_blocks: pd.Series):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    monthly = prices.tz_convert(config.MARKET_TZ).resample("MS").mean()
    monthly.index = monthly.index.tz_localize(None)  # plot local months, not UTC timestamps
    axes[0].plot(monthly.index, monthly, color=BLUE, lw=2)
    axes[0].set_title("Monthly average day-ahead price (€/MWh)", loc="left", fontsize=11)
    sp = spread.resample("MS").median()
    axes[1].plot(sp.index, sp, color=ORANGE, lw=2)
    axes[1].set_title("Monthly median daily spread, max − min (€/MWh)", loc="left", fontsize=11)
    f = fcr_blocks.groupby(pd.Grouper(level="day", freq="MS")).mean()
    axes[2].plot(f.index, f, color=VIOLET, lw=2)
    axes[2].set_title("Monthly average FCR price (€/MW/h)", loc="left", fontsize=11)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), color=GREY, alpha=0.08)
        style(ax)
    axes[0].text(pd.Timestamp("2024-02-01"), axes[0].get_ylim()[1] * 0.92, "calibration",
                 fontsize=8, color="#52514e")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bloc 1
# ---------------------------------------------------------------------------

def plot_week(schedule: pd.DataFrame, start: str, battery=config.BATTERY):
    """Price, power, SoC and cumulative profit of a perfect-foresight schedule over a week."""
    s = schedule.copy()
    s.index = pd.to_datetime(s.index, utc=True).tz_convert(config.MARKET_TZ)
    start = pd.Timestamp(start).tz_localize(config.MARKET_TZ)
    week = s[(s.index >= start) & (s.index < start + pd.Timedelta(days=7))]
    end_h = week.index + pd.Timedelta(hours=1)
    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2, 3, 3]})
    a1.step(week.index, week["price"], where="post", color=BLUE, lw=1.8)
    a1.axhline(0, color=GREY, lw=0.8)
    a1.set_ylabel("Price (€/MWh)")
    w = 0.9 / 24
    a2.bar(week.index, week["discharge_mw"], width=w, align="edge", color=ORANGE, label="Discharge (sell)")
    a2.bar(week.index, -week["charge_mw"], width=w, align="edge", color=GREEN, label="Charge (buy)")
    a2.axhline(0, color=GREY, lw=0.8)
    a2.set_ylabel("MW")
    a2.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, ncol=2, fontsize=9)
    a3.plot(end_h, week["energy_mwh"] / battery.capacity_mwh * 100, color=BLUE, lw=1.8)
    for lvl in (battery.soc_min, battery.soc_max):
        a3.axhline(lvl * 100, color=GREY, lw=1, ls="--")
    a3.set_ylim(0, 100)
    a3.set_ylabel("SoC (%)")
    cum = week["net_profit_eur"].cumsum() / 1000
    a4.plot(end_h, cum, color=BLUE, lw=1.8)
    a4.fill_between(end_h, cum, 0, color=BLUE, alpha=0.12, lw=0)
    a4.axhline(0, color=GREY, lw=0.8)
    a4.set_ylabel("Cumulative profit (k€)")
    for ax in (a1, a2, a3, a4):
        style(ax)
        for d in pd.date_range(start, start + pd.Timedelta(days=7), freq="D"):
            ax.axvline(d, color="#d0cfca", lw=0.8)
    a4.xaxis.set_major_locator(mdates.DayLocator(tz=week.index.tz))
    a4.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m", tz=week.index.tz))
    fig.suptitle(f"Perfect foresight over one week (from {start:%d %b %Y})", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bloc 2
# ---------------------------------------------------------------------------

def plot_example_scenarios(example: pd.DataFrame):
    idx = pd.to_datetime(example.index, utc=True).tz_convert(config.MARKET_TZ)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, method, title in ((axes[0], "bootstrap", "Bootstrap (past days of errors)"),
                              (axes[1], "gaussian", "Gaussian Monte Carlo (Cholesky)")):
        cols = [c for c in example.columns if c.startswith(method)]
        ax.plot(idx, example[cols].to_numpy(), color="#b9b8b2", lw=0.8, alpha=0.7)
        ax.plot(idx, example["forecast"], color=BLUE, lw=2.2, label="Central forecast (ARX)")
        ax.plot(idx, example["actual"], color="#0b0b0b", lw=2, label="Actual price")
        ax.plot([], [], color="#b9b8b2", label=f"{len(cols)} scenarios")
        ax.set_title(title, loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Hh", tz=idx.tz))
        style(ax)
    axes[0].set_ylabel("€/MWh")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Price scenarios for {idx[0]:%d %b %Y}", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_error_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = range(0, 24, 3)
    ax.set_xticks(ticks, [f"{h}h" for h in ticks])
    ax.set_yticks(ticks, [f"{h}h" for h in ticks])
    ax.set_title("Correlation of forecast errors between hours (2024)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_convergence(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, alpha in zip(axes[:2], (0.9, 0.95)):
        s = summary.loc[alpha]
        k = s.index.to_numpy()
        ax.plot(k, s["in_obj"] / 1000, ls="--", marker="o", color=GREY, lw=1.8,
                label="What the optimizer expects")
        ax.plot(k, s["out_obj"] / 1000, marker="o", color=BLUE, lw=1.8,
                label="What the schedule delivers (evaluation)")
        ax.fill_between(k, (s["out_obj"] - s["out_obj_std"]) / 1000,
                        (s["out_obj"] + s["out_obj_std"]) / 1000, color=BLUE, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(k, [str(v) for v in k])
        ax.minorticks_off()
        ax.set_title(f"α = {alpha:.0%}", loc="left", fontsize=11)
        ax.set_xlabel("Number of scenarios K")
        style(ax, "both")
    axes[0].set_ylabel("Objective summed over 12 days (k€)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    s = summary.loc[0.9]
    axes[2].plot(s.index, s["solve_s"], marker="o", color=ORANGE, lw=1.8)
    axes[2].set_xscale("log")
    axes[2].set_xticks(s.index, [str(v) for v in s.index])
    axes[2].minorticks_off()
    axes[2].set_title("Solve time per day (s)", loc="left", fontsize=11)
    axes[2].set_xlabel("Number of scenarios K")
    style(axes[2], "both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bloc 3
# ---------------------------------------------------------------------------

def plot_frontier_and_rule(m_cal: pd.DataFrame, m_test: pd.DataFrame, lam_star: float):
    fig, (ax_f, ax_l) = plt.subplots(1, 2, figsize=(14, 5))
    for m, label, color in ((m_cal, "Calibration 2024", BLUE), (m_test, "Test 2025", ORANGE)):
        rows = m[m.index.str.startswith("stochastic_lam")].copy()
        rows.index = [float(i.split("lam")[1]) for i in rows.index]
        rows = rows.sort_index()
        ax_f.plot(rows["cvar90_daily_eur"] / 1000, rows["mean_daily_eur"] / 1000, marker="o",
                  color=color, lw=1.8, label=label)
        for lam, r in rows.iterrows():
            ax_f.annotate(f"λ={lam:g}", (r["cvar90_daily_eur"] / 1000, r["mean_daily_eur"] / 1000),
                          xytext=(5, 4), textcoords="offset points", fontsize=7, color="#52514e")
        d = m.loc["deterministic"]
        ax_f.scatter(d["cvar90_daily_eur"] / 1000, d["mean_daily_eur"] / 1000, marker="D", s=45,
                     color=color, edgecolor="white", zorder=3)
        ax_l.plot(rows.index, rows["loss_days_pct"] * 100, marker="o", color=color, lw=1.8, label=label)
    ax_f.scatter([], [], marker="D", color=GREY, label="Deterministic (diamond)")
    ax_f.set_xlabel("CVaR 90% of realized daily profits (k€)  → less risk")
    ax_f.set_ylabel("Mean realized daily profit (k€)")
    ax_f.set_title("Return / risk frontier on actual prices", loc="left", fontsize=11)
    ax_f.legend(frameon=False, fontsize=9)
    ax_l.axhline(config.MAX_LOSS_DAYS * 100, color=RED, ls="--", lw=1.2)
    ax_l.text(0, config.MAX_LOSS_DAYS * 100, " budget: 1 day in 7", va="bottom", fontsize=8, color=RED)
    ax_l.axvline(lam_star, color=GREY, ls=":", lw=1)
    ax_l.set_xlabel("λ (weight of the CVaR)")
    ax_l.set_ylabel("Loss days (%)")
    ax_l.set_title(f"λ selection rule (chosen: {lam_star:g})", loc="left", fontsize=11)
    ax_l.legend(frameon=False, fontsize=9, loc="lower left")
    for ax in (ax_f, ax_l):
        style(ax, "both")
    fig.tight_layout()
    return fig


def plot_cumulative(results: pd.DataFrame, strategies: list[str], title: str,
                    lam_star=None, dd_star=None):
    daily = results.pivot(index="day", columns="strategy", values="realized_eur").sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in strategies:
        label, color = strategy_style(name, lam_star, dd_star)
        cum = daily[name].cumsum() / 1e6
        ax.plot(cum.index, cum, lw=2, color=color, ls="--" if name == "perfect_foresight" else "-",
                label=f"{label}: {cum.iloc[-1]:.2f} M€")
    ax.set_ylabel("Cumulative realized profit (M€)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, "both")
    fig.tight_layout()
    return fig


def plot_volatility(vq: pd.DataFrame, strategies: list[str], title: str, lam_star=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    first = vq[vq["strategy"] == strategies[0]]
    x = np.arange(len(first))
    labels = [f"{q}\n{s:.0f} €/MWh" for q, s in zip(first["quintile"], first["median_spread"])]
    for name in strategies:
        r = vq[vq["strategy"] == name]
        label, color = strategy_style(name, lam_star)
        a1.plot(x, r["mean_daily_eur"] / 1000, marker="o", lw=2, color=color, label=label,
                ls="--" if name == "perfect_foresight" else "-")
        if name != "perfect_foresight":
            a2.plot(x, r["loss_days_pct"] * 100, marker="o", lw=2, color=color, label=label)
    for ax in (a1, a2):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Volatility quintile (median daily spread, max − min)")
        style(ax, "both")
    a1.set_ylabel("Mean daily profit (k€)")
    a1.set_title("Profit by daily volatility", loc="left", fontsize=11)
    a1.legend(frameon=False, fontsize=8)
    a2.set_ylabel("Loss days (%)")
    a2.set_title("Loss days by daily volatility", loc="left", fontsize=11)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_fcr_contribution(vq_arb: pd.DataFrame, vq_fcr: pd.DataFrame, fcr_by_q: pd.DataFrame,
                          strategy: str, title: str):
    a = vq_arb[vq_arb["strategy"] == strategy].reset_index(drop=True)
    f = vq_fcr[vq_fcr["strategy"] == strategy].reset_index(drop=True)
    x = np.arange(len(a))
    labels = [f"{q}\n{s:.0f} €/MWh" for q, s in zip(a["quintile"], a["median_spread"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.38
    for off, data, color, label in ((-w / 2, a, BLUE, "Arbitrage only"),
                                    (w / 2, f, ORANGE, "Arbitrage + FCR")):
        vals = data["share_of_total"] * 100
        a1.bar(x + off, vals, width=w * 0.95, color=color, label=label)
        for xi, v in zip(x + off, vals):
            a1.text(xi, v + (1 if v >= 0 else -4), f"{v:.0f}%", ha="center", fontsize=8,
                    color="#52514e")
    a1.axhline(0, color=GREY, lw=0.8)
    a1.set_ylabel("Share of annual profit (%)")
    a1.set_title("Contribution of each volatility quintile", loc="left", fontsize=11)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a2.bar(x, fcr_by_q["fcr_keur"] / 1000, width=0.6, color=YELLOW, label="FCR revenue")
    arb = fcr_by_q["profit_keur"] - fcr_by_q["fcr_keur"]
    a2.bar(x, arb / 1000, width=0.6, bottom=np.where(arb >= 0, fcr_by_q["fcr_keur"] / 1000, 0),
           color=GREEN, label="Day-ahead arbitrage (net)")
    a2.set_ylabel("Profit (M€)")
    a2.set_title("With FCR: where the profit comes from", loc="left", fontsize=11)
    a3 = a2.twinx()
    a3.plot(x, fcr_by_q["avg_fcr_mw"], color="#52514e", marker="o", lw=1.5, label="MW reserved for FCR")
    a3.set_ylim(0, 80)
    a3.set_ylabel("MW reserved for FCR")
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    for ax in (a1, a2):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Day-ahead volatility quintile")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top"]].set_visible(False)
    a1.spines[["right"]].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig
