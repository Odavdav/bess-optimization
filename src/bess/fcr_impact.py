"""Price impact of the battery on the French FCR price, from the anonymous bid lists.

regelleistung.net publishes every FCR bid (price, offered and allocated MW, country), but only
once a month for the month before. Two uses, with two different information sets:

  - evaluation (ex post): replay each auction with the battery offering q MW at 0 EUR/MW and
    read the new French price -> one price curve p_b(q) per 4-hour block;
  - decision (ex ante): one factor f per month, f = sum p(P) / sum p(0) over the month (P = power),
    used two months later (f of month M-2 for decisions in month M).

Auction (pay-as-clear): demand is fixed by the TSOs; the price is the most expensive
allocated bid. A bid at 0 EUR/MW is always allocated, so adding q MW amounts to removing q MW
from the top of the allocated stack.
  - island block (France at its export limit): the battery only displaces French bids;
  - otherwise France is cleared together with the countries at the same price (the pool):
    France exports up to its remaining room r and the battery displaces pool bids; if q > r,
    France reaches its limit and the rest displaces French bids.
Export limit: 30% of the French demand (checked on the data: every island block allocates
exactly demand + floor(0.3 x demand) MW of French bids). With it, the replay without the battery
gives back the published French price in every block of 2023-2025.
"""
import numpy as np
import pandas as pd

from . import config

FCR_BLOCK_H = 4
Q_STEP_MW = 5
# MW offered by the battery: 0 to its power, every 5 MW (follows config.BATTERY).
# Changing the power requires rebuilding the curves (01_prepare_data.py, with the bid lists).
Q_GRID = np.unique(np.append(np.arange(0, config.BATTERY.power_mw, Q_STEP_MW), config.BATTERY.power_mw))
if np.all(Q_GRID == np.round(Q_GRID)):
    Q_GRID = Q_GRID.astype(int)  # column names q0, q5, ...
EXPORT_SHARE = 0.3


def read_bids(path) -> pd.DataFrame:
    """Allocated bids, one row per bid: day, block, country, price (EUR/MW per block), MW."""
    df = pd.read_excel(path)
    df = df[df["ALLOCATED_CAPACITY_[MW]"] > 0]
    start_hour = df["PRODUCT"].str.extract(r"_(\d{2})_\d{2}$")[0].astype(int)
    return pd.DataFrame({
        "day": pd.to_datetime(df["DATE_FROM"]).dt.normalize(),
        "block": start_hour // FCR_BLOCK_H,
        "country": df["COUNTRY"],
        "price": df["OFFERED_CAPACITY_PRICE_[EUR/MW]"],
        "mw": df["ALLOCATED_CAPACITY_[MW]"],
        "settlement": df["SETTLEMENTCAPACITY_PRICE_[EUR/MW]"],
    })


def read_french_demand(overview_path) -> pd.Series:
    """French FCR demand (MW) per (day, block), from the results overview file."""
    df = pd.read_excel(overview_path)
    df = df[df["PRODUCT_TYPE"] == "FCR"].drop_duplicates(["DATE_FROM", "PRODUCTNAME"])
    start_hour = df["PRODUCTNAME"].str.extract(r"_(\d{2})_\d{2}$")[0].astype(int)
    return pd.Series(df["FRANCE_DEMAND_[MW]"].to_numpy(dtype=float),
                     index=pd.MultiIndex.from_arrays(
                         [pd.to_datetime(df["DATE_FROM"]).dt.normalize().to_numpy(),
                          (start_hour // FCR_BLOCK_H).to_numpy()], names=["day", "block"]))


def price_after_removing(prices: np.ndarray, mw: np.ndarray, q: float) -> float:
    """Marginal price of an allocated stack once q MW are removed from its top."""
    order = np.argsort(-prices)
    top_down = np.cumsum(mw[order])
    j = np.searchsorted(top_down, q, side="right")  # first bid not entirely removed
    return float(prices[order][j]) if j < len(prices) else 0.0


def block_curve(block: pd.DataFrame, demand: float) -> tuple[np.ndarray, bool]:
    """French price (EUR/MW per block) for each q of Q_GRID, and whether the block is an island.

    Island: France allocates demand + export limit, so it cannot export more.
    Otherwise France belongs to the group of countries cleared at the same price (the pool).
    """
    fr = block[block["country"] == "FR"]
    fr_price = fr["settlement"].iloc[0]
    pool = block[(block["settlement"] - fr_price).abs() < 0.011]
    room = max(0.0, demand + np.floor(EXPORT_SHARE * demand) - fr["mw"].sum())
    island = room == 0
    curve = []
    for q in Q_GRID:
        if island:
            # At q = 0 France keeps its settlement price: when it sits exactly at its export
            # limit it can still be priced with the pool, above its own most expensive bid.
            # Any MW added cannot be exported, so French bids set the price.
            curve.append(fr_price if q == 0 else
                         price_after_removing(fr["price"].to_numpy(), fr["mw"].to_numpy(), q))
            continue
        exported = min(q, room)
        p = price_after_removing(pool["price"].to_numpy(), pool["mw"].to_numpy(), exported)
        if q > room:  # France reaches its export limit: the rest displaces French bids
            p = min(p, price_after_removing(fr["price"].to_numpy(), fr["mw"].to_numpy(), q - room))
        curve.append(p)
    return np.array(curve), island


def price_curves(bids: pd.DataFrame, demand: pd.Series) -> pd.DataFrame:
    """One row per (day, block): French price in EUR/MW per HOUR for every q of Q_GRID."""
    rows, index = [], []
    for key, block in bids.groupby(["day", "block"]):
        curve, island = block_curve(block, demand.loc[key])
        rows.append([*(curve / FCR_BLOCK_H), island])
        index.append(key)
    cols = [f"q{q}" for q in Q_GRID] + ["island"]
    out = pd.DataFrame(rows, columns=cols,
                       index=pd.MultiIndex.from_tuples(index, names=["day", "block"]))
    out["island"] = out["island"].astype(bool)
    return out


def monthly_factors(curves: pd.DataFrame) -> pd.Series:
    """f per month = sum of prices with the full power / sum of prices without the battery."""
    month = curves.index.get_level_values("day").to_period("M")
    check_grid(curves)
    g = curves.groupby(month)
    return (g[f"q{Q_GRID[-1]}"].sum() / g["q0"].sum()).rename("f")


def decision_factor(factors: pd.Series, day, lag_months: int = 2) -> float:
    """Factor usable for decisions on `day`: month M-2 (published by then); 1 if not available."""
    month = pd.Timestamp(day).to_period("M") - lag_months
    return float(factors.get(month, 1.0))


def realized_price(curves: pd.DataFrame, index_utc: pd.DatetimeIndex, fcr_mw: np.ndarray,
                   market_tz: str) -> np.ndarray:
    """Realized FCR price (EUR/MW/h) of each hour, given the MW reserved in that hour.

    Linear interpolation of the block curve between the 5-MW points.
    """
    local = index_utc.tz_convert(market_tz)
    keys = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(local.date), local.hour // FCR_BLOCK_H], names=["day", "block"])
    check_grid(curves)
    fcr_mw = np.asarray(fcr_mw, dtype=float)
    if (fcr_mw > Q_GRID[-1] + 1e-6).any():
        raise ValueError(f"{fcr_mw.max():g} MW reserved, but the price curves stop at {Q_GRID[-1]:g} MW")
    table = curves.reindex(keys)[[f"q{q}" for q in Q_GRID]].to_numpy()
    if np.isnan(table).any():
        raise ValueError("FCR price curves missing for some hours")
    return np.array([np.interp(q, Q_GRID, row) for q, row in zip(fcr_mw, table)])


def check_grid(curves: pd.DataFrame) -> None:
    """The stored curves must cover the battery power (np.interp would silently cap beyond)."""
    missing = [f"q{q}" for q in Q_GRID if f"q{q}" not in curves.columns]
    if missing:
        raise ValueError(f"FCR price curves have no column {missing}: the battery power changed, rebuild "
                         "them with scripts/01_prepare_data.py and the anonymous bid lists in data/raw/")
