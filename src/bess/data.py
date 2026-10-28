"""Data loading and cleaning: French day-ahead prices and FCR capacity prices.

Day-ahead prices: Energy-Charts export (Fraunhofer ISE). Since delivery day 1 Oct 2025
the auction has 15-minute prices; they are averaged into hourly prices.
Licence: French day-ahead prices are published under CC BY 4.0,
source "Bundesnetzagentur | SMARD.de" (to be cited when results are shared).

FCR prices: regelleistung.net "RESULT_OVERVIEW_CAPACITY_MARKET_FCR" exports, one row per
day and 4-hour block, settlement price in EUR/MW for the whole block.
"""
import numpy as np
import pandas as pd

from . import config

FCR_BLOCK_H = 4


# ---------------------------------------------------------------------------
# Day-ahead prices
# ---------------------------------------------------------------------------

def read_energy_charts(path) -> pd.Series:
    """Raw Energy-Charts file -> hourly UTC price series (EUR/MWh)."""
    with open(path, encoding="utf-8") as f:
        header_line = next(i for i, line in enumerate(f) if line.startswith("Date (TC)"))
    df = pd.read_csv(path, skiprows=header_line, header=0).iloc[1:]  # drop the units row
    times = pd.to_datetime(df.iloc[:, 0], utc=True)
    prices = pd.to_numeric(df["Day Ahead Auction (FR)"], errors="coerce")
    s = pd.Series(prices.to_numpy(), index=times).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return fill_gaps(s.resample("h").mean())  # 15-minute prices -> hourly average


def fill_gaps(s: pd.Series) -> pd.Series:
    """Fill missing hours: time interpolation, then mean of the same hour at D-1 and D+1.

    This was benchmarked on artificial gaps (WAPE) on the first data set of the project;
    the Energy-Charts files used here have no gap, so it is only a safety net.
    """
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    filled = full.interpolate(method="time", limit=3)
    daily = (full.shift(24) + full.shift(-24)) / 2
    return filled.fillna(daily).rename("price")


def load_prices(years=config.YEARS) -> pd.Series:
    """Processed hourly prices (UTC) for the given years, as one continuous series."""
    parts = []
    for year in years:
        df = pd.read_csv(config.DATA_PROCESSED / config.PRICE_FILE.format(year=year),
                         index_col=0, parse_dates=True)
        s = df.iloc[:, 0]
        s.index = s.index.tz_localize("UTC")
        parts.append(s)
    prices = pd.concat(parts).sort_index()
    prices = prices[~prices.index.duplicated(keep="first")].rename("price")
    check_hourly(prices)
    return prices


def check_hourly(prices: pd.Series) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("prices must have a timezone-aware DatetimeIndex")
    if prices.isna().any():
        raise ValueError(f"{int(prices.isna().sum())} missing prices")
    steps = prices.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("prices must be on a continuous hourly index")


# ---------------------------------------------------------------------------
# FCR prices
# ---------------------------------------------------------------------------

def read_fcr(path) -> pd.Series:
    """Raw regelleistung.net file -> French block prices in EUR/MW per HOUR.

    Indexed by (local day, block 0..5). For each block, keep the first tender with a
    valid French price (a second tender is exceptional and usually has no French price).
    """
    df = pd.read_excel(path)
    price_col = next(c for c in df.columns
                     if c.startswith(("FR_", "FRANCE_")) and "SETTLEMENTCAPACITY_PRICE" in c)
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df["price"].notna()]
    df = df[~((df["TENDER_NUMBER"] > 1) & (df["price"] == 0))]
    df = df.sort_values("TENDER_NUMBER").drop_duplicates(["DATE_FROM", "PRODUCTNAME"])
    if len(df) != 6 * df["DATE_FROM"].nunique():
        raise ValueError(f"{path}: incomplete FCR blocks")
    start_hour = df["PRODUCTNAME"].str.extract(r"_(\d{2})_\d{2}$")[0].astype(int)
    return pd.Series(
        df["price"].to_numpy() / FCR_BLOCK_H,  # EUR/MW per block -> EUR/MW per hour
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(df["DATE_FROM"]).dt.normalize().to_numpy(),
             (start_hour // FCR_BLOCK_H).to_numpy()], names=["day", "block"]),
        name="fcr_eur_per_mw_h",
    ).sort_index()


def load_fcr() -> pd.Series:
    df = pd.read_csv(config.DATA_PROCESSED / config.FCR_FILE, parse_dates=["day"])
    return df.set_index(["day", "block"])["fcr_eur_per_mw_h"]


def load_fcr_curves() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / config.FCR_CURVES_FILE, parse_dates=["day"])
    return df.set_index(["day", "block"])


def fcr_hourly(blocks: pd.Series, index_utc: pd.DatetimeIndex) -> pd.Series:
    """FCR price (EUR/MW/h) of each hour, from the LOCAL-time block it belongs to."""
    local = index_utc.tz_convert(config.MARKET_TZ)
    keys = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(local.date), local.hour // FCR_BLOCK_H], names=["day", "block"])
    values = blocks.reindex(keys).to_numpy()
    if np.isnan(values).any():
        raise ValueError("FCR prices missing for some hours")
    return pd.Series(values, index=index_utc, name="fcr")


def fcr_hourly_forecast(blocks: pd.Series, index_utc: pd.DatetimeIndex, day) -> pd.Series:
    """Naive FCR forecast for `day`: same block of the day before.

    The FCR auction for `day` closes at 8:00 the day before, when the results of the
    previous auction (for the day before `day`) are already known.
    """
    prev = blocks.loc[pd.Timestamp(day).normalize() - pd.Timedelta(days=1)].reindex(range(6))
    local_hours = index_utc.tz_convert(config.MARKET_TZ).hour
    return pd.Series(prev.to_numpy()[local_hours // FCR_BLOCK_H], index=index_utc, name="fcr_fc")


def daily_spread(prices: pd.Series) -> pd.Series:
    """Daily price spread (max - min over the local market day), EUR/MWh."""
    local = prices.tz_convert(config.MARKET_TZ)
    s = local.groupby(local.index.date).agg(lambda x: x.max() - x.min())
    s.index = pd.to_datetime(s.index)
    return s
