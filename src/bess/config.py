"""All project parameters in one place.

Protocol: 2023 = warm-up (forecast model and error pool), 2024 = calibration
(every setting is chosen on 2024), 2025 = out-of-sample test (nothing is tuned).
"""
from pathlib import Path

from .battery import Battery, DegradationModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

YEARS = (2023, 2024, 2025)
RAW_PRICE_FILE = "energy_charts_{year}.csv"   # Energy-Charts export, FR day-ahead
RAW_FCR_FILE = "fcr_{year}.xlsx"              # regelleistung.net FCR results
PRICE_FILE = "prices_{year}.csv"              # processed hourly prices (UTC)
FCR_FILE = "fcr_blocks.csv"                   # processed FCR block prices
RAW_BIDS_FILE = "fcr_bids_{year}.xlsx"        # regelleistung.net anonymous FCR bid lists
BID_YEARS = (2023, 2024, 2025)
FCR_CURVES_FILE = "fcr_price_curves.csv"      # French FCR price vs MW added by the battery
FCR_FACTOR_LAG_MONTHS = 2                     # bids are published monthly: use month M-2

# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
MARKET_TZ = "Europe/Paris"
CALIBRATION = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2025-12-31")

# ---------------------------------------------------------------------------
# Battery (central case) and degradation (2026 assumptions)
# ---------------------------------------------------------------------------
BATTERY = Battery(
    capacity_mwh=100,           # 2-hour battery
    power_mw=50,
    soc_min=0.1,
    soc_max=0.9,
    round_trip_efficiency=0.88,
    initial_soc=0.5,
)
DEGRADATION = DegradationModel(
    capex_eur_per_mwh=100_000,  # LFP DC block replacement, Europe 2026
    cycle_life=7000,            # ~10,000 cycles to 70% SoH in datasheets, derated
    eol_soh=0.8,
)

# ---------------------------------------------------------------------------
# Forecast and scenarios
# ---------------------------------------------------------------------------
TRAIN_DAYS = 56          # rolling window of the hourly ARX regressions (8 weeks)
ERROR_POOL_DAYS = 60     # rolling window of past forecast errors
SCENARIO_METHOD = "gaussian"   # "bootstrap" was compared and dropped (see notebook)
N_SCENARIOS = 200        # from the convergence study
ALPHA = 0.9              # CVaR level: average of the 10% worst scenarios
SEED = 0

# ---------------------------------------------------------------------------
# Strategies and calibration rules (fixed BEFORE looking at 2025)
# ---------------------------------------------------------------------------
LAMBDAS = [0, 0.1, 0.25, 0.375, 0.5, 0.75, 1]   # weight of the CVaR (0.375: midpoint, finer grid)
MAX_LOSS_DAYS = 1 / 7                     # lambda rule: at most 1 loss day out of 7
DD_PCTS = [0.005, 0.01, 0.02]            # drawdown budget: share of accumulated gains
DD_FLOOR_EUR_PER_MWH = 50                 # drawdown floor: 5 kEUR for a 100 MWh battery
LOSS_THRESHOLD_EUR = -1.0                 # a loss day loses more than 1 EUR

# ---------------------------------------------------------------------------
# Crisis replay (first version of the project): 2021 = calibration, 2022 = test
# ---------------------------------------------------------------------------
# Same source as the main study (Energy-Charts, FR day-ahead, UTC). No FCR in this replay.
CRISIS_YEARS = (2021, 2022)
CRISIS_CALIBRATION = ("2021-05-01", "2021-12-31")   # Jan-Apr 2021 = forecast warm-up
CRISIS_TEST = ("2022-01-01", "2022-12-31")
