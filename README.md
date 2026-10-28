# Battery storage optimization on the French power market

Day-ahead arbitrage, price uncertainty (Monte Carlo, mean-CVaR) and primary reserve (FCR) for a
**50 MW / 100 MWh** battery, on French market data **2023-2025**. Every setting is chosen on 2024; 2025 is an
out-of-sample test. The full story, with figures, is in [`notebooks/bess_analysis.ipynb`](notebooks/bess_analysis.ipynb). The same content
without code is in [`notebooks/report.pdf`](notebooks/report.pdf).

## Results (test 2025)

| | Profit | |
|---|---|---|
| Perfect foresight, arbitrage only | €1.76M | theoretical ceiling |
| Deterministic (ARX forecast) | €1.23M | 70% of the ceiling |
| Mean-CVaR, λ = 0.375 (chosen on 2024) | €1.21M | −2% profit (not significant), +18% CVaR on the 10% worst days (significant) |
| Arbitrage + FCR, λ = 0.375 | €3.64M | 84% from FCR, with the battery's own impact on the FCR price |

- **Risk sits on calm days**: there, the deterministic strategy loses money one day in two.
- **The battery moves the FCR price**: −26% at 50 MW in 2025 (−36% in 2024), measured by replaying every auction
  from the anonymous bid lists. A price-taker model would have said €4.56M instead of €3.64M.

## Approach

1. **Deterministic MILP** (PuLP + HiGHS): power, capacity, efficiency, binary against simultaneous charge and
   discharge, linear degradation cost per MWh discharged; perfect-foresight benchmark.
2. **Uncertainty**: hourly ARX forecast, Monte Carlo scenarios of whole days of errors (Gaussian, Cholesky),
   convergence study (200 scenarios), mean-CVaR linearized with Rockafellar-Uryasev.
3. **Out-of-sample backtest**: decisions with the information of the day before, calibration 2024 / test 2025,
   λ rule fixed in advance, moving-block bootstrap for significance, analysis by volatility regime.
   A drawdown-budget variant is also tested, including on a replay of the 2022 crisis.
4. **FCR extension**: 4-hour capacity blocks stacked with arbitrage. Decisions use the naive FCR forecast × a
   price-impact factor of month M−2 (bid lists are published monthly); reserved MW are paid the replayed price.

## Structure

```
├── src/bess/       # model, optimizers, scenarios, backtest, FCR replay, analysis, figures; config.py holds every parameter
├── scripts/        # heavy computations, run in order (run_all.sh, about one hour on 2 cores), write to results/
├── results/        # CSV read by the notebook
├── notebooks/      # the project story
├── tests/          # consistency tests (pytest)
└── data/           # raw/ (downloads, not versioned) and processed/
```

**Data** (`data/raw/`, not versioned; `data/processed/` is what the code reads):

- French day-ahead prices 2021-2025, [energy-charts.info](https://energy-charts.info) exports in UTC,
  `energy_charts_{year}.csv`. Licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), source
  **Bundesnetzagentur | SMARD.de**. Changes: 15-minute prices (since October 2025) averaged to hourly; stored as
  hourly UTC series in `data/processed/prices_{year}.csv`. 2021-2022 are only used for the crisis replay.
- FCR results and anonymous bid lists from [regelleistung.net](https://www.regelleistung.net),
  `fcr_{2023,2024,2025}.xlsx` and `fcr_bids_{2023,2024,2025}.xlsx` (2023 bids only give the price-impact factors of
  January-February 2024). The bid lists are not redistributed; the replayed price curves derived from them are in
  `data/processed/fcr_price_curves.csv`, so the whole pipeline after the bid replay runs without them.

## Limitations

- Forecast from past prices only (no load, weather or gas); scenarios overestimate the daily spread by ~20%.
- FCR price impact is static (other bidders do not react); activation energy is neglected.
- Daily horizon, hourly prices (15-minute prices since October 2025 are averaged), linear degradation.

## Licence

Data keep their own licences: day-ahead prices CC BY 4.0 (Bundesnetzagentur | SMARD.de);
FCR data from regelleistung.net are not redistributed, only results derived from them.
