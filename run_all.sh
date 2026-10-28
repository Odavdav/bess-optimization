#!/usr/bin/env bash
# Full pipeline (about 1 hour on 2 cores). Each step saves its results in results/.
set -euo pipefail
cd "$(dirname "$0")/scripts"
python 01_prepare_data.py
python 02_scenarios.py
python 03_perfect_foresight.py
python 04_backtest.py
python 05_fcr.py
python 06_crisis_2022.py
