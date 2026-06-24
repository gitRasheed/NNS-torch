#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python examples/classification_margin.py
python benchmarks/bench_pm_cor.py --smoke
