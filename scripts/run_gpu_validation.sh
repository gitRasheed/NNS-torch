#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.cuda.get_device_name())
PY

python -m pytest -q -m "cuda or compile"
python benchmarks/bench_pm_cor.py --smoke --device cuda
