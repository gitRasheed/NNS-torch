# NNS-Torch

PyTorch-native partial-moment tensor operators for differentiable training loops.

```python
import torch
import nns_torch as nt

x = torch.randn(32, 128, 768)
y = torch.randn(32, 128, 768)

downside = nt.lpm(x, target=0.0, degree=2.0, dim=-1)
upside = nt.upm(x, target=0.0, degree=2.0, dim=-1)
corr = nt.pm_cor(x, y, degree=2.0, dim=-1)

loss = nt.nn.CoPMAuxLoss(degree_init=1.3)([x], y)
```

`degree` is the total partial-moment order. For co-moments and `pm_cor`, each side uses `degree / 2`, so the product has total order `degree`.

The package is pure PyTorch: no NNS-Python runtime dependency and no required CUDA extension.

## API Semantics

- `target` is the benchmark threshold. Values exactly equal to `target` contribute zero.
- `degree` is the partial-moment order and must be non-negative for Python scalar inputs. Learned module degrees are constrained with `softplus`.
- `lpm` and `upm` support `reduction="mean"`, `"sum"`, and `"none"` plus PyTorch `dim` / `keepdim` semantics.
- `pm_cor` is currently mean-centered along `dim`; it does not accept explicit targets yet.
- `co_lpm`, `co_upm`, and `pm_cor` use `degree / 2` per side so the product has total order `degree`.

## Development

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python numpy pytest
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest -q
```

CPU validation, matching normal CI:

```bash
scripts/run_cpu_validation.sh
```

Manual GPU validation, for a self-hosted CUDA runner:

```bash
scripts/run_gpu_validation.sh
```

CUDA tests skip automatically on CPU-only machines. The manual GPU workflow fails early if CUDA is unavailable.

## Docs

- [Asymmetric CoPM design note](docs/design/asymmetric_copm.md)
- [GPU benchmark notes](docs/benchmarks/gpu_benchmark_notes.md)
- [Current status report](docs/reports/nns_torch_status.md)

CPU benchmark:

```bash
python benchmarks/bench_pm_cor.py
python benchmarks/bench_copm_loss.py
```

`bench_pm_cor.py --smoke` stays tiny for CI.
