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
