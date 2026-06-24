# NNS-Torch Status

## Current Scope

NNS-Torch is an MVP PyTorch-native partial-moment tensor library. It currently
focuses on differentiable training primitives rather than broad NNS-Python API
coverage.

Core primitives:

- `lpm_elem`, `upm_elem`
- `lpm`, `upm`
- `co_lpm`, `co_upm`
- `pm_cor`
- explicit reference `pm_cor_reference`

Neural modules:

- `LPMAuxLoss`
- `CoPMAuxLoss`
- `PMMarginLoss`
- `PMAActivation`

## Validation

Latest A100 validation:

- `python -m pytest -q`: 27 passed
- `scripts/run_cpu_validation.sh`: 27 passed
- `scripts/run_gpu_validation.sh`: 4 passed, 23 deselected

The PM-Network compatibility smoke path covers:

- hidden states as a list of `[B, T, C]` tensors
- hidden states as a stacked `[L, B, T, C]` tensor
- final hidden `[B, T, C]`
- scalar finite `CoPMAuxLoss`
- scalar finite `LPMAuxLoss`
- backward and finite gradients
- stacked `CoPMAuxLoss` under `torch.compile` when available

## A100 Primitive Benchmarks

Environment:

- GPU: NVIDIA A100-SXM4-80GB
- Torch: 2.4.1+cu124
- CUDA available: true

`pm_cor`, CUDA forward:

| Shape | Explicit | Optimized eager | Optimized compiled |
| --- | ---: | ---: | ---: |
| `1024x256` | 0.466 ms | 0.229 ms | 0.059 ms |
| `4096x768` | 0.583 ms | 0.315 ms | 0.112 ms |
| `8192x1024` | 1.543 ms | 0.837 ms | 0.201 ms |

`pm_cor`, CUDA forward+backward:

| Shape | Explicit | Optimized eager | Optimized compiled |
| --- | ---: | ---: | ---: |
| `1024x256` | 1.661 ms | 0.990 ms | 0.424 ms |
| `4096x768` | 1.785 ms | 1.064 ms | 0.423 ms |
| `8192x1024` | 4.257 ms | 2.639 ms | 0.454 ms |

`CoPMAuxLoss`, `L=12 B=8 T=128 C=768`, CUDA forward+backward:

| Layout | Explicit | Optimized eager | Optimized compiled |
| --- | ---: | ---: | ---: |
| list | 13.127 ms | 2.599 ms | 1.304 ms |
| stacked | 13.330 ms | 2.403 ms | 1.162 ms |

Stacked tensors are slightly faster on A100, but list input remains viable.

## PM-Network 500-Step Gate

This was run as a PM-Network end-to-end timing gate, not as an NNS-Torch
integration. Primitive timings above are not apples-to-apples with full
training-loop timings because the PM-Network run includes the model forward,
attention, optimizer, data movement, evaluation, `torch.compile` startup, and
data/token cache setup.

Source:

- PM-Network branch: `main`
- PM-Network source hash: `abe1e4c`
- Source tree was dirty when copied to the pod.

Command:

```bash
cd /workspace/PM-Network/experiments
BATCH=8 GRAD_ACCUM=8 MODES="aux_copm" SEEDS="0" STEPS=500 EXTRA="--compile" PY=python3 RESUME=0 ./run_wikitext103_125m_sweep.sh
```

Run:

- Mode: `aux_copm`
- Dataset/tokenization: `wikitext103/bpe`
- Model: 12 layers, 768 dim, 12 heads, 1024 context, 124.4M params
- Precision: bf16
- GPU: NVIDIA A100-SXM4-80GB
- Artifact folder: `/workspace/pmnet_artifacts/nns_torch_500step_gate`

Results:

- Outer wall clock: 815.797 s
- Training-loop time reported by PM-Network: 352.935 s
- Training seconds/step: 0.7059 s/step
- Effective tokens/step: 65,536
- Training throughput: about 92,840 tokens/s
- Step 500 train loss: 4.7319
- Step 500 val loss: 4.7998
- Learned aux degree: 0.9775
- Learned aux tau: -0.0133
- Learned CoPM degree: 1.1083

Historical PM-Network anchors:

- original/loop-ish: about 386 s
- fusion: about 358 s
- optimized PM-Network: about 343 s
- best old optimized anchor: about 0.685 s/step

The 500-step training-loop result here was 352.935 s, or 0.7059 s/step. It is
close to the old optimized PM-Network timing but not faster. Because this run
used PM-Network itself rather than NNS-Torch-integrated modules, it does not
prove NNS-Torch end-to-end speedup. It does show the performance target for an
NNS-Torch integration: improve the CoPM path without disturbing the rest of the
training loop.

## Current Read

NNS-Torch primitive timings are strong enough to justify an integration trial
inside PM-Network, especially around compiled `pm_cor` and stacked
`CoPMAuxLoss`. The next useful step is a minimal PM-Network branch that swaps
only the existing PM ops/modules to NNS-Torch equivalents and reruns this same
500-step gate.
