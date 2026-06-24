# GPU Benchmark Notes

Environment:

- RunPod `innovative_blue_panther`
- GPU: NVIDIA A100-SXM4-80GB
- Torch: 2.4.1+cu124
- CUDA available: true

Validation:

- `python -m pytest -q`: 25 passed
- `scripts/run_gpu_validation.sh`: 3 passed, 22 deselected
- No OOMs at the current benchmark shapes

Key CUDA forward numbers after the signed-power formulation:

| Benchmark | Explicit | Optimized eager | Optimized compiled | Eager speedup | Compiled speedup | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pm_cor 1024x256` | 0.466 ms | 0.229 ms | 0.059 ms | 2.03x | 7.84x | 1.04e-7 |
| `pm_cor 4096x768` | 0.583 ms | 0.315 ms | 0.112 ms | 1.85x | 5.22x | 9.69e-8 |
| `pm_cor 8192x1024` | 1.543 ms | 0.837 ms | 0.201 ms | 1.84x | 7.68x | 9.69e-8 |
| `CoPMAuxLoss list` | 3.983 ms | 0.738 ms | 0.526 ms | 5.40x | 7.58x | 7.28e-11 |
| `CoPMAuxLoss stacked` | 3.961 ms | 0.686 ms | 0.435 ms | 5.78x | 9.11x | 5.82e-11 |

Key CUDA forward+backward numbers:

| Benchmark | Explicit | Optimized eager | Optimized compiled | Eager speedup | Compiled speedup | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pm_cor 1024x256` | 1.661 ms | 0.990 ms | 0.424 ms | 1.68x | 3.92x | 8.94e-8 |
| `pm_cor 4096x768` | 1.785 ms | 1.064 ms | 0.423 ms | 1.68x | 4.22x | 1.04e-7 |
| `pm_cor 8192x1024` | 4.257 ms | 2.639 ms | 0.454 ms | 1.61x | 9.37x | 8.94e-8 |
| `CoPMAuxLoss list` | 13.423 ms | 2.598 ms | 1.172 ms | 5.17x | 11.45x | 0 |
| `CoPMAuxLoss stacked` | 12.468 ms | 2.402 ms | 1.105 ms | 5.19x | 11.28x | 0 |

Profiler bottlenecks are mostly elementwise/autograd work:

- `aten::mul`
- `aten::where`
- `PowBackward1`
- `MulBackward0`
- vectorized elementwise kernels

Interpretation:

- The signed-power `pm_cor` optimization carries to CUDA.
- `CoPMAuxLoss` benefits strongly from avoiding the explicit per-layer loop.
- List and stacked hidden-state inputs are very close on CUDA.
- CPU pod timings are noisy and not the main optimization target.

Next optimization target:

1. Make `torch.compile` benchmarking first-class for `pm_cor` and `CoPMAuxLoss`.
2. Try lower-intermediate pure PyTorch formulations that reduce elementwise kernels and saved autograd tensors.
3. Defer custom autograd and Triton until pure PyTorch plus compile stops improving.
