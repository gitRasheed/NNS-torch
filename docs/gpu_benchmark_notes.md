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

Key CUDA forward numbers:

| Benchmark | Explicit | Optimized eager | Optimized compiled | Eager speedup | Compiled speedup | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pm_cor 1024x256` | 0.470 ms | 0.302 ms | 0.064 ms | 1.55x | 7.38x | 8.2e-8 |
| `pm_cor 4096x768` | 0.582 ms | 0.412 ms | 0.095 ms | 1.41x | 6.10x | 1.06e-7 |
| `pm_cor 8192x1024` | 1.464 ms | 1.052 ms | 0.167 ms | 1.39x | 8.74x | 1.12e-7 |
| `CoPMAuxLoss list` | 4.991 ms | 0.960 ms | 0.512 ms | 5.20x | 9.74x | 2.91e-11 |
| `CoPMAuxLoss stacked` | 4.931 ms | 0.853 ms | 0.424 ms | 5.78x | 11.63x | 8.73e-11 |

Key CUDA forward+backward numbers:

| Benchmark | Explicit | Optimized eager | Optimized compiled | Eager speedup | Compiled speedup | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pm_cor 1024x256` | 1.668 ms | 1.127 ms | 0.412 ms | 1.48x | 4.05x | 8.94e-8 |
| `pm_cor 4096x768` | 1.787 ms | 1.289 ms | 0.459 ms | 1.39x | 3.89x | 1.04e-7 |
| `pm_cor 8192x1024` | 4.527 ms | 3.228 ms | 0.582 ms | 1.40x | 7.78x | 1.01e-7 |
| `CoPMAuxLoss list` | 15.146 ms | 3.019 ms | 1.290 ms | 5.02x | 11.74x | 5.82e-11 |
| `CoPMAuxLoss stacked` | 15.385 ms | 2.798 ms | 1.104 ms | 5.50x | 13.94x | 1.46e-11 |

Profiler bottlenecks are mostly elementwise/autograd work:

- `aten::mul`
- `aten::where`
- `PowBackward1`
- `MulBackward0`
- vectorized elementwise kernels

Interpretation:

- The signed-magnitude `pm_cor` optimization carries to CUDA.
- `CoPMAuxLoss` benefits strongly from avoiding the explicit per-layer loop.
- List and stacked hidden-state inputs are very close on CUDA.
- CPU pod timings are noisy and not the main optimization target.

Next optimization target:

1. Make `torch.compile` benchmarking first-class for `pm_cor` and `CoPMAuxLoss`.
2. Try lower-intermediate pure PyTorch formulations that reduce elementwise kernels and saved autograd tensors.
3. Defer custom autograd and Triton until pure PyTorch plus compile stops improving.
