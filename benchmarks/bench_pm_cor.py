import argparse
import time

import torch

import nns_torch as nt
from nns_torch.testing import reference as ref


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, x, y, degree, dim, iters):
    for _ in range(5):
        fn(x, y, degree=degree, dim=dim)
    sync(x.device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(x, y, degree=degree, dim=dim)
    sync(x.device)
    return out, (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "bfloat16"), default="float32")
    parser.add_argument("--shape", type=int, nargs=2, default=(4096, 768))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--degree", type=float, default=1.3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    shape = (64, 32) if args.smoke else tuple(args.shape)
    iters = 2 if args.smoke else args.iters
    dtype = getattr(torch, args.dtype)
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA device does not support bfloat16")

    x = torch.randn(*shape, device=device, dtype=dtype)
    y = torch.randn(*shape, device=device, dtype=dtype)

    cases = [("explicit", ref.pm_cor_reference), ("optimized", nt.pm_cor)]
    if hasattr(torch, "compile") and not args.smoke:
        cases.append(("optimized_compile", torch.compile(nt.pm_cor)))

    baseline = None
    for name, fn in cases:
        out, seconds = bench(fn, x, y, args.degree, -1, iters)
        baseline = out if baseline is None else baseline
        max_diff = (out - baseline).abs().max().item()
        print(f"{name:18s} {seconds * 1e3:8.3f} ms/iter max_diff={max_diff:.3g}")


if __name__ == "__main__":
    main()
