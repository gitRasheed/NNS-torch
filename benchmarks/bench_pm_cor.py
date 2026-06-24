import argparse
import time

import torch

import nns_torch as nt
from nns_torch.testing import reference as ref

DEFAULT_SHAPES = [(1024, 256), (4096, 768), (8192, 1024)]


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


def run_shape(shape, args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(*shape, device=device, dtype=dtype)
    y = torch.randn(*shape, device=device, dtype=dtype)

    explicit, explicit_seconds = bench(ref.pm_cor_reference, x, y, args.degree, -1, args.iters)
    optimized, optimized_seconds = bench(nt.pm_cor, x, y, args.degree, -1, args.iters)
    max_diff = (optimized - explicit).abs().max().item()
    speedup = explicit_seconds / optimized_seconds
    print(
        f"{shape[0]}x{shape[1]} explicit={explicit_seconds * 1e3:.3f}ms "
        f"optimized={optimized_seconds * 1e3:.3f}ms speedup={speedup:.2f}x max_diff={max_diff:.3g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "bfloat16"), default="float32")
    parser.add_argument("--shape", type=int, nargs=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--degree", type=float, default=1.3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    dtype = getattr(torch, args.dtype)
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA device does not support bfloat16")

    if args.smoke:
        args.iters = 2
        run_shape((64, 32), args)
        return

    shapes = [tuple(args.shape)] if args.shape else DEFAULT_SHAPES
    for shape in shapes:
        run_shape(shape, args)


if __name__ == "__main__":
    main()
