import argparse
import time

import torch

import nns_torch as nt
from nns_torch.testing import reference as ref

DEFAULT_SHAPES = [(1024, 256), (4096, 768), (8192, 1024)]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_once(fn, x, y, degree, backward):
    if x.grad is not None:
        x.grad = None
    if y.grad is not None:
        y.grad = None
    out = fn(x, y, degree=degree, dim=-1)
    if backward:
        out.sum().backward()
    return out.detach()


def bench(fn, x, y, degree, backward, iters):
    for _ in range(3):
        out = run_once(fn, x, y, degree, backward)
    sync(x.device)
    start = time.perf_counter()
    for _ in range(iters):
        out = run_once(fn, x, y, degree, backward)
    sync(x.device)
    return out, (time.perf_counter() - start) / iters


def run_shape(shape, args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=args.backward)
    y = torch.randn(*shape, device=device, dtype=dtype, requires_grad=args.backward)

    cases = [("explicit", ref.pm_cor_reference), ("optimized", nt.pm_cor)]
    if args.compile:
        cases.append(("compiled", torch.compile(nt.pm_cor)))

    baseline = None
    timings = {}
    diffs = {}
    for name, fn in cases:
        out, seconds = bench(fn, x, y, args.degree, args.backward, args.iters)
        if baseline is None:
            baseline = out
        timings[name] = seconds
        diffs[name] = (out - baseline).abs().max().item()

    mode = "backward" if args.backward else "forward"
    line = [f"{shape[0]}x{shape[1]}", mode]
    for case_name, _ in cases:
        line.append(f"{case_name}={timings[case_name] * 1e3:.3f}ms")
        if case_name != "explicit":
            line.append(f"{case_name}_speedup={timings['explicit'] / timings[case_name]:.2f}x")
            line.append(f"{case_name}_max_diff={diffs[case_name]:.3g}")
    print(" ".join(line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "bfloat16"), default="float32")
    parser.add_argument("--shape", type=int, nargs=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--degree", type=float, default=1.3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.forward_only and args.backward:
        raise SystemExit("choose either --forward-only or --backward")

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
