import argparse
import time

import torch

from nns_torch import experimental as exp
from nns_torch.nn import CoPMAuxLoss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def clear_grads(hidden, final, *mods):
    if isinstance(hidden, list):
        for h in hidden:
            h.grad = None
    else:
        hidden.grad = None
    final.grad = None
    for mod in mods:
        mod.zero_grad(set_to_none=True)


def run_once(fn, hidden, final, backward, *mods):
    clear_grads(hidden, final, *mods)
    loss = fn(hidden, final)
    if backward:
        loss.backward()
    return loss.detach()


def bench(fn, hidden, final, backward, iters, *mods):
    for _ in range(3):
        loss = run_once(fn, hidden, final, backward, *mods)
    sync(final.device)
    start = time.perf_counter()
    for _ in range(iters):
        loss = run_once(fn, hidden, final, backward, *mods)
    sync(final.device)
    return loss, (time.perf_counter() - start) / iters


def make_inputs(args, stacked):
    device = torch.device(args.device)
    base = torch.randn(args.layers, args.batch, args.tokens, args.channels, device=device)
    final = torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=args.backward)
    if stacked:
        return base.detach().clone().requires_grad_(args.backward), final
    return [base[i].detach().clone().requires_grad_(args.backward) for i in range(args.layers)], final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.smoke:
        args.layers, args.batch, args.tokens, args.channels, args.iters = 2, 2, 16, 64, 2

    mode = "backward" if args.backward else "forward"
    for stacked in (False, True):
        hidden, final = make_inputs(args, stacked)
        sym = CoPMAuxLoss(degree_init=1.3).to(device)
        asym = exp.AsymmetricCoPMLoss(degree_init=1.3).to(device)
        layout = "stacked" if stacked else "list"

        cases = [
            ("symmetric", lambda h, f: sym(h, f), sym),
            ("asym", lambda h, f: asym(h, f), asym),
        ]
        if args.compile:
            cases.append(("asym_compiled", torch.compile(lambda h, f: asym(h, f)), asym))

        baseline = None
        timings = {}
        diffs = {}
        for name, fn, mod in cases:
            loss, seconds = bench(fn, hidden, final, args.backward, args.iters, sym, asym)
            if baseline is None:
                baseline = loss
            timings[name] = seconds
            diffs[name] = (loss - baseline).abs().item()

        line = [f"L={args.layers}", f"B={args.batch}", f"T={args.tokens}", f"C={args.channels}", layout, mode]
        for name, _, _ in cases:
            line.append(f"{name}={timings[name] * 1e3:.3f}ms")
            if name != "symmetric":
                line.append(f"{name}_ratio={timings[name] / timings['symmetric']:.2f}x")
                line.append(f"{name}_max_diff={diffs[name]:.3g}")
        print(" ".join(line))


if __name__ == "__main__":
    main()
