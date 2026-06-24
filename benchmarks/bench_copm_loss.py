import argparse
import time

import torch

import nns_torch as nt
from nns_torch.nn import CoPMAuxLoss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def explicit_loss(hidden, final, degree):
    final = final.reshape(-1, final.size(-1)).detach()
    losses = []
    for h in hidden:
        losses.append(nt.pm_cor(h.reshape(-1, h.size(-1)), final, degree=degree, dim=-1))
    return -torch.stack(losses).mean()


def clear_grads(hidden, final, loss_fn):
    if isinstance(hidden, list):
        for h in hidden:
            h.grad = None
    else:
        hidden.grad = None
    final.grad = None
    loss_fn.zero_grad(set_to_none=True)


def run_once(fn, hidden, final, loss_fn, backward):
    clear_grads(hidden, final, loss_fn)
    loss = fn(hidden, final, loss_fn)
    if backward:
        loss.backward()
    return loss.detach()


def bench(fn, hidden, final, loss_fn, backward, iters):
    for _ in range(3):
        loss = run_once(fn, hidden, final, loss_fn, backward)
    sync(final.device)
    start = time.perf_counter()
    for _ in range(iters):
        loss = run_once(fn, hidden, final, loss_fn, backward)
    sync(final.device)
    return loss, (time.perf_counter() - start) / iters


def make_inputs(args, stacked):
    device = torch.device(args.device)
    base = torch.randn(args.layers, args.batch, args.tokens, args.channels, device=device)
    final = torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=args.backward)
    if stacked:
        hidden = base.detach().clone().requires_grad_(args.backward)
    else:
        hidden = [base[i].detach().clone().requires_grad_(args.backward) for i in range(args.layers)]
    return hidden, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--iters", type=int, default=5)
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
    if args.smoke:
        args.layers, args.batch, args.tokens, args.channels, args.iters = 2, 2, 16, 64, 2

    mode = "backward" if args.backward else "forward"
    for stacked in (False, True):
        hidden, final = make_inputs(args, stacked)
        loss_fn = CoPMAuxLoss(degree_init=1.3).to(device)
        layout = "stacked" if stacked else "list"

        cases = [
            ("explicit", lambda h, f, m: explicit_loss(h, f, m.degree)),
            ("optimized", lambda h, f, m: m(h, f)),
        ]
        if args.compile:
            compiled = torch.compile(lambda h, f, m: m(h, f))
            cases.append(("compiled", compiled))

        baseline = None
        timings = {}
        diffs = {}
        for name, fn in cases:
            loss, seconds = bench(fn, hidden, final, loss_fn, args.backward, args.iters)
            if baseline is None:
                baseline = loss
            timings[name] = seconds
            diffs[name] = (loss - baseline).abs().item()

        line = [f"L={args.layers}", f"B={args.batch}", f"T={args.tokens}", f"C={args.channels}", layout, mode]
        for name, _ in cases:
            line.append(f"{name}={timings[name] * 1e3:.3f}ms")
            if name != "explicit":
                line.append(f"{name}_speedup={timings['explicit'] / timings[name]:.2f}x")
                line.append(f"{name}_max_diff={diffs[name]:.3g}")
        print(" ".join(line))


if __name__ == "__main__":
    main()
