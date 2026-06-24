import argparse
import time

import torch

import nns_torch as nt
from nns_torch.nn import CoPMAuxLoss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def loop_loss(hidden, final, degree):
    flat_final = final.reshape(-1, final.size(-1)).detach()
    losses = []
    for h in hidden:
        losses.append(nt.pm_cor(h.reshape(-1, h.size(-1)), flat_final, degree=degree, dim=-1))
    return -torch.stack(losses).mean()


def bench(fn, hidden, final, loss_fn, iters):
    for _ in range(2):
        loss = fn(hidden, final, loss_fn)
        loss.backward()
        for h in hidden:
            h.grad = None
        loss_fn.zero_grad(set_to_none=True)
    sync(final.device)
    start = time.perf_counter()
    for _ in range(iters):
        loss = fn(hidden, final, loss_fn)
        loss.backward()
        for h in hidden:
            h.grad = None
        loss_fn.zero_grad(set_to_none=True)
    sync(final.device)
    return loss.detach(), (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    hidden = [
        torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=True)
        for _ in range(args.layers)
    ]
    final = torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=True)
    loss_fn = CoPMAuxLoss(degree_init=1.3).to(device)

    explicit, explicit_seconds = bench(lambda h, f, m: loop_loss(h, f, m.degree), hidden, final, loss_fn, args.iters)
    optimized, optimized_seconds = bench(lambda h, f, m: m(h, f), hidden, final, loss_fn, args.iters)
    speedup = explicit_seconds / optimized_seconds
    max_diff = (optimized - explicit).abs().item()
    print(
        f"L={args.layers} B={args.batch} T={args.tokens} C={args.channels} "
        f"explicit={explicit_seconds * 1e3:.3f}ms optimized={optimized_seconds * 1e3:.3f}ms "
        f"speedup={speedup:.2f}x max_diff={max_diff:.3g}"
    )


if __name__ == "__main__":
    main()
