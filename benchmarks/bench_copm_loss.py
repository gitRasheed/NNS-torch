import argparse
import time

import torch

from nns_torch.nn import CoPMAuxLoss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--iters", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    hidden = [
        torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=True)
        for _ in range(args.layers)
    ]
    final = torch.randn(args.batch, args.tokens, args.channels, device=device, requires_grad=True)
    loss_fn = CoPMAuxLoss(degree_init=1.3).to(device)

    for _ in range(3):
        loss_fn(hidden, final).backward()
        for h in hidden:
            h.grad = None
    sync(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        loss = loss_fn(hidden, final)
        loss.backward()
        for h in hidden:
            h.grad = None
    sync(device)
    print(f"copm_loss {(time.perf_counter() - start) * 1e3 / args.iters:.3f} ms/iter loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

