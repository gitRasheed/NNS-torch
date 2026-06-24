import torch
import torch.nn.functional as F

from nns_torch.nn import PMMarginLoss


def make_data(n=256):
    torch.manual_seed(0)
    x0 = torch.randn(n // 2, 2) * 0.6 + torch.tensor([-1.0, -1.0])
    x1 = torch.randn(n // 2, 2) * 0.6 + torch.tensor([1.0, 1.0])
    x = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    return x, y


def main():
    x, y = make_data()
    model = torch.nn.Linear(2, 2)
    margin_loss = PMMarginLoss(margin=1.0, degree_init=2.0)
    opt = torch.optim.Adam([*model.parameters(), *margin_loss.parameters()], lr=0.05)

    for _ in range(100):
        logits = model(x)
        loss = F.cross_entropy(logits, y) + 0.1 * margin_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = (model(x).argmax(dim=-1) == y).float().mean()
    print(f"loss={loss.item():.4f} acc={acc.item():.3f} margin={margin_loss.margin:.3f}")


if __name__ == "__main__":
    main()
