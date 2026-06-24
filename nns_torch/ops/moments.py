from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _reduce(
    x: torch.Tensor,
    dim=None,
    keepdim: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum() if dim is None else x.sum(dim=dim, keepdim=keepdim)
    if reduction == "mean":
        return x.mean() if dim is None else x.mean(dim=dim, keepdim=keepdim)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def upm_elem(
    x: torch.Tensor,
    target: float | torch.Tensor = 0.0,
    degree: float | torch.Tensor = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Elementwise upper partial moment: ``max(x - target, 0) ** degree``."""
    hinge = F.relu(x - target)
    return torch.where(hinge > 0, hinge.clamp_min(eps).pow(degree), torch.zeros_like(hinge))


def lpm_elem(
    x: torch.Tensor,
    target: float | torch.Tensor = 0.0,
    degree: float | torch.Tensor = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Elementwise lower partial moment: ``max(target - x, 0) ** degree``."""
    hinge = F.relu(target - x)
    return torch.where(hinge > 0, hinge.clamp_min(eps).pow(degree), torch.zeros_like(hinge))


def upm(
    x: torch.Tensor,
    target: float | torch.Tensor = 0.0,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """Upper partial moment reduced by mean, sum, or not reduced."""
    return _reduce(upm_elem(x, target=target, degree=degree, eps=eps), dim, keepdim, reduction)


def lpm(
    x: torch.Tensor,
    target: float | torch.Tensor = 0.0,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """Lower partial moment reduced by mean, sum, or not reduced."""
    return _reduce(lpm_elem(x, target=target, degree=degree, eps=eps), dim, keepdim, reduction)
