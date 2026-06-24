from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _reduce(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    return x.mean() if dim is None else x.mean(dim=dim, keepdim=keepdim)


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
    eps: float = EPS,
) -> torch.Tensor:
    """Upper partial moment reduced by mean."""
    return _reduce(upm_elem(x, target=target, degree=degree, eps=eps), dim, keepdim)


def lpm(
    x: torch.Tensor,
    target: float | torch.Tensor = 0.0,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Lower partial moment reduced by mean."""
    return _reduce(lpm_elem(x, target=target, degree=degree, eps=eps), dim, keepdim)

