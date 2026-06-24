from __future__ import annotations

import torch

from nns_torch.ops.moments import EPS


def _reduce(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    return x.mean() if dim is None else x.mean(dim=dim, keepdim=keepdim)


def lpm_elem_reference(x, target=0.0, degree=1.0, eps: float = EPS):
    below = target - x
    return torch.where(below > 0, below.clamp_min(eps).pow(degree), torch.zeros_like(x))


def upm_elem_reference(x, target=0.0, degree=1.0, eps: float = EPS):
    above = x - target
    return torch.where(above > 0, above.clamp_min(eps).pow(degree), torch.zeros_like(x))


def lpm_reference(x, target=0.0, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    return _reduce(lpm_elem_reference(x, target, degree, eps), dim, keepdim)


def upm_reference(x, target=0.0, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    return _reduce(upm_elem_reference(x, target, degree, eps), dim, keepdim)


def _targets(x, y, dim):
    if dim is None:
        return x.mean(), y.mean()
    return x.mean(dim=dim, keepdim=True), y.mean(dim=dim, keepdim=True)


def _quadrants(x, y, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    tx, ty = _targets(x, y, dim)
    half = degree / 2.0
    x_dn = lpm_elem_reference(x, tx, half, eps)
    x_up = upm_elem_reference(x, tx, half, eps)
    y_dn = lpm_elem_reference(y, ty, half, eps)
    y_up = upm_elem_reference(y, ty, half, eps)
    return (
        _reduce(x_dn * y_dn, dim, keepdim),
        _reduce(x_up * y_up, dim, keepdim),
        _reduce(x_dn * y_up, dim, keepdim),
        _reduce(x_up * y_dn, dim, keepdim),
    )


def co_lpm_reference(x, y, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    return _quadrants(x, y, degree, dim, keepdim, eps)[0]


def co_upm_reference(x, y, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    return _quadrants(x, y, degree, dim, keepdim, eps)[1]


def pm_cor_reference(x, y, degree=1.0, dim=None, keepdim: bool = False, eps: float = EPS):
    clpm, cupm, dlpm, dupm = _quadrants(x, y, degree, dim, keepdim, eps)
    return (clpm + cupm - dlpm - dupm) / (clpm + cupm + dlpm + dupm + eps)

