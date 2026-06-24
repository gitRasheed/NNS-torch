from __future__ import annotations

import torch

from .moments import EPS, _check_degree, _reduce, _signed_power, lpm_elem, upm_elem


def _target(x: torch.Tensor, target, dim, keepdim: bool) -> torch.Tensor | float:
    if target is not None:
        return target
    return x.mean() if dim is None else x.mean(dim=dim, keepdim=True)


def _co_quadrants(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: float | torch.Tensor = 1.0,
    target_x: float | torch.Tensor | None = None,
    target_y: float | torch.Tensor | None = None,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
):
    tx = _target(x, target_x, dim, keepdim)
    ty = _target(y, target_y, dim, keepdim)
    half = degree / 2.0
    x_dn = lpm_elem(x, target=tx, degree=half, eps=eps)
    x_up = upm_elem(x, target=tx, degree=half, eps=eps)
    y_dn = lpm_elem(y, target=ty, degree=half, eps=eps)
    y_up = upm_elem(y, target=ty, degree=half, eps=eps)
    return (
        _reduce(x_dn * y_dn, dim, keepdim),
        _reduce(x_up * y_up, dim, keepdim),
        _reduce(x_dn * y_up, dim, keepdim),
        _reduce(x_up * y_dn, dim, keepdim),
    )


def co_lpm(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: float | torch.Tensor | None = None,
    target_y: float | torch.Tensor | None = None,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Concordant lower co-partial moment, reduced by mean."""
    return _co_quadrants(x, y, degree, target_x, target_y, dim, keepdim, eps)[0]


def co_upm(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: float | torch.Tensor | None = None,
    target_y: float | torch.Tensor | None = None,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Concordant upper co-partial moment, reduced by mean."""
    return _co_quadrants(x, y, degree, target_x, target_y, dim, keepdim, eps)[1]


def pm_cor_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Explicit quadrant formula: ``CLPM + CUPM - DLPM - DUPM`` over total mass."""
    clpm, cupm, dlpm, dupm = _co_quadrants(x, y, degree, None, None, dim, keepdim, eps)
    return (clpm + cupm - dlpm - dupm) / (clpm + cupm + dlpm + dupm + eps)


def pm_cor(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: float | torch.Tensor = 1.0,
    dim=None,
    keepdim: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Optimized signed-magnitude partial-moment correlation."""
    _check_degree(degree)
    target_x = _target(x, None, dim, keepdim)
    target_y = _target(y, None, dim, keepdim)
    half = degree / 2.0

    dx = x - target_x
    dy = y - target_y
    x_signed = _signed_power(dx, half, eps)
    y_signed = _signed_power(dy, half, eps)
    signed_mass = x_signed * y_signed
    numerator = _reduce(signed_mass, dim, keepdim)
    denominator = _reduce(signed_mass.abs(), dim, keepdim)
    return numerator / (denominator + eps)
