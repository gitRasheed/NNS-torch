from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nns_torch.ops.correlation import _target
from nns_torch.ops.moments import EPS, lpm_elem, upm_elem


class AsymPMComponents(NamedTuple):
    uu: torch.Tensor
    ll: torch.Tensor
    ul: torch.Tensor
    lu: torch.Tensor


def _reduce(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    return x.mean() if dim is None else x.mean(dim=dim, keepdim=keepdim)


def _split_degree(degree):
    if isinstance(degree, tuple) or isinstance(degree, list):
        return degree
    half = degree / 2.0
    return half, half


def _weights(weights, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(weights, dtype=like.dtype, device=like.device)


def asym_pm_components(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: float | torch.Tensor | None = 0.0,
    target_y: float | torch.Tensor | None = 0.0,
    degree: float | torch.Tensor | tuple = 2.0,
    dim=-1,
    keepdim: bool = False,
    eps: float = EPS,
) -> AsymPMComponents:
    degree_x, degree_y = _split_degree(degree)
    tx = _target(x, target_x, dim, keepdim)
    ty = _target(y, target_y, dim, keepdim)
    ux = upm_elem(x, tx, degree_x, eps)
    lx = lpm_elem(x, tx, degree_x, eps)
    uy = upm_elem(y, ty, degree_y, eps)
    ly = lpm_elem(y, ty, degree_y, eps)
    return AsymPMComponents(
        uu=_reduce(ux * uy, dim, keepdim),
        ll=_reduce(lx * ly, dim, keepdim),
        ul=_reduce(ux * ly, dim, keepdim),
        lu=_reduce(lx * uy, dim, keepdim),
    )


def asym_pm_cor(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: float | torch.Tensor | None = None,
    target_y: float | torch.Tensor | None = None,
    degree: float | torch.Tensor | tuple = 2.0,
    weights=(1.0, 1.0, -1.0, -1.0),
    dim=-1,
    keepdim: bool = False,
    normalize: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    comps = asym_pm_components(x, y, target_x, target_y, degree, dim, keepdim, eps)
    w = _weights(weights, comps.uu)
    score = w[0] * comps.uu + w[1] * comps.ll + w[2] * comps.ul + w[3] * comps.lu
    if not normalize:
        return score
    mag = comps.uu + comps.ll + comps.ul + comps.lu
    return score / (mag + eps)


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("degree_init must be positive")
    return math.log(math.expm1(x))


class AsymmetricCoPMLoss(nn.Module):
    """Experimental quadrant-weighted CoPM loss."""

    def __init__(
        self,
        degree_init: float = 1.0,
        weights=(1.0, 1.0, -1.0, -1.0),
        learnable_weights: bool = False,
        weight_transform: str | None = None,
        detach_final: bool = True,
        eps: float = EPS,
    ):
        super().__init__()
        self.degree_raw = nn.Parameter(torch.tensor(_inv_softplus(degree_init)))
        w = torch.tensor(weights, dtype=torch.float32)
        if learnable_weights:
            self.weight_raw = nn.Parameter(w)
        else:
            self.register_buffer("weight_raw", w)
        if weight_transform not in (None, "tanh", "sigmoid", "softmax"):
            raise ValueError("weight_transform must be None, 'tanh', 'sigmoid', or 'softmax'")
        self.weight_transform = weight_transform
        self.detach_final = detach_final
        self.eps = eps

    @property
    def degree(self):
        return F.softplus(self.degree_raw)

    @property
    def weights(self):
        if self.weight_transform == "tanh":
            return torch.tanh(self.weight_raw)
        if self.weight_transform == "sigmoid":
            return torch.sigmoid(self.weight_raw)
        if self.weight_transform == "softmax":
            return torch.softmax(self.weight_raw, dim=0)
        return self.weight_raw

    def forward(self, hidden_states, final_hidden: torch.Tensor) -> torch.Tensor:
        final = final_hidden.reshape(-1, final_hidden.size(-1))
        if self.detach_final:
            final = final.detach()

        if isinstance(hidden_states, torch.Tensor):
            h = hidden_states.reshape(hidden_states.size(0), -1, hidden_states.size(-1))
        else:
            if len(hidden_states) == 0:
                return final_hidden.new_zeros(())
            h = torch.stack([x.reshape(-1, x.size(-1)) for x in hidden_states], dim=0)

        return -asym_pm_cor(h, final, degree=self.degree, weights=self.weights, dim=-1, eps=self.eps).mean()
