from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nns_torch.functional import upm_elem


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("degree_init must be positive")
    return math.log(math.expm1(x))


class PMAActivation(nn.Module):
    """Learned upper partial-moment activation."""

    def __init__(self, dim: int, degree_init: float = 1.0, target_init: float = 0.0):
        super().__init__()
        self.degree_raw = nn.Parameter(torch.tensor(_inv_softplus(degree_init)))
        self.target = nn.Parameter(torch.full((dim,), float(target_init)))

    @property
    def degree(self):
        return F.softplus(self.degree_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upm_elem(x, target=self.target, degree=self.degree)

