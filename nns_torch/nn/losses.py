from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nns_torch.functional import lpm, lpm_elem, pm_cor, upm_elem
from nns_torch.ops.moments import EPS


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("degree_init must be positive")
    return math.log(math.expm1(x))


class LPMAuxLoss(nn.Module):
    """Normalized partial-moment utility on the correct-class logit."""

    def __init__(self, degree_init: float = 1.0, target_init: float = 0.0, eps: float = EPS):
        super().__init__()
        self.degree_raw = nn.Parameter(torch.tensor(_inv_softplus(degree_init)))
        self.target = nn.Parameter(torch.tensor(float(target_init)))
        self.eps = eps

    @property
    def degree(self):
        return F.softplus(self.degree_raw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        correct = flat_logits[torch.arange(flat_targets.numel(), device=flat_logits.device), flat_targets]
        up = upm_elem(correct, target=self.target, degree=self.degree, eps=self.eps)
        down = lpm_elem(correct, target=self.target, degree=self.degree, eps=self.eps)
        return -((up - down) / (up + down + self.eps)).mean()


class CoPMAuxLoss(nn.Module):
    """Cross-layer partial-moment correlation against the final representation."""

    def __init__(self, degree_init: float = 1.0, detach_final: bool = True, eps: float = EPS):
        super().__init__()
        self.degree_raw = nn.Parameter(torch.tensor(_inv_softplus(degree_init)))
        self.detach_final = detach_final
        self.eps = eps

    @property
    def degree(self):
        return F.softplus(self.degree_raw)

    def forward(self, hidden_states, final_hidden: torch.Tensor) -> torch.Tensor:
        final = final_hidden.reshape(-1, final_hidden.size(-1))
        if self.detach_final:
            final = final.detach()

        if isinstance(hidden_states, torch.Tensor):
            H = hidden_states.reshape(hidden_states.size(0), -1, hidden_states.size(-1))
        else:
            if len(hidden_states) == 0:
                return final_hidden.new_zeros(())
            H = torch.stack([h.reshape(-1, h.size(-1)) for h in hidden_states], dim=0)

        return -pm_cor(H, final, degree=self.degree, dim=-1, eps=self.eps).mean()


class PMMarginLoss(nn.Module):
    """Lower partial-moment loss on the correct-vs-best-other logit margin."""

    def __init__(self, margin: float = 1.0, degree_init: float = 1.0, eps: float = EPS):
        super().__init__()
        self.margin = float(margin)
        self.degree_raw = nn.Parameter(torch.tensor(_inv_softplus(degree_init)))
        self.eps = eps

    @property
    def degree(self):
        return F.softplus(self.degree_raw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        row = torch.arange(flat_targets.numel(), device=flat_logits.device)
        correct = flat_logits[row, flat_targets]
        other = flat_logits.masked_fill(F.one_hot(flat_targets, flat_logits.size(-1)).bool(), -torch.inf)
        margins = correct - other.max(dim=-1).values
        return lpm(margins, target=self.margin, degree=self.degree, eps=self.eps)
