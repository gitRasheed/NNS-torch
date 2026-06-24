# Asymmetric CoPM Design Note

Current `CoPMAuxLoss` is symmetric: it optimizes the signed `pm_cor` score where concordant quadrants increase the score and divergent quadrants decrease it.

An asymmetric variant should stay experimental until training evidence says it helps. The obvious reference form is the four explicit quadrant moments:

- `UU`: x upper, y upper
- `LL`: x lower, y lower
- `UL`: x upper, y lower
- `LU`: x lower, y upper

A minimal learnable version could apply four scalar quadrant weights before summing. That gives the model separate control over upside concordance, downside concordance, and the two divergent tails.

Experimental API:

```python
from nns_torch import experimental as ntx

components = ntx.asym_pm_components(x, y, target_x=0.0, target_y=0.0, degree=2.0, dim=-1)
score = ntx.asym_pm_cor(x, y, degree=2.0, weights=(1.0, 1.0, -1.0, -1.0), dim=-1)
loss = ntx.AsymmetricCoPMLoss(weights=(1.0, 1.0, -1.0, -1.0), learnable_weights=True)
```

`asym_pm_components` returns `(uu, ll, ul, lu)`. The scalar `degree` is split across x and y as `degree / 2` per side, matching the existing CoPM convention. A two-item degree tuple can be passed when each side should use a different order.

The default score uses:

```text
(UU + LL - UL - LU) / (UU + LL + UL + LU + eps)
```

That matches the current signed symmetric `pm_cor` when the same targets are used. `asym_pm_cor` defaults to mean targets, like `pm_cor`; pass explicit `target_x` and `target_y` to study fixed-threshold asymmetry.

Why experimental:

- The signs and useful parameterization of quadrant weights are still research questions.
- Learnable weights can change the objective meaning, not just its speed.
- The explicit quadrant form does more elementwise work than optimized signed `pm_cor`.

Performance expectation:

- Symmetric `pm_cor` remains the fast core path.
- Asymmetric CoPM is useful only if separate `UL` and `LU` behavior improves training enough to pay for the extra quadrant work.

Use it for small controlled experiments first, not as the default PM-Network objective.

Tiny sanity example:

```python
import torch
from nns_torch import experimental as exp

x = torch.tensor([-2.0, -1.0, 1.0, 3.0])
y = torch.tensor([2.0, -3.0, 4.0, -1.0])

components = exp.asym_pm_components(x, y, target_x=0.0, target_y=0.0, degree=2.0, dim=0)
symmetric = exp.asym_pm_cor(x, y, 0.0, 0.0, degree=2.0, weights=(1.0, 1.0, -1.0, -1.0), dim=0)
ul_sensitive = exp.asym_pm_cor(x, y, 0.0, 0.0, degree=2.0, weights=(1.0, 1.0, -2.0, 0.0), dim=0)
```
