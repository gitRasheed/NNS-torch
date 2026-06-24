# Asymmetric CoPM Design Note

Current `CoPMAuxLoss` is symmetric: it optimizes the signed `pm_cor` score where concordant quadrants increase the score and divergent quadrants decrease it.

An asymmetric variant should stay experimental until training evidence says it helps. The obvious reference form is the four explicit quadrant moments:

- upper / upper
- lower / lower
- upper / lower
- lower / upper

A minimal learnable version could apply four scalar quadrant weights before summing. That gives the model separate control over upside concordance, downside concordance, and the two divergent tails.

Keep it out of the core API until the reference formula, gradients, and a tiny CPU example show useful behavior.
