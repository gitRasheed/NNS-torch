import pytest
import torch

import nns_torch as nt
from nns_torch.testing import reference as ref


@pytest.mark.compile
def test_pm_cor_compile_if_available():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    compiled = torch.compile(nt.pm_cor)

    torch.testing.assert_close(compiled(x, y, degree=1.4, dim=-1), ref.pm_cor_reference(x, y, degree=1.4, dim=-1))


@pytest.mark.compile
def test_pm_cor_bf16_autocast_cpu_if_available():
    if not hasattr(torch, "autocast"):
        pytest.skip("autocast unavailable")

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = nt.pm_cor(x, y, degree=1.4, dim=-1)

    assert torch.isfinite(got).all()
