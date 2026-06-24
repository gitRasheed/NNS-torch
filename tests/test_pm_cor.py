import torch
import pytest

import nns_torch as nt
from nns_torch.testing import reference as ref


def test_pm_cor_hand_values():
    x = torch.tensor([-1.0, 0.0, 1.0])

    torch.testing.assert_close(nt.pm_cor(x, x, degree=2.0), torch.tensor(1.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(nt.pm_cor(x, -x, degree=2.0), torch.tensor(-1.0), atol=1e-6, rtol=0.0)


def test_pm_cor_matches_explicit_quadrants_values_and_shape():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    y = torch.randn(1, 3, 5, dtype=torch.float64)

    got = nt.pm_cor(x, y, degree=1.6, dim=-1, keepdim=True)
    want = ref.pm_cor_reference(x, y, degree=1.6, dim=-1, keepdim=True)

    assert got.shape == (2, 3, 1)
    torch.testing.assert_close(got, want)


def test_pm_cor_constant_inputs_return_zero():
    x = torch.ones(5, dtype=torch.float64, requires_grad=True)
    y = torch.arange(5, dtype=torch.float64, requires_grad=True)

    got = nt.pm_cor(x, y, degree=2.0)
    want = ref.pm_cor_reference(x, y, degree=2.0)
    grads = torch.autograd.grad(got, (x, y), allow_unused=True)

    torch.testing.assert_close(got, torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(got, want)
    for grad in grads:
        if grad is not None:
            assert torch.isfinite(grad).all()


def test_pm_cor_gradient_parity_for_x_y_and_degree():
    torch.manual_seed(3)
    x = torch.randn(7, dtype=torch.float64, requires_grad=True)
    y = torch.randn(7, dtype=torch.float64, requires_grad=True)
    degree = torch.tensor(1.8, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    yr = y.detach().clone().requires_grad_()
    dr = degree.detach().clone().requires_grad_()

    got = nt.pm_cor(x, y, degree=degree)
    want = ref.pm_cor_reference(xr, yr, degree=dr)
    got_grads = torch.autograd.grad(got, (x, y, degree))
    want_grads = torch.autograd.grad(want, (xr, yr, dr))

    torch.testing.assert_close(got, want)
    for got_grad, want_grad in zip(got_grads, want_grads):
        torch.testing.assert_close(got_grad, want_grad)


def test_co_moment_parity():
    torch.manual_seed(4)
    x = torch.randn(4, 6)
    y = torch.randn(4, 6)

    torch.testing.assert_close(nt.co_lpm(x, y, degree=2.0, dim=-1), ref.co_lpm_reference(x, y, degree=2.0, dim=-1))
    torch.testing.assert_close(nt.co_upm(x, y, degree=2.0, dim=-1), ref.co_upm_reference(x, y, degree=2.0, dim=-1))


def test_co_moments_broadcast_targets_and_target_grads():
    x = torch.tensor([[-2.0, 1.0, 3.0], [0.0, -1.0, 2.0]], requires_grad=True)
    y = torch.tensor([[1.0, -2.0, 4.0], [2.0, 0.0, -1.0]], requires_grad=True)
    target_x = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
    target_y = torch.tensor(0.0, requires_grad=True)

    lower = nt.co_lpm(x, y, target_x=target_x, target_y=target_y, degree=2.0, dim=0)
    upper = nt.co_upm(x, y, target_x=target_x, target_y=target_y, degree=2.0, dim=0)
    loss = lower.sum() + upper.sum()
    grads = torch.autograd.grad(loss, (x, y, target_x, target_y))

    assert lower.shape == (3,)
    assert upper.shape == (3,)
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_pm_cor_tuple_dim_reduction_matches_reference():
    torch.manual_seed(8)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    y = torch.randn(2, 3, 4, dtype=torch.float64)

    got = nt.pm_cor(x, y, degree=1.4, dim=(1, 2))
    want = ref.pm_cor_reference(x, y, degree=1.4, dim=(1, 2))

    assert got.shape == (2,)
    torch.testing.assert_close(got, want)


def test_negative_float_degree_raises():
    x = torch.tensor([-1.0, 1.0])

    with pytest.raises(ValueError, match="degree must be non-negative"):
        nt.lpm_elem(x, degree=-1.0)
    with pytest.raises(ValueError, match="degree must be non-negative"):
        nt.upm(x, degree=-1.0)
    with pytest.raises(ValueError, match="degree must be non-negative"):
        nt.pm_cor(x, x, degree=-1.0)


def test_negative_tensor_degree_raises():
    x = torch.tensor([-1.0, 1.0])

    with pytest.raises(AssertionError, match="degree must be non-negative"):
        nt.lpm_elem(x, degree=torch.tensor(-1.0))
