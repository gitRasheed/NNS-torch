import pytest
import torch

import nns_torch as nt
from nns_torch import experimental as exp


def test_asym_pm_components_hand_values():
    x = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    y = torch.tensor([2.0, -3.0, 4.0, -1.0])

    comps = exp.asym_pm_components(x, y, target_x=0.0, target_y=0.0, degree=2.0, dim=0)

    torch.testing.assert_close(comps.uu, torch.tensor(1.0))
    torch.testing.assert_close(comps.ll, torch.tensor(0.75))
    torch.testing.assert_close(comps.ul, torch.tensor(0.75))
    torch.testing.assert_close(comps.lu, torch.tensor(1.0))


def test_asym_pm_cor_matches_pm_cor_with_symmetric_weights():
    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]])
    y = torch.tensor([[1.0, 0.0, -1.0], [1.0, -1.0, 0.0]])

    got = exp.asym_pm_cor(x, y, degree=2.0, weights=(1.0, 1.0, -1.0, -1.0), dim=-1)
    want = nt.pm_cor(x, y, degree=2.0, dim=-1)

    torch.testing.assert_close(got, want)


def test_asym_pm_components_swap_ul_lu():
    x = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    y = torch.tensor([2.0, -3.0, 4.0, -1.0])

    xy = exp.asym_pm_components(x, y, target_x=0.0, target_y=0.0, degree=2.0, dim=0)
    yx = exp.asym_pm_components(y, x, target_x=0.0, target_y=0.0, degree=2.0, dim=0)

    torch.testing.assert_close(xy.uu, yx.uu)
    torch.testing.assert_close(xy.ll, yx.ll)
    torch.testing.assert_close(xy.ul, yx.lu)
    torch.testing.assert_close(xy.lu, yx.ul)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_asym_pm_cor_gradients_are_finite(dtype):
    torch.manual_seed(11)
    x = torch.randn(4, 5, dtype=dtype, requires_grad=True)
    y = torch.randn(4, 5, dtype=dtype, requires_grad=True)
    weights = torch.tensor([1.0, 0.7, -0.4, -1.2], dtype=dtype, requires_grad=True)

    loss = exp.asym_pm_cor(x, y, degree=1.6, weights=weights, dim=-1).sum()
    grads = torch.autograd.grad(loss, (x, y, weights))

    for grad in grads:
        assert torch.isfinite(grad).all()


def test_asymmetric_copm_loss_list_and_stacked_match():
    torch.manual_seed(12)
    L, B, T, C = 4, 2, 8, 16
    final = torch.randn(B, T, C)
    hidden_base = torch.randn(L, B, T, C)
    loss_fn = exp.AsymmetricCoPMLoss(degree_init=1.3)

    hidden_list = [hidden_base[i].detach().clone().requires_grad_() for i in range(L)]
    list_loss = loss_fn(hidden_list, final)
    list_loss.backward()

    hidden_stacked = hidden_base.detach().clone().requires_grad_()
    stacked_loss = loss_fn(hidden_stacked, final)
    stacked_loss.backward()

    assert list_loss.shape == ()
    assert stacked_loss.shape == ()
    assert all(h.grad is not None and torch.isfinite(h.grad).all() for h in hidden_list)
    assert hidden_stacked.grad is not None
    assert torch.isfinite(hidden_stacked.grad).all()
    torch.testing.assert_close(list_loss.detach(), stacked_loss.detach())


@pytest.mark.cuda
def test_asym_pm_cor_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    y = torch.randn(4, 8, device="cuda", requires_grad=True)
    loss = exp.asym_pm_cor(x, y, degree=1.4, dim=-1).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


@pytest.mark.compile
def test_asymmetric_copm_loss_compile_if_available():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    torch.manual_seed(13)
    final = torch.randn(2, 8, 16)
    hidden = torch.randn(4, 2, 8, 16, requires_grad=True)
    loss_fn = exp.AsymmetricCoPMLoss(degree_init=1.3)

    loss = torch.compile(loss_fn)(hidden, final)
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
