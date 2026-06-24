import pytest
import torch

from nns_torch.nn import CoPMAuxLoss, LPMAuxLoss, PMMarginLoss


def test_lpm_aux_loss_backprops_to_logits_target_and_degree():
    logits = torch.tensor([[2.0, -1.0], [0.1, 0.8]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss_fn = LPMAuxLoss(degree_init=1.2, target_init=0.0)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss < 0
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(loss_fn.target.grad)
    assert torch.isfinite(loss_fn.degree_raw.grad)


def test_copm_aux_loss_prefers_concordant_layers_and_detaches_final_by_default():
    torch.manual_seed(5)
    final = torch.randn(2, 3, 4, requires_grad=True)
    concordant = [(final.detach() + 0.01 * torch.randn_like(final)).requires_grad_()]
    discordant = [(-final.detach() + 0.01 * torch.randn_like(final)).requires_grad_()]
    loss_fn = CoPMAuxLoss(degree_init=1.3)

    assert loss_fn(concordant, final) < loss_fn(discordant, final)

    loss = loss_fn(concordant, final)
    loss.backward()
    assert concordant[0].grad is not None
    assert final.grad is None


def test_copm_aux_loss_no_detach_backprops_to_final():
    torch.manual_seed(6)
    final = torch.randn(2, 3, 4, requires_grad=True)
    hidden = [torch.randn(2, 3, 4, requires_grad=True)]
    loss = CoPMAuxLoss(detach_final=False)(hidden, final)

    loss.backward()
    assert hidden[0].grad is not None
    assert final.grad is not None


def test_copm_aux_loss_accepts_pm_network_list_and_stacked_shapes():
    torch.manual_seed(7)
    L, B, T, C = 4, 2, 8, 16
    final = torch.randn(B, T, C)
    hidden_base = torch.randn(L, B, T, C)
    loss_fn = CoPMAuxLoss(degree_init=1.3)

    hidden_list = [hidden_base[i].detach().clone().requires_grad_() for i in range(L)]
    list_loss = loss_fn(hidden_list, final)
    list_loss.backward()
    assert list_loss.shape == ()
    assert torch.isfinite(list_loss)
    assert all(h.grad is not None and torch.isfinite(h.grad).all() for h in hidden_list)

    hidden_stacked = hidden_base.detach().clone().requires_grad_()
    stacked_loss = loss_fn(hidden_stacked, final)
    stacked_loss.backward()
    assert stacked_loss.shape == ()
    assert torch.isfinite(stacked_loss)
    assert hidden_stacked.grad is not None
    assert torch.isfinite(hidden_stacked.grad).all()
    torch.testing.assert_close(list_loss.detach(), stacked_loss.detach())

    logits = torch.randn(B, T, 11, requires_grad=True)
    targets = torch.randint(0, logits.size(-1), (B, T))
    aux_loss = LPMAuxLoss(degree_init=1.2)(logits, targets)
    aux_loss.backward()
    assert aux_loss.shape == ()
    assert torch.isfinite(aux_loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.compile
def test_copm_aux_loss_compile_stacked_pm_network_shape_if_available():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    torch.manual_seed(70)
    L, B, T, C = 4, 2, 8, 16
    final = torch.randn(B, T, C)
    hidden = torch.randn(L, B, T, C, requires_grad=True)
    loss_fn = CoPMAuxLoss(degree_init=1.3)

    loss = torch.compile(loss_fn)(hidden, final)
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_pm_margin_loss_penalizes_small_margins():
    logits_good = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    logits_bad = torch.tensor([[0.2, 0.0], [0.0, 0.2]])
    targets = torch.tensor([0, 1])
    loss_fn = PMMarginLoss(margin=1.0, degree_init=2.0)

    assert loss_fn(logits_good, targets) < loss_fn(logits_bad, targets)
