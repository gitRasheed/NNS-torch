import torch
import pytest

import nns_torch as nt
from nns_torch.testing import reference as ref


def test_lpm_upm_hand_values_preserve_exact_target_zero():
    x = torch.tensor([-2.0, 0.0, 1.0, 3.0])

    torch.testing.assert_close(nt.lpm_elem(x, target=1.0, degree=2.0), torch.tensor([9.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(nt.upm_elem(x, target=1.0, degree=2.0), torch.tensor([0.0, 0.0, 0.0, 4.0]))
    torch.testing.assert_close(nt.lpm(x, target=1.0, degree=2.0), torch.tensor(2.5))
    torch.testing.assert_close(nt.upm(x, target=1.0, degree=2.0), torch.tensor(1.0))


def test_moment_reduction_and_broadcasting_match_reference():
    x = torch.tensor([[-2.0, 0.5, 4.0], [3.0, -1.0, 2.0]])
    target = torch.tensor([0.0, 1.0, 2.0])
    degree = torch.tensor([1.0, 2.0, 3.0])

    torch.testing.assert_close(nt.lpm(x, target=target, degree=degree, dim=0), ref.lpm_reference(x, target, degree, dim=0))
    torch.testing.assert_close(
        nt.upm(x, target=target, degree=degree, dim=1, keepdim=True),
        ref.upm_reference(x, target, degree, dim=1, keepdim=True),
    )


def test_moment_value_parity_cpu_float64_and_float32():
    torch.manual_seed(0)
    for dtype in (torch.float64, torch.float32):
        x = torch.randn(4, 5, dtype=dtype)
        target = torch.randn(5, dtype=dtype)
        degree = torch.full((5,), 1.7, dtype=dtype)

        torch.testing.assert_close(nt.lpm_elem(x, target, degree), ref.lpm_elem_reference(x, target, degree))
        torch.testing.assert_close(nt.upm_elem(x, target, degree), ref.upm_elem_reference(x, target, degree))
        torch.testing.assert_close(nt.lpm(x, target, degree, dim=0), ref.lpm_reference(x, target, degree, dim=0))
        torch.testing.assert_close(nt.upm(x, target, degree, dim=0), ref.upm_reference(x, target, degree, dim=0))


def test_moment_gradient_parity_for_input_target_and_degree():
    torch.manual_seed(1)
    x = torch.randn(6, dtype=torch.float64, requires_grad=True) + 0.3
    target = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    degree = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone().requires_grad_()
    degree_ref = degree.detach().clone().requires_grad_()

    got = nt.lpm(x, target=target, degree=degree) + nt.upm(x, target=target, degree=degree)
    want = ref.lpm_reference(x_ref, target_ref, degree_ref) + ref.upm_reference(x_ref, target_ref, degree_ref)

    got_grads = torch.autograd.grad(got, (x, target, degree))
    want_grads = torch.autograd.grad(want, (x_ref, target_ref, degree_ref))
    torch.testing.assert_close(got, want)
    for got_grad, want_grad in zip(got_grads, want_grads):
        torch.testing.assert_close(got_grad, want_grad)


@pytest.mark.cuda
def test_moments_cuda_float32_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(8, 9, device="cuda")
    target = torch.randn(9, device="cuda")
    torch.testing.assert_close(nt.lpm(x, target=target, degree=2.0, dim=0), ref.lpm_reference(x, target, 2.0, dim=0))
