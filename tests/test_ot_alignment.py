import pytest
import torch

from simba_plus.loss.ot_alignment import (
    sinkhorn_uniform_logdomain,
    _pairwise_sqeuclidean,
    ot_cost_from_plan_sqeuclidean,
    build_two_sample_cell_index,
)

def test_sinkhorn_marginals_close_uniform():
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    y = torch.randn(7, 3)
    C = _pairwise_sqeuclidean(x, y)
    P = sinkhorn_uniform_logdomain(C, eps=0.1, n_iters=200)

    # sums
    rs = P.sum(dim=1)  # ~ 1/5
    cs = P.sum(dim=0)  # ~ 1/7

    assert torch.allclose(rs, torch.full_like(rs, 1/5), atol=5e-2, rtol=0)
    assert torch.allclose(cs, torch.full_like(cs, 1/7), atol=5e-2, rtol=0)
    assert torch.all(P >= 0)

def test_ot_cost_matches_explicit_sum_PC():
    torch.manual_seed(1)
    x = torch.randn(4, 2)
    y = torch.randn(6, 2)
    C = _pairwise_sqeuclidean(x, y)
    P = sinkhorn_uniform_logdomain(C, eps=0.2, n_iters=200).detach()

    cost_fast = ot_cost_from_plan_sqeuclidean(x, y, P)
    cost_explicit = (P * C).sum()

    assert torch.isclose(cost_fast, cost_explicit, atol=1e-4)

def test_build_two_sample_cell_index_requires_two_samples():
    cell_sample = torch.tensor([0, 0, 1, 1])
    idx0, idx1 = build_two_sample_cell_index(cell_sample)
    assert idx0.numel() == 2
    assert idx1.numel() == 2

    with pytest.raises(NotImplementedError):
        _ = build_two_sample_cell_index(torch.tensor([0, 1, 2]))
