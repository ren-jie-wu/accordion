import pytest
import torch

from simba_plus.loss.ot_alignment import (
    sinkhorn_uniform_logdomain,
    _pairwise_sqeuclidean,
    ot_cost_from_plan_sqeuclidean,
    build_two_sample_cell_index,          # deprecated but keep tested
    build_sample_cell_index,
    make_all_sample_pairs,
    compute_two_sample_ot_state,
    compute_multi_sample_ot_states,
    ot_cost_two_sample,
    ot_cost_multi_sample,
)


def test_sinkhorn_marginals_close_uniform():
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    y = torch.randn(7, 3)
    C = _pairwise_sqeuclidean(x, y)
    P = sinkhorn_uniform_logdomain(C, eps=0.1, n_iters=200)

    rs = P.sum(dim=1)  # ~ 1/5
    cs = P.sum(dim=0)  # ~ 1/7

    assert torch.allclose(rs, torch.full_like(rs, 1 / 5), atol=5e-2, rtol=0)
    assert torch.allclose(cs, torch.full_like(cs, 1 / 7), atol=5e-2, rtol=0)
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


def test_build_sample_cell_index_groups_cover_and_disjoint():
    # 3 samples with uneven counts
    cell_sample = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    groups = build_sample_cell_index(cell_sample)

    assert set(groups.keys()) == {0, 1, 2}
    assert groups[0].numel() == 3
    assert groups[1].numel() == 2
    assert groups[2].numel() == 4

    # cover all indices
    all_idx = torch.cat([v for v in groups.values()], dim=0)
    assert all_idx.numel() == cell_sample.numel()
    assert torch.unique(all_idx).numel() == cell_sample.numel()  # disjoint & cover

    # correct membership
    for sid, idx in groups.items():
        assert torch.all(cell_sample[idx] == sid)


def test_make_all_sample_pairs():
    pairs = make_all_sample_pairs([0, 1, 2, 3])
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # order doesn't matter in input
    pairs2 = make_all_sample_pairs([3, 2, 1, 0, 0])
    assert pairs2 == pairs


def test_compute_multi_sample_ot_states_shapes_and_count():
    torch.manual_seed(0)
    # 3 samples: sizes 6,5,7 => total 18 cells
    cell_sample = torch.tensor([0] * 6 + [1] * 5 + [2] * 7, dtype=torch.long)
    groups = build_sample_cell_index(cell_sample)

    d = 4
    cell_mu = torch.randn(cell_sample.numel(), d)

    states = compute_multi_sample_ot_states(
        cell_mu=cell_mu,
        groups=groups,
        subset_size=5,
        eps=0.2,
        n_iters=50,
        generator=None,
    )

    # C(3,2)=3 pairs
    assert len(states) == 3
    got_pairs = {(s.s0, s.s1) for s in states}
    assert got_pairs == {(0, 1), (0, 2), (1, 2)}

    for s in states:
        st = s.state
        assert st.idx0.ndim == 1
        assert st.idx1.ndim == 1
        assert st.P.ndim == 2
        assert st.P.shape == (st.idx0.numel(), st.idx1.numel())
        assert torch.all(st.P >= 0)
        assert torch.isfinite(st.P).all()
        # Sum should be ~1 (entropic OT with uniform marginals)
        psum = st.P.sum()
        assert torch.isfinite(psum)
        assert torch.allclose(psum, psum.new_tensor(1.0))


def test_ot_cost_multi_sample_equals_mean_of_pair_costs():
    torch.manual_seed(1)
    cell_sample = torch.tensor([0] * 5 + [1] * 6 + [2] * 7, dtype=torch.long)
    groups = build_sample_cell_index(cell_sample)

    cell_mu = torch.randn(cell_sample.numel(), 3)
    states = compute_multi_sample_ot_states(
        cell_mu=cell_mu,
        groups=groups,
        subset_size=4,
        eps=0.3,
        n_iters=80,
    )

    cost_multi = ot_cost_multi_sample(cell_mu, states, P_detach=True)

    costs = []
    for s in states:
        st = s.state
        costs.append(
            ot_cost_two_sample(
                cell_mu=cell_mu,
                ot_idx0=st.idx0,
                ot_idx1=st.idx1,
                ot_P=st.P,
                P_detach=True,
            )
        )
    cost_manual = torch.stack(costs).mean()
    assert torch.allclose(cost_multi, cost_manual, atol=1e-6, rtol=0)


def test_ot_grad_flow_with_detached_plan_matches_training_usage():
    """
    Mimic training:
      - compute OT plan with detached embeddings (no grad)
      - compute OT loss with trainable embeddings, treating P as constant
      - verify backward produces finite grads
    """
    torch.manual_seed(42)

    cell_sample = torch.tensor([0] * 8 + [1] * 9 + [2] * 7, dtype=torch.long)
    groups = build_sample_cell_index(cell_sample)

    # trainable embeddings
    cell_mu = torch.randn(cell_sample.numel(), 5, requires_grad=True)

    # plan computed from detached embeddings (as in LightningProxModel._compute_ot_plan)
    with torch.no_grad():
        states = compute_multi_sample_ot_states(
            cell_mu=cell_mu.detach(),
            groups=groups,
            subset_size=6,
            eps=0.2,
            n_iters=60,
        )

    loss = ot_cost_multi_sample(cell_mu, states, P_detach=True)
    loss.backward()

    assert cell_mu.grad is not None
    assert torch.isfinite(cell_mu.grad).all()
    # should have some non-zero gradients in general
    assert cell_mu.grad.abs().sum().item() > 0


def test_ot_cost_multi_sample_empty_returns_zero_on_device():
    cell_mu = torch.randn(10, 3)
    z = ot_cost_multi_sample(cell_mu, [], P_detach=True)
    assert z.device == cell_mu.device
    assert torch.isfinite(z)
    assert z.item() == 0.0

def test_multi_sample_ot_single_sample_returns_empty_and_zero():
    torch.manual_seed(0)
    cell_sample = torch.zeros(12, dtype=torch.long)  # only sample 0
    groups = build_sample_cell_index(cell_sample)
    assert set(groups.keys()) == {0}
    assert groups[0].numel() == 12

    cell_mu = torch.randn(12, 4)

    states = compute_multi_sample_ot_states(
        cell_mu=cell_mu,
        groups=groups,
        subset_size=5,
        eps=0.2,
        n_iters=20,
    )
    assert states == [] or len(states) == 0

    cost = ot_cost_multi_sample(cell_mu, states, P_detach=True)
    assert cost.device == cell_mu.device
    assert torch.isfinite(cost)
    assert torch.allclose(cost, cell_mu.new_tensor(0.0))
