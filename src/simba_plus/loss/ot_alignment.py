from __future__ import annotations
from dataclasses import dataclass
import torch
from simba_plus.util_modules import _CheckHeteroDataCodes as chk
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TwoSampleOTState:
    idx0: torch.Tensor  # (n0,) global cell indices for sample 0 (subset)
    idx1: torch.Tensor  # (n1,) global cell indices for sample 1 (subset)
    P: torch.Tensor     # (n0, n1) OT coupling (rows/cols sum to uniform marginals approximately)


@dataclass(frozen=True)
class PairOTState:
    """
    OT state for a specific (s0, s1) sample pair.
    """
    s0: int
    s1: int
    state: TwoSampleOTState


# deprecated
def build_two_sample_cell_index(
    cell_sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return all cell indices for sample 0 and sample 1.
    Currently strict: expects exactly samples {0,1}.
    """
    uniq = chk._as_sorted_unique_list(cell_sample)
    if len(uniq) != 2:
        raise NotImplementedError(
            f"OT alignment currently supports exactly 2 samples. Got unique samples={uniq}"
        )
    s0, s1 = uniq
    if (s0, s1) != (0, 1):
        raise NotImplementedError(
            f"Expected sample codes (0,1). Got ({s0},{s1})."
        )
    idx0 = (cell_sample == 0).nonzero(as_tuple=False).view(-1)
    idx1 = (cell_sample == 1).nonzero(as_tuple=False).view(-1)
    return idx0, idx1


# return `groups``
def build_sample_cell_index(
    cell_sample: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Return all cell indices grouped by sample id.
    cell_sample: (N_cells,)
    Returns: {sample_id: idx_tensor (global cell indices)}
    """
    if cell_sample.ndim != 1:
        raise ValueError("cell_sample must be a 1D tensor")
    groups: dict[int, torch.Tensor] = {}
    uniq = chk._as_sorted_unique_list(cell_sample)
    for sid in uniq:
        sid = int(sid)
        idx = (cell_sample == sid).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            groups[sid] = idx
    return groups


# take in `groups.keys()` and return `pairs`
def make_all_sample_pairs(sample_ids: Iterable[int]) -> List[Tuple[int, int]]:
    """
    sample_ids: iterable of sample ids (e.g., groups.keys()).
    Return all unordered pairs (s0, s1) with s0 < s1.
    """
    sids = sorted(int(x) for x in set(sample_ids))
    pairs: List[Tuple[int, int]] = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            pairs.append((sids[i], sids[j]))
    return pairs


def _pairwise_sqeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: (n, d), y: (m, d)
    return C: (n, m) with C_ij = ||x_i - y_j||^2
    """
    x2 = (x * x).sum(dim=1, keepdim=True)          # (n, 1)
    y2 = (y * y).sum(dim=1, keepdim=True).t()      # (1, m)
    # (n, m)
    return x2 + y2 - 2.0 * (x @ y.t())


def sinkhorn_uniform_logdomain(
    C: torch.Tensor,
    eps: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """
    Log-domain Sinkhorn iterations for entropic OT (uniform marginals).

    Parameters:
        C: (n, m) cost matrix
        eps: entropic regularization parameter
        n_iters: number of Sinkhorn iterations
    
    Returns:
        P: (n, m) coupling matrix with sum(P)=1 approximately
    
    Notes:
        Algorithm: Sinkhorn-Knopp scaling with entropic regularization (Cuturi, 2013).
        Log-domain stabilization follows standard derivations (e.g., Peyré & Cuturi, 2019).
        Implementation is a minimal PyTorch variant; conceptually similar to POT's `sinkhorn_log`.
    """
    if C.ndim != 2:
        raise ValueError("C must be 2D")
    n, m = C.shape
    if n == 0 or m == 0:
        return C.new_zeros((n, m))

    # logK = -C/eps
    logK = -C / float(eps)

    # uniform marginals
    log_a = -torch.log(torch.tensor(float(n), device=C.device, dtype=C.dtype))
    log_b = -torch.log(torch.tensor(float(m), device=C.device, dtype=C.dtype))

    log_u = torch.zeros((n,), device=C.device, dtype=C.dtype)
    log_v = torch.zeros((m,), device=C.device, dtype=C.dtype)

    # Sinkhorn iterations in log domain
    for _ in range(int(n_iters)):
        # log_u = log_a - logsumexp(logK + log_v[None,:], dim=1)
        log_u = log_a - torch.logsumexp(logK + log_v.view(1, m), dim=1)
        # log_v = log_b - logsumexp(logK^T + log_u[None,:], dim=1)
        log_v = log_b - torch.logsumexp(logK.t() + log_u.view(1, n), dim=1)

    # P = exp(log_u[:,None] + logK + log_v[None,:])
    P = torch.exp(log_u.view(n, 1) + logK + log_v.view(1, m))
    return P


def compute_two_sample_ot_state(
    cell_mu: torch.Tensor,
    idx0_all: torch.Tensor,
    idx1_all: torch.Tensor,
    subset_size: int = 2000,
    eps: float = 0.05,
    n_iters: int = 50,
    generator: torch.Generator | None = None,
) -> TwoSampleOTState:
    """
    Compute OT coupling between a subset of sample0 and sample1 cells.
    cell_mu: (N_cells, d)  (global)
    """
    device = cell_mu.device
    idx0_all = idx0_all.to(device)
    idx1_all = idx1_all.to(device)

    # pick subset
    if subset_size is not None and subset_size > 0:
        n0 = min(int(subset_size), idx0_all.numel())
        n1 = min(int(subset_size), idx1_all.numel())
        perm0 = torch.randperm(idx0_all.numel(), device=device, generator=generator)[:n0]
        perm1 = torch.randperm(idx1_all.numel(), device=device, generator=generator)[:n1]
        idx0 = idx0_all[perm0]
        idx1 = idx1_all[perm1]
    else:
        idx0, idx1 = idx0_all, idx1_all

    x = cell_mu[idx0]  # (n0, d)
    y = cell_mu[idx1]  # (n1, d)

    C = _pairwise_sqeuclidean(x, y)  # (n0, n1)
    P = sinkhorn_uniform_logdomain(C, eps=eps, n_iters=n_iters)  # (n0, n1)
    return TwoSampleOTState(idx0=idx0, idx1=idx1, P=P)


# take in `groups` and return `ot_states`
def compute_multi_sample_ot_states(
    cell_mu: torch.Tensor,
    groups: dict[int, torch.Tensor],
    subset_size: int = 2000,
    eps: float = 0.05,
    n_iters: int = 50,
    generator: torch.Generator | None = None,
) -> List[PairOTState]:
    """
    Compute OT states for multiple sample pairs.
    groups: {sid: idx_all}
    Returns: [PairOTState(s0, s1, TwoSampleOTState), ...]
    """
    out: List[PairOTState] = []
    for s0, s1 in make_all_sample_pairs(groups.keys()):
        idx0_all = groups[s0]
        idx1_all = groups[s1]
        st = compute_two_sample_ot_state(
            cell_mu=cell_mu,
            idx0_all=idx0_all,
            idx1_all=idx1_all,
            subset_size=subset_size,
            eps=eps,
            n_iters=n_iters,
            generator=generator,
        )
        out.append(PairOTState(s0=s0, s1=s1, state=st))
    return out


def ot_cost_from_plan_sqeuclidean(
    x: torch.Tensor,      # (n, d)
    y: torch.Tensor,      # (m, d)
    P: torch.Tensor,      # (n, m), sum(P)=1 approx
) -> torch.Tensor:
    """
    Compute sum_ij P_ij ||x_i - y_j||^2 WITHOUT building full C again.
    Differentiable w.r.t x,y, treating P as constant.
    """
    # row/col mass
    r = P.sum(dim=1)  # (n,)
    c = P.sum(dim=0)  # (m,)

    x2 = (x * x).sum(dim=1)  # (n,)
    y2 = (y * y).sum(dim=1)  # (m,)

    # term1 + term2 - 2 * sum_i x_i · (sum_j P_ij y_j)
    Py = P @ y              # (n, d)
    term1 = (r * x2).sum()
    term2 = (c * y2).sum()
    term3 = 2.0 * (x * Py).sum()
    return term1 + term2 - term3


def ot_cost_two_sample(
    cell_mu: torch.Tensor,
    ot_idx0: torch.Tensor,
    ot_idx1: torch.Tensor,
    ot_P: torch.Tensor,
    P_detach: bool = True,
) -> torch.Tensor:
    """
    Compute OT cost for a two sample OT state.
    cell_mu: (N_cells, d)  (global)
    ot_idx0: (n0,) global cell indices for sample 0 (subset)
    ot_idx1: (n1,) global cell indices for sample 1 (subset)
    ot_P: (n0, n1) OT coupling (rows/cols sum to uniform marginals approximately)
    """
    x = cell_mu[ot_idx0]  # (n0, d)
    y = cell_mu[ot_idx1]  # (n1, d)
    if P_detach:
        ot_P = ot_P.detach()
    return ot_cost_from_plan_sqeuclidean(x, y, ot_P)


# take in `ot_states` and return `cost`
def ot_cost_multi_sample(
    cell_mu: torch.Tensor,
    ot_states: List[PairOTState],
    P_detach: bool = True,
) -> torch.Tensor:
    """
    Compute OT cost for multiple sample pairs.
    cell_mu: (N_cells, d)  (global)
    ot_states: [PairOTState(s0, s1, TwoSampleOTState), ...]
    """
    if ot_states is None or len(ot_states) == 0:
        return cell_mu.new_tensor(0.0)
    costs = [
        ot_cost_two_sample(
            cell_mu=cell_mu,
            ot_idx0=s.state.idx0,
            ot_idx1=s.state.idx1,
            ot_P=s.state.P,
            P_detach=P_detach,
        ) for s in ot_states
    ]
    return torch.stack(costs).mean()
