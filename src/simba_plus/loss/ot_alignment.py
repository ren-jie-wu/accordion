from __future__ import annotations
from dataclasses import dataclass
import torch
from simba_plus.util_modules import _CheckHeteroDataCodes as chk
from typing import Iterable, List, Tuple
import random


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


@dataclass(frozen=True)
class PairOTStateKeyed:
    k0: Tuple[int, int]
    k1: Tuple[int, int]
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


def build_sample_batch_cell_index(
    cell_sample: torch.Tensor,
    cell_batch_local: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Group cell indices by (sample_id, batch_local).
    Returns: {(sid, bid): idx_tensor}
    """
    if cell_sample.ndim != 1 or cell_batch_local.ndim != 1:
        raise ValueError("cell_sample and cell_batch_local must be 1D tensors")
    if cell_sample.numel() != cell_batch_local.numel():
        raise ValueError("cell_sample and cell_batch_local must have same length")

    groups: dict[tuple[int, int], torch.Tensor] = {}
    # sample ids are consecutive by your checker
    for sid in chk._as_sorted_unique_list(cell_sample):
        sid = int(sid)
        mask_s = (cell_sample == sid)
        if not mask_s.any():
            continue
        b_uniqs = chk._as_sorted_unique_list(cell_batch_local[mask_s])
        for bid in b_uniqs:
            bid = int(bid)
            idx = (mask_s & (cell_batch_local == bid)).nonzero(as_tuple=False).view(-1)
            if idx.numel() > 0:
                groups[(sid, bid)] = idx
    return groups


# take in `groups.keys()` and return `pairs`
def make_all_sample_pairs(
    sample_ids: Iterable[int],
    mode: str = "all_pairs",
    k = 0,
) -> List[Tuple[int, int]]:
    """
    sample_ids: iterable of sample ids (e.g., groups.keys()).
    Return all unordered pairs (s0, s1) with s0 < s1.
    """
    sids = sorted(int(x) for x in set(sample_ids))
    pairs: List[Tuple[int, int]] = []
    if mode == "all_pairs":
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                pairs.append((sids[i], sids[j]))
    elif mode == "star":
        ref = sids[0]
        for sid in sids[1:]:
            pairs.append((ref, sid))
    elif mode == "random_k":
        all_pairs = make_all_sample_pairs(sample_ids, mode="all_pairs")
        if k == 0:
            print("Warning in make_all_sample_pairs: k=0 means no random sampling, using all pairs")
            pairs = all_pairs
        elif k >= len(all_pairs):
            print(f"Warning in make_all_sample_pairs: k({k}) >= the number of all pairs({len(all_pairs)}), using all pairs")
            pairs = all_pairs
        elif k < 0:
            raise ValueError(f"k={k} is negative")
        elif 0 < k < 1: # fraction
            _k = max(1, int(k * len(all_pairs)))
            pairs = random.sample(all_pairs, _k)
        else: # number
            pairs = random.sample(all_pairs, int(k))
    else:
        raise ValueError(f"Unknown mode={mode}")
    return pairs


def make_all_sample_batch_pairs(
    sample_batch_ids: Iterable[tuple[int, int]],
    mode: str = "all_pairs",
    k = 0,
) -> dict[int, List[Tuple[int, int]]]:
    """
    sample_batch_ids: iterable of (sample_id, batch_id) pairs (e.g., groups_sb.keys()).
    Return all unordered pairs ((s0, b0), (s0, b1)) with b0 < b1.
    """
    by_sample: dict[int, List[int]] = {}
    by_sample_pairs: dict[int, List[Tuple[int, int]]] = {}
    for sid, bid in sample_batch_ids:
        sid, bid = int(sid), int(bid)
        by_sample.setdefault(sid, []).append(bid)
    for sid in by_sample.keys():
        by_sample_pairs[sid] = make_all_sample_pairs(by_sample[sid], mode=mode, k=k)
    return by_sample_pairs

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
    pairs: List[Tuple[int, int]] | None = None,
    subset_size: int = 2000,
    eps: float = 0.05,
    n_iters: int = 50,
    generator: torch.Generator | None = None,
) -> List[PairOTState]:
    """
    Compute OT states for multiple sample pairs.
    groups: {sid: idx_all}
    If `pairs` is provided, only compute OT states for the given pairs. Otherwise, compute OT states for all pairs.
    Returns: [PairOTState(s0, s1, TwoSampleOTState), ...]
    """
    out: List[PairOTState] = []
    all_pairs = make_all_sample_pairs(groups.keys(), mode="all_pairs")
    if pairs is not None and len(pairs) > 0:
        pairs = sorted(set(pairs) & set(all_pairs))
        if len(pairs) == 0:
            print(f"Warning in compute_multi_sample_ot_states: no valid pairs found with the provided pairs, using all pairs")
            pairs = all_pairs
    else:
        pairs = all_pairs
    for s0, s1 in pairs:
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


def compute_multi_batch_ot_states(
    cell_mu: torch.Tensor,
    groups_sb: dict[tuple[int, int], torch.Tensor],
    pairs_sb: dict[int, List[Tuple[int, int]]] | None = None,
    subset_size: int = 2000,
    eps: float = 0.05,
    n_iters: int = 50,
    generator: torch.Generator | None = None,
) -> List[PairOTStateKeyed]:
    """
    Compute OT states for batch pairs within each sample.
    groups_sb: {(sid, bid): idx_all}
    Returns: [PairOTStateKeyed(((sid,b0),(sid,b1)), state), ...]
    """
    # group keys by sample_id
    by_sample_all_pairs = make_all_sample_batch_pairs(groups_sb.keys(), mode="all_pairs")
    if pairs_sb is not None and sum(len(pairs) for pairs in pairs_sb.values()) > 0:
        pairs_sids = pairs_sb.keys() & by_sample_all_pairs.keys()
        pairs_sb = {sid: sorted(set(pairs_sb[sid]) & set(by_sample_all_pairs[sid])) for sid in pairs_sids}
        if sum(len(pairs) for pairs in pairs_sb.values()) == 0:
            print(f"Warning in compute_multi_batch_ot_states: no valid pairs found with the provided pairs_sb, using all pairs")
            pairs_sb = by_sample_all_pairs
    else:
        pairs_sb = by_sample_all_pairs

    out: List[PairOTStateKeyed] = []
    for sid, bid_pairs in pairs_sb.items():
        for b0, b1 in bid_pairs:
            st = compute_two_sample_ot_state(
                cell_mu=cell_mu,
                idx0_all=groups_sb[(sid, b0)],
                idx1_all=groups_sb[(sid, b1)],
                subset_size=subset_size,
                eps=eps,
                n_iters=n_iters,
                generator=generator,
            )
            out.append(PairOTStateKeyed(k0=(sid, b0), k1=(sid, b1), state=st))
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


def ot_cost_two(
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
def ot_cost_multi(
    cell_mu: torch.Tensor,
    ot_states: List[PairOTState | PairOTStateKeyed],
    P_detach: bool = True,
) -> torch.Tensor:
    """
    Compute OT cost for multiple sample pairs.
    cell_mu: (N_cells, d)  (global)
    ot_states: [PairOTState(s0, s1, TwoSampleOTState), ...] or [PairOTStateKeyed(((sid,b0),(sid,b1)), state), ...]
    """
    if ot_states is None or len(ot_states) == 0:
        return cell_mu.new_tensor(0.0)
    costs = [
        ot_cost_two(
            cell_mu=cell_mu,
            ot_idx0=s.state.idx0,
            ot_idx1=s.state.idx1,
            ot_P=s.state.P,
            P_detach=P_detach,
        ) for s in ot_states
    ]
    return torch.stack(costs).mean()
