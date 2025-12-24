from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TwoSampleOTState:
    idx0: torch.Tensor  # (n0,) global cell indices for sample 0 (subset)
    idx1: torch.Tensor  # (n1,) global cell indices for sample 1 (subset)
    P: torch.Tensor     # (n0, n1) OT coupling (rows/cols sum to uniform marginals approximately)


def build_two_sample_cell_index(
    cell_sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return all cell indices for sample 0 and sample 1.
    Currently strict: expects exactly samples {0,1}.
    """
    uniq = torch.unique(cell_sample).cpu().tolist()
    if len(uniq) != 2:
        raise NotImplementedError(
            f"OT alignment currently supports exactly 2 samples. Got unique samples={uniq}"
        )
    s0, s1 = sorted(uniq)
    if (s0, s1) != (0, 1):
        raise NotImplementedError(
            f"Expected sample codes (0,1). Got ({s0},{s1})."
        )
    idx0 = (cell_sample == 0).nonzero(as_tuple=False).view(-1)
    idx1 = (cell_sample == 1).nonzero(as_tuple=False).view(-1)
    return idx0, idx1


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
    Entropic OT with uniform marginals using log-domain Sinkhorn for stability.
    C: (n, m) cost matrix
    Returns coupling P: (n, m) with sum(P)=1 approximately
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
