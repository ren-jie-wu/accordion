from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TwoSampleGenePairs:
    """Index pairs (global gene node indices) for shared genes between sample 0 and sample 1."""
    idx0: torch.Tensor  # (n_shared,)
    idx1: torch.Tensor  # (n_shared,)


def build_two_sample_gene_pairs(
    gene_sample: torch.Tensor,
    gene_id: torch.Tensor,
) -> TwoSampleGenePairs:
    """
    Build index pairs for shared gene_id between sample 0 and 1.
    Requires: exactly two samples labeled 0 and 1 (or at least contains both).
    Assumes within each sample, gene_id is unique (true if var_names unique per sample).
    """
    if gene_sample.ndim != 1 or gene_id.ndim != 1:
        raise ValueError("gene_sample and gene_id must be 1D tensors")
    if gene_sample.numel() != gene_id.numel():
        raise ValueError("gene_sample and gene_id must have the same length")

    uniq = torch.unique(gene_sample).cpu().tolist()
    if len(uniq) != 2:
        raise NotImplementedError(
            f"Gene alignment is implemented for exactly 2 samples for now. Got unique samples={uniq}"
        )

    # We *assume* samples are encoded as 0/1 in your loader (they are).
    # If in the future it isn't guaranteed, you can map uniq->(0,1) here.
    s0, s1 = sorted(uniq)
    if (s0, s1) != (0, 1):
        # keep it strict for now, to avoid silent mismatches
        raise NotImplementedError(
            f"Expected sample codes (0,1). Got ({s0},{s1}). "
            "If you want to support arbitrary two codes, map them explicitly."
        )

    # Build global index lookup by gene_id per sample
    n_gid = int(gene_id.max().item()) + 1 if gene_id.numel() > 0 else 0
    device = gene_id.device

    map0 = torch.full((n_gid,), -1, dtype=torch.long, device=device)
    map1 = torch.full((n_gid,), -1, dtype=torch.long, device=device)

    idx_s0 = (gene_sample == 0).nonzero(as_tuple=False).view(-1)
    idx_s1 = (gene_sample == 1).nonzero(as_tuple=False).view(-1)

    if idx_s0.numel() > 0:
        map0[gene_id[idx_s0]] = idx_s0
    if idx_s1.numel() > 0:
        map1[gene_id[idx_s1]] = idx_s1

    shared = (map0 >= 0) & (map1 >= 0)
    idx0 = map0[shared]
    idx1 = map1[shared]

    return TwoSampleGenePairs(idx0=idx0, idx1=idx1)


def gene_alignment_msd_loss(
    gene_mu: torch.Tensor,
    pairs: TwoSampleGenePairs,
) -> torch.Tensor:
    """
    Mean Squared Distance between gene_mu[pairs.idx0] and gene_mu[pairs.idx1].
    gene_mu: (n_genes_total, latent_dim)
    """
    if pairs.idx0.numel() == 0:
        return torch.tensor(0.0)

    diff = gene_mu[pairs.idx0] - gene_mu[pairs.idx1]  # (n_shared, d)
    per_gene = diff.pow(2).sum(dim=-1) # (n_shared,)
    return per_gene.mean()
