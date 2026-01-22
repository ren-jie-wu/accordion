from __future__ import annotations
import itertools
from dataclasses import dataclass
import torch


# deprecated
@dataclass(frozen=True)
class TwoSampleGenePairs:
    """Index pairs (global gene node indices) for shared genes between sample 0 and sample 1."""
    idx0: torch.Tensor  # (n_shared,)
    idx1: torch.Tensor  # (n_shared,)


# deprecated
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
    if gene_id.numel() == 0:
        return TwoSampleGenePairs(idx0=gene_id.new_empty((0,), dtype=torch.long),
                                  idx1=gene_id.new_empty((0,), dtype=torch.long))

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


@dataclass(frozen=True)
class MultiSampleGenePairs:
    """Generic index pairs for shared genes across multiple samples."""
    idx0: torch.Tensor
    idx1: torch.Tensor


def build_multi_sample_gene_pairs(
    gene_sample: torch.Tensor,
    gene_id: torch.Tensor,
    *,
    mode: str = "all_pairs",     # "all_pairs" | "star"
) -> MultiSampleGenePairs:
    """
    Build index pairs (global gene node indices) for shared genes across >=2 samples.

    Assumptions:
      - gene_sample, gene_id are 1D and same length
      - within each sample, gene_id is unique (true if var_names unique per sample)
    """
    if gene_sample.ndim != 1 or gene_id.ndim != 1:
        raise ValueError("gene_sample and gene_id must be 1D tensors")
    if gene_sample.numel() != gene_id.numel():
        raise ValueError("gene_sample and gene_id must have the same length")
    if gene_id.numel() == 0:
        return MultiSampleGenePairs(idx0=gene_id.new_empty((0,), dtype=torch.long),
                                    idx1=gene_id.new_empty((0,), dtype=torch.long))

    # Sort by gene_id so we can iterate groups cheaply
    perm = torch.argsort(gene_id)
    gid_sorted = gene_id[perm]
    # group boundaries
    is_new = torch.ones_like(gid_sorted, dtype=torch.bool)
    is_new[1:] = gid_sorted[1:] != gid_sorted[:-1] # whether the current gene_id is different from the previous one
    starts = torch.nonzero(is_new, as_tuple=False).view(-1)
    ends = torch.cat([starts[1:], gid_sorted.new_tensor([gid_sorted.numel()])], dim=0)

    idx0_list = []
    idx1_list = []

    for st, ed in zip(starts.tolist(), ends.tolist()):
        grp_perm = perm[st:ed]  # global indices of gene nodes for this gene_id
        if grp_perm.numel() < 2:
            continue

        if mode == "star":
            # choose reference node index inside this group
            samp_grp = gene_sample[grp_perm].long()
            ref_idx = grp_perm[torch.argmin(samp_grp)]
            others = grp_perm[grp_perm != ref_idx]
            if others.numel() == 0:
                continue
            idx0_list.append(ref_idx.expand_as(others))
            idx1_list.append(others)

        elif mode == "all_pairs":
            # all combinations inside group
            # assume no duplicate gene_id within each sample
            grp = grp_perm.tolist()
            for a, b in itertools.combinations(grp, 2):
                idx0_list.append(torch.tensor([a], device=gene_id.device, dtype=torch.long))
                idx1_list.append(torch.tensor([b], device=gene_id.device, dtype=torch.long))
        else:
            raise ValueError(f"Unknown mode={mode}")

    if len(idx0_list) == 0:
        empty = gene_id.new_empty((0,), dtype=torch.long)
        return MultiSampleGenePairs(idx0=empty, idx1=empty)

    idx0 = torch.cat(idx0_list, dim=0)
    idx1 = torch.cat(idx1_list, dim=0)
    return MultiSampleGenePairs(idx0=idx0, idx1=idx1)


def gene_alignment_msd_loss(
    gene_mu: torch.Tensor,
    pairs: TwoSampleGenePairs,
) -> torch.Tensor:
    """
    Mean Squared Distance between gene_mu[pairs.idx0] and gene_mu[pairs.idx1].
    gene_mu: (n_genes_total, latent_dim)
    """
    if pairs.idx0.numel() == 0:
        return gene_mu.new_tensor(0.0) # make sure it's on the right device

    diff = gene_mu[pairs.idx0] - gene_mu[pairs.idx1]  # (n_shared, d)
    per_gene = diff.pow(2).sum(dim=-1) # (n_shared,)
    return per_gene.mean()
