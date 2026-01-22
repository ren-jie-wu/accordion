import pytest
import torch

from simba_plus.loss.gene_alignment import (
    MultiSampleGenePairs,
    build_multi_sample_gene_pairs,
    gene_alignment_msd_loss,
)

def test_two_sample_gene_alignment_pairs_and_loss_shared_only():
    # union gene_id: 0(A),1(B),2(C)
    # sample0 has A,B ; sample1 has B,C
    gene_sample = torch.tensor([0, 0, 1, 1])
    gene_id = torch.tensor([0, 1, 1, 2])

    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id)
    # only B is shared => should pick indices (1) and (2)
    assert pairs.idx0.tolist() == [1]
    assert pairs.idx1.tolist() == [2]

    # make mu so that RMSD between mu[1] and mu[2] is known
    # latent_dim=2: mu1=(0,0), mu2=(3,4) => diff^2 mean = (9+16)/1 = 25
    gene_mu = torch.tensor([
        [9.0, 9.0],   # A in sample0 (unused)
        [0.0, 0.0],   # B in sample0
        [3.0, 4.0],   # B in sample1
        [1.0, 1.0],   # C in sample1 (unused)
    ])
    loss = gene_alignment_msd_loss(gene_mu, pairs)
    assert torch.isclose(loss, torch.tensor(25.0), atol=1e-6)

def test_two_sample_gene_alignment_no_shared_returns_zero():
    gene_sample = torch.tensor([0, 0, 1, 1])
    gene_id = torch.tensor([0, 1, 2, 3])  # no overlap
    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id)
    assert pairs.idx0.numel() == 0
    gene_mu = torch.randn(4, 3)
    loss = gene_alignment_msd_loss(gene_mu, pairs)
    assert loss.item() == 0.0

def test_build_multi_sample_gene_pairs_all_pairs_counts():
    # 3 samples, each has gene_id [0,1,2]
    gene_sample = torch.tensor([0, 0, 0,  1, 1, 1,  2, 2, 2], dtype=torch.long)
    gene_id     = torch.tensor([0, 1, 2,  0, 1, 2,  0, 1, 2], dtype=torch.long)

    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id, mode="all_pairs")
    # For each gene: C(3,2)=3 pairs, total = 3 genes * 3 = 9
    assert pairs.idx0.numel() == 9
    assert pairs.idx1.numel() == 9

    # idx should be within [0, len(gene_id))
    assert int(pairs.idx0.min()) >= 0
    assert int(pairs.idx1.min()) >= 0
    assert int(pairs.idx0.max()) < gene_id.numel()
    assert int(pairs.idx1.max()) < gene_id.numel()

def test_build_multi_sample_gene_pairs_star_counts():
    gene_sample = torch.tensor([0, 0, 0,  1, 1, 1,  2, 2, 2], dtype=torch.long)
    gene_id     = torch.tensor([0, 1, 2,  0, 1, 2,  0, 1, 2], dtype=torch.long)

    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id, mode="star")
    # For each gene: connect ref -> other 2 samples => 2 pairs, total = 3 genes * 2 = 6
    assert pairs.idx0.numel() == 6
    assert pairs.idx1.numel() == 6

    # all idx0 should be in sample 0 OR if gene missing in ref then could differ
    # In this toy, ref exists for all genes, so idx0 should always come from sample 0.
    idx0_samples = gene_sample[pairs.idx0]
    assert torch.all(idx0_samples == 0)

def test_multi_sample_gene_alignment_pairs_and_loss_shared_only_star():
    # global gene nodes (in this order):
    # 0: A in sample0 (gene_id=0)
    # 1: B in sample0 (gene_id=1)
    # 2: B in sample1 (gene_id=1)
    # 3: C in sample1 (gene_id=2)
    # 4: B in sample2 (gene_id=1)
    # 5: D in sample2 (gene_id=3)
    gene_sample = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    gene_id     = torch.tensor([0, 1, 1, 2, 1, 3], dtype=torch.long)

    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id, mode="star")

    # Only gene_id=1 (B) is shared across multiple samples.
    # In star mode with ref=0, ref is B in sample0 => global index 1.
    # It should connect to B in sample1 (index 2) and B in sample2 (index 4).
    assert pairs.idx0.tolist() == [1, 1]
    assert pairs.idx1.tolist() == [2, 4]

    gene_mu = torch.tensor([
        [9.0, 9.0],   # A sample0 (unused)
        [0.0, 0.0],   # B sample0 (ref)
        [3.0, 4.0],   # B sample1 => dist^2=25
        [1.0, 1.0],   # C sample1 (unused)
        [6.0, 8.0],   # B sample2 => dist^2=100
        [2.0, 2.0],   # D sample2 (unused)
    ], dtype=torch.float32)

    loss = gene_alignment_msd_loss(gene_mu, pairs)
    assert torch.isclose(loss, torch.tensor(62.5), atol=1e-6)

def test_multi_sample_gene_alignment_no_shared_returns_zero():
    # 3 samples, all gene_id unique globally => no shared genes across samples
    gene_sample = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    gene_id     = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    pairs = build_multi_sample_gene_pairs(gene_sample, gene_id, mode="star")
    assert pairs.idx0.numel() == 0
    assert pairs.idx1.numel() == 0

    gene_mu = torch.randn(6, 3)
    loss = gene_alignment_msd_loss(gene_mu, pairs)

    # Should be an exact 0 scalar (and ideally on the same device as gene_mu if you applied the fix)
    assert loss.item() == 0.0

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gene_alignment_msd_loss_empty_pairs_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    gene_mu = torch.randn(10, 8, device=device, dtype=torch.float32)
    empty_pairs = MultiSampleGenePairs(
        idx0=torch.empty(0, dtype=torch.long, device=device),
        idx1=torch.empty(0, dtype=torch.long, device=device),
    )
    loss = gene_alignment_msd_loss(gene_mu, empty_pairs)
    assert loss.shape == torch.Size([])
    assert loss.device.type == gene_mu.device.type
    assert loss.dtype == gene_mu.dtype
    assert float(loss.item()) == 0.0