import pytest
import torch

from simba_plus.loss.gene_alignment import (
    build_two_sample_gene_pairs,
    gene_alignment_msd_loss,
)

def test_gene_alignment_pairs_and_loss_shared_only():
    # union gene_id: 0(A),1(B),2(C)
    # sample0 has A,B ; sample1 has B,C
    gene_sample = torch.tensor([0, 0, 1, 1])
    gene_id = torch.tensor([0, 1, 1, 2])

    pairs = build_two_sample_gene_pairs(gene_sample, gene_id)
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

def test_gene_alignment_no_shared_returns_zero():
    gene_sample = torch.tensor([0, 0, 1, 1])
    gene_id = torch.tensor([0, 1, 2, 3])  # no overlap
    pairs = build_two_sample_gene_pairs(gene_sample, gene_id)
    assert pairs.idx0.numel() == 0
    gene_mu = torch.randn(4, 3)
    loss = gene_alignment_msd_loss(gene_mu, pairs)
    assert loss.item() == 0.0

def test_gene_alignment_more_than_two_samples_raises():
    gene_sample = torch.tensor([0, 1, 2])
    gene_id = torch.tensor([0, 0, 0])
    with pytest.raises(NotImplementedError):
        _ = build_two_sample_gene_pairs(gene_sample, gene_id)
