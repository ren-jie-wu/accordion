import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch

from simba_plus.load_data import make_sc_HetData_multi_rna

def test_make_sc_HetData_multi_rna_two_samples_block_diagonal_and_gene_id():
    # sample0: 3 cells x 4 genes
    X0 = sp.csr_matrix(
        np.array([
            [1, 0, 2, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 4],
        ], dtype=np.float32)
    )
    ad0 = ad.AnnData(X0)
    ad0.var_names = ["G1", "G2", "G3", "G4"]

    # sample1: 2 cells x 3 genes (overlaps on G2,G3, new G5)
    X1 = sp.csr_matrix(
        np.array([
            [0, 5, 0],
            [6, 0, 7],
        ], dtype=np.float32)
    )
    ad1 = ad.AnnData(X1)
    ad1.var_names = ["G2", "G3", "G5"]

    dat = make_sc_HetData_multi_rna(
        adata_CG_list=[ad0, ad1],
        cell_cat_cov_list=None,
        n_dims=8,
    )

    assert dat["cell"].num_nodes == 5
    assert dat["gene"].num_nodes == 7
    assert set(dat["cell"].sample.unique().tolist()) == {0, 1}
    assert set(dat["gene"].sample.unique().tolist()) == {0, 1}

    edge_type = ("cell", "expresses", "gene")
    ei = dat[edge_type].edge_index
    assert ei.shape[0] == 2
    assert dat[edge_type].edge_attr.numel() == ei.shape[1]

    # block diagonal check: edge endpoints must share sample id
    cs = dat["cell"].sample[ei[0]]
    gs = dat["gene"].sample[ei[1]]
    assert torch.all(cs == gs)

    # gene_id check: same gene name across samples should have same gene_id
    # gene nodes are concatenated in dataset order; sample0 genes indices [0..3], sample1 genes indices [4..6]
    gid = dat["gene"].gene_id
    # "G2": sample0 idx=1, sample1 idx=4
    assert gid[1].item() == gid[4].item()
    # "G3": sample0 idx=2, sample1 idx=5
    assert gid[2].item() == gid[5].item()
    # distinct genes should not share gene_id
    assert gid[0].item() != gid[6].item()  # G1 vs G5
