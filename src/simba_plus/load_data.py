# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
from typing import Dict, Iterable
import numpy as np
import torch
from torch_geometric.data import HeteroData
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import anndata as ad
from simba_plus.utils import _make_tensor, is_integer_valued, _to_coo
import argparse
from typing import List, Optional
import os

def validate_input(adata_CG, adata_CP):
    """
    Validate and extract dimensions from input AnnData objects.
    It returns the number of cells, genes, peaks, and motifs (-1 if not determined).

    Args:
        adata_CG (anndata.AnnData or None): AnnData for cell by gene data.
        adata_CP (anndata.AnnData or None): AnnData for cell by peak data.

    Returns:
        tuple:
            n_cells (int): Number of cells determined from provided AnnDatas.
            n_genes (int): Number of genes (-1 if adata_CG not given).
            n_peaks (int): Number of peaks (-1 if adata_CP not given).
            n_motifs (int): Placeholder, always -1.
    """
    n_cells = n_genes = n_peaks = n_motifs = -1
    if adata_CG is not None:
        n_cells, n_genes = adata_CG.shape
        if adata_CP is not None:
            _, n_peaks = adata_CP.shape
    elif adata_CP is not None:
        n_cells, n_peaks = adata_CP.shape
    if adata_CG is not None and adata_CP is not None:
        if not (adata_CG.obs.index == adata_CP.obs.index).all():
            adata_CP = adata_CP[adata_CG.obs.index, :]
        assert (adata_CG.obs.index == adata_CG.obs.index).all()
    return n_cells, n_genes, n_peaks, n_motifs


def type_attribute(data):
    for node_type in data.node_types:
        data[node_type].x = torch.tensor(data[node_type].x, dtype=torch.float)
    return data


def make_sc_HetData(
    adata_CG: ad.AnnData = None,
    adata_CP: ad.AnnData = None,
    cell_cont_covariate_to_include: Dict[str, Iterable[str]] = None,
    cell_cat_cov: str = None,
):
    if not adata_CG and not adata_CP:
        raise ValueError("No data provided for edge construction")
    n_cells, n_genes, n_peaks, n_motifs = validate_input(adata_CG, adata_CP)
    data = HeteroData()
    n_dims = 50
    if n_cells > 0:
        if adata_CG is not None:
            data["cell"].x = torch.zeros((adata_CG.n_obs, n_dims))
            data["cell"].size_factor = adata_CG.X.toarray().sum(axis=1) / np.median(
                adata_CG.X.toarray().sum(axis=1)
            )
            if cell_cat_cov is not None:
                data["cell"].batch = torch.tensor(
                    adata_CG.obs[cell_cat_cov].astype("category").cat.codes.values,
                    dtype=torch.long,
                )
            data["gene"].size_factor = adata_CG.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CG.obs[
                    cell_cont_covariate_to_include["CG"]
                ].values
            else:
                data["cell"].cont_cov = None
        elif adata_CP is not None:
            data["cell"].x = torch.zeros((adata_CP.n_obs, n_dims))
            # data["cell"].size_factor = adata_CP.X.toarray().sum(axis=1) / np.median(
            #     adata_CP.X.toarray().sum(axis=1)
            # )
            if cell_cat_cov is not None:
                data["cell"].batch = torch.tensor(
                    adata_CP.obs[cell_cat_cov].astype("category").cat.codes.values,
                    dtype=torch.long,
                )
            # data["peak"].size_factor = adata_CP.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CP.obs[
                    cell_cont_covariate_to_include["CP"]
                ].values
            else:
                data["cell"].cont_cov = None

    if n_genes > 0:
        data["gene"].x = torch.zeros((adata_CG.n_vars, n_dims))
    if n_peaks > 0:
        data["peak"].x = torch.zeros((adata_CP.n_vars, n_dims))
    if adata_CG is not None:
        data["cell", "expresses", "gene"].edge_index = torch.from_numpy(
            np.stack(adata_CG.X.nonzero(), axis=0)
        ).long()  # [2, num_edges_expresses]
        data["cell", "expresses", "gene"].edge_attr = torch.from_numpy(
            adata_CG.X[adata_CG.X.nonzero()]
        ).squeeze()  # [num_edges_expresses, num_features_expresses]
        data["cell", "expresses", "gene"].edge_dist = (
            "NegativeBinomial"
            if is_integer_valued(adata_CG.X)
            else "Normal"
        )

    if adata_CP is not None:
        data["cell", "has_accessible", "peak"].edge_index = torch.from_numpy(
            np.stack(adata_CP.X.nonzero(), axis=0)
        ).long()  # [2, num_edges_has_accessible]
        data["cell", "has_accessible", "peak"].edge_attr = torch.ones(
            data["cell", "has_accessible", "peak"].num_edges
        )
        data["cell", "has_accessible", "peak"].edge_dist = (
            "Bernoulli"  # Move to NB later on
        )

    for node_type in data.node_types:
        data[node_type].x = data[node_type].x.detach().clone().float()
    return data


def _align_gene_names(genes_list: List[List[str]]): #TODO
    """
    Align the gene names of the multiple RNA-seq datasets in case they use different naming systems.
    """
    pass #TODO


def _check_no_duplicates(*genes_list: List[List[str]], name="RNA"):
    """
    Check if there are any duplicate gene names in the multiple RNA-seq datasets.
    """
    for i, genes in enumerate(genes_list):
        if len(genes) != len(set(genes)):
            raise ValueError(f"Duplicate feature names found in the {name} datasets {i} (index from 0)")
    return True

def make_sc_HetData_multi_rna(
    adata_CG_list: Optional[List[Optional[ad.AnnData]]] = None,
    cell_cat_cov_list: Optional[List[Optional[str]]] = None,
    n_dims: int = 50,
):
    """
    Construct a HeteroData object from multiple scRNA-seq datasets (cell-by-gene).

    Notes
    -----
    - Cells from different samples are concatenated into one "cell" node type,
      with `data["cell"].sample` indicating sample membership.
    - Genes are also concatenated (sample-specific gene nodes), with
      `data["gene"].sample` and `data["gene"].gene_id` enabling cross-sample alignment.
    - For batch correction:
        * `data["cell"].batch_local`: per-sample batch ids (0..B_s-1)
        * `data["cell"].batch`: global batch ids with offsets, unique across samples
    """

    if adata_CG_list is None or \
        len(adata_CG_list) == 0 or \
        all(adata is None for adata in adata_CG_list):
        raise ValueError("No data provided for edge construction")
    
    if cell_cat_cov_list is None or len(cell_cat_cov_list) == 0:
        cell_cat_cov_list = [None] * len(adata_CG_list)
    if len(cell_cat_cov_list) == 1: # broadcast
        cell_cat_cov_list = cell_cat_cov_list * len(adata_CG_list)
    assert len(adata_CG_list) == len(cell_cat_cov_list), "The number of RNA-seq datasets and cell batch covariates must be the same"
    
    data = HeteroData()

    ## prepare genes
    # taking care of None in adata_CG_list because we may allow providing only one of RNA and ATAC in the future
    genes_list = [list(map(str, adata.var_names)) if adata is not None else [] for adata in adata_CG_list]
    _check_no_duplicates(*genes_list)
    _align_gene_names(genes_list)
    union_genes = sorted(set().union(*[set(genes) for genes in genes_list if len(genes) > 0]))
    gene_to_id = {gene: i for i, gene in enumerate(union_genes)} # common gene names for future gene alignment use
    
    g_list = [adata.n_vars if adata is not None else 0 for adata in adata_CG_list]
    n_list = [adata.n_obs if adata is not None else 0 for adata in adata_CG_list]

    if sum(g_list) == 0 or sum(n_list) == 0:
        raise ValueError("No genes or cells found in the RNA-seq datasets")
    
    ## assign x, batch, sample, gene_id
    ## size_factor is IGNORED since it is never used in current implementation

    data["cell"].x = torch.zeros((sum(n_list), n_dims), dtype=torch.float)
    data["gene"].x = torch.zeros((sum(g_list), n_dims), dtype=torch.float)

    if any(cell_cat_cov is not None for cell_cat_cov in cell_cat_cov_list):
        batch_local_all = torch.zeros((sum(n_list),), dtype=torch.long)
        batch_global_all = torch.zeros((sum(n_list),), dtype=torch.long)
        offset = 0
        for i, (cell_cat_cov, adata) in enumerate(zip(cell_cat_cov_list, adata_CG_list)):
            if adata is None:
                continue
            idx_start, idx_end = sum(n_list[:i]), sum(n_list[:i+1])
            if cell_cat_cov is not None and cell_cat_cov in adata.obs.columns:
                codes_local = torch.from_numpy(adata.obs[cell_cat_cov]
                    .astype("category")
                    .cat.codes
                    .to_numpy()
                    .astype(np.int64)
                ).long()
                batch_local_all[idx_start:idx_end] = codes_local
                batch_global_all[idx_start:idx_end] = codes_local + offset
                offset += int(codes_local.max()) + 1 if codes_local.numel() > 0 else 0
            else:
                batch_local_all[idx_start:idx_end] = 0
                batch_global_all[idx_start:idx_end] = offset
                offset += 1
        data["cell"].batch_local = batch_local_all
        data["cell"].batch = batch_global_all

    data["cell"].sample = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_list) if n > 0],
        dim=0,
    )
    data["gene"].sample = torch.cat(
        [torch.full((g,), i, dtype=torch.long) for i, g in enumerate(g_list) if g > 0],
        dim=0,
    )

    data["gene"].gene_id = torch.cat([
        torch.tensor(
            [gene_to_id[gene] for gene in genes], dtype=torch.long
        ) for genes in genes_list
    ], dim=0)

    data["gene"].local_id = torch.cat([
        torch.arange(g, dtype=torch.long) for g in g_list
    ], dim=0)

    ## build edges

    edge_type = ("cell", "expresses", "gene")

    coo_list = [_to_coo(adata.X) if adata is not None else None for adata in adata_CG_list]
    _get_edge_index = lambda coo, cell_off, gene_off: torch.vstack(
        [
            torch.from_numpy(coo.row.astype(np.int64)) + cell_off,
            torch.from_numpy(coo.col.astype(np.int64)) + gene_off,
        ]
    )

    if len(coo_list) == 0 or all(coo is None for coo in coo_list):
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.hstack(
            [
                _get_edge_index(coo, cell_off=sum(n_list[:i]), gene_off=sum(g_list[:i])) 
                    for i, coo in enumerate(coo_list) if coo is not None
            ]
        ).long()
        edge_attr = torch.cat(
            [
                torch.from_numpy(np.asarray(coo.data).astype(np.float32)).view(-1)
                    for coo in coo_list if coo is not None
            ], dim=0,
        )

    is_integer_list = [is_integer_valued(adata.X) for adata in adata_CG_list if adata is not None]
    if all(is_integer_list):
        edge_dist = "NegativeBinomial"
    elif any(is_integer_list): # mixed integer and non-integer data
        print("[WARNING] Mixed integer and non-integer data detected. Edge distribution will be set to Normal.")
        edge_dist = "Normal"
    else:
        edge_dist = "Normal"

    data[edge_type].edge_index = edge_index
    data[edge_type].edge_attr = edge_attr
    data[edge_type].edge_dist = edge_dist

    return data

def load_from_path(path: str, device=None) -> HeteroData:
    if device is not None:
        warnings.warn("device argument is deprecated and might be removed in future versions. Currently load data stage does not support device transfer.")
    data = torch.load(path, weights_only=False)
    data.generate_ids()
    # _make_tensor(data, device=device)
    return data


def add_argument(parser):
    parser.description = "Prepare a HeteroData object from AnnData of RNA-seq and ATAC-seq data."
    parser.add_argument(
        "--gene-adata",
        type=str,
        nargs="+", # allow 1 or more gene adatas
        help="Path to the cell by gene AnnData file(s) (e.g., .h5ad).",
    )
    parser.add_argument(
        "--peak-adata",
        type=str,
        help="Path to the cell by gene AnnData file (e.g., .h5ad).",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        nargs="+", # allow 1 or more batch columns for multiple RNA-seq datasets
        help="Batch column(s) in AnnData.obs of gene AnnData(s). "\
             "Should match the number of gene AnnData(s), or be a single value to broadcast to all gene AnnData(s). "\
             "If gene AnnData is not provided, peak AnnData will be used.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Path to the saved HeteroData object (e.g., .dat file).",
    )
    return parser


def main(args):
    gene_paths = args.gene_adata
    if gene_paths is None or len(gene_paths) == 0:
        print("No gene adatas provided.")
        gene_adatas = None
    else:
        gene_adatas = []
        for gene_path in gene_paths:
            try:
                adata = ad.read_h5ad(gene_path)
                gene_adatas.append(adata)
            except Exception as e:
                print(f"Skipping {gene_path} due to error: {e}")
                gene_adatas.append(None)
    
    dat = make_sc_HetData_multi_rna(
        adata_CG_list=gene_adatas,
        cell_cat_cov_list=args.batch_col,
        n_dims=1, # since .x is deleted in train.py, this can be arbitrary
    )
    
    # dat = make_sc_HetData(
    #     adata_CG=ad.read_h5ad(args.gene_adata) if args.gene_adata else None,
    #     adata_CP=ad.read_h5ad(args.peak_adata) if args.peak_adata else None,
    #     cell_cat_cov=args.batch_col,
    # )
    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(dat, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
