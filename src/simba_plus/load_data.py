# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
from typing import Dict, Iterable
import numpy as np
import torch
from torch_geometric.data import HeteroData
import anndata as ad
from simba_plus.utils import _assign_node_id, _make_tensor


def validate_input(adata_CG, adata_CP):
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
    cell_cont_covariate_to_include: Dict[str, Iterable[str]] = {"CG": "pct_counts_mt"},
    cell_cat_covariate_to_include: Dict[str, Iterable[str]] = None,
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
            data["gene"].size_factor = adata_CG.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CG.obs[
                    cell_cont_covariate_to_include["CG"]
                ].values
            else:
                data["cell"].cont_cov = None
            if cell_cat_covariate_to_include:
                data["cell"].cat_cov = adata_CG.obs[
                    cell_cat_covariate_to_include["CG"]
                ].values
            else:
                data["cell"].cat_cov = None
        elif adata_CP is not None:
            data["cell"].x = torch.zeros((adata_CG.n_obs, n_dims))
            data["cell"].size_factor = adata_CP.X.toarray().sum(axis=1) / np.median(
                adata_CP.X.toarray().sum(axis=1)
            )
            data["peak"].size_factor = adata_CP.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CP.obs[
                    cell_cont_covariate_to_include["CP"]
                ].values
            else:
                data["cell"].cont_cov = None
            if cell_cat_covariate_to_include:
                data["cell"].cat_cov = adata_CP.obs[
                    cell_cat_covariate_to_include["CP"]
                ].values
            else:
                data["cell"].cat_cov = None

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
            if np.issubdtype(adata_CG.X.dtype, np.integer)
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
        data[node_type].x = torch.tensor(data[node_type].x, dtype=torch.float)
    return data


def load_from_path(path: str, device="cpu") -> HeteroData:
    data = torch.load(path, map_location=device, weights_only=False)
    _assign_node_id(data)
    _make_tensor(data, device=device)
    return data
