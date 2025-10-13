"""Utility functions and classes from SIMBA"""

from typing import Tuple
import numpy as np
from anndata import AnnData
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage
import tables


def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g["lr"] = new_lr


def add_cov_to_latent(node_storage: NodeStorage, z: torch.Tensor):
    cat_z = z
    if "cont_cov" in node_storage:
        if z.dim() != node_storage.cont_cov.dim():
            cat_z = torch.cat([z, node_storage.cont_cov.unsqueeze(-1)], dim=-1)
        else:
            cat_z = torch.cat([z, node_storage.cont_cov], dim=-1)

    # if node_storage.cat_covs is not None:
    #     categorical_input = torch.split(node_storage.cat_covs, 1, dim=1)
    # else:
    #     categorical_input = tuple() # TODO: categorical covariate not considered properly
    return cat_z


def count_additional_latent_dims_for_covs(data: HeteroData):
    additional_dims = 0
    if "cont_cov" in data["cell"]:
        if data["cell"].cont_cov.ndim == 1:
            additional_dims += 1
        elif data["cell"].cont_cov.ndim == 2:
            additional_dims += data["cell"].cont_cov.shape[-1]
        else:
            raise ValueError(
                "Cell has continuous covariate with more than 2 dimensions: check data again."
            )
    if "cat_cov" in data["cell"]:
        if data["cell"].cat_cov.ndim == 1:
            additional_dims += len(data["cell"].cat_cov.unique())
        elif data["cell"].cat_cov.ndim == 2:
            for i in data["cell"].cat_cov.shape[-1]:
                additional_dims += len(data["cell"].cat_cov[:, i].unique())
        else:
            raise ValueError(
                "Cell has categorical covariate with more than 2 dimensions: check data again."
            )
    return additional_dims


def pairwise_wasserstein_distance(means, stds):
    """
    Calculate pairwise Wasserstein distance between all pairs of mean-field Gaussian distributions.

    Parameters:
    - means: numpy array of shape (n_samples, n_features) containing the means
    - stds: numpy array of shape (n_samples, n_features) containing the standard deviations

    Returns:
    - distances: numpy array of shape (n_samples, n_samples) containing pairwise distances
    """
    n_samples, n_features = means.shape

    # Compute squared differences of means
    mean_diff_sq = np.sum(
        (means[:, np.newaxis, :] - means[np.newaxis, :, :]) ** 2, axis=2
    )

    # Compute sum of variances and their square roots
    var_sum = stds[:, np.newaxis, :] ** 2 + stds[np.newaxis, :, :] ** 2
    std_product = stds[:, np.newaxis, :] ** 2 * stds[np.newaxis, :, :] ** 2

    # Compute the trace term
    trace_term = np.sum(var_sum - 2 * np.sqrt(std_product), axis=2)

    # Combine terms and take square root
    distances = np.sqrt(mean_diff_sq + trace_term)

    return distances


def make_key(node_type: str, edge_type: Tuple[str]):
    return f"{node_type}__{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"


# modifed from
# scanpy https://github.com/theislab/scanpy/blob/master/scanpy/readwrite.py
def _read_v3_10x_h5(filename):
    """
    Read hdf5 file from Cell Ranger v3 or later versions.
    """
    with tables.open_file(str(filename), "r") as f:
        try:
            dsets = {}
            for node in f.walk_nodes("/matrix", "Array"):
                dsets[node.name] = node.read()
            from scipy.sparse import csr_matrix

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == np.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets["barcodes"].astype(str)),
                var=dict(
                    var_names=dsets["name"].astype(str),
                    gene_ids=dsets["id"].astype(str),
                    feature_types=dsets["feature_type"].astype(str),
                    genome=dsets["genome"].astype(str),
                ),
            )
            return adata
        except KeyError:
            raise Exception("File is missing one or more required datasets.")


def _read_legacy_10x_h5(filename, genome=None):
    """
    Read hdf5 file from Cell Ranger v2 or earlier versions.
    """
    with tables.open_file(str(filename), "r") as f:
        try:
            children = [x._v_name for x in f.list_nodes(f.root)]
            if not genome:
                if len(children) > 1:
                    raise ValueError(
                        f"'{filename}' contains more than one genome. "
                        "For legacy 10x h5 "
                        "files you must specify the genome "
                        "if more than one is present. "
                        f"Available genomes are: {children}"
                    )
                genome = children[0]
            elif genome not in children:
                raise ValueError(
                    f"Could not find genome '{genome}' in '{filename}'. "
                    f"Available genomes are: {children}"
                )
            dsets = {}
            for node in f.walk_nodes("/" + genome, "Array"):
                dsets[node.name] = node.read()
            # AnnData works with csr matrices
            # 10x stores the transposed data, so we do the transposition
            from scipy.sparse import csr_matrix

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == np.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )
            # the csc matrix is automatically the transposed csr matrix
            # as scanpy expects it, so, no need for a further transpostion
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets["barcodes"].astype(str)),
                var=dict(
                    var_names=dsets["gene_names"].astype(str),
                    gene_ids=dsets["genes"].astype(str),
                ),
            )
            return adata
        except KeyError:
            raise Exception("File is missing one or more required datasets.")
