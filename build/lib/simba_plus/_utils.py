"""Utility functions and classes from SIMBA"""

from typing import Tuple
import numpy as np
from anndata import AnnData
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage


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
