from typing import Optional, List
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils.num_nodes import maybe_num_nodes
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import pickle as pkl
from simba_plus.loader import (
    CustomNSMultiIndexDataset,
    CustomMultiIndexDataset,
    collate_graph,
    collate,
)
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import scipy.sparse as sp


def negative_sampling(edge_index, num_nodes, num_neg_samples_fold=1):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = (
        torch.rand(edge_index.size(1) * num_neg_samples_fold, device=edge_index.device)
        < 0.5
    )
    mask_2 = torch.logical_not(mask_1)
    neg_edge_index = edge_index.repeat(1, num_neg_samples_fold)
    neg_edge_index[0, mask_1] = torch.randint(
        0, num_nodes[0], (mask_1.sum(),), device=edge_index.device
    )
    neg_edge_index[1, mask_2] = torch.randint(
        0, num_nodes[1], (mask_2.sum(),), device=edge_index.device
    )
    return neg_edge_index


class MyEarlyStopping(EarlyStopping):
    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        if trainer.current_epoch > pl_module.n_kl_warmup * 2:
            self._run_early_stopping_check(trainer)
        else:
            self.wait_count = 0
            torch_inf = torch.tensor(torch.inf)
            self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf


class MyDataModule(L.LightningDataModule):
    """
    Lightning DataModule wrapping train/validation DataLoaders with resampled negatives.

    On each call to `train_dataloader()` or `val_dataloader()`, this module
    triggers `sample_negative()` on the underlying `CustomNSMultiIndexDataset`,
    so that negative edges are resampled once per epoch. The returned DataLoader
    then yields batched subgraphs built via the custom `collate` function.
    """
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        """Resample negative edges for training and return the DataLoader."""
        self.train_loader.dataset.sample_negative()
        return self.train_loader

    def val_dataloader(self):
        """Resample negative edges for validation and return the DataLoader."""
        self.val_loader.dataset.sample_negative()
        return self.val_loader


def _make_tensor(data: HeteroData, device="cpu"):
    data.apply(
        lambda x: (
            torch.tensor(x).to(device)
            if isinstance(x, np.ndarray)
            else x.clone().detach()
        )
    )


def structured_negative_sampling(
    edge_index, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = True
):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:

        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0),), dtype=torch.long).to(
        edge_index.device
    )
    neg_idx = row * num_nodes + rand

    mask = torch.isin(neg_idx, pos_idx)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0),), dtype=torch.long).to(
            edge_index.device
        )
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.isin(neg_idx, pos_idx)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


def get_edge_split_data(data, data_path, edge_types, negative_sampling_fold, logger):
    """
    Create or load train/validation edge splits for a heterogeneous graph and wrap them
    into `CustomNSMultiIndexDataset` objects.

    For each edge type in `edge_types`, randomly permute all observed edges and plit the 
    permuted indices into approximately 96% train, 2% validation, and 2% test. Save these 
    index splits as a pickle file at ``<data_path without .dat>_data_idx.pkl``.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        Heterogeneous graph containing edges for each edge type. Each edge type
        is expected to have an `edge_index` tensor and a `num_edges` attribute.
    data_path : str
        Path to the serialized HetData file (e.g. ``..._HetData.dat``). This path
        is used to derive the name of the index file
        (``<data_path without .dat>_data_idx.pkl``). If the file already exists, 
        it will be loaded and the train/val/test split will be reconstructed.
    edge_types : Iterable[Tuple[str, str, str]]
        Collection of edge types (as 3-tuples, e.g.
        ``("cell", "expresses", "gene")``) for which splits should be created
        and datasets constructed.
    negative_sampling_fold : int
        Number of negative samples to draw per positive edge inside
        `CustomNSMultiIndexDataset`. This controls the ratio of negative to
        positive edges used during training/validation.
    logger : logging.Logger
        Logger used to report progress and information about the created splits.

    Returns
    -------
    train_data : CustomNSMultiIndexDataset
        Dataset wrapping the training positive-edge indices for all specified
        edge types and performing on-the-fly negative sampling.
    val_data : CustomNSMultiIndexDataset
        Dataset wrapping the validation positive-edge indices for all specified
        edge types and performing on-the-fly negative sampling.
    """

    data_idx_path = f"{data_path.split('.dat')[0]}_data_idx.pkl"
    if os.path.exists(data_idx_path):
        # Load existing train/val/test split
        train_idxs = {}
        val_idxs = {}
        with open(data_idx_path, "rb") as f:
            saved_splits = pkl.load(f)
            train_edge_index_dict = saved_splits["train"]
            val_edge_index_dict = saved_splits["val"]
            test_edge_index_dict = saved_splits["test"]
            # Reconstruct train indices as complement of val+test
            for edge_type in edge_types:
                edge_key = "__".join(edge_type)
                train_idxs[edge_type] = train_edge_index_dict[edge_key]
                val_idxs[edge_type] = val_edge_index_dict[edge_key]
        train_data = CustomNSMultiIndexDataset(train_idxs, data, negative_sampling_fold)
        val_data = CustomNSMultiIndexDataset(
            val_idxs,
            data,
            negative_sampling_fold,
        )
    else:
        train_index_dict, val_index_dict, test_index_dict = {}, {}, {}
        for edge_type in edge_types:
            num_edges = data[edge_type].num_edges
            edge_index = data[edge_type].edge_index
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # Find indices that cover all source and target nodes
            selected_indices = set()

            remaining_indices = torch.arange(num_edges)[torch.randperm(num_edges)]
            train_size = int(num_edges * 0.96)
            val_size = int(num_edges * 0.02)
            selected_indices = list(selected_indices)
            train_index_dict["__".join(edge_type)] = torch.tensor(
                # selected_indices
                # +
                remaining_indices[: train_size - len(selected_indices)],
                dtype=torch.long,
            )
            val_index_dict["__".join(edge_type)] = torch.tensor(
                remaining_indices[
                    (train_size - len(selected_indices)) : (
                        train_size - len(selected_indices) + val_size
                    )
                ],
                dtype=torch.long,
            )
            test_index_dict["__".join(edge_type)] = torch.tensor(
                remaining_indices[(train_size - len(selected_indices) + val_size) :],
                dtype=torch.long,
            )
        logger.info(f"Saving data indices to {data_idx_path}...")
        idx_dict = {
            "train": train_index_dict,
            "val": val_index_dict,
            "test": test_index_dict,
        }
        with open(data_idx_path, "wb") as f:
            pkl.dump(
                {
                    "train": train_index_dict,
                    "val": val_index_dict,
                    "test": test_index_dict,
                },
                f,
            )
            print(
                [
                    {k: len(v) for k, v in idx_dict[s].items()}
                    for s in ["train", "val", "test"]
                ]
            )
        train_data = CustomNSMultiIndexDataset(
            pos_idx_dict={
                edge_type: torch.sort(train_index_dict["__".join(edge_type)])[0]
                for edge_type in edge_types
            },
            data=data,
            negative_sampling_fold=negative_sampling_fold,
        )
        val_data = CustomNSMultiIndexDataset(
            pos_idx_dict={
                edge_type: torch.sort(val_index_dict["__".join(edge_type)])[0]
                for edge_type in edge_types
            },
            data=data,
            negative_sampling_fold=negative_sampling_fold,
        )
    logger.info("Finished preparing train/val data splits.")
    return train_data, val_data


def get_edge_split_datamodule(
    data,
    data_path,
    edge_types,
    batch_size,
    num_workers,
    negative_sampling_fold,
    logger,
):
    """
    Build a Lightning DataModule for training and validation on a 
    heterogeneous graph with positive and optionally negative edges.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        Full heterogeneous graph.
    data_path : str
        Path to the serialized HetData file, used for persistent train/val/test splits.
    edge_types : Iterable[EdgeType]
        Edge types to include in training/validation.
    batch_size : int
        Number of edges per batch (across all edge types).
    num_workers : int
        Number of worker processes for the DataLoader.
    negative_sampling_fold : int
        Number of negatives per positive edge used by the dataset.
    logger : logging.Logger
        Logger for reporting progress.

    Returns
    -------
    MyDataModule
        Lightning DataModule providing train/validation dataloaders.
    """
    train_data, val_data = get_edge_split_data(
        data=data,
        edge_types=edge_types,
        data_path=data_path,
        negative_sampling_fold=negative_sampling_fold,
        logger=logger,
    )

    # resample before dataloader creation in case the dataloader only sees 
    # the length of the dataset as positive edge number
    if negative_sampling_fold > 0:
        train_data.sample_negative()
        val_data.sample_negative()

    train_loader, val_loader = get_dataloader(
        train_data=train_data,
        val_data=val_data,
        data=data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    pldata = MyDataModule(train_loader, val_loader)
    return pldata


def get_dataloader(train_data, val_data, data, batch_size, num_workers: int = 30):
    """
    Construct PyTorch DataLoaders for train and validation datasets.

    Uses the custom `collate` function to turn a batch of
    `(edge_type, edge_index, full_data)` samples from `CustomNSMultiIndexDataset`
    into a subgraph `HeteroData` object.

    Parameters
    ----------
    train_data : CustomNSMultiIndexDataset
        Dataset used for training.
    val_data : CustomNSMultiIndexDataset
        Dataset used for validation.
    data : torch_geometric.data.HeteroData
        Full heterogeneous graph (not used directly here, kept for compatibility).
    batch_size : int
        Number of edges per batch.
    num_workers : int, optional
        Number of worker processes for the DataLoaders, by default 30.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding per-batch subgraphs for training.
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding per-batch subgraphs for validation.
    """
    # collate_ = partial(collate, data=data)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        # collate_fn=partial(collate_graph, data=data),
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        # collate_fn=partial(collate_graph, data=data),
        collate_fn=collate,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_node_weights(
    data: HeteroData,
    pldata: L.LightningDataModule,
    checkpoint_dir: str,
    logger,
    device: torch.device,
):
    node_weights_path = os.path.join(checkpoint_dir, "node_weights_dict.pt")
    if os.path.exists(node_weights_path):
        try:
            loaded = torch.load(node_weights_path, map_location="cpu", weights_only=False)
            # Convert loaded values to device tensors if needed
            node_weights_dict = {
                k: (
                    v.to(device)
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, device=device)
                )
                for k, v in loaded.items()
            }
            logger.info(f"Loaded node_weights_dict from {node_weights_path}")
        except Exception as e:  # pragma: no cover - best-effort load
            logger.info(
                f"Failed to load node_weights_dict from {node_weights_path}: {e}"
            )
    else:
        logger.info("Computing node weights...")
        node_counts_dict = {
            node_type: torch.ones(x.shape[0], device=device)
            for (node_type, x) in data.n_id_dict.items()
        }
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            node_counts_dict[src] += degree(data[edge_type].edge_index[0].to(device))
            node_counts_dict[dst] += degree(data[edge_type].edge_index[1].to(device))

        node_weights_dict = {k: 1.0 / v for k, v in node_counts_dict.items()}
        logger.info(f"Saving node weights to {node_weights_path}...")
        torch.save(node_weights_dict, node_weights_path)
    return node_weights_dict


def get_nll_scales(
    data,
    pldata,
    edge_types,
    batch_size,
    n_batches,
    n_val_batches,
    logger,
    scale_with_pos_edges_only: bool = True
):
    """
    Compute scaling factors as n_dense_edges/n_edges to rescale sampled NLL to a dense-edge equivalent

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        Full heterogeneous graph.
    pldata : L.LightningDataModule
        Lightning DataModule providing train/validation dataloaders.
    edge_types : Iterable[EdgeType]
        Edge types to include in training/validation.
    logger : logging.Logger
    scale_with_pos_edges_only : bool
        Whether to scale with only positive edges.

    Returns
    -------
    nll_scale : float
        Scaling factor for NLL.
    val_nll_scale : float
        Scaling factor for validation NLL.

    Notes
    -----
    Previous version in fact scale with only positive edges, thus this is set as default. 
    Consider changing default to False in the future. #TODO
    """
    n_dense_edges = 0
    for src_nodetype, _, dst_nodetype in edge_types:
        n_dense_edges += data[src_nodetype].num_nodes * data[dst_nodetype].num_nodes
    train_data = pldata.train_loader.dataset
    val_data = pldata.val_loader.dataset
    n_edges = train_data.total_length
    n_pos_edges = sum(len(train_data.pos_idx_dict[etype]) for etype in train_data.edge_types)
    n_val_edges = val_data.total_length
    n_val_pos_edges = sum(len(val_data.pos_idx_dict[etype]) for etype in val_data.edge_types)

    if scale_with_pos_edges_only:
        nll_scale = n_dense_edges / n_pos_edges
        val_nll_scale = n_dense_edges / n_val_pos_edges
    else:
        nll_scale = n_dense_edges / n_edges
        val_nll_scale = n_dense_edges / n_val_edges
    logger.info(
        f"Scaling KL divergence loss with NLL scaling factor:{nll_scale}, {val_nll_scale}"
    )
    return nll_scale, val_nll_scale


def write_bed(adata, filename=None):
    """Write peaks into .bed file

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with peaks as variables.
    use_top_pcs: `bool`, optional (default: True)
        Use top-PCs-associated features
    filename: `str`, optional (default: None)
        Filename name for peaks.
        By default, a file named 'peaks.bed' will be written to
        `.settings.workdir`
    """
    for x in ["chr", "start", "end"]:
        if x not in adata.var_keys():
            raise ValueError(f"could not find {x} in `adata.var_keys()`")
    peaks_selected = adata.var[["chr", "start", "end"]]
    peaks_selected.to_csv(filename, sep="\t", header=False, index=False)
    fp, fn = os.path.split(filename)
    print(f'"{fn}" was written to "{fp}".')


def setup_logging(
    name: str = "simba_plus",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    file_name: Optional[str] = None,
    console: bool = True,
):
    """Create and return a configured logger for use across subcommands.

    - name: logger name (typically the subcommand or package name).
    - log_dir: when provided, a rotating file handler will be created there.
    - level: logging level string, e.g. "INFO", "DEBUG".
    - file_name: optional filename for file logging (defaults to "<name>.log").
    - console: whether to add a stdout stream handler.

    Repeated calls are idempotent (existing handlers for the same logger are cleared
    first) so this can be called from different subcommands safely.
    """

    if file_name is None:
        file_name = f"{name}.log"

    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicated logs when called multiple times.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(p / file_name, maxBytes=10_000_000, backupCount=5)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False

    # Print ASCII art banner when logger starts
    ascii_art = r"""
                  _                 
                 | |            _   
  _ __🦁_ __ ___ | |__   __ _ _| |_ 
 / __| | '_ ` _ \| '_ \ / _` |_   _|
 \__ \ | | | | | | |_) | (_| | |_|  
 |___/_|_| |_| |_|_.__/ \__,_|      
    """
    logger.info(ascii_art)
    return logger
def is_integer_valued(X, tol=1e-8, max_check=100_000):
    """
    Check if the values in X are integer-valued.
    Args:
        X (numpy.ndarray or scipy.sparse.spmatrix): The input data.
        tol (float): The tolerance for the comparison.
        max_check (int): The maximum number of values to check. This is for time efficiency.
    Returns:
        bool: True if the values in X are integer-valued, False otherwise.
    """
    if np.issubdtype(X.dtype, np.integer):
        return True
    
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    assert data.size > 0, "X is empty"

    if data.size > max_check:
        idx = np.random.choice(data.size, max_check, replace=False)
        data = data[idx]
    
    return np.allclose(data, np.round(data), atol=tol)
