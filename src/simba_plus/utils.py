from typing import Optional, List
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import pickle as pkl
from simba_plus.loader import CustomMultiIndexDataset, collate
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def negative_sampling(edge_index, num_nodes, num_neg_samples_fold=2):
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
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def setup(self, stage: str):
        if stage == "fit":
            self.train_loader.dataset.setup()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
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


def get_edge_split_data(data, data_path, edge_types, logger):
    data_idx_path = f"{data_path.split(".dat")[0]}_data_idx.pkl"
    if os.path.exists(data_idx_path):
        # Load existing train/val/test split
        train_idxs = []
        val_idxs = []
        with open(data_idx_path, "rb") as f:
            saved_splits = pkl.load(f)
            val_edge_index_dict = saved_splits["val"]
            test_edge_index_dict = saved_splits["test"]
            # Reconstruct train indices as complement of val+test
            train_edge_index_dict = {}
            for edge_type in edge_types:
                edge_key = "__".join(edge_type)
                all_edges = set(range(data[edge_type].num_edges))
                val_test_edges = set(
                    val_edge_index_dict[edge_key].tolist()
                    + test_edge_index_dict[edge_key].tolist()
                )
                train_edges = list(all_edges - val_test_edges)
                train_edge_index_dict[edge_key] = torch.tensor(train_edges)
                train_idxs.append(train_edge_index_dict[edge_key])
                val_idxs.append(val_edge_index_dict[edge_key])
        train_data = CustomMultiIndexDataset(edge_types, train_idxs)
        val_data = CustomMultiIndexDataset(edge_types, val_idxs)
    else:
        train_index_dict, val_index_dict, test_index_dict = {}, {}, {}
        for edge_type in edge_types:
            num_edges = data[edge_type].num_edges
            edge_index = data[edge_type].edge_index
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # Find indices that cover all source and target nodes
            selected_indices = set()

            indices = torch.arange(num_edges)[torch.randperm(num_edges)]
            logger.info("Selecting source node")
            selected_indices.update(
                np.unique(src_nodes[indices].cpu().numpy(), return_index=True)[
                    1
                ].tolist()
            )
            logger.info("Selecting destination node")
            selected_indices.update(
                np.unique(dst_nodes[indices].cpu().numpy(), return_index=True)[
                    1
                ].tolist()
            )
            logger.info("Selected indices")

            # Fill up to 90% for train, 5% for val
            remaining_indices = [
                i for i in indices.cpu().numpy() if i not in selected_indices
            ]
            logger.info("Got remaining indices")
            train_size = int(num_edges * 0.9)
            val_size = int(num_edges * 0.05)
            selected_indices = list(selected_indices)
            train_index_dict["__".join(edge_type)] = torch.tensor(
                selected_indices
                + remaining_indices[: train_size - len(selected_indices)]
            )
            val_index_dict["__".join(edge_type)] = torch.tensor(
                remaining_indices[
                    (train_size - len(selected_indices)) : (
                        train_size - len(selected_indices) + val_size
                    )
                ]
            )
            test_index_dict["__".join(edge_type)] = torch.tensor(
                remaining_indices[(train_size - len(selected_indices) + val_size) :]
            )
        logger.info(f"Saving data indices to {data_idx_path}...")
        with open(data_idx_path, "wb") as f:
            pkl.dump({"val": val_index_dict, "test": test_index_dict}, f)
        train_data = CustomMultiIndexDataset(
            edge_types, list(train_index_dict.values())
        )
        val_data = CustomMultiIndexDataset(edge_types, list(val_index_dict.values()))
    return train_data, val_data


def get_edge_split_datamodule(
    data,
    data_path,
    edge_types,
    batch_size,
    num_workers,
    checkpoint_dir,
    logger,
):
    train_data, val_data = get_edge_split_data(
        data=data, edge_types=edge_types, data_path=data_path, logger=logger
    )
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
    collate_ = partial(collate, data=data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, collate_fn=collate_, num_workers=num_workers
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
            loaded = torch.load(node_weights_path, map_location="cpu")
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
        node_counts_dict = {
            node_type: torch.ones(x.shape[0], device=device)
            for (node_type, x) in data.x_dict.items()
        }
        for batch in tqdm(pldata.train_loader):
            for node_type in batch.node_types:
                node_counts_dict[node_type][batch[node_type].n_id] += 1
        node_weights_dict = {k: 1.0 / v for k, v in node_counts_dict.items()}
        torch.save(node_weights_dict, node_weights_path)
    return node_weights_dict


def get_nll_scales(
    data,
    edge_types,
    num_neg_samples_fold,
    batch_size,
    n_batches,
    n_val_batches,
    logger,
):
    n_dense_edges = 0
    for src_nodetype, _, dst_nodetype in edge_types:
        n_dense_edges += data[src_nodetype].num_nodes * data[dst_nodetype].num_nodes

    nll_scale = n_dense_edges / ((num_neg_samples_fold + 1) * batch_size * n_batches)
    val_nll_scale = n_dense_edges / (
        (num_neg_samples_fold + 1) * batch_size * n_val_batches
    )
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
