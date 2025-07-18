from typing import Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils.num_nodes import maybe_num_nodes
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


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


def _make_tensor(data: HeteroData, device="cpu"):
    data.apply(
        lambda x: (
            torch.tensor(x).to(device)
            if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)
            else x
        )
    )


def _assign_node_id(data: HeteroData):
    for node_type in data.node_types:
        data[node_type].n_id = torch.arange(data[node_type].num_nodes)


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
