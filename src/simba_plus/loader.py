from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.typing import EdgeType
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
from torch_geometric.transforms.to_device import ToDevice
from copy import copy
import pickle as pkl
import time

from simba_plus.negative_sampling import negative_sampling_same_sample_bipartite


class CustomIndexDataset(Dataset):
    def __init__(
        self,
        edge_attr_type: EdgeType,
        index: torch.Tensor,
    ):
        self.edge_attr_type = edge_attr_type
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.edge_attr_type, self.index[idx]


class CustomNSMultiIndexDataset(Dataset):
    """
    Multi-edge-type dataset with optional negative sampling for heterogeneous graphs.
    """
    def __init__(
        self,
        pos_idx_dict: dict[EdgeType, torch.Tensor],
        data: torch_geometric.data.HeteroData,
        negative_sampling_fold: int = 1,
    ):
        """
        Initialize the dataset.  
        If negative_sampling_fold is 0, no negative sampling is performed; 
        use all positive edges and permute them.

        Parameters
        ----------
        pos_idx_dict : dict[EdgeType, torch.Tensor]
            Dictionary of positive edge indices for each edge type.
        data : torch_geometric.data.HeteroData
            HeteroData object containing the graph data.
        negative_sampling_fold : int, optional
            Number of negative samples to draw per positive edge.

        Returns
        -------
        None
        """
        assert len(pos_idx_dict) > 0, "pos_idx_dict must not be empty"
        self.pos_idx_dict = pos_idx_dict
        self.edge_types = list(pos_idx_dict.keys())
        self.lengths = [
            len(self.pos_idx_dict[edge_type]) for edge_type in self.edge_types
        ]
        self.total_length = sum(self.lengths)
        self.num_types = len(self.edge_types)
        self.setup_count = 0
        self.negative_sampling_fold = negative_sampling_fold
        self.data = data
        if negative_sampling_fold == 0: # No negative sampling; use all positive edges and permute them
            self.edge_index_dict = self.pos_idx_dict
            print("Permuting edges...")
            self.lengths = [
                len(self.edge_index_dict[edge_type]) for edge_type in self.edge_types
            ]
            self.total_length = sum(self.lengths)
            perm_idx = torch.randperm(self.total_length)
            self.type_assignments = torch.cat(
                [torch.full((length,), i) for i, length in enumerate(self.lengths)]
            )[perm_idx] # total_length list of edge type indices
            self.permuted_edge_idx = {
                k: v[torch.randperm(len(v))] for k, v in self.edge_index_dict.items()
            }
            self.all_edge_idx = torch.zeros(self.total_length, dtype=torch.long)
            for i, edge_type in enumerate(self.edge_types):
                self.all_edge_idx[self.type_assignments == i] = self.permuted_edge_idx[
                    edge_type
                ] # total_length list of edge indices

    def sample_negative(self, device=None):
        """
        (Re)generate negative edges and build a flattened, permuted index.

        Copies the original `HeteroData` into `self.full_data`. If
        `negative_sampling_fold > 0`, for each edge type it samples additional
        negative edges using `torch_geometric.utils.negative_sampling`, appends
        them (with edge_attr = 0) to `full_data[edge_type]`, and updates
        `edge_index_dict`, `type_assignments`, and `all_edge_idx` to include both
        positive and negative edges.

        Parameters
        ----------
        device : torch.device, optional
            Dummy parameter for now.

        Returns
        -------
        None

        Notes
        -----
        Should be called at least once before training; can be called once per
        epoch to resample negatives.
        """
        t1 = time.time()
        self.setup_count += 1
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_geometric.seed.seed_everything(self.setup_count)
        self.setup_count += 1
        self.full_data = copy(self.data)
        if self.negative_sampling_fold > 0:
            print("Sampling negative edges...")
            self.edge_index_dict = {}
            for edge_type in self.edge_types:
                src, _, dst = edge_type
                if hasattr(self.data[src], "sample") and hasattr(self.data[dst], "sample"):
                    neg_edge_index = negative_sampling_same_sample_bipartite(
                        self.data[edge_type].edge_index,
                        src_sample=self.data[src].sample,
                        dst_sample=self.data[dst].sample,
                        num_neg_samples_fold=self.negative_sampling_fold,
                        method="sparse",
                    )
                else:
                    neg_edge_index = torch_geometric.utils.negative_sampling(
                        self.data[edge_type].edge_index,
                        num_nodes=(
                            self.data[src].num_nodes,
                            self.data[dst].num_nodes,
                        ),
                        num_neg_samples=len(self.pos_idx_dict[edge_type])
                        * self.negative_sampling_fold,
                        method="sparse",
                    ).to(self.data[edge_type].edge_index.device)
                self.full_data[edge_type].edge_index = torch.cat(
                    [self.data[edge_type].edge_index, neg_edge_index], dim=1
                )
                self.full_data[edge_type].edge_attr = torch.cat(
                    [
                        self.data[edge_type].edge_attr,
                        torch.zeros(
                            neg_edge_index.size(1),
                            dtype=self.data[edge_type].edge_attr.dtype,
                            device=self.data[edge_type].edge_attr.device,
                        ),
                    ],
                )
                if hasattr(self.full_data[edge_type], "e_id"):
                    del self.full_data[edge_type].e_id  # Remove eid to avoid confusion
                self.edge_index_dict[edge_type] = torch.cat(
                    [
                        self.pos_idx_dict[edge_type],
                        torch.arange( # negative edges are assigned indices n_pos, ..., n_pos+n_neg-1
                            self.data[edge_type].edge_index.size(1),
                            (
                                self.data[edge_type].edge_index.size(1)
                                + neg_edge_index.size(1)
                            ),
                            dtype=torch.long,
                            device=self.pos_idx_dict[edge_type].device,
                        ),
                    ],
                    dim=0,
                )
            print(f"Negative sampling time: {time.time() - t1:.4f}s")
            print("Permuting edges...")
            self.lengths = [
                len(self.edge_index_dict[edge_type]) for edge_type in self.edge_types
            ] # include both positive and negative edges
            self.total_length = sum(self.lengths)
            perm_idx = torch.randperm(self.total_length)
            self.type_assignments = torch.cat(
                [torch.full((length,), i) for i, length in enumerate(self.lengths)]
            )[perm_idx]
            self.permuted_edge_idx = {
                k: v[torch.randperm(len(v))] for k, v in self.edge_index_dict.items()
            }
            self.all_edge_idx = torch.zeros(self.total_length, dtype=torch.long)
            for i, edge_type in enumerate(self.edge_types):
                self.all_edge_idx[self.type_assignments == i] = self.permuted_edge_idx[
                    edge_type
                ]
            print("Negative sampling and permutation done.")

    def __len__(self):
        """Total number of edges (positive + negative) across all edge types"""
        return self.total_length

    def __getitem__(self, idx):
        """
        Get a single edge sample (positive or negative) by index.

        Parameters
        ----------
        idx : int
            Index of the edge to get.

        Returns
        -------
        edge_type : EdgeType
            The edge type (e.g. `("cell", "expresses", "gene")`) of this sample.
        sample_idx : int
            Index from `full_data[edge_type].edge_index`.  
            If it is a negative edge index, then `sample_idx` >= `self.data[edge_type].edge_index.size(1)`
            and `edge_attr` is zero.
        full_data : HeteroData
            The graph containing all positive and (optionally) negative edges.
        """
        edge_type = self.edge_types[self.type_assignments[idx]]
        sample_idx = self.all_edge_idx[idx]
        res = (edge_type, sample_idx.item(), self.full_data)
        return res


def collate(batches):
    """
    Collate function: build a batched subgraph from edge-level samples.

    Parameters
    ----------
    batches : list[tuple]
        List of `(edge_type, edge_index, full_data)` samples.

    Returns
    -------
    torch_geometric.data.HeteroData
        A subgraph containing only the edges and nodes involved in this batch.
    """
    graph_batch = {}
    full_data = batches[0][2]

    for batch in batches:
        edge_type, edge_index, _ = batch
        if edge_type not in graph_batch:
            graph_batch[edge_type] = [edge_index]
        else:
            graph_batch[edge_type].append(edge_index)

    dat_return = RemoveIsolatedNodes()(
        full_data.edge_subgraph(
            {
                k: torch.tensor(
                    v,
                    dtype=torch.long,
                    device=full_data[k].edge_index.device,
                )
                for k, v in graph_batch.items()
            }
        )
    )

    return dat_return


class CustomMultiIndexDataset(Dataset):
    def __init__(
        self,
        pos_idx_dict: dict[EdgeType, torch.Tensor],
        data: torch_geometric.data.HeteroData,
        negative_sampling_fold: int = 1,
    ):
        assert len(pos_idx_dict) > 0, "pos_idx_dict must not be empty"
        self.pos_idx_dict = pos_idx_dict
        self.edge_types = list(pos_idx_dict.keys())
        self.lengths = [
            len(self.pos_idx_dict[edge_type]) for edge_type in self.edge_types
        ]
        self.total_length = sum(self.lengths)
        self.num_types = len(self.edge_types)
        self.negative_sampling_fold = negative_sampling_fold
        self.data = data
        self.edge_index_dict = self.pos_idx_dict
        self.lengths = [
            len(self.edge_index_dict[edge_type]) for edge_type in self.edge_types
        ]
        self.total_length = sum(self.lengths)
        perm_idx = torch.randperm(self.total_length)
        self.type_assignments = torch.cat(
            [torch.full((length,), i) for i, length in enumerate(self.lengths)]
        )[perm_idx]
        self.all_edge_idx = torch.zeros(self.total_length, dtype=torch.long)
        for i, edge_type in enumerate(self.edge_types):
            self.all_edge_idx[self.type_assignments == i] = pos_idx_dict[edge_type]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        edge_type = self.edge_types[self.type_assignments[idx]]
        sample_idx = self.all_edge_idx[idx]
        res = (
            edge_type,
            sample_idx.item(),
        )
        return res


def collate_graph(batches, data):
    """To be used with CustomMultiIndexDataset without negative sampling"""
    graph_batch = defaultdict(list)
    t0 = time.time()
    for edge_type, edge_index in batches:
        graph_batch[edge_type].append(edge_index)
    t1 = time.time()
    print(f"Collate time: {t1 - t0:.4f}s")
    dat = data.edge_subgraph(
        {
            k: torch.tensor(
                v,
                dtype=torch.long,
            )
            for k, v in graph_batch.items()
        }
    )
    t2 = time.time()
    print(f"edge subgraph time: {t2-t1:.4f}")

    dat_return = RemoveIsolatedNodes()(dat)
    t3 = time.time()
    print(f"remove time: {t3-t2:.4f}")
    dat_all_edges = data.subgraph(dat_return.n_id_dict)
    print(f"subgraph time: {time.time()- t3:.4f}s")
    return dat_return, dat_all_edges
