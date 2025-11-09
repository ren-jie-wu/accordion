import torch
from torch.utils.data import Dataset
from torch_geometric.typing import EdgeType
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes


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


class CustomMultiIndexDataset(Dataset):
    def __init__(
        self,
        edge_attr_types: list[EdgeType],
        indexes: list[torch.Tensor],
    ):
        assert len(edge_attr_types) == len(
            indexes
        ), "edge_attr_types and indexes must have the same length"
        self.edge_attr_types = edge_attr_types
        self.indexes = indexes
        self.lengths = [len(index) for index in indexes]
        self.total_length = sum(self.lengths)
        self.num_types = len(self.edge_attr_types)
        perm_idx = torch.randperm(self.total_length)
        self.type_assignments = torch.cat(
            [torch.full((length,), i) for i, length in enumerate(self.lengths)]
        )[perm_idx]
        self.type_idxs = torch.zeros(len(self.lengths), dtype=torch.long)
        self.setup_count = 0

    def setup(self):
        print(f"setup called {self.setup_count + 1} times")
        self.setup_count += 1
        perm_idx = torch.randperm(
            self.total_length, generator=torch.Generator().manual_seed(self.setup_count)
        )
        self.type_assignments = torch.cat(
            [torch.full((length,), i) for i, length in enumerate(self.lengths)]
        )[perm_idx]
        self.type_idxs = torch.zeros(len(self.lengths), dtype=torch.long)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        type_idx = self.type_assignments[idx]
        sample_idx = self.type_idxs[type_idx]
        res = self.edge_attr_types[type_idx], self.indexes[type_idx][sample_idx]
        self.type_idxs[type_idx] += 1
        return res


def collate(idx, data):
    batch = {}
    for d in idx:
        if d[0] not in batch:
            batch[d[0]] = [d[1]]
        else:
            batch[d[0]].append(d[1])
    return RemoveIsolatedNodes()(
        data.edge_subgraph({k: torch.tensor(v) for k, v in batch.items()})
    )
