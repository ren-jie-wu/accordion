import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class TransEncoder(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        n_latent_dims: int,
        *args,
        **kwargs,
    ):
        """
        Initialize parameter dicts mu and logstd. The keys are the node types and the values are num_nodes_per_node_type x n_latent_dims tensors.
        """
        super().__init__()
        # better initialization: create empty parameters and use a controlled initializer
        self.__mu_dict__ = torch.nn.ParameterDict()
        for node_type in data.node_types:
            num_nodes = data.n_id_dict[node_type].shape[0]
            p = torch.nn.Parameter(torch.empty(num_nodes, n_latent_dims))
            # stable small normal init (you can swap to xavier_uniform_ if preferred)
            torch.nn.init.normal_(p, mean=0.0, std=1)
            self.__mu_dict__[node_type] = p
        self.__logstd_dict__ = torch.nn.ParameterDict(
            {
                node_type: torch.nn.Parameter(
                    torch.zeros(data.n_id_dict[node_type].shape[0], n_latent_dims)
                )
                for node_type in data.node_types
            }
        )

    def encode(self, batch, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.forward(batch, *args, **kwargs)

    def forward(self, batch, *args, **kwargs):
        """Returns the mu and logstd dicts for the given batch."""
        mu_dict = {
            node_type: self.__mu_dict__[node_type][batch[node_type].n_id, :]
            for node_type in batch.node_types
        }
        logstd_dict = {
            node_type: self.__logstd_dict__[node_type][batch[node_type].n_id, :]
            for node_type in batch.node_types
        }
        return mu_dict, logstd_dict
