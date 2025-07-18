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
        super().__init__()
        print(f"@N LATENT DIMS:{n_latent_dims}")
        self.__mu_dict__ = torch.nn.ParameterDict(
            {
                node_type: torch.nn.Parameter(
                    data.x_dict[node_type]
                    if False  # data.x_dict[node_type].shape[1] == n_latent_dims
                    else torch.randn(data.x_dict[node_type].shape[0], n_latent_dims)
                )
                for node_type in data.node_types
            }
        )
        self.__logstd_dict__ = torch.nn.ParameterDict(
            {
                node_type: torch.nn.Parameter(
                    torch.zeros(data.x_dict[node_type].shape[0], n_latent_dims)
                )
                for node_type in data.node_types
            }
        )

    def encode(self, batch, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.forward(batch, *args, **kwargs)

    def forward(self, batch, *args, **kwargs):
        mu_dict = {
            node_type: self.__mu_dict__[node_type][batch[node_type].n_id, :]
            for node_type in batch.node_types
        }
        logstd_dict = {
            node_type: self.__logstd_dict__[node_type][batch[node_type].n_id, :]
            for node_type in batch.node_types
        }
        return mu_dict, logstd_dict
