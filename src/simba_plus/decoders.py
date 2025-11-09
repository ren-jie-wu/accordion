from typing import Optional, Mapping, Dict, Union, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Identity, LeakyReLU
from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch.distributions import Distribution
from simba_plus._utils import make_key
import simba_plus.prob_decoders as pr

# ---------------------------------- Adoptedd from SCGLUE ----------------------------------

_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", pr.NormalDataDecoder)
register_prob_model("Poisson", pr.PoissonDataDecoder)
register_prob_model("Bernoulli", pr.BernoulliDataDecoder)
register_prob_model("NegativeBinomial", pr.NegativeBinomialDataDecoder)
register_prob_model("Gamma", pr.GammaDataDecoder)


# ----------------------------------------------------------------------------------------------------


class RelationalEdgeDistributionDecoder(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        device="cpu",
    ) -> None:
        """Initialize the decoder with shared projection matrix per relation type.
        See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hgt_conv.html#HGTConv

        Args:
            data: HeteroData with node types
            encoded_channels: Number of dimensions of latent vector that will be decoded
            add_covariate: add covariate to cell node
        """
        super().__init__()
        self.device = device
        self.prob_dict = torch.nn.ModuleDict()
        for edge_type in data.edge_types:
            if edge_type in data.edge_dist_dict.keys():
                self.prob_dict[",".join(edge_type)] = _DECODER_MAP[
                    data.edge_dist_dict[edge_type]
                ]()

    def forward(
        self,
        batch,
        z_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        src_logscale_dict,
        src_bias_dict,
        src_std_dict,
        dst_logscale_dict,
        dst_bias_dict,
        dst_std_dict,
    ) -> Dict[EdgeType, Distribution]:
        """Decodes the latent variable per edge type"""
        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            prob_decoder = self.prob_dict[",".join(edge_type)]
            src_z = z_dict[src_type]
            dst_z = z_dict[dst_type]

            u = src_z[edge_index[0], :]
            v = dst_z[edge_index[1], :]

            out_dict[edge_type] = prob_decoder.forward(
                u=u,
                v=v,
                src_logscale=src_logscale_dict[edge_type],
                src_bias=src_bias_dict[edge_type],
                src_std=src_std_dict[edge_type],
                dst_logscale=dst_logscale_dict[edge_type],
                dst_bias=dst_bias_dict[edge_type],
                dst_std=dst_std_dict[edge_type],
            )
        return out_dict
