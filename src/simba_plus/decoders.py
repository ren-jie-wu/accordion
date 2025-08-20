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
register_prob_model("Beta", pr.BetaDataDecoder)
register_prob_model("NegativeBinomial", pr.NegativeBinomialDataDecoder)


# ----------------------------------------------------------------------------------------------------


class RelationalEdgeDistributionDecoder(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        encoded_channels: int,
        projected_channels: Optional[int] = None,
        add_covariate: bool = False,
        device="cpu",
        project=True,
        edgetype_specific_bias: bool = True,
        edgetype_specific_scale: bool = True,
        edgetype_specific_std: bool = True,
    ) -> None:
        """Initialize the decoder with shared projection matrix per relation type.
        See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hgt_conv.html#HGTConv

        Args:
            data: HeteroData with node types
            encoded_channels: Number of dimensions of latent vector that will be decoded
            projected_channels: Number of dimensions of projected latent vector onto the relation-specific space
            add_covariate: add covariate to cell node
        """
        super().__init__()
        self.device = device
        if projected_channels is not None:
            projected_channels = encoded_channels
        self.edgetype_specific_bias = edgetype_specific_bias
        self.edgetype_specific_scale = edgetype_specific_scale
        self.edgetype_specific_std = edgetype_specific_std
        self.prob_dict = torch.nn.ModuleDict()
        if project:
            self.proj_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                if node_type == "cell" and add_covariate:
                    self.proj_dict[node_type] = Linear(
                        encoded_channels,  # + count_additional_latent_dims_for_covs(data),
                        projected_channels,
                    )
                elif node_type == "cell":
                    self.proj_dict[node_type] = Identity()
                else:
                    self.proj_dict[node_type] = torch.nn.Sequential(
                        Linear(
                            encoded_channels,
                            projected_channels,
                        ),
                        LeakyReLU(),
                    )
        # import pdb; pdb.set_trace()
        for edge_type in data.edge_types:
            if edge_type in data.edge_dist_dict.keys():
                self.prob_dict[",".join(edge_type)] = _DECODER_MAP[
                    data.edge_dist_dict[edge_type]
                ](
                    projected_channels
                )  # TODO: add attribute setting in graph construction step. What if the distribution decoder has external parameter to be specified?
                # ModuleDict only takes string as its key.
        self.add_covariate = add_covariate
        if hasattr(data["cell"], "cat_covs"):
            # data["cell"].cat_cov.shape == (n_cat_cov, n_cells)
            n_cat_covs = data["cell"].cat_cov.shape[0]
            self.cat_cov_dict = nn.ParameterDict(
                {
                    str(k): nn.Parameter(
                        torch.zeros(
                            (
                                len(torch.unique(data["cell"].cat_cov[k, :])),
                                projected_channels,
                            )
                        )
                    )
                    for k in range(n_cat_covs)
                }
            )

        if hasattr(data["cell"], "cont_covs"):
            # data["cell"].cont_cov.shape == (n_cont_cov, n_cells)
            n_cont_covs = data["cell"].cont_cov.shape[0]
            self.cat_cov_dict = nn.ParameterDict(
                {
                    str(k): nn.Parameter(torch.zeros((n_cont_covs, projected_channels)))
                    for k in range(n_cont_covs)
                }
            )

    def project(
        self, src_z: Tensor, dst_z: Tensor, src_type: NodeType, dst_type: NodeType
    ) -> Tuple[Tensor, Tensor]:
        # print(
        #     f"@DECODER PROJECTION: {self.proj_dict[dst_type].weight.dtype}{dst_z.dtype}"
        # )
        if hasattr(self, "proj_dict"):
            return (self.proj_dict[src_type](src_z), self.proj_dict[dst_type](dst_z))
        else:
            return src_z, dst_z

    def forward(
        self,
        batch,
        z_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        scale_dict=None,
        bias_dict=None,
        std_dict=None,
    ) -> Dict[EdgeType, Distribution]:
        """Decodes the latent variable per edge type"""
        out_dict = {}
        # import pdb; pdb.set_trace()
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_z = z_dict[src_type]
            dst_z = z_dict[dst_type]
            src_z = src_z[edge_index[0], :]
            dst_z = dst_z[edge_index[1], :]
            u, v = self.project(src_z, dst_z, src_type, dst_type)
            prob_decoder = self.prob_dict[",".join(edge_type)]
            if self.edgetype_specific_bias:
                src_bias_key = make_key(src_type, edge_type)
                dst_bias_key = make_key(dst_type, edge_type)
            else:
                src_bias_key = src_type
                dst_bias_key = dst_type
            if self.edgetype_specific_scale:
                src_scale_key = make_key(src_type, edge_type)
                dst_scale_key = make_key(dst_type, edge_type)
            else:
                src_scale_key = src_type
                dst_scale_key = dst_type
            if self.edgetype_specific_std:
                src_std_key = make_key(src_type, edge_type)
                dst_std_key = make_key(dst_type, edge_type)
            else:
                src_std_key = src_type
                dst_std_key = dst_type

            use_batch = False
            if hasattr(batch["cell"], "batch"):
                use_batch = True
                batch_feature = ["gene", "peak"]

            src_node_id = batch[src_type].n_id[edge_index[0]]
            if use_batch and src_type in batch_feature:
                if dst_type == "cell":
                    batches = batch[dst_type].batch[edge_index[1]].long()
                    src_scale = scale_dict[src_scale_key][batches, src_node_id]
                    src_bias = bias_dict[src_bias_key][batches, src_node_id]
                    src_std = std_dict[src_std_key][batches, src_node_id]
                else:
                    src_scale = scale_dict[src_scale_key][0, src_node_id]
                    src_bias = bias_dict[src_bias_key][0, src_node_id]
                    src_std = std_dict[src_std_key][0, src_node_id]
            else:
                src_scale = scale_dict[src_scale_key][src_node_id]
                src_bias = bias_dict[src_bias_key][src_node_id]
                src_std = std_dict[src_std_key][src_node_id]

            dst_node_id = batch[dst_type].n_id[edge_index[1]]
            if use_batch and dst_type in batch_feature:
                if src_type == "cell":
                    batches = batch["cell"].batch[edge_index[0]].long()
                    dst_scale = scale_dict[dst_scale_key][batches, dst_node_id]
                    dst_bias = bias_dict[dst_bias_key][batches, dst_node_id]
                    dst_std = std_dict[dst_std_key][batches, dst_node_id]
                else:
                    dst_scale = scale_dict[dst_scale_key][0, dst_node_id]
                    dst_bias = bias_dict[dst_bias_key][0, dst_node_id]
                    dst_std = std_dict[dst_std_key][0, dst_node_id]
            else:
                dst_scale = scale_dict[dst_scale_key][dst_node_id]
                dst_bias = bias_dict[dst_bias_key][dst_node_id]
                dst_std = std_dict[dst_std_key][dst_node_id]

            out_dict[edge_type] = prob_decoder.forward(
                u,
                v,
                None,
                None,
                src_scale=src_scale,
                src_bias=src_bias,
                src_std=src_std,
                dst_scale=dst_scale,
                dst_bias=dst_bias,
                dst_std=dst_std,
            )
        return out_dict
