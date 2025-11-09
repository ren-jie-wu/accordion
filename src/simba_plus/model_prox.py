from typing import Optional, Dict, Union, Tuple
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from torch.distributions import Distribution
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
from torch_geometric.transforms.to_device import ToDevice
from simba_plus.encoders import TransEncoder
from simba_plus.constants import MIN_LOGSTD, MAX_LOGSTD
from simba_plus.utils import negative_sampling

# from torch_geometric.utils import negative_sampling
from simba_plus.losses import bernoulli_kl_loss
from simba_plus.decoders import RelationalEdgeDistributionDecoder
import time
from simba_plus._utils import (
    add_cov_to_latent,
    make_key,
    update_lr,
)

from simba_plus.evaluation_utils import (
    compute_reconstruction_gene_metrics,
    compute_classification_metrics,
    plot_nll_distributions,
)


class AuxParams(nn.Module):
    def __init__(self, data: HeteroData, edgetype_specific: bool = True) -> None:
        super().__init__()
        # Batch correction for RNA-seq data
        self.bias_dict = nn.ParameterDict()
        self.logscale_dict = nn.ParameterDict()
        self.std_dict = nn.ParameterDict()
        self.use_batch = False
        self.edgetype_specific = edgetype_specific
        if (
            hasattr(data["cell"], "batch")
            and torch.unique(data["cell"].batch).size(0) > 1
        ):
            self.use_batch = True
        self.bias_logstd_dict = nn.ParameterDict()
        self.logscale_logstd_dict = nn.ParameterDict()
        self.std_logstd_dict = nn.ParameterDict()
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            if edgetype_specific:
                src_key = make_key(src, edge_type)
                dst_key = make_key(dst, edge_type)
            else:
                src_key = src
                dst_key = dst
            self.logscale_dict[src_key] = nn.Parameter(
                torch.zeros(
                    data[src].num_nodes,
                )
            )
            self.bias_dict[src_key] = nn.Parameter(
                torch.zeros(
                    data[src].num_nodes,
                )
            )
            self.std_dict[src_key] = nn.Parameter(
                torch.zeros(
                    data[src].num_nodes,
                )
            )

            if self.use_batch:
                n_batches = len(data["cell"].batch.unique())
                self.logscale_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                        # device=self.device,
                    )
                )
                self.bias_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                        # device=self.device,
                    )
                )
                self.std_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                        # device=self.device,
                    )
                )

                self.logscale_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                    ),
                )
                self.bias_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                        # device=self.device,
                    )
                )
                self.std_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        (n_batches, data[dst].num_nodes),
                        # device=self.device,
                    )
                )
            else:
                self.logscale_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                    )
                )
                self.bias_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                    )
                )

                self.std_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                    )
                )
                self.logscale_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                    ),
                )
                self.bias_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                        # device=self.device,
                    )
                )
                self.std_logstd_dict[dst_key] = nn.Parameter(
                    torch.zeros(
                        data[dst].num_nodes,
                        # device=self.device,
                    )
                )

    def _kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
            weight (Tensor, optional): weights of each position in the tensor.
        """

        def weighted_sum(x, w):
            # if not w.any():
            #     return 0
            if w is None:
                return x.sum()
            return (x * w).sum()  # / w.long().sum()

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        kls = 1 + 2 * logstd - mu**2 - logstd.exp() ** 2
        return -0.5 * weighted_sum(kls, weight)

    def batched(self, param, param_logstd):
        return_param = param
        if self.training:
            return_param = return_param + torch.randn_like(param_logstd) * torch.exp(
                param_logstd
            )
        return return_param

    def forward(self, batch, edge_index_dict):
        src_logscale_dict, src_bias_dict, src_std_dict = {}, {}, {}
        dst_logscale_dict, dst_bias_dict, dst_std_dict = {}, {}, {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            (
                src_key,
                dst_key,
            ) = self.get_keys(src_type, dst_type, edge_type)

            src_node_id = batch[src_type].n_id[edge_index[0]]
            src_logscale_dict[edge_type] = self.logscale_dict[src_key][src_node_id]
            src_bias_dict[edge_type] = self.bias_dict[src_key][src_node_id]
            src_std_dict[edge_type] = self.std_dict[src_key][src_node_id]

            dst_node_id = batch[dst_type].n_id[edge_index[1]]
            if self.use_batch:
                batches = batch["cell"].batch[edge_index[0]].long()
                dst_logscale_dict[edge_type] = self.batched(
                    self.logscale_dict[dst_key],
                    self.scale_logstd_dict[dst_key],
                )[batches, dst_node_id]
                dst_bias_dict[edge_type] = self.batched(
                    self.bias_dict[dst_key], self.bias_logstd_dict[dst_key]
                )[batches, dst_node_id]
                dst_std_dict[edge_type] = self.batched(
                    self.std_dict[dst_key], self.std_logstd_dict[dst_key]
                )[batches, dst_node_id]
            else:
                dst_logscale_dict[edge_type] = self.batched(
                    self.logscale_dict[dst_key],
                    self.logscale_logstd_dict[dst_key],
                )[dst_node_id]
                dst_bias_dict[edge_type] = self.batched(
                    self.bias_dict[dst_key], self.bias_logstd_dict[dst_key]
                )[dst_node_id]
                dst_std_dict[edge_type] = self.batched(
                    self.std_dict[dst_key], self.std_logstd_dict[dst_key]
                )[dst_node_id]
        return {
            "src_logscale_dict": src_logscale_dict,
            "src_bias_dict": src_bias_dict,
            "src_std_dict": src_std_dict,
            "dst_logscale_dict": dst_logscale_dict,
            "dst_bias_dict": dst_bias_dict,
            "dst_std_dict": dst_std_dict,
        }

    def get_keys(self, src_type, dst_type, edge_type):
        if self.edgetype_specific:
            src_key = make_key(src_type, edge_type)
            dst_key = make_key(dst_type, edge_type)
        else:
            src_key = src_type
            dst_key = dst_type
        return (
            src_key,
            dst_key,
        )

    def kl_div_loss(self, batch, node_weights_dict):
        l = torch.tensor(0.0, device=next(self.parameters()).device)
        for edge_type in batch.edge_types:
            edge_index = batch[edge_type].edge_index
            src_type, _, dst_type = edge_type
            (
                src_key,
                dst_key,
            ) = self.get_keys(src_type, dst_type, edge_type)
            if self.use_batch:
                batches = batch["cell"].batch[edge_index[0]].long()
                dst_node_id = batch[dst_type].n_id[edge_index[1]]
                # obtain unique index

                # sub_idx = torch.where(batches != 0)[0]
                # if len(sub_idx) == 0:
                #     continue
                # batches = batches[sub_idx]
                # dst_node_id = dst_node_id[sub_idx]
                unique_index = torch.unique(torch.stack([batches, dst_node_id]), dim=0)

                batches = unique_index[0, :]  # because batch 0 is baseline
                dst_node_id = unique_index[1, :]
                weight = node_weights_dict[dst_type][dst_node_id]
                m_logscale = self.logscale_dict[dst_key][batches, dst_node_id]
                logstd_logscale = self.logscale_logstd_dict[dst_key][
                    batches, dst_node_id
                ]
                l += self._kl_loss(m_logscale, logstd_logscale, weight)
                m_bias = self.bias_dict[dst_key][batches, dst_node_id]
                logstd_bias = self.bias_logstd_dict[dst_key][batches, dst_node_id]
                l += self._kl_loss(m_bias, logstd_bias, weight)
                m_std = self.std_dict[dst_key][batches, dst_node_id]
                logstd_std = self.std_logstd_dict[dst_key][batches, dst_node_id]
                l += self._kl_loss(m_std, logstd_std, weight)
        return l


class LightningProxModel(L.LightningModule):
    def __init__(
        self,
        data: HeteroData,
        encoder_class: torch.nn.Module = TransEncoder,
        n_latent_dims: int = 50,
        decoder_class: torch.nn.Module = RelationalEdgeDistributionDecoder,
        device="cpu",
        num_neg_samples_fold: int = 1,
        edgetype_specific: bool = True,
        edge_types: Optional[Tuple[str]] = None,
        hsic: Optional[nn.Module] = None,
        herit_loss: Optional[nn.Module] = None,
        herit_loss_lam: float = 1,
        n_no_kl: int = 10,
        n_kl_warmup: int = 1,
        nll_scale: float = 1.0,
        val_nll_scale: float = 1.0,
        learning_rate=1e-2,
        node_weights_dict=None,
        nonneg=False,
        reweight_rarecell: bool = False,
        reweight_rarecell_neighbors: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.nonneg = nonneg
        self.data = data
        self.learning_rate = learning_rate
        self.encoder = encoder_class(
            data,
            n_latent_dims,
        )
        self.decoder = decoder_class(
            data,
            device=device,
        )
        self.hsic = hsic
        if self.hsic is not None:
            self.hsic_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=hsic.lam
            )
        self.n_no_kl = n_no_kl
        self.n_kl_warmup = n_kl_warmup
        self.nll_scale = nll_scale
        self.val_nll_scale = val_nll_scale
        self.num_nodes_dict = {
            node_type: x.shape[0] for (node_type, x) in data.x_dict.items()
        }
        self.reweight_rarecell = reweight_rarecell
        if self.reweight_rarecell:
            self.cell_weights = torch.ones(data["cell"].num_nodes, device=device)
            if reweight_rarecell_neighbors is None:
                reweight_rarecell_neighbors = max(20, int(data["cell"].num_nodes / 100))
            self.reweight_rarecell_neighbors = reweight_rarecell_neighbors
        if edge_types is None:
            self.edge_types = data.edge_types
        else:
            self.edge_types = edge_types
        self.edgetype_loss_weight_dict = {
            edgetype: data.num_edges
            / len(data.edge_types)  # mean $ edges
            / len(data[edgetype].edge_index[0])
            for edgetype in self.edge_types
        }

        self.node_weights_dict = node_weights_dict
        self.herit_loss = herit_loss
        self.herit_loss_lam = herit_loss_lam

        self.num_neg_samples_fold = num_neg_samples_fold
        self.validation_step_outputs = []
        self.aux_params = AuxParams(data, edgetype_specific=edgetype_specific)

    def on_train_start(self):
        self.node_weights_dict = {
            k: v.to(self.device) for k, v in self.node_weights_dict.items()
        }
        if self.hsic is not None:
            update_lr(self.hsic_optimizer, self.hsic.lam * self.learning_rate)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def reparametrize(
        self,
        mu_dict: Dict[NodeType, Tensor],
        logstd_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        """Generate random z from mu, logstd"""
        out_dict = {}
        assert mu_dict.keys() == logstd_dict.keys()
        for node_type in mu_dict.keys():
            if self.training:
                out_dict[node_type] = mu_dict[node_type] + torch.randn_like(
                    logstd_dict[node_type]
                ) * torch.exp(logstd_dict[node_type])
            else:
                out_dict[node_type] = mu_dict[node_type]
        return out_dict

    def relational_recon_loss(
        self,
        batch: HeteroData,
        z_dict: Dict[NodeType, Tensor],
        pos_edge_index_dict: Dict[EdgeType, Tensor],
        pos_edge_weight_dict: Dict[EdgeType, Tensor],
        neg_edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
        neg_sample=True,
        plot=False,
        get_metric=False,
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict]:
        """Calculate reconstruction loss by maximizing log_prob of observing edge weight in pos_edge_index_dict and 0 weight in neg_edge_index_dict

        Args
        z_dict: encoded vector
        pos_edge_index_dict: Dictionary of Tensors with shape (2, n_pos_edges)
        pos_edge_weight_dict: Dictionary of Tensors with shape (n_pos_edges,) encoding the weight of each edges.
        neg_edge_index_dict: Dictionary of Tensors with shape (2, n_neg_edges) to be used as negative edges
        num_neg_samples: If neg_edge_index is None and
            num_neg_samples is None, This number of negative edges are sampled. Otherwise, the same number as the positive edges are sample.
        """
        pos_dist_dict: Dict[EdgeType, Distribution] = self.decoder(
            batch,
            z_dict,
            pos_edge_index_dict,
            **self.aux_params(batch, pos_edge_index_dict),
        )

        if neg_sample:
            if neg_edge_index_dict is None:
                neg_edge_index_dict = {}
                for edge_type, pos_edge_index in pos_edge_index_dict.items():
                    if len(pos_edge_index) == 0:
                        continue
                    src_type, _, dst_type = edge_type
                    (
                        neg_src_idx,
                        neg_dst_idx,
                    ) = negative_sampling(
                        batch.edge_index_dict[edge_type],
                        num_nodes=(
                            batch[src_type].num_nodes,
                            batch[dst_type].num_nodes,
                        ),
                        num_neg_samples_fold=self.num_neg_samples_fold,
                    )
                    neg_edge_index_dict[edge_type] = torch.stack(
                        [neg_src_idx, neg_dst_idx]
                    )

            neg_dist_dict: Dict[EdgeType, Tensor] = self.decoder(
                batch,
                z_dict,
                neg_edge_index_dict,
                **self.aux_params(batch, neg_edge_index_dict),
            )

        loss_dict = {}
        metric_dict = {}
        for edge_type, pos_dist in pos_dist_dict.items():

            src_type, _, dst_type = edge_type
            pos_edge_weights = pos_edge_weight_dict[edge_type]
            pos_loss = -pos_dist.log_prob(pos_edge_weights).sum()

            if neg_sample:
                neg_log_probs = -neg_dist_dict[edge_type].log_prob(
                    torch.tensor(0.0, device=batch[src_type].x.device)
                )
                neg_loss = neg_log_probs.sum()
                loss_dict[edge_type] = pos_loss + neg_loss
            else:
                loss_dict[edge_type] = pos_loss

        return loss_dict, neg_edge_index_dict, metric_dict

    def _kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
            weight (Tensor, optional): weights of each position in the tensor.
        """

        def weighted_sum(x, w):
            # if not w.any():
            #     return 0
            if w is None:
                return x.sum()
            return (x * w).sum()  # / w.long().sum()

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        kls = torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1)
        return -0.5 * weighted_sum(kls, weight)

    def relational_kl_divergence(
        self,
        mu_dict: Dict[NodeType, Tensor],
        logstd_dict: Dict[NodeType, Tensor],
        node_index_dict: Dict[NodeType, Tensor],
        node_weights_dict: Optional[Dict[NodeType, Tensor]] = None,
    ) -> Dict[EdgeType, Tensor]:
        """Sums KL divergence across relations.

        Args
            z_dict: encoded vector

        """
        loss_dict = {}
        for node_type in mu_dict.keys():
            if node_weights_dict is None:
                weight = torch.ones(mu_dict[node_type].shape[0], device=self.device)
            else:
                weight = node_weights_dict[node_type][node_index_dict[node_type]]
            loss_dict[node_type] = self._kl_loss(
                mu_dict[node_type], logstd_dict[node_type], weight
            )
        return loss_dict

    def kl_div_loss(
        self,
        batch,
        mu_dict: Dict[NodeType, Tensor],
        logstd_dict: Dict[NodeType, Tensor],
        node_index_dict: Dict[NodeType, Tensor],
        node_weights_dict: Optional[Dict[NodeType, Tensor]] = None,
        nodetype_loss_weight_dict: Optional[Dict[EdgeType, float]] = None,
    ) -> Tensor:
        """
        Args
        mu_dict: mu of encoded batch
        logstd_dict: logstd of encoded batch
        node_index_dict: node index of the batch
        node_counts_dict: For entire dataset, counts how many times each node is used for KL div calculation. If None, assume no node has been used for the calculation.

        """
        kl_div_dict: Dict[NodeType, Tensor] = self.relational_kl_divergence(
            mu_dict, logstd_dict, node_index_dict, node_weights_dict=node_weights_dict
        )
        l = torch.tensor(0.0, device=self.device)
        for node_type, kl_div in kl_div_dict.items():
            if nodetype_loss_weight_dict:
                nodetype_weight = nodetype_loss_weight_dict[node_type]
            else:
                nodetype_weight = 1.0
            l += kl_div * nodetype_weight
        l += self.aux_params.kl_div_loss(batch, node_weights_dict)
        return l

    def nll_loss(
        self,
        batch: HeteroData,
        z_dict: Dict[NodeType, Tensor],
        pos_edge_index_dict: Dict[EdgeType, Tensor],
        pos_edge_weight_dict: Dict[EdgeType, Tensor],
        neg_edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
        # num_neg_samples_fold: Optional[int] = 1,
        edgetype_loss_weight_dict: Optional[Dict[EdgeType, float]] = None,
        neg_sample=False,
    ):
        nll_dict, neg_edge_index_dict, metric_dict = self.relational_recon_loss(
            batch=batch,
            z_dict=z_dict,
            pos_edge_index_dict=pos_edge_index_dict,
            pos_edge_weight_dict=pos_edge_weight_dict,
            neg_edge_index_dict=neg_edge_index_dict,
            neg_sample=neg_sample,
        )
        l = torch.tensor(0.0, device=self.device)
        for edge_type, nll in nll_dict.items():
            self.log(
                f"{'train' if self.training else 'val'}_nll_loss/{edge_type}",
                nll,
                batch_size=batch[edge_type].edge_index.shape[1],
                on_step=True,
                on_epoch=True,
            )
            if edgetype_loss_weight_dict:
                edgetype_weight = edgetype_loss_weight_dict[edge_type]
            else:
                edgetype_weight = torch.tensor(1.0, device=self.device)
            l += nll * edgetype_weight
        return l, neg_edge_index_dict, metric_dict

    def training_step(self, batch, batch_idx):
        t0 = time.time()
        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)
        t0 = time.time()
        batch_nll_loss, neg_edge_index_dict, _ = self.nll_loss(
            batch=batch,
            z_dict=z_dict,
            pos_edge_index_dict=batch.edge_index_dict,
            pos_edge_weight_dict=batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
        )
        t1 = time.time()
        self.log("time:nll_loss", t1 - t0, on_step=True, on_epoch=False)
        if self.current_epoch >= self.n_no_kl:
            t0 = time.time()
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
            t1 = time.time()

            batch_kl_div_loss *= min(
                self.current_epoch + 1, self.n_kl_warmup - self.n_no_kl
            ) / (self.n_kl_warmup - self.n_no_kl)
        else:
            batch_kl_div_loss = 0.0
        if self.herit_loss is not None:
            t0 = time.time()
            if "peak" in batch.node_types:
                pid = batch["peak"].n_id.cpu()
                herit_loss_value = self.herit_loss_lam * self.herit_loss(
                    mu_dict["peak"],
                    pid,
                )
            else:
                herit_loss_value = torch.tensor(0.0)
            t1 = time.time()
            self.log("time:herit_loss", t1 - t0, on_step=True, on_epoch=False)
            self.log(
                "herit_loss",
                herit_loss_value,
                batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
                on_step=True,
                on_epoch=True,
            )
        else:
            herit_loss_value = torch.tensor(0.0)
        self.log(
            "nll_loss",
            batch_nll_loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "kl_div_loss",
            batch_kl_div_loss / self.nll_scale,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=True,
            on_epoch=True,
        )
        loss = batch_nll_loss  # + batch_kl_div_loss / self.nll_scale + #herit_loss_value  #
        self.log(
            "loss",
            loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)
        batch_nll_loss, neg_edge_index_dict, metric_dict = self.nll_loss(
            batch,
            z_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
            neg_sample=False,
        )

        if self.current_epoch >= self.n_no_kl:
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
        else:
            batch_kl_div_loss = 0.0
        self.log(
            "val_nll_loss",
            batch_nll_loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        if self.herit_loss is not None:
            if "peak" in batch.node_types:
                pid = batch["peak"].n_id.cpu()
                herit_loss_value = self.herit_loss_lam * self.herit_loss(
                    mu_dict["peak"],
                    pid,
                )
            else:
                herit_loss_value = torch.tensor(0.0)
            self.log(
                "val_herit_loss",
                herit_loss_value,
                batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
                on_step=False,
                on_epoch=True,
            )
        else:
            herit_loss_value = torch.tensor(0.0)
        loss = batch_nll_loss
        # + batch_kl_div_loss / self.val_nll_scale + herit_loss_value
        # )
        self.log(
            "val_nll_loss_monitored",
            batch_nll_loss
            + (torch.inf if self.current_epoch < self.n_kl_warmup * 2 else 0),
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val_kl_div_loss",
            batch_kl_div_loss / self.val_nll_scale,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "valid_loss",
            loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )

        self.validation_step_outputs.append(loss)
        return loss

    def mean_cell_neighbor_distance(self, k=50):
        """
        Calculates the mean distance between each cell node and its k nearest neighbors.
        Returns a tensor of mean distances for each cell.
        """
        cell_mu = self.encoder.__mu_dict__["cell"].detach()  # (num_cells, latent_dim)
        # Compute pairwise Euclidean distances
        dist_matrix = torch.cdist(cell_mu, cell_mu, p=2)  # (num_cells, num_cells)
        # Exclude self-distance by setting diagonal to infinity
        dist_matrix.fill_diagonal_(float("inf"))
        # Find k nearest neighbors for each cell
        knn_distances, _ = torch.topk(dist_matrix, k, largest=False, dim=1)
        # Mean distance to k nearest neighbors for each cell
        mean_distances = knn_distances.mean(dim=1)
        weights = mean_distances / mean_distances.mean()
        return weights  # shape: (num_cells,)

    def on_train_epoch_start(self):
        self.validation_step_outputs = []
        if self.trainer.is_last_batch and self.hsic is not None:
            hsic_loss = self.hsic.custom_train(
                self.encoder.__mu_dict__["cell"], optimizer=self.hsic_optimizer
            )
            self.log("hsic_loss", hsic_loss, on_epoch=True)
        if self.hsic is not None:
            self.log(
                "hsic_lr",
                self.hsic_optimizer.param_groups[0]["lr"],
                on_epoch=True,
            )
        if self.reweight_rarecell:
            self.cell_weights = self.mean_cell_neighbor_distance(
                self.reweight_rarecell_neighbors
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=3,
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_nll_loss",
                "strict": False,
            }
        ]

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
        if self.hsic is not None:
            update_lr(
                self.hsic_optimizer,
                self.hsic.lam
                * self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[
                    0
                ]["lr"],
            )

    def on_save_checkpoint(self, checkpoint):
        # Remove the argument from the checkpoint before it is saved
        checkpoint.pop("train_data_dict", None)
