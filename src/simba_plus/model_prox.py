from typing import Optional, Dict, Union, Tuple
import math
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
from torch_geometric.utils import negative_sampling
from simba_plus.encoders import TransEncoder
from simba_plus.constants import MIN_LOGSTD, MAX_LOGSTD

# from simba_plus.utils import negative_sampling

# from torch_geometric.utils import negative_sampling
from simba_plus.losses import bernoulli_kl_loss
from simba_plus.decoders import RelationalEdgeDistributionDecoder
import time
from simba_plus._utils import (
    add_cov_to_latent,
    make_key,
    update_lr,
)
from simba_plus.negative_sampling import negative_sampling_same_sample_bipartite
from simba_plus.loss.gene_alignment import TwoSampleGenePairs, build_two_sample_gene_pairs, gene_alignment_msd_loss


class AuxParams(nn.Module):
    def __init__(self, data: HeteroData, edgetype_specific: bool = True) -> None:
        """
        Prepare parameter dictionaries with batch correction for RNA-seq data
        
        Args:
            data (HeteroData): The data object containing the graph data.
            edgetype_specific (bool): Whether to use different parameters for different edge types.
        
        Notes:
            An example keylist for a parameter dictionary when edgetype_specific is True is:
            [
                "cell__cell_expresses_gene",      # value shape: (num_cells,)
                "cell__cell_has_accessible_peak", # value shape: (num_cells,)
                "gene__cell_expresses_gene",      # value shape: (num_genes, num_batches) if use_batch else (num_genes,)
                "peak__cell_has_accessible_peak"  # value shape: (num_peaks, num_batches) if use_batch else (num_peaks,)
            ]
            But for *_logstd_dict, there is only destination node type in the key.
            [
                "gene__cell_expresses_gene",
                "peak__cell_has_accessible_peak"
            ]
        """
        super().__init__()
        # Batch correction for RNA-seq data
        self.bias_dict = nn.ParameterDict()
        self.logscale_dict = nn.ParameterDict()
        self.std_dict = nn.ParameterDict()

        self.use_batch = False
        self.edgetype_specific = edgetype_specific
        # if there are more than one batch, turn on use_batch
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
            # if edgetype_specific is True, same node type but with different edge type will have different sets of parameters
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

            # if use_batch, same dst node & edge type but with different batch will have different sets of parameters
            if self.use_batch:
                # number of batches
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
            if w is None:
                return x.sum()
            return (x * w).sum()

        kls = 1 + 2 * logstd - mu**2 - logstd.exp() ** 2
        return -0.5 * weighted_sum(kls, weight)

    def batched(self, param, param_logstd):
        if self.training and self.use_batch:
            return_param = torch.cat(
                [
                    param[[0], :],
                    param[1:, :]
                    + param[[0], :]
                    + torch.randn_like(param_logstd[1:, :])
                    * torch.exp(param_logstd[1:, :]),
                ],
                axis=0,
            )
            return return_param
        return param

    def regularization_loss(self, batch):
        l = torch.tensor(0.0, device=next(self.parameters()).device)
        for edge_type in batch.edge_types:
            src_type, _, dst_type = edge_type

            (
                src_key,
                dst_key,
            ) = self.get_keys(src_type, dst_type, edge_type)

            src_node_id = batch[src_type].n_id
            l += (self.logscale_dict[src_key][src_node_id]).pow(2).sum()
            l += (self.bias_dict[src_key][src_node_id]).pow(2).sum()
            l += (self.std_dict[src_key][src_node_id]).pow(2).sum()

            if self.use_batch:
                dst_node_id = batch[dst_type].n_id
                l += (self.logscale_dict[dst_key][:, dst_node_id]).pow(2).sum()
                l += (self.bias_dict[dst_key][:, dst_node_id]).pow(2).sum()
                l += (self.std_dict[dst_key][:, dst_node_id]).pow(2).sum()
            else:
                dst_node_id = batch[dst_type].n_id
                l += (self.logscale_dict[dst_key][dst_node_id]).pow(2).sum()
                l += (self.bias_dict[dst_key][dst_node_id]).pow(2).sum()
                l += (self.std_dict[dst_key][dst_node_id]).pow(2).sum()
        return l

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
                    self.logscale_logstd_dict[dst_key],
                )[batches, dst_node_id]
                dst_bias_dict[edge_type] = self.batched(
                    self.bias_dict[dst_key], self.bias_logstd_dict[dst_key]
                )[batches, dst_node_id]
                dst_std_dict[edge_type] = self.batched(
                    self.std_dict[dst_key], self.std_logstd_dict[dst_key]
                )[batches, dst_node_id]
            else:
                dst_logscale_dict[edge_type] = self.logscale_dict[dst_key][dst_node_id]
                dst_bias_dict[edge_type] = self.bias_dict[dst_key][dst_node_id]
                dst_std_dict[edge_type] = self.std_dict[dst_key][dst_node_id]
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
            src_type, _, dst_type = edge_type
            (
                src_key,
                dst_key,
            ) = self.get_keys(src_type, dst_type, edge_type)
            if self.use_batch:
                dst_node_id = batch[dst_type].n_id
                weight = node_weights_dict[dst_type][dst_node_id]
                m_logscale = self.logscale_dict[dst_key][1:, dst_node_id]
                logstd_logscale = self.logscale_logstd_dict[dst_key][1:, dst_node_id]
                l += self._kl_loss(m_logscale, logstd_logscale, weight)
                m_bias = self.bias_dict[dst_key][1:, dst_node_id]
                logstd_bias = self.bias_logstd_dict[dst_key][1:, dst_node_id]
                l += self._kl_loss(m_bias, logstd_bias, weight)
                m_std = self.std_dict[dst_key][1:, dst_node_id]
                logstd_std = self.std_logstd_dict[dst_key][1:, dst_node_id]
                l += self._kl_loss(m_std, logstd_std, weight)
        return l


class LightningProxModel(L.LightningModule):
    def __init__(
        self,
        data: HeteroData,
        logger=None,
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
        n_no_kl: int = 30,
        n_kl_warmup: int = 50,
        nll_scale: float = 1.0,
        val_nll_scale: float = 1.0,
        learning_rate=1e-2,
        node_weights_dict=None,
        nonneg=False, #TODO: not used
        reweight_rarecell: bool = False,
        reweight_rarecell_neighbors: Optional[int] = None,
        verbose: bool = False,
        batch_negative: bool = False, # make the default same as in the CLI
        monitor_key: str = None,

        lambda_gene_align: float = 0.0,
        gene_align_n_no: int = 0,
        gene_align_n_warmup: int = 10,
        lambda_ot: float = 0.0,
        ot_n_no: int = 15,
        ot_n_warmup: int = 30,
        ot_k: int = 256,
        ot_eps: float = 0.05,
        ot_iter: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.nonneg = nonneg
        self.data = data
        self.logger2 = logger
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
        self.num_nodes_dict = data.num_nodes_dict
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
        self.verbose = verbose
        self.batch_negative = batch_negative

        self.monitor_key = monitor_key

        self.lambda_gene_align = lambda_gene_align
        self.gene_align_n_no = gene_align_n_no
        self.gene_align_n_warmup = gene_align_n_warmup
        self.lambda_ot = lambda_ot
        self.ot_n_no = ot_n_no
        self.ot_n_warmup = ot_n_warmup
        self.ot_k = ot_k
        self.ot_eps = ot_eps
        self.ot_iter = ot_iter

        self._register_gene_pairs()
    
    def _register_gene_pairs(self):
        self._gene_pairs: TwoSampleGenePairs | None = None
        if self.lambda_gene_align > 0:
            if not (hasattr(self.data["gene"], "sample") and hasattr(self.data["gene"], "gene_id")):
                raise ValueError(
                    "lambda_gene_align > 0 requires data['gene'].sample and data['gene'].gene_id "
                    "(constructed in make_sc_HetData_multi_rna)."
                )
            
            pairs = build_two_sample_gene_pairs(
                gene_sample=self.data["gene"].sample,
                gene_id=self.data["gene"].gene_id,
            )
            
            self.register_buffer("gene_align_idx0", pairs.idx0, persistent=True)
            self.register_buffer("gene_align_idx1", pairs.idx1, persistent=True)
            self._gene_pairs = TwoSampleGenePairs(idx0=self.gene_align_idx0, idx1=self.gene_align_idx1)

    def _stage(self) -> str:
        return "train" if self.training else "val"
    
    @staticmethod
    def _num_edges_in_batch(batch) -> int:
        if hasattr(batch, "edge_index_dict"):
            return int(sum(v.size(1) for v in batch.edge_index_dict.values()))
        return 1
    
    def _edge_type_key(self, edge_type: EdgeType) -> str:
        try:
            return "_".join(edge_type)
        except TypeError:
            return str(edge_type)
    
    def _log_metric(
        self,
        name: str,
        value,
        *,
        stage = None,
        batch_size = None,
        on_step = False,
        on_epoch = True,
        prog_bar = False,
    ) -> None:
        stage = self._stage() if stage is None else stage
        key = f"{stage}/{name}"
        batch_size = 1 if batch_size is None else batch_size
        value = float("nan") if (isinstance(value, float) and not math.isfinite(value)) else value

        self.log(
            key,
            value,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=True
        )
    
    def _log_perf(self, name, seconds, *, on_step = True) -> None:
        self._log_metric(f"{name}/perf", seconds, on_step=on_step, on_epoch=not on_step, prog_bar=False)
    
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
            if (
                self.training and self.current_epoch > self.n_no_kl
            ):  # Start reparameterization later
                out_dict[node_type] = mu_dict[node_type] + torch.randn_like(
                    logstd_dict[node_type]
                ) * torch.exp(logstd_dict[node_type])
            else:
                out_dict[node_type] = mu_dict[node_type]  # Use mean only initially
        return out_dict

    def relational_recon_loss(
        self,
        batch: HeteroData,
        z_dict: Dict[NodeType, Tensor],
        pos_edge_index_dict: Dict[EdgeType, Tensor],
        pos_edge_weight_dict: Dict[EdgeType, Tensor],
        neg_edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
        batch_alledges: HeteroData = None,
        neg_sample=False,
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
                for edge_type in pos_edge_index_dict.keys():
                    src, _, dst = edge_type
                    if hasattr(batch[src], "sample") and hasattr(batch[dst], "sample"):
                        neg_edge_index = negative_sampling_same_sample_bipartite(
                            pos_edge_index_dict[edge_type],
                            src_sample=batch[src].sample,
                            dst_sample=batch[dst].sample,
                            num_neg_samples_fold=self.num_neg_samples_fold,
                            method="sparse",
                        )
                    else:
                        n_pos_edges = pos_edge_index_dict[edge_type].shape[1]
                        neg_edge_index = negative_sampling(
                            batch[edge_type].edge_index,
                            num_nodes=(
                                batch[src].num_nodes,
                                batch[dst].num_nodes,
                            ),
                            num_neg_samples=n_pos_edges * self.num_neg_samples_fold,
                        )
                    neg_edge_index_dict[edge_type] = neg_edge_index
            neg_dist_dict: Dict[EdgeType, Distribution] = self.decoder(
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
            loss_dict[edge_type] = pos_loss
            if neg_sample:
                neg_dist = neg_dist_dict[edge_type]
                neg_edge_weights = torch.zeros( #TODO: check this: is it okay to use zeros as negative edge weights?
                    neg_dist.event_shape, device=pos_edge_weights.device
                )
                neg_loss = -neg_dist.log_prob(neg_edge_weights).sum()
                loss_dict[edge_type] += neg_loss

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
        batch_alledges: HeteroData = None,
        neg_sample=False,
    ):
        nll_dict, neg_edge_index_dict, metric_dict = self.relational_recon_loss(
            batch=batch,
            z_dict=z_dict,
            pos_edge_index_dict=pos_edge_index_dict,
            pos_edge_weight_dict=pos_edge_weight_dict,
            neg_edge_index_dict=neg_edge_index_dict,
            batch_alledges=batch_alledges,
            neg_sample=neg_sample,
        )
        l = torch.tensor(0.0, device=self.device)
        weighted_nll_dict = {}
        for edge_type, nll in nll_dict.items():
            if edgetype_loss_weight_dict:
                edgetype_weight = edgetype_loss_weight_dict[edge_type]
            else:
                edgetype_weight = torch.tensor(1.0, device=self.device)
            l += nll * edgetype_weight
            weighted_nll_dict[edge_type] = nll * edgetype_weight
        return l, weighted_nll_dict, neg_edge_index_dict, metric_dict

    def gene_alignment_loss(self) -> torch.Tensor:
        if self.lambda_gene_align <= 0 or self._gene_pairs is None:
            return torch.tensor(0.0)
        
        gene_mu = self.encoder.__mu_dict__["gene"] # (n_genes, n_latent_dims)
        pairs = self._gene_pairs
        return gene_alignment_msd_loss(gene_mu, pairs)

    def training_step(self, batch, batch_idx):
        t_all0 = time.time()

        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)

        bs_edges = self._num_edges_in_batch(batch)

        # ---- NLL ----
        t0 = time.time()
        batch_nll_loss, weighted_nll_dict, neg_edge_index_dict, _ = self.nll_loss(
            batch=batch,
            z_dict=z_dict,
            pos_edge_index_dict=batch.edge_index_dict,
            pos_edge_weight_dict=batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
            neg_sample=self.batch_negative
        )

        self._log_metric("nll", batch_nll_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
        self._log_perf("nll", time.time() - t0, on_step=True)
        for edge_type, nll_w in weighted_nll_dict.items():
            edge_key = self._edge_type_key(edge_type)
            edge_bs = int(batch[edge_type].edge_index.size(1))
            self._log_metric(f"nll/{edge_key}", nll_w, batch_size=edge_bs, on_step=False, on_epoch=True, prog_bar=False)

        # ---- KL ----
        if self.current_epoch >= self.n_no_kl:
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
            # aux_kl_div_loss = self.aux_params.kl_div_loss(batch, self.node_weights_dict)
            t1 = time.time()
            kl_scale = self._linear_warmup_scale(self.n_no_kl, self.n_kl_warmup)
            # batch_kl_div_loss *= kl_scale
        else:
            batch_kl_div_loss = 0.0
            kl_scale = 0.0
        
        self._log_metric("kl", batch_kl_div_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
        self._log_metric("kl_weighted", 1/self.nll_scale * kl_scale * batch_kl_div_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # ---- Herit Loss ----
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
            self._log_metric("herit_loss", herit_loss_value, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            self._log_perf("herit_loss", time.time() - t0, on_step=True)
        else:
            herit_loss_value = torch.tensor(0.0)
        
        # ---- Gene Alignment ----
        gene_align_loss = self.gene_alignment_loss()
        gene_align_scale = self._linear_warmup_scale(self.gene_align_n_no, self.gene_align_n_warmup)

        self._log_metric("gene_align", gene_align_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
        self._log_metric("gene_align_weighted", self.lambda_gene_align * gene_align_scale * gene_align_loss, on_step=True, on_epoch=True, prog_bar=False)

        loss = (
            batch_nll_loss
            + 1/self.nll_scale * kl_scale * batch_kl_div_loss
            # + aux_kl_div_loss / self.nll_scale
            + herit_loss_value
            # + aux_reg_loss
            + self.lambda_gene_align * gene_align_scale * gene_align_loss
        )

        self._log_metric("loss", loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=True)
        self._log_perf("step_total", time.time() - t_all0, on_step=True)

        # Debug: Monitor gradients and embeddings every 100 steps
        if self.current_epoch % 10 == 0 and self.local_step == 0 and self.verbose:
            self._debug_embeddings_only(mu_dict, logstd_dict)

        self.local_step += 1

        return loss

    def on_after_backward(self):
        """Called after loss.backward() and before optimizers are stepped."""

        # Gradient clipping to prevent aux_params from dominating
        # torch.nn.utils.clip_grad_norm_(self.aux_params.parameters(), max_norm=1.0)

        # Only check gradients every 100 steps to avoid spam
        if self.current_epoch % 10 == 0 and self.local_step == 0 and self.verbose:
            self._debug_gradients()

    def _debug_embeddings_only(self, mu_dict, logstd_dict):
        """Debug method to monitor embedding quality only (no gradients)."""

        self.logger2.info(
            f"\n=== EMBEDDING DEBUG - Epoch {self.current_epoch}, Step {self.global_step} ==="
        )

        # 1. Check KL Divergence and Variance
        for node_type in mu_dict.keys():
            mu_std = torch.std(mu_dict[node_type]).item()
            logstd_mean = torch.mean(logstd_dict[node_type]).item()

            self.logger2.info(
                f"{node_type} - mu std: {mu_std:.4f}, logstd mean: {logstd_mean:.4f}"
            )

            # Check for collapse
            if mu_std < 0.01:
                self.logger2.info(f"WARNING: Potential mu collapse in {node_type}")
            if logstd_mean < -5:
                self.logger2.info(
                    f"WARNING: Potential variance collapse in {node_type}"
                )

        # 2. Check latent representation quality
        with torch.no_grad():
            z_dict = self.reparametrize(mu_dict, logstd_dict)

            for node_type, z in z_dict.items():
                # Check variance across dimensions
                dim_vars = torch.var(z, dim=0)
                active_dims = (dim_vars > 0.01).sum().item()
                total_dims = z.shape[1]

                self.logger2.info(
                    f"{node_type} - Active dims: {active_dims}/{total_dims}"
                )
                self.logger2.info(
                    f"  Min/Max variance: {dim_vars.min().item():.4f}/{dim_vars.max().item():.4f}"
                )

                if active_dims < total_dims * 0.1:
                    self.logger2.info(
                        f"WARNING: Very few active dimensions in {node_type}"
                    )

        self.logger2.info("=" * 50)

    def _debug_gradients(self):
        """Debug method to monitor gradients after backward pass."""

        self.logger2.info(
            f"\n=== GRADIENT DEBUG - Epoch {self.current_epoch}, Step {self.global_step} ==="
        )

        # Check gradient norms after backward pass
        encoder_grad_norm = 0
        decoder_grad_norm = 0
        aux_grad_norm = 0

        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item() ** 2
                if "encoder" in name:
                    encoder_grad_norm += grad_norm
                elif "decoder" in name:
                    decoder_grad_norm += grad_norm
                elif "aux_params" in name:
                    aux_grad_norm += grad_norm
            else:
                self.logger2.info(f"No gradient for: {name}")

        encoder_grad_norm = encoder_grad_norm**0.5
        decoder_grad_norm = decoder_grad_norm**0.5
        aux_grad_norm = aux_grad_norm**0.5

        self.logger2.info(f"Encoder grad norm: {encoder_grad_norm:.6f}")
        self.logger2.info(f"Decoder grad norm: {decoder_grad_norm:.6f}")
        self.logger2.info(f"Aux params grad norm: {aux_grad_norm:.6f}")

        # Check gradient ratios
        total_grad_norm = encoder_grad_norm + decoder_grad_norm + aux_grad_norm
        if total_grad_norm > 0:
            self.logger2.info(
                f"Encoder grad %: {100*encoder_grad_norm/total_grad_norm:.1f}%"
            )
            self.logger2.info(
                f"Decoder grad %: {100*decoder_grad_norm/total_grad_norm:.1f}%"
            )
            self.logger2.info(
                f"Aux params grad %: {100*aux_grad_norm/total_grad_norm:.1f}%"
            )

            if encoder_grad_norm > 0 and decoder_grad_norm > 0:
                ratio = encoder_grad_norm / decoder_grad_norm
                self.logger2.info(f"Encoder/Decoder grad ratio: {ratio:.4f}")
                if ratio < 0.01:
                    self.logger2.info(
                        "WARNING: Encoder gradients much smaller than decoder"
                    )
        else:
            self.logger2.info("WARNING: No gradients found!")

        self.logger2.info("=" * 60)

    def validation_step(self, batch, batch_idx):
        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)
        
        bs_edges = self._num_edges_in_batch(batch)

        # ---- NLL ----
        batch_nll_loss, weighted_nll_dict, neg_edge_index_dict, metric_dict = self.nll_loss(
            batch,
            z_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
            neg_sample=self.batch_negative
        )

        self._log_metric("nll", batch_nll_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        for edge_type, nll_w in weighted_nll_dict.items():
            edge_key = self._edge_type_key(edge_type)
            edge_bs = int(batch[edge_type].edge_index.size(1))
            self._log_metric(f"nll/{edge_key}", nll_w, batch_size=edge_bs, on_step=False, on_epoch=True, prog_bar=False)

        # ---- KL ----
        if self.current_epoch >= self.n_no_kl:
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
            kl_scale = self._linear_warmup_scale(self.n_no_kl, self.n_kl_warmup)
            # batch_kl_div_loss *= kl_scale
        else:
            batch_kl_div_loss = 0.0
            kl_scale = 0.0
        
        self._log_metric("kl", batch_kl_div_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        self._log_metric("kl_weighted", 1/self.val_nll_scale * kl_scale * batch_kl_div_loss, on_step=False, on_epoch=True, prog_bar=False)

        # ---- Herit Loss ----
        if self.herit_loss is not None:
            if "peak" in batch.node_types:
                pid = batch["peak"].n_id.cpu()
                herit_loss_value = self.herit_loss_lam * self.herit_loss(
                    mu_dict["peak"],
                    pid,
                )
            else:
                herit_loss_value = torch.tensor(0.0)
            self._log_metric("herit_loss", herit_loss_value, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        else:
            herit_loss_value = torch.tensor(0.0)
        
        # ---- Gene Alignment ----
        gene_align_loss = self.gene_alignment_loss()
        gene_align_scale = self._linear_warmup_scale(self.gene_align_n_no, self.gene_align_n_warmup)
        
        self._log_metric("gene_align", gene_align_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        self._log_metric("gene_align_weighted", self.lambda_gene_align * gene_align_scale * gene_align_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        loss = (
            batch_nll_loss + 
            1/self.val_nll_scale * kl_scale * batch_kl_div_loss + 
            herit_loss_value
            + self.lambda_gene_align * gene_align_scale * gene_align_loss
        )

        self._log_metric("loss", loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=True)

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
            self._log_metric("hsic_loss", hsic_loss, on_epoch=True)
        if self.hsic is not None:
            self._log_metric("hsic_lr", self.hsic_optimizer.param_groups[0]["lr"], on_epoch=True)
        if self.reweight_rarecell:
            self.cell_weights = self.mean_cell_neighbor_distance(
                self.reweight_rarecell_neighbors
            )
        self.local_step = 0

    def on_train_batch_start(self, batch, batch_idx):
        if hasattr(self, "t"):
            self.log("data_loading_time", time.time() - self.t)

    def on_train_batch_end(self, output, batch, batch_idx):
        self.t = time.time()

    def configure_optimizers(self):
        # Different learning rates for encoder vs aux_params
        all_params = list(self.parameters())
        encoder_params = list(self.encoder.parameters())
        aux_params = list(self.aux_params.parameters())

        optimizer = torch.optim.Adam(
            # currently there are only encoder and aux parameters, no decoder parameters
            [
                {"params": encoder_params, "lr": self.learning_rate},
                {
                    "params": aux_params,
                    "lr": self.learning_rate * 0.1,
                },  # 10x smaller LR for aux params
            ]
            # [{"params": all_params, "lr": self.learning_rate}]
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=3,
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.monitor_key,
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
    
    def _linear_warmup_scale(self, n_no: int, n_warmup: int) -> float:
        if self.current_epoch < n_no:
            return 0.0
        if n_warmup <= 0:
            return 1.0
        return float(min(self.current_epoch + 1 - n_no, n_warmup) / n_warmup)

    def on_save_checkpoint(self, checkpoint):
        # Remove the argument from the checkpoint before it is saved
        checkpoint.pop("train_data_dict", None)
