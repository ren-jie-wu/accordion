from typing import Optional, Dict, Union, Tuple
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from torch.distributions import Distribution
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_geometric.typing import EdgeType, NodeType
from simba_plus.encoders import TransEncoder
from simba_plus.utils import negative_sampling
from simba_plus.losses import bernoulli_kl_loss
from simba_plus.decoders import RelationalEdgeDistributionDecoder
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


class LightningProxModel(L.LightningModule):
    def __init__(
        self,
        data: HeteroData,
        encoder_class: torch.nn.Module = TransEncoder,
        n_hidden_dims: int = 128,
        n_latent_dims: int = 50,
        decoder_class: torch.nn.Module = RelationalEdgeDistributionDecoder,
        device="cpu",
        num_neg_samples_fold: int = 2,
        num_layers: int = 1,
        num_heads: int = 1,
        project_decoder: bool = True,
        edgetype_specific_bias: bool = True,
        edgetype_specific_scale: bool = True,
        edgetype_specific_std: bool = True,
        edge_types: Optional[Tuple[str]] = None,
        hsic: nn.Module = None,
        n_no_kl: int = 1,
        n_count_nodes: int = 20,
        n_kl_warmup: int = 50,
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
        data.generate_ids()
        self.nonneg = nonneg
        self.data = data
        self.learning_rate = learning_rate
        self.encoder = encoder_class(
            data,
            n_hidden_dims,
            n_latent_dims,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = decoder_class(
            data,
            n_latent_dims,
            n_latent_dims,
            device=device,
            project=project_decoder,
            edgetype_specific_bias=edgetype_specific_bias,
            edgetype_specific_scale=edgetype_specific_scale,
            edgetype_specific_std=edgetype_specific_std,
        )
        self.hsic = hsic
        if self.hsic is not None:
            self.hsic_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=hsic.lam
            )
        self.n_no_kl = n_no_kl
        self.n_count_nodes = n_count_nodes
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

        # Batch correction for RNA-seq data
        self.bias_dict = nn.ParameterDict()
        self.scale_dict = nn.ParameterDict()
        self.std_dict = nn.ParameterDict()
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            if edgetype_specific_bias:
                src_bias_key = make_key(src, edge_type)
                dst_bias_key = make_key(dst, edge_type)
            else:
                src_bias_key = src
                dst_bias_key = dst
            if edgetype_specific_scale:
                src_scale_key = make_key(src, edge_type)
                dst_scale_key = make_key(dst, edge_type)
            else:
                src_scale_key = src
                dst_scale_key = dst
            if edgetype_specific_std:
                src_std_key = make_key(src, edge_type)
                dst_std_key = make_key(dst, edge_type)
            else:
                src_std_key = src
                dst_std_key = dst
            self.bias_dict[src_bias_key] = nn.Parameter(
                torch.zeros(
                    data[src].num_nodes,
                )
            )
            self.bias_dict[dst_bias_key] = nn.Parameter(
                torch.zeros(
                    data[dst].num_nodes,
                )
            )
            self.scale_dict[src_scale_key] = nn.Parameter(
                torch.ones(
                    data[src].num_nodes,
                )
            )
            self.scale_dict[dst_scale_key] = nn.Parameter(
                torch.ones(
                    data[dst].num_nodes,
                )
            )
            self.std_dict[src_std_key] = nn.Parameter(
                torch.zeros(
                    data[src].num_nodes,
                )
            )

            self.std_dict[dst_std_key] = nn.Parameter(
                torch.zeros(
                    data[dst].num_nodes,
                )
            )

            if hasattr(data["cell"], "batch"):
                n_batches = len(data["cell"].batch.unique())
                batch_features = ["gene", "peak"]
                for batch_feature in batch_features:
                    if batch_feature == src:
                        self.scale_dict[src_scale_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[src].num_nodes),
                            )
                        )
                        self.bias_dict[src_bias_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[batch_feature].num_nodes),
                                # device=self.device,
                            )
                        )
                        self.std_dict[src_bias_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[batch_feature].num_nodes),
                                # device=self.device,
                            )
                        )
                    if batch_feature == dst:
                        self.scale_dict[dst_scale_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[dst].num_nodes),
                                # device=self.device,
                            )
                        )
                        self.bias_dict[dst_bias_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[dst].num_nodes),
                                # device=self.device,
                            )
                        )
                        self.std_dict[dst_bias_key] = nn.Parameter(
                            torch.ones(
                                (n_batches, data[dst].num_nodes),
                                # device=self.device,
                            )
                        )
        self.num_neg_samples_fold = num_neg_samples_fold
        self.validation_step_outputs = []

    def on_train_start(self):
        self.node_weights_dict = {
            k: v.to(self.device) for k, v in self.node_weights_dict.items()
        }
        if self.hsic is not None:
            update_lr(self.hsic_optimizer, self.hsic.lam * self.learning_rate)

    # def initialize_node_embedding(self, data: HeteroData, requires_grad: bool = True):
    #     for ntype in data.node_types:
    #         emb = nn.Parameter(data[ntype].x, requires_grad=requires_grad)
    #         data[ntype].x = emb
    #     return data

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
        if self.nonneg:
            transf = nn.ReLU()
            out_dict = {k: transf(v) for k, v in out_dict.items()}
        return out_dict

    def project(
        self, data: HeteroData, z_dict: Dict[NodeType, Tensor], edge_type: EdgeType
    ) -> Tuple[Tensor, Tensor]:
        """Projects all node in z_dict of edge_type to edge-type-specific space."""
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type]
        dst_z = z_dict[dst_type]
        if src_type == "cell" and self.decoder.add_covariate:
            src_z = add_cov_to_latent(data[src_type], src_z)
        if dst_type == "cell" and self.decoder.add_covariate:
            dst_z = add_cov_to_latent(data[dst_type], dst_z)
        return self.decoder.project(src_z, dst_z, src_type, dst_type)

    def relational_recon_loss(
        self,
        batch: HeteroData,
        z_dict: Dict[NodeType, Tensor],
        pos_edge_index_dict: Dict[EdgeType, Tensor],
        pos_edge_weight_dict: Dict[EdgeType, Tensor],
        neg_edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
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
        # import pdb; pdb.set_trace()
        pos_dist_dict: Dict[EdgeType, Distribution] = self.decoder(
            batch,
            z_dict,
            pos_edge_index_dict,
            scale_dict=self.scale_dict,
            bias_dict=self.bias_dict,
            std_dict=self.std_dict,
        )
        if neg_edge_index_dict is None:
            neg_edge_index_dict = {}
            for edge_type, pos_edge_index in pos_edge_index_dict.items():
                src_type, _, dst_type = edge_type
                (
                    neg_src_idx,
                    neg_dst_idx,
                ) = negative_sampling(
                    pos_edge_index,
                    num_nodes=(batch[src_type].num_nodes, batch[dst_type].num_nodes),
                    num_neg_samples_fold=self.num_neg_samples_fold,
                    # method="dense",
                )
                if (neg_src_idx > batch[src_type].num_nodes).any():  # pragma: no cover
                    raise ValueError(
                        f"Negative sampling produced indices larger than the number of nodes in {src_type}. "
                        f"Please check your data and the negative sampling parameters."
                    )
                neg_edge_index_dict[edge_type] = torch.stack([neg_src_idx, neg_dst_idx])
        neg_dist_dict: Dict[EdgeType, Tensor] = self.decoder(
            batch,
            z_dict,
            neg_edge_index_dict,
            scale_dict=self.scale_dict,
            bias_dict=self.bias_dict,
            std_dict=self.std_dict,
        )

        loss_dict = {}
        metric_dict = {}
        for edge_type, pos_dist in pos_dist_dict.items():
            src_type, _, dst_type = edge_type
            pos_edge_weights = pos_edge_weight_dict[edge_type]
            pos_log_probs = -pos_dist.log_prob(pos_edge_weights)
            neg_log_probs = -neg_dist_dict[edge_type].log_prob(
                torch.tensor(0.0, device=batch[src_type].x.device)
                if batch[edge_type].edge_dist != "Beta"
                else torch.tensor(1e-6, device=batch[src_type].x.device)
            )
            if self.reweight_rarecell and src_type == "cell":
                pos_log_probs *= self.cell_weights[batch[src_type].n_id]
                neg_log_probs *= self.cell_weights[batch[src_type].n_id[neg_src_idx]]
            pos_loss = pos_log_probs.sum()
            neg_loss = neg_log_probs.sum()
            loss_dict[edge_type] = pos_loss + neg_loss

            if get_metric:
                if edge_type == ("cell", "expresses", "gene"):
                    # import pdb; pdb.set_trace()
                    metric_dict[edge_type] = compute_reconstruction_gene_metrics(
                        pos_edge_weights,
                        pos_dist_dict[edge_type].mean,
                        pos_dist_dict[edge_type].stddev,
                        plot=plot,
                    )
                    # plot the pos, neg and overall log p distribution
                    if plot:
                        plot_nll_distributions(
                            pos_log_probs.detach().cpu().numpy(),
                            neg_log_probs.detach().cpu().numpy(),
                            name="cell_express_gene",
                        )

                elif edge_type == ("cell", "has_accessible", "peak"):
                    # import pdb; pdb.set_trace()
                    if get_metric:
                        neg_size = neg_edge_index_dict[edge_type].shape[1]
                        ground_truth = (
                            torch.cat(
                                [
                                    pos_edge_weights,
                                    torch.zeros(
                                        neg_size, device=batch[src_type].x.device
                                    ),
                                ],
                                dim=0,
                            )
                            .detach()
                            .cpu()
                        )
                        pred = (
                            torch.cat(
                                [
                                    pos_dist_dict[edge_type].logits,
                                    neg_dist_dict[edge_type].logits,
                                ],
                                dim=0,
                            )
                            .detach()
                            .cpu()
                        )
                        pred = torch.sigmoid(pred)
                        metric_dict[edge_type] = compute_classification_metrics(
                            ground_truth, pred, plot=plot
                        )
                    if plot:
                        plot_nll_distributions(
                            pos_log_probs.detach().cpu().numpy(),
                            neg_log_probs.detach().cpu().numpy(),
                            name="cell_has_peak",
                        )

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
            weight (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
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
        if hasattr(self.decoder, "variant_emb_locs"):
            l += self._kl_loss(
                self.decoder.variant_emb_locs,
                self.decoder.variant_emb_logstd,
            )
            l += bernoulli_kl_loss(
                torch.tensor(-2), self.decoder.variant_emb_mask_logitp
            ).sum()
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
        plot=False,
        get_metric=False,
    ):
        nll_dict, neg_edge_index_dict, metric_dict = self.relational_recon_loss(
            batch,
            z_dict,
            pos_edge_index_dict,
            pos_edge_weight_dict,
            neg_edge_index_dict,
            # num_neg_samples_fold,
            plot=plot,
            get_metric=get_metric,
        )
        l = torch.tensor(0.0, device=self.device)
        for edge_type, nll in nll_dict.items():
            if edgetype_loss_weight_dict:
                edgetype_weight = edgetype_loss_weight_dict[edge_type]
            else:
                edgetype_weight = torch.tensor(1.0, device=self.device)
            l += nll * edgetype_weight
        return l, neg_edge_index_dict, metric_dict

    def training_step(self, batch, batch_idx):
        edge_attr_type, idx = batch
        edge_attr_type = tuple(edge_attr_type)
        batch = (
            self.data.edge_type_subgraph([edge_attr_type])
            .edge_subgraph({edge_attr_type: idx.to("cpu")})
            .to(self.device)
        )
        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)
        batch_nll_loss, neg_edge_index_dict, _ = self.nll_loss(
            batch,
            z_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
        )
        if self.current_epoch >= self.n_no_kl:
            batch_kl_div_loss = self.kl_div_loss(
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )

            batch_kl_div_loss *= min(
                self.current_epoch, self.n_kl_warmup - self.n_no_kl
            ) / (self.n_kl_warmup - self.n_no_kl)
        else:
            batch_kl_div_loss = 0.0

        self.log(
            "nll_loss",
            batch_nll_loss * self.nll_scale,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "kl_div_loss",
            batch_kl_div_loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        loss = batch_nll_loss * self.nll_scale + batch_kl_div_loss
        self.log(
            "loss",
            loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        edge_attr_type, idx = batch
        edge_attr_type = tuple(edge_attr_type)
        batch = (
            self.data.edge_type_subgraph([edge_attr_type])
            .edge_subgraph({edge_attr_type: idx.to("cpu")})
            .to(self.device)
        )

        self.eval()
        mu_dict, logstd_dict = self.encode(batch)
        z_dict = self.reparametrize(mu_dict, logstd_dict)
        batch_nll_loss, neg_edge_index_dict, metric_dict = self.nll_loss(
            batch,
            z_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            edgetype_loss_weight_dict=self.edgetype_loss_weight_dict,
            get_metric=True,
        )

        if self.current_epoch >= self.n_no_kl:
            batch_kl_div_loss = self.kl_div_loss(
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
        else:
            batch_kl_div_loss = 0.0
        loss = batch_nll_loss * self.val_nll_scale + batch_kl_div_loss
        self.log(
            "val_nll_loss",
            batch_nll_loss * self.val_nll_scale,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_nll_loss_monitored",
            batch_nll_loss * self.val_nll_scale
            + (torch.inf if self.current_epoch < self.n_kl_warmup * 2 else 0),
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_kl_div_loss",
            batch_kl_div_loss,
            batch_size=sum([v.shape[1] for v in batch.edge_index_dict.values()]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_loss",
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
            patience=15,
        )
        # scheduler = CyclicLR(
        #     optimizer,
        #     self.learning_rate / 2,
        #     self.learning_rate * 2,
        #     step_size_up=10,
        #     mode="triangular2",
        # )
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
