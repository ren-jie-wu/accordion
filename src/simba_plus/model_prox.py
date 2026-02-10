from typing import Optional, Dict, Union, Tuple, List
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
import warnings
import pandas as pd
import os
from datetime import datetime

# from simba_plus.utils import negative_sampling

# from torch_geometric.utils import negative_sampling
from simba_plus.loss.hsic import bernoulli_kl_loss
from simba_plus.decoders import RelationalEdgeDistributionDecoder
import time
from simba_plus._utils import (
    add_cov_to_latent,
    make_key,
    update_lr,
)
from simba_plus.util_modules import BufferDict, _CheckHeteroDataCodes as chk, _logger
from simba_plus.negative_sampling import negative_sampling_same_sample_bipartite
from simba_plus.loss.gene_alignment import (
    TwoSampleGenePairs, 
    build_two_sample_gene_pairs, 
    MultiSampleGenePairs,
    build_multi_sample_gene_pairs,
    gene_alignment_msd_loss,
)
from simba_plus.loss.ot_alignment import (
    build_two_sample_cell_index,
    compute_two_sample_ot_state,
    ot_cost_two,
    TwoSampleOTState,
    PairOTState,
    build_sample_cell_index,
    make_all_sample_pairs,
    compute_multi_sample_ot_states,
    ot_cost_multi,
    PairOTStateKeyed,
    build_sample_batch_cell_index,
    make_all_sample_batch_pairs,
    compute_multi_batch_ot_states,
)


class _PerSampleBatchedParam(nn.Module):
    """
    Store per-sample batched parameters:
      - param[s]: (B_s, N_s)
      - param_logstd[s]: (B_s, N_s)
    """

    def __init__(
        self,
        n_batches_per_sample: Dict[int, int],
        n_nodes_per_sample: Dict[int, int],
        sample_ids: Tuple[int, ...],
        has_logstd: bool = True,
        init: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_ids = sample_ids
        self.sample_to_listidx = {sid: i for i, sid in enumerate(sample_ids)}

        self.param = nn.ParameterList()
        self.has_logstd = has_logstd
        if self.has_logstd:
            self.param_logstd = nn.ParameterList()
        for sid in sample_ids:
            B = int(n_batches_per_sample.get(sid, 1))
            N = int(n_nodes_per_sample.get(sid, 0))
            self.param.append(nn.Parameter(torch.full((B, N), float(init))))
            if self.has_logstd:
                self.param_logstd.append(nn.Parameter(torch.zeros((B, N))))

    def get(self, sample_id: int) -> Tuple[Tensor, Tensor]:
        i = self.sample_to_listidx[int(sample_id)]
        if self.has_logstd:
            return self.param[i], self.param_logstd[i]
        else:
            return self.param[i]


class AuxParams(nn.Module):
    def __init__(self, data: HeteroData, edgetype_specific: bool = True, check_data: bool = False, use_noise: bool = True) -> None:
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
                "gene__cell_expresses_gene",      # value shape: (num_batches, num_genes) if use_batch else (num_genes,)
                "peak__cell_has_accessible_peak"  # value shape: (num_batches, num_peaks) if use_batch else (num_peaks,)
            ]
            But for *_logstd_dict, there is only destination node type in the key.
            [
                "gene__cell_expresses_gene",
                "peak__cell_has_accessible_peak"
            ]
        """
        super().__init__()
        self.edgetype_specific = edgetype_specific
        self.use_noise = use_noise

        srcs, dsts = zip(*[(src, dst) for src, _, dst in data.edge_types])
        assert set(srcs) <= {"cell"} and set(dsts) <= {"gene", "peak"}, "Only cell-gene and cell-peak edges are supported"

        if check_data:
            ok, msgs = chk.check_heterodata(data)
            if not ok:
                raise ValueError(f"Invalid heterodata: {msgs}")
            else:
                print(f"HeteroData codes are valid: {msgs}")

        # multi-sample detection
        if hasattr(data["cell"], "sample"):
            self.sample_ids = tuple(sorted(
                torch.unique(data["cell"].sample).detach().cpu().tolist()
            ))
        else:
            self.sample_ids = (0,)
        self.multi_sample = len(self.sample_ids) > 1

        # if there are more than one batch, turn on use_batch
        self.use_batch = (
            (hasattr(data["cell"], "batch_local") 
            and torch.unique(data["cell"].batch_local).numel() > 1)
            or (hasattr(data["cell"], "batch")
            and torch.unique(data["cell"].batch).numel() > len(self.sample_ids))
        )

        # Batch correction for RNA-seq data
        self.bias_dict = nn.ParameterDict()
        self.logscale_dict = nn.ParameterDict()
        self.std_dict = nn.ParameterDict()

        self.bias_logstd_dict = nn.ParameterDict()
        self.logscale_logstd_dict = nn.ParameterDict()
        self.std_logstd_dict = nn.ParameterDict()

        self.bias_dict_per_sample = nn.ModuleDict()
        self.logscale_dict_per_sample = nn.ModuleDict()
        self.std_dict_per_sample = nn.ModuleDict()

        if self.multi_sample and self.use_batch:
            # precompute n_batches per sample
            n_batches_per_sample: Dict[int, int] = {sid: 1 for sid in self.sample_ids}
            for sid in self.sample_ids:
                mask = data["cell"].sample == sid
                if mask.any():
                    bmax = int(data["cell"].batch_local[mask].max().item())
                    n_batches_per_sample[sid] = bmax + 1
                else:
                    n_batches_per_sample[sid] = 1
            
            # precompute n_nodes per sample for each feature node type
            n_nodes_per_sample_per_type: Dict[str, Dict[int, int]] = {}
            for node_type in data.node_types:
                if node_type == "cell":
                    continue
                if not hasattr(data[node_type], "sample"):
                    raise ValueError(f"Node type {node_type} does not have sample attribute while there are multiple samples")
                n_nodes_per_sample_per_type[node_type] = {sid: 0 for sid in self.sample_ids}
                for sid in self.sample_ids:
                    mask = data[node_type].sample == sid
                    n_nodes_per_sample_per_type[node_type][sid] = int(mask.sum().item())
        
        # build parameter containers per edge type
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            src_key, dst_key = self.get_keys(src, dst, edge_type)

            num_src_nodes = data[src].num_nodes
            num_dst_nodes = data[dst].num_nodes

            # src params: always per-node
            if src_key not in self.bias_dict:
                self.bias_dict[src_key] = nn.Parameter(torch.zeros(num_src_nodes))
                self.logscale_dict[src_key] = nn.Parameter(torch.zeros(num_src_nodes))
                self.std_dict[src_key] = nn.Parameter(torch.zeros(num_src_nodes))
            
            # dst params: depending on use_batch and multi_sample
            if not self.use_batch:
                if dst_key not in self.bias_dict:
                    self.bias_dict[dst_key] = nn.Parameter(torch.zeros(num_dst_nodes))
                    self.logscale_dict[dst_key] = nn.Parameter(torch.zeros(num_dst_nodes))
                    self.std_dict[dst_key] = nn.Parameter(torch.zeros(num_dst_nodes))
            elif not self.multi_sample:
                _batch = data["cell"].batch if hasattr(data["cell"], "batch") else data["cell"].batch_local
                n_batches = int(torch.unique(_batch).numel())
                if dst_key not in self.bias_dict:
                    self.bias_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
                    self.logscale_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
                    self.std_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
                    self.bias_logstd_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
                    self.logscale_logstd_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
                    self.std_logstd_dict[dst_key] = nn.Parameter(torch.zeros(n_batches, num_dst_nodes))
            else: # multi-sample and use_batch
                if dst_key not in self.bias_dict_per_sample:
                    self.bias_dict_per_sample[dst_key] = _PerSampleBatchedParam(
                        n_batches_per_sample=n_batches_per_sample,
                        n_nodes_per_sample=n_nodes_per_sample_per_type[dst],
                        sample_ids=self.sample_ids,
                        init=0.0,
                    )
                    self.logscale_dict_per_sample[dst_key] = _PerSampleBatchedParam(
                        n_batches_per_sample=n_batches_per_sample,
                        n_nodes_per_sample=n_nodes_per_sample_per_type[dst],
                        sample_ids=self.sample_ids,
                        init=0.0,
                    )
                    self.std_dict_per_sample[dst_key] = _PerSampleBatchedParam(
                        n_batches_per_sample=n_batches_per_sample,
                        n_nodes_per_sample=n_nodes_per_sample_per_type[dst],
                        sample_ids=self.sample_ids,
                        init=0.0,
                    )

    def batched(self, param, param_logstd):
        """
        Generate a batch of stochastic parameters based on baseline parameters, offsets, and deviations.

        Args:
            param: Baseline and offset parameters, shape: (num_batches, num_nodes)
            param_logstd: Log standard deviations, shape: (num_batches, num_nodes)

        Returns:
            Stochastic parameters, shape: (num_batches, num_nodes)
        """
        if self.use_noise and self.training and self.use_batch:
            assert param.dim() == 2, f"param must be a 2D tensor when use_batch is True, but got {param.dim()}D tensor"
            if param.size(0) <= 1:
                return param
            return_param = torch.cat(
                [
                    param[[0], :],
                    param[1:, :] + param[[0], :] + torch.randn_like(param_logstd[1:, :]) * torch.exp(param_logstd[1:, :]),
                ],
                dim=0,
            )
            return return_param
        return param

    def _dst_values_per_edge(
        self,
        batch: HeteroData,
        edge_index: Tensor,
        dst_type: str,
        dst_key: str,
        which: str,
    ) -> Tensor:
        """
        Return a (num_edges,) tensor of dst parameters for this edge_type.

        which in {"logscale","bias","std"}.
        """
        device = edge_index.device
        num_edges = edge_index.size(1)

        # not use_batch
        if not self.use_batch:
            dst_global = batch[dst_type].n_id[edge_index[1]]
            if which == "logscale":
                return self.logscale_dict[dst_key][dst_global]
            if which == "bias":
                return self.bias_dict[dst_key][dst_global]
            if which == "std":
                return self.std_dict[dst_key][dst_global]
            raise ValueError(f"Unknown which={which}")

        # use_batch and multi_sample
        if self.multi_sample:
            # Per-sample: index by (cell.sample, cell.batch_local, dst.local_id)
            samples = batch["cell"].sample[edge_index[0]].long()
            b_local = batch["cell"].batch_local[edge_index[0]].long()
            dst_local = batch[dst_type].local_id[edge_index[1]].long()

            out = torch.empty((num_edges,), device=device, dtype=torch.float32)

            if which == "logscale":
                mod = self.logscale_dict_per_sample[dst_key]
            elif which == "bias":
                mod = self.bias_dict_per_sample[dst_key]
            elif which == "std":
                mod = self.std_dict_per_sample[dst_key]
            else:
                raise ValueError(f"Unknown which={which}")

            for sid in torch.unique(samples).detach().cpu().tolist():
                sid = int(sid)
                mask = (samples == sid)
                if not mask.any():
                    continue
                p, p_logstd = mod.get(sid)
                p_eff = self.batched(p, p_logstd)
                out[mask] = p_eff[b_local[mask], dst_local[mask]].to(out.dtype)
            return out

        # use_batch and not multi_sample
        dst_global = batch[dst_type].n_id[edge_index[1]]
        _batch = batch["cell"].batch if hasattr(batch["cell"], "batch") else batch["cell"].batch_local
        b = _batch[edge_index[0]].long()

        if which == "logscale":
            return self.batched(self.logscale_dict[dst_key], self.logscale_logstd_dict[dst_key])[b, dst_global]
        if which == "bias":
            return self.batched(self.bias_dict[dst_key], self.bias_logstd_dict[dst_key])[b, dst_global]
        if which == "std":
            return self.batched(self.std_dict[dst_key], self.std_logstd_dict[dst_key])[b, dst_global]
        raise ValueError(f"Unknown which={which}")

    def forward(self, batch, edge_index_dict):
        """
        Get the parameters for the given batch and edge index dictionary.

        Args:
            batch: Batch object, providing
                - batch[node_type].n_id: help convert local node IDs in `edge_index_dict` to global node IDs
                - batch["cell"].batch: provide batch information for the cells
            edge_index_dict: Edge index dictionary, each value is a tensor of shape (2, num_edges) with local node IDs

        Returns:
            Dictionary of parameters for the given batch and edge index dictionary
        """
        src_logscale_dict, src_bias_dict, src_std_dict = {}, {}, {}
        dst_logscale_dict, dst_bias_dict, dst_std_dict = {}, {}, {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_key, dst_key = self.get_keys(src_type, dst_type, edge_type)

            src_node_id = batch[src_type].n_id[edge_index[0]] # local id to global id
            src_logscale_dict[edge_type] = self.logscale_dict[src_key][src_node_id]
            src_bias_dict[edge_type] = self.bias_dict[src_key][src_node_id]
            src_std_dict[edge_type] = self.std_dict[src_key][src_node_id]

            dst_logscale_dict[edge_type] = self._dst_values_per_edge(batch, edge_index, dst_type, dst_key, "logscale")
            dst_bias_dict[edge_type] = self._dst_values_per_edge(batch, edge_index, dst_type, dst_key, "bias")
            dst_std_dict[edge_type] = self._dst_values_per_edge(batch, edge_index, dst_type, dst_key, "std")
        
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


class LightningProxModel(L.LightningModule):
    def __init__(
        self,
        data: HeteroData,
        logger=None,
        encoder_class: torch.nn.Module = TransEncoder,
        n_latent_dims: int = 50,
        decoder_class: torch.nn.Module = RelationalEdgeDistributionDecoder,
        device=None,
        num_neg_samples_fold: int = 1,
        edgetype_specific: bool = True,
        edge_types: Optional[Tuple[str]] = None,
        hsic: Optional[nn.Module] = None,
        herit_loss: Optional[nn.Module] = None,
        herit_loss_lam: float = 1,
        kl_lambda: float = 0.05,
        kl_n_no: int = 30,
        kl_n_warmup: int = 50,
        # nll_scale: float = 1.0,
        # val_nll_scale: float = 1.0,
        learning_rate=1e-2,
        node_weights_dict=None,
        # nonneg=False, #not used
        reweight_rarecell: bool = False,
        reweight_rarecell_neighbors: Optional[int] = None,
        verbose: bool = False,
        batch_negative: bool = False, # make the default same as in the CLI
        monitor_key: str = None,

        gene_align_lambda: float = 0.0,
        gene_align_n_no: int = 0,
        gene_align_n_warmup: int = 10,
        ot_cs_lambda: float = 0.0,
        ot_cs_n_no: int = 7,
        ot_cs_n_warmup: int = 7,
        ot_cs_k: int = 256,
        ot_cb_lambda: float = 0.0,
        ot_cb_n_no: int = 7,
        ot_cb_n_warmup: int = 7,
        ot_cb_k: int = 256,
        ot_eps: float = 0.05,
        ot_iter: int = 50,
        ot_plan_every_n_epochs: int = 1, # invalid if 0
        ot_loss_every_n_steps: int = 1,  # if 0, once per epoch (first batch)

        # DEBUG
        use_aux: bool = True,
        use_aux_noise: bool = False,
    ):
        super().__init__()
        if device is not None:
            warnings.warn(f"device is deprecated and will be removed in future versions."\
                "Please use `model.to(device)` to move the model to the desired device instead.")
        
        self.save_hyperparameters()
        # self.nonneg = nonneg
        self.data = data
        self.logger2 = logger
        
        if self.logger2 is None:
            self.logger2 = _logger() # fallback to print
        
        ok, msgs = chk.check_heterodata(data)
        if not ok:
            raise ValueError(f"Invalid heterodata: {msgs}")
        else:
            self.logger2.info(f"HeteroData codes are valid: {msgs}")

        self.learning_rate = learning_rate
        self.encoder = encoder_class(
            data,
            n_latent_dims,
        )
        self.decoder = decoder_class(
            data,
        )
        self.hsic = hsic
        if self.hsic is not None:
            self.hsic_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=hsic.lam
            )
        self.kl_lambda = kl_lambda
        self.kl_n_no = kl_n_no
        self.kl_n_warmup = kl_n_warmup
        # self.nll_scale = nll_scale
        # self.val_nll_scale = val_nll_scale
        # self.num_nodes_dict = data.num_nodes_dict
        self.cell_weights = torch.ones(data["cell"].num_nodes)
        self.reweight_rarecell = reweight_rarecell
        if self.reweight_rarecell: # not used
            if reweight_rarecell_neighbors is None:
                reweight_rarecell_neighbors = max(20, int(data["cell"].num_nodes / 100))
            self.reweight_rarecell_neighbors = reweight_rarecell_neighbors
        if edge_types is None:
            self.edge_types = data.edge_types
        else:
            self.edge_types = edge_types
        self.edgetype_loss_weight_dict = {
            edgetype: torch.tensor(1.0 / len(self.edge_types)) 
            for edgetype in self.edge_types
        }

        self.node_weights_dict = node_weights_dict
        self.herit_loss = herit_loss
        self.herit_loss_lam = herit_loss_lam

        self.num_neg_samples_fold = num_neg_samples_fold
        self.validation_step_outputs = []
        self.aux_params = AuxParams(data, edgetype_specific=edgetype_specific, 
                                    check_data=False,  # already checked above
                                    use_noise=use_aux_noise    # DEBUG
        )
        if not use_aux:
            self._freeze_aux_params()

        self.verbose = verbose
        self.batch_negative = batch_negative

        self.monitor_key = monitor_key

        self.gene_align_lambda = gene_align_lambda
        self.gene_align_n_no = gene_align_n_no
        self.gene_align_n_warmup = gene_align_n_warmup
        self.ot_cs_lambda = ot_cs_lambda
        self.ot_cs_n_no = ot_cs_n_no
        self.ot_cs_n_warmup = ot_cs_n_warmup
        self.ot_cs_k = ot_cs_k
        self.ot_cb_lambda = ot_cb_lambda
        self.ot_cb_n_no = ot_cb_n_no
        self.ot_cb_n_warmup = ot_cb_n_warmup
        self.ot_cb_k = ot_cb_k
        self.ot_eps = ot_eps
        self.ot_iter = ot_iter
        self.ot_plan_every_n_epochs = ot_plan_every_n_epochs
        self.ot_loss_every_n_steps = ot_loss_every_n_steps

        self._register_gene_pairs()
        self._register_ot_state()
        self._register_others()

        self.log_every_n_steps = 50
    
    def _freeze_aux_params(self):
        for param in self.aux_params.parameters():
            param.requires_grad_(False)
    
    def _register_gene_pairs(self):
        if self.gene_align_lambda > 0:
            if not (hasattr(self.data["gene"], "sample") and hasattr(self.data["gene"], "gene_id")):
                raise ValueError(
                    "gene_align_lambda > 0 requires data['gene'].sample and data['gene'].gene_id "
                    "(constructed in make_sc_HetData_multi_rna)."
                )
            
            pairs = build_multi_sample_gene_pairs(
                gene_sample=self.data["gene"].sample,
                gene_id=self.data["gene"].gene_id,
            )
            
            self.register_buffer("gene_align_idx0", pairs.idx0, persistent=True)
            self.register_buffer("gene_align_idx1", pairs.idx1, persistent=True)
            self._gene_pairs = MultiSampleGenePairs(idx0=self.gene_align_idx0, idx1=self.gene_align_idx1)
    
    def _register_ot_state(self):
        # cross-sample OT state
        if self.ot_cs_lambda > 0:
            if not hasattr(self.data["cell"], "sample"):
                raise ValueError("ot_cs_lambda > 0 requires data['cell'].sample")
            
            # not buffers; we'll compute OT related variables on the fly
            self._ot_cs_groups: dict[int, torch.Tensor] = build_sample_cell_index(self.data["cell"].sample)
            self._ot_cs_pairs: List[Tuple[int, int]] = make_all_sample_pairs(self._ot_cs_groups.keys(), mode="all_pairs")
            self._ot_cs_states: List[PairOTState] = []
        
        # cross-batch OT state
        if self.ot_cb_lambda > 0:
            if not hasattr(self.data["cell"], "batch_local"):
                if not hasattr(self.data["cell"], "batch"):
                    raise ValueError("ot_cb_lambda > 0 requires data['cell'].batch_local or data['cell'].batch")
                else:
                    if not hasattr(self.data["cell"], "sample"):
                        self.data["cell"].sample = torch.zeros_like(self.data["cell"].batch)
                    self.data["cell"].batch_local = chk._make_batch_local_from_global(self.data["cell"].sample, self.data["cell"].batch)
            self._ot_cb_groups: dict[tuple[int, int], torch.Tensor] = build_sample_batch_cell_index(self.data["cell"].sample, self.data["cell"].batch_local)
            self._ot_cb_pairs: dict[int, List[Tuple[int, int]]] = make_all_sample_batch_pairs(self._ot_cb_groups.keys(), mode="all_pairs")
            self._ot_cb_states: List[PairOTStateKeyed] = []

    def _register_others(self):
        # self.register_buffer("cell_weights", self.cell_weights) # not used
        self.edgetype_loss_weight_dict = BufferDict(self.edgetype_loss_weight_dict)
        if self.node_weights_dict is not None:
            self.node_weights_dict = BufferDict(self.node_weights_dict)

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
                self.training and self.current_epoch > self.kl_n_no
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
        fold_tensor = torch.tensor(self.num_neg_samples_fold, device=self.device)
        weight_tensor = torch.tensor(1.0 / (1.0 + self.num_neg_samples_fold), device=self.device)
        for edge_type, pos_dist in pos_dist_dict.items():
            src_type, _, dst_type = edge_type
            pos_edge_weights = pos_edge_weight_dict[edge_type]
            pos_loss = -pos_dist.log_prob(pos_edge_weights).mean()
            loss_dict[edge_type] = pos_loss
            if neg_sample:
                neg_dist = neg_dist_dict[edge_type]
                neg_edge_weights = torch.zeros(
                    neg_dist.event_shape, device=self.device
                )
                neg_loss = -neg_dist.log_prob(neg_edge_weights).mean()
                loss_dict[edge_type] = (loss_dict[edge_type] + neg_loss * fold_tensor) * weight_tensor

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

        def weighted_mean(x, w):
            # if not w.any():
            #     return 0
            if w is None:
                return x.mean()
            return (x * w).sum() / (w.sum() + 1e-12)

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        kls = torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1)
        return -0.5 * weighted_mean(kls, weight)

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
        if node_weights_dict is None:
            node_weights_dict = {node_type: torch.ones(mu_dict[node_type].shape[0], device=self.device) for node_type in mu_dict.keys()}
        for node_type in mu_dict.keys():
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
        if nodetype_loss_weight_dict is None:
            nodetype_loss_weight_dict = {node_type: torch.tensor(1.0 / len(kl_div_dict), device=self.device) for node_type in kl_div_dict.keys()}
        else:
            s = torch.stack([nodetype_loss_weight_dict[node_type] for node_type in kl_div_dict.keys()]).sum()
            nodetype_loss_weight_dict = {node_type: nodetype_loss_weight_dict[node_type] / s for node_type in kl_div_dict.keys()}
        for node_type, kl_div in kl_div_dict.items():
            l += kl_div * nodetype_loss_weight_dict[node_type]
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
        if edgetype_loss_weight_dict is None:
            edgetype_loss_weight_dict = {edge_type: torch.tensor(1.0 / len(nll_dict), device=self.device) for edge_type in nll_dict.keys()}
        else:
            s = torch.stack([edgetype_loss_weight_dict[edge_type] for edge_type in nll_dict.keys()]).sum()
            edgetype_loss_weight_dict = {edge_type: edgetype_loss_weight_dict[edge_type] / s for edge_type in nll_dict.keys()}
        for edge_type, nll in nll_dict.items():
            l += nll * edgetype_loss_weight_dict[edge_type]
            weighted_nll_dict[edge_type] = nll * edgetype_loss_weight_dict[edge_type]
        return l, weighted_nll_dict, neg_edge_index_dict, metric_dict

    def gene_alignment_loss(self) -> torch.Tensor:        
        gene_mu = self.encoder.__mu_dict__["gene"] # (n_genes, n_latent_dims)
        pairs = self._gene_pairs
        return gene_alignment_msd_loss(gene_mu, pairs)

    def ot_alignment_loss(self, type: str) -> torch.Tensor:
        # current trainable mu (NOT detach) => gradients flow to those cells
        cell_mu = self.encoder.__mu_dict__["cell"]
        if type == "cs":
            return ot_cost_multi(cell_mu, self._ot_cs_states, P_detach=True)
        elif type == "cb":
            return ot_cost_multi(cell_mu, self._ot_cb_states, P_detach=True)
        else:
            raise ValueError(f"Unknown OT type={type}")

    def training_step(self, batch, batch_idx):
        # batch = batch.to(self.device)
        
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
        # self._log_perf("nll", time.time() - t0, on_step=True)
        if len(weighted_nll_dict) > 1:
            for edge_type, nll_w in weighted_nll_dict.items():
                edge_key = self._edge_type_key(edge_type)
                edge_bs = int(batch[edge_type].edge_index.size(1))
                self._log_metric(f"nll/{edge_key}", nll_w, batch_size=edge_bs, on_step=False, on_epoch=True, prog_bar=False)
        
        # ---- KL ----
        compute_kl_now, kl_weight = self._compute_kl_now(return_weight=True)
        batch_kl_div_loss = torch.tensor(0.0, device=self.device)
        if compute_kl_now:
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
            self._log_metric("kl", batch_kl_div_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            self._log_metric("kl_weighted", kl_weight * batch_kl_div_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
        
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
                herit_loss_value = torch.tensor(0.0, device=self.device)
            self._log_metric("herit_loss", herit_loss_value, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            # self._log_perf("herit_loss", time.time() - t0, on_step=True)
        else:
            herit_loss_value = torch.tensor(0.0, device=self.device)
        
        # ---- Gene Alignment ----
        compute_gene_align_now, gene_align_weight = self._compute_gene_align_now(return_weight=True)

        gene_align_loss = torch.tensor(0.0, device=self.device)
        if compute_gene_align_now:
            gene_align_loss = self.gene_alignment_loss()
            self._log_metric("gene_align", gene_align_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            self._log_metric("gene_align_weighted", gene_align_weight * gene_align_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)

        # ---- OT loss ----
        compute_ot_cs_now, ot_cs_weight = self._compute_ot_cs_now(return_weight=True)

        ot_cs_loss = torch.tensor(0.0, device=self.device)
        if compute_ot_cs_now:
            t0 = time.time()
            ot_cs_loss = self.ot_alignment_loss(type="cs")
            self._log_metric("ot_cs_loss", ot_cs_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            self._log_metric("ot_cs_weighted", ot_cs_weight * ot_cs_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            # self._log_perf("ot_cs_loss", time.time() - t0, on_step=True)
        
        compute_ot_cb_now, ot_cb_weight = self._compute_ot_cb_now(return_weight=True)
        ot_cb_loss = torch.tensor(0.0, device=self.device)
        if compute_ot_cb_now:
            t0 = time.time()
            ot_cb_loss = self.ot_alignment_loss(type="cb")
            self._log_metric("ot_cb_loss", ot_cb_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            self._log_metric("ot_cb_weighted", ot_cb_weight * ot_cb_loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=False)
            # self._log_perf("ot_cb_loss", time.time() - t0, on_step=True)
        
        loss = (
            batch_nll_loss
            + kl_weight * batch_kl_div_loss
            # + aux_kl_div_loss / self.nll_scale
            + herit_loss_value
            # + aux_reg_loss
            + gene_align_weight * gene_align_loss
            + ot_cs_weight * ot_cs_loss
            + ot_cb_weight * ot_cb_loss
        )

        self._log_metric("loss", loss, batch_size=bs_edges, on_step=True, on_epoch=True, prog_bar=True)
        # self._log_perf("step_total", time.time() - t_all0, on_step=True)

        # Debug: Monitor gradients and embeddings every 10 steps
        if self.current_epoch % 10 == 0 and self.local_step == 0 and self.verbose:
            self._debug_embeddings_only(mu_dict, logstd_dict)

        self.local_step += 1

        return loss

    def on_after_backward(self):
        """Called after loss.backward() and before optimizers are stepped."""

        # Gradient clipping to prevent aux_params from dominating
        # torch.nn.utils.clip_grad_norm_(self.aux_params.parameters(), max_norm=1.0)

        if self.current_epoch % 10 == 0 and self.local_step == 1 and self.verbose:
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
        # batch = batch.to(self.device)
        
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
        if len(weighted_nll_dict) > 1:
            for edge_type, nll_w in weighted_nll_dict.items():
                edge_key = self._edge_type_key(edge_type)
                edge_bs = int(batch[edge_type].edge_index.size(1))
                self._log_metric(f"nll/{edge_key}", nll_w, batch_size=edge_bs, on_step=False, on_epoch=True, prog_bar=False)

        # ---- KL ----
        compute_kl_now, kl_weight = self._compute_kl_now(return_weight=True)
        batch_kl_div_loss = torch.tensor(0.0, device=self.device)
        if compute_kl_now:
            batch_kl_div_loss = self.kl_div_loss(
                batch,
                mu_dict,
                logstd_dict,
                batch.n_id_dict,
                node_weights_dict=self.node_weights_dict,
            )
            self._log_metric("kl", batch_kl_div_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
            self._log_metric("kl_weighted", kl_weight * batch_kl_div_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)

        # ---- Herit Loss ----
        if self.herit_loss is not None:
            if "peak" in batch.node_types:
                pid = batch["peak"].n_id.cpu()
                herit_loss_value = self.herit_loss_lam * self.herit_loss(
                    mu_dict["peak"],
                    pid,
                )
            else:
                herit_loss_value = torch.tensor(0.0, device=self.device)
            self._log_metric("herit_loss", herit_loss_value, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        else:
            herit_loss_value = torch.tensor(0.0, device=self.device)
        
        # ---- Gene Alignment ----
        compute_gene_align_now, gene_align_weight = self._compute_gene_align_now(return_weight=True)

        gene_align_loss = torch.tensor(0.0, device=self.device)
        if compute_gene_align_now:
            gene_align_loss = self.gene_alignment_loss()
            self._log_metric("gene_align", gene_align_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
            self._log_metric("gene_align_weighted", gene_align_weight * gene_align_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)

        # ---- OT loss ----
        compute_ot_cs_now, ot_cs_weight = self._compute_ot_cs_now(return_weight=True)

        ot_cs_loss = torch.tensor(0.0, device=self.device)
        if compute_ot_cs_now:
            t0 = time.time()
            ot_cs_loss = self.ot_alignment_loss(type="cs")
            self._log_metric("ot_cs_loss", ot_cs_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
            self._log_metric("ot_cs_weighted", ot_cs_weight * ot_cs_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
        
        compute_ot_cb_now, ot_cb_weight = self._compute_ot_cb_now(return_weight=True)
        ot_cb_loss = torch.tensor(0.0, device=self.device)
        if compute_ot_cb_now:
            t0 = time.time()
            ot_cb_loss = self.ot_alignment_loss(type="cb")
            self._log_metric("ot_cb_loss", ot_cb_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)
            self._log_metric("ot_cb_weighted", ot_cb_weight * ot_cb_loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=False)

        loss = (
            batch_nll_loss + 
            kl_weight * batch_kl_div_loss + 
            herit_loss_value
            + gene_align_weight * gene_align_loss
            + ot_cs_weight * ot_cs_loss
            + ot_cb_weight * ot_cb_loss
        )

        self._log_metric("loss", loss, batch_size=bs_edges, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(loss)
        self.local_step_val += 1

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
    
    def _compute_kl_now(self, return_weight=False):
        """Decide whether to compute KL loss for this training/validation step."""
        kl_weight = self.kl_lambda * self._linear_warmup_scale(self.kl_n_no, self.kl_n_warmup)
        compute_kl_now = kl_weight > 0
        if return_weight:
            return compute_kl_now, kl_weight
        else:
            return compute_kl_now
    
    def _compute_gene_align_now(self, return_weight=False):
        """Decide whether to compute gene alignment loss for this training/validation step."""
        gene_align_weight = self.gene_align_lambda * self._linear_warmup_scale(self.gene_align_n_no, self.gene_align_n_warmup)
        compute_gene_align_now = False
        if gene_align_weight > 0 and hasattr(self, "_gene_pairs") and self._gene_pairs is not None:
            compute_gene_align_now = True
        if return_weight:
            return compute_gene_align_now, gene_align_weight
        else:
            return compute_gene_align_now

    def _compute_ot_plan(self):
        need_plan = self.current_epoch % self.ot_plan_every_n_epochs == 0
        ot_cs_weight = self.ot_cs_lambda * self._linear_warmup_scale(self.ot_cs_n_no, self.ot_cs_n_warmup)
        ot_cb_weight = self.ot_cb_lambda * self._linear_warmup_scale(self.ot_cb_n_no, self.ot_cb_n_warmup)

        if ot_cs_weight > 0 and need_plan:
            t0 = time.time()
            with torch.no_grad():
                cell_mu = self.encoder.__mu_dict__["cell"].detach()
                self._ot_cs_states = compute_multi_sample_ot_states(
                    cell_mu=cell_mu,
                    groups=self._ot_cs_groups,
                    pairs=self._ot_cs_pairs,
                    subset_size=self.ot_cs_k,
                    eps=self.ot_eps,
                    n_iters=self.ot_iter,
                )
            t1 = time.time()
            self.log("time:ot_cs_plan", t1 - t0, on_step=False, on_epoch=True)
        
        if ot_cb_weight > 0 and need_plan:
            t0 = time.time()
            with torch.no_grad():
                cell_mu = self.encoder.__mu_dict__["cell"].detach()
                self._ot_cb_states = compute_multi_batch_ot_states(
                    cell_mu=cell_mu,
                    groups_sb=self._ot_cb_groups,
                    pairs_sb=self._ot_cb_pairs,
                    subset_size=self.ot_cb_k,
                    eps=self.ot_eps,
                    n_iters=self.ot_iter,
                )
            t1 = time.time()
            self.log("time:ot_cb_plan", t1 - t0, on_step=False, on_epoch=True)

    def _compute_ot_cs_now(self, return_weight=False):
        """Decide whether to compute cross-sample OT loss for this training/validation step."""
        ot_cs_weight = self.ot_cs_lambda * self._linear_warmup_scale(self.ot_cs_n_no, self.ot_cs_n_warmup)
        compute_ot_cs_now = False
        if ot_cs_weight > 0 and hasattr(self, "_ot_cs_states") and self._ot_cs_states is not None and len(self._ot_cs_states) > 0:
            step = self.local_step if self.training else self.local_step_val
            if self.ot_loss_every_n_steps > 0:
                compute_ot_cs_now = (step % self.ot_loss_every_n_steps == 0)
            else:
                # default: once per epoch (first batch)
                compute_ot_cs_now = (step == 0)
        if return_weight:
            return compute_ot_cs_now, ot_cs_weight
        else:
            return compute_ot_cs_now
    
    def _compute_ot_cb_now(self, return_weight=False):
        """Decide whether to compute cross-batch OT loss for this training/validation step."""
        ot_cb_weight = self.ot_cb_lambda * self._linear_warmup_scale(self.ot_cb_n_no, self.ot_cb_n_warmup)
        compute_ot_cb_now = False
        if ot_cb_weight > 0 and hasattr(self, "_ot_cb_states") and self._ot_cb_states is not None and len(self._ot_cb_states) > 0:
            step = self.local_step if self.training else self.local_step_val
            if self.ot_loss_every_n_steps > 0:
                compute_ot_cb_now = (step % self.ot_loss_every_n_steps == 0)
            else:
                # default: once per epoch (first batch)
                compute_ot_cb_now = (step == 0)
        if return_weight:
            return compute_ot_cb_now, ot_cb_weight
        else:
            return compute_ot_cb_now

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
        t0 = time.time()
        self._compute_ot_plan()
        # self._log_perf("ot_plan", time.time() - t0, on_step=False)

    def on_train_batch_start(self, batch, batch_idx):
        if hasattr(self, "t"):
            self.log("data_loading_time", time.time() - self.t)

    def on_train_batch_end(self, output, batch, batch_idx):
        self.t = time.time()

    def on_validation_epoch_start(self):
        self.local_step_val = 0
    
    # def on_validation_epoch_end(self):
    #     # record distance for each gene pair
    #     pairs = self._gene_pairs if hasattr(self, "_gene_pairs") else None
    #     if pairs is not None and hasattr(pairs, "idx0") and pairs.idx0 is not None:
    #         gene_mu = self.encoder.__mu_dict__["gene"]
    #         gene_id = self.data["gene"].gene_id[pairs.idx0]
    #         dist = (gene_mu[pairs.idx0] - gene_mu[pairs.idx1]).pow(2).sum(dim=-1).sqrt()

    #         gene_id = gene_id.detach().cpu().numpy()
    #         dist = dist.detach().cpu().numpy()

    #         col = f"dist_{self.current_epoch}"
    #         if not hasattr(self, "_gene_align_dist_cols"):
    #             self._gene_align_dist_cols = {}
            
    #         self._gene_align_dist_cols[col] = pd.Series(dist, index=gene_id, name=col)

    # def on_fit_end(self):
    #     if not hasattr(self, "_gene_align_dist_cols"):
    #         return
    #     df = pd.concat(self._gene_align_dist_cols.values(), axis=1)

    #     log_dir = None
    #     if self.trainer is not None:
    #         log_dir = getattr(self.trainer, "log_dir", None)
    #     if log_dir is None and getattr(self, "logger", None) is not None:
    #         log_dir = getattr(self.logger, "log_dir", None)
    #     if log_dir is None:
    #         log_dir = "."
        
    #     os.makedirs(log_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = os.path.join(log_dir, f"gene_align_dist_df_{timestamp}.csv")
    #     df.to_csv(filename)
    #     self.logger2.info(f"Gene alignment distance dataframe saved to {filename}")

    def configure_optimizers(self):
        # Different learning rates for encoder vs aux_params
        all_params = list(self.parameters())
        encoder_params = list(self.encoder.parameters())
        aux_params = list(self.aux_params.parameters())
        assert len(encoder_params) + len(aux_params) == len(all_params), "Parameters other than encoder and aux_params exist in the model" # sanity check

        optimizer = torch.optim.Adam(
            # currently there are only encoder and aux parameters, no decoder parameters
            [
                {"params": encoder_params, "lr": self.learning_rate},
                {
                    "params": aux_params,
                    "lr": self.learning_rate * 0.1,
                },  # 10x smaller LR for aux params
            ] if self.use_aux else
            [{"params": encoder_params, "lr": self.learning_rate}]
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
