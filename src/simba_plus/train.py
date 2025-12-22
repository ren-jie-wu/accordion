import os
import time
import logging
from typing import Optional
import pandas as pd
from datetime import datetime
import argparse
import warnings
import pickle as pkl
import math

warnings.simplefilter(action="ignore", category=FutureWarning)
import anndata as ad
import numpy as np
from simba_plus.model_prox import LightningProxModel
import torch
import torch_geometric
from torch_geometric.transforms.to_device import ToDevice
import lightning as L
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    OnExceptionCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import wandb
import scanpy as sc
import simba_plus.load_data
from simba_plus.encoders import TransEncoder
from simba_plus.losses import HSIC, SumstatResidualLoss

from simba_plus.utils import (
    MyEarlyStopping,
    get_edge_split_datamodule,
    get_node_weights,
    get_nll_scales,
)
from simba_plus.heritability.get_residual import (
    get_residual,
    get_peak_residual,
)
from simba_plus.evaluate import evaluate_model, pretty_print
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
torch_geometric.seed_everything(2026)
# https://pytorch-lightning.readthedocs.io/en/stable/model/train_model_basic.html


def setup_logging(checkpoint_dir):
    """Setup logging to both file and console"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(checkpoint_dir, f"train_{timestamp}.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with detailed formatting
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)

    # Console handler with simpler formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)

    # Add both handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_run_id(
    batch_size: int = 1_000_000,
    n_batch_sampling: int = 1,
    data_path: str = None,
    hidden_dims: int = 50,
    hsic_lam: float = 0.0,
    edgetype_specific: bool = True,
    nonneg: bool = False,
    sumstats: Optional[str] = None,
    sumstats_lam: float = 1.0,
    **kwargs,
):
    return f"simba+{os.path.basename(data_path).split('_hetdata.dat')[0]}_{human_format(batch_size)}{'x'+str(n_batch_sampling) if n_batch_sampling > 1 else ''}{'.indep2_' + format(hsic_lam, '1.0e') if hsic_lam != 0 else ''}{os.path.basename(sumstats).split('.txt')[0] if sumstats is not None else ''}{'_'+format(sumstats_lam, '1.0e') if sumstats is not None and sumstats_lam != 1.0 else ''}{'.d' + str(hidden_dims) if hidden_dims != 50 else ''}{'.en' if not edgetype_specific else ''}{'.nn' if nonneg else ''}.randinit"


def run(
    batch_size: int = 1_000_000,
    n_batch_sampling: int = 1,
    output_dir: str = "rna",
    data_path: str = None,
    load_checkpoint: bool = False,
    checkpoint_suffix: str = "",
    num_workers: int = 2,
    n_no_kl: int = 0,
    n_kl_warmup: int = 10,
    hidden_dims: int = 50,
    hsic_lam: float = 0.0,
    edgetype_specific: bool = True,
    nonneg: bool = False,
    adata_CG: str = None,
    adata_CP: str = None,
    get_adata: bool = False,
    ldsc_res: Optional[pd.DataFrame] = None,
    sumstats: Optional[str] = None,
    sumstats_lam: float = 1.0,
    early_stopping_steps: int = 10,
    max_epochs: int = 1000,
    verbose: bool = False,
    negative_sampling_fold: int = 1,
    batch_negative: bool = True,
    no_wandb: bool = False,

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
    """
    Train the model with the given parameters.

    Args:
        # can be specified in CLI
        data_path (str): The path to the data file (hetdata.dat or similar).
        adata_CG (str): The path to the cell adata file. Only used to provide cell metadata and gene metadata.
        adata_CP (str): The path to the peak adata file. Only used to provide cell metadata (if not provided in adata_CG) and peak metadata.
        batch_size (int): The batch size of edges for the DataLoader. If batch_negative is True, this is the number of positive edges per batch.
        batch_negative (bool, default=True): If True, datamodule construction will not do negative sampling; instead, negative sampling will be done in each training step and validation step.
        output_dir (str): The output directory.
        sumstats (str): The path to a TSV file with trait name and path to GWAS summary statistics file per line.
        sumstats_lam (float): The weight of the loss for sumstats residual.
        negative_sampling_fold (int, default=1): The ratio of negative samples to positive samples when constructing the datamodule. If 0, datamodule construction will not do negative sampling.
        load_checkpoint (bool, default=False): Whether to load the checkpoint. If True, model as well as best learning rate found previously will be loaded from the last checkpoint
        checkpoint_suffix (str, default=""): The suffix of the checkpoint filename. The full path should be f"{checkpoint_dir}/last{checkpoint_suffix}.ckpt"
        hidden_dims (int): The number of hidden dimensions.
        hsic_lam (float): The HSIC regularization lambda. Will be scaled by the number of batches.
        get_adata (bool, default=False): If True, only fetch the last checkpoint, save AnnData outputs and exit.
        num_workers (int, default=2): The number of workers for data loading and LDSC.
        early_stopping_steps (int, default=10): The number of epochs for early stopping patience.
        max_epochs (int, default=1000): The number of max epochs for training.
        verbose (bool, default=False): If True, enables verbose logging.
        
        # other parameters
        n_batch_sampling (int): Dummy parameter for now.
        n_no_kl (int): ... passed to LightningProxModel.
        n_kl_warmup (int): ... passed to LightningProxModel.
        edgetype_specific (bool): Whether to use different parameters for the same node type but different edge types. Passed to LightningProxModel.
        nonneg (bool): Passed to LightningProxModel but not used for now.
        ldsc_res (pd.DataFrame): per-SNP LD score regression residuals from get_residual(). Used to promote peak loading explaining GWAS residual.
        
        # multi-align related parameters
        lambda_gene_align (float): Weight of the gene alignment loss.
        gene_align_n_no (int): Number of epochs to wait before starting gene alignment.
        gene_align_n_warmup (int): Number of epochs for gene alignment warmup.
        lambda_ot (float): Weight of the Optimal Transportation loss (among cells).
        ot_n_no (int): Number of epochs to wait before starting Optimal Transportation loss.
        ot_n_warmup (int): Number of epochs for Optimal Transportation loss warmup.
        ot_k (int): Number of cells to sample for Optimal Transportation loss.
        ot_eps (float): Epsilon for Optimal Transportation loss.
        ot_iter (int): Number of iterations for Optimal Transportation loss.

    Returns:
        None
    """

    # I/O preparation

    run_id = get_run_id(
        batch_size=batch_size,
        n_batch_sampling=n_batch_sampling,
        data_path=data_path,
        hidden_dims=hidden_dims,
        hsic_lam=hsic_lam,
        edgetype_specific=edgetype_specific,
        nonneg=nonneg,
        sumstats=sumstats,
        sumstats_lam=sumstats_lam,
    )
    prefix = os.path.normpath(output_dir)
    checkpoint_dir = os.path.join(prefix, f"{run_id}.checkpoints/")
    logger = setup_logging(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Using {device} as device...")
    # torch.set_default_device(device)
    data = simba_plus.load_data.load_from_path(data_path, device=device)
    data.generate_ids()
    for node_type in data.node_types:
        del data[node_type].x

    logger.info(f"Data loaded to {data['cell'].n_id.device}: {data}")
    dim_u = hidden_dims

    # Only get adata from the last checkpoint
    if get_adata:
        logger.info(
            "With `--get-adata` flag, only getting adata output from the last checkpoint..."
        )
        save_files(
            os.path.join(prefix, run_id),
            cell_adata=adata_CG,
            peak_adata=adata_CP,
            checkpoint_suffix=checkpoint_suffix,
            logger=logger,
        )
        last_model_path = f"{checkpoint_dir}/last{checkpoint_suffix}.ckpt"
        return last_model_path, logger

    edge_types = data.edge_types
    if "peak" in data.node_types and "gene" in data.node_types:
        edge_types = [("cell", "has_accessible", "peak"), ("cell", "expresses", "gene")]

    # construct train/val dataloaders; batch_negative=True won't do negative sampling
    pldata = get_edge_split_datamodule(
        data=data,  # ToDevice(device)(data),
        data_path=data_path,
        edge_types=edge_types,
        batch_size=batch_size,
        num_workers=num_workers,
        negative_sampling_fold=0 if batch_negative else negative_sampling_fold,
        logger=logger,
    )

    # calculate some model parameters
    node_weights_dict = get_node_weights(
        data=data,
        pldata=pldata,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        device=device,
    )

    # n_batches and n_val_batches are actually not used in get_nll_scales()
    n_batches = math.ceil(pldata.train_loader.dataset.total_length / batch_size)
    logger.info(f"@N_BATCHES:{n_batches}")
    n_val_batches = math.ceil(pldata.val_loader.dataset.total_length / batch_size)
    nll_scale, val_nll_scale = get_nll_scales(
        data=data,
        pldata=pldata,
        edge_types=edge_types,
        batch_size=batch_size,
        n_batches=n_batches,
        n_val_batches=n_val_batches,
        logger=logger,
    )

    # For now I skip the HSIC part and thus have not checked it. #TODO
    if hsic_lam != 0:
        # cell_edge_types = []
        # for edge_type in edge_types:
        #     if edge_type[0] == "cell" or edge_type[1] == "cell":
        #         cell_edge_types.append(edge_type)
        hsic_lam = n_batches * hsic_lam
        logger.info(f"lambda_HSIC= {hsic_lam}")
        hsic = HSIC(
            subset_samples=min(3000, max(1000, data["cell"].num_nodes // n_batches)),
            lam=hsic_lam,
        )
    else:
        hsic = None
    if ldsc_res is not None:
        peak_res = get_peak_residual(ldsc_res, adata_CP, checkpoint_dir, logger)
        herit_loss = SumstatResidualLoss(
            peak_res, device, n_factors=dim_u, lam=sumstats_lam
        )
    else:
        herit_loss = None

    rpvgae = LightningProxModel(
        data,
        encoder_class=TransEncoder,
        n_latent_dims=dim_u,
        device=device,
        edgetype_specific=edgetype_specific,
        hsic=hsic, # calculated
        herit_loss=herit_loss, # calculated
        herit_loss_lam=sumstats_lam,
        n_no_kl=n_no_kl,
        n_kl_warmup=n_kl_warmup,
        nll_scale=nll_scale, # calculated
        val_nll_scale=val_nll_scale, # calculated
        node_weights_dict=node_weights_dict, # calculated
        nonneg=nonneg,
        logger=logger,
        verbose=verbose,
        num_neg_samples_fold=negative_sampling_fold,
        batch_negative=batch_negative,

        lambda_gene_align=lambda_gene_align,
        gene_align_n_no=gene_align_n_no,
        gene_align_n_warmup=gene_align_n_warmup,
        lambda_ot=lambda_ot,
        ot_n_no=ot_n_no,
        ot_n_warmup=ot_n_warmup,
        ot_k=ot_k,
        ot_eps=ot_eps,
        ot_iter=ot_iter,
    ).to(device)

    def train(
        rpvgae,
        early_stopping_steps=early_stopping_steps,
        run_id="",
        use_wandb=True,
    ):
        # prepare wandb logger
        code_directory_path = os.path.dirname(os.path.abspath(__file__))

        if use_wandb:
            wandb_logger = WandbLogger(project=f"pyg_simba_{output_dir.replace('/', '_')}")
            code_artifact = wandb.Artifact("my-python-code", type="code")
            code_artifact.add_dir(code_directory_path)
            wandb_logger.experiment.log_artifact(code_artifact)
            wandb_logger.experiment.config.update(
                {
                    "run_id": run_id,
                    "batch_size": batch_size,
                    "n_no_kl": n_no_kl,
                    "n_kl_warmup": n_kl_warmup,
                    "early_stopping_steps": early_stopping_steps,
                    "HSIC_loss": hsic_lam,
                    "negative_sampling_fold": negative_sampling_fold,
                    "edgetype_specific": edgetype_specific,
                }
            )
            pl_logger = wandb_logger
        else:
            logger.info("W&B logging disabled; using CSV logger instead.")
            pl_logger = CSVLogger(save_dir=checkpoint_dir, name="pl_logs")

        # prepare callbacks
        early_stopping_callback = MyEarlyStopping(
            monitor="val_nll_loss",
            mode="min",
            strict=True,
            patience=early_stopping_steps,
        )
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            monitor="val_nll_loss",
            mode="min",
            save_last=True,
        )
        lrmonitor_callback = LearningRateMonitor()
        onexception_callback = OnExceptionCheckpoint(checkpoint_dir)
        
        # core process: Tuner(trainer).lr_find() -> trainer.fit()
        trainer = L.Trainer(
            callbacks=[
                early_stopping_callback,
                checkpoint_callback,
                lrmonitor_callback,
                # onexception_callback,
            ],
            logger=pl_logger,
            devices=1,
            default_root_dir=checkpoint_dir,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=5,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            max_epochs=max_epochs,
        )
        if not load_checkpoint:
            tuner = Tuner(trainer)
            tuner.lr_find(
                rpvgae,
                pldata,
                min_lr=0.001,
                max_lr=0.01,
                num_training=30,
                early_stop_threshold=None,
            )
            # rpvgae.learning_rate = 0.1

            logger.info(f"@TRAIN: LR={rpvgae.learning_rate}")
        # trainer.validate(rpvgae, datamodule=pldata)
        trainer.fit(
            model=rpvgae,
            datamodule=pldata,
            ckpt_path=(
                f"{checkpoint_dir}/last{checkpoint_suffix}.ckpt"
                if load_checkpoint
                else None
            ),
        )
        return checkpoint_callback.last_model_path

    last_model_path = train(
        rpvgae,
        run_id=run_id,
        use_wandb=not no_wandb,
    )
    torch.save(rpvgae.state_dict(), os.path.join(prefix, f"{run_id}.model"))
    save_files(
        os.path.join(prefix, run_id),
        last_model_path,
        cell_adata=adata_CG,
        peak_adata=adata_CP,
        logger=logger,
    )
    return last_model_path, logger

    logger.info("Starting evaluation with test data...")
    run_eval()


def human_format(num):
    """
    Convert a number to a human-readable format (e.g. 1000 -> 1K, 1000000 -> 1M)

    Args:
        num (int): The number to convert.
    
    Returns:
        str: The human-readable format of the number.
    """
    num = float(f"{num:.3g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{str(num).rstrip('0').rstrip('.')}{['', 'K', 'M', 'B', 'T'][magnitude]}"


def save_files(
    run_id: str,
    last_model_path: str = None,
    cell_adata: str = None,
    peak_adata: str = None,
    checkpoint_suffix: str = "",
    logger=None,
):
    """
    Save the model parameters and the adata files (cell, peak, and gene embeddings) from the last checkpoint to the checkpoint directory.
    
    Args:
        run_id (str): The run ID (with prefix). Used to load the last checkpoint and save the model parameters and adata files.
        cell_adata (str): The path to the cell-gene adata file. Optional. Only used to provide cell metadata and gene metadata.
        peak_adata (str): The path to the cell-peak adata file. Optional. Only used to provide cell metadata (if not provided in cell_adata) and peak metadata.
    """
    adata_CG = ad.read_h5ad(cell_adata) if cell_adata is not None else None
    adata_CP = ad.read_h5ad(peak_adata) if peak_adata is not None else None
    if last_model_path is None:
        last_model_path = f"{run_id}.checkpoints/last{checkpoint_suffix}.ckpt"
    else:
        checkpoint_suffix = (
            os.path.basename(last_model_path).split("last")[1].split(".ckpt")[0]
        )
    logger.info(
        f"Saving model outputs from checkpoint {last_model_path} into into AnnDatas..."
    )
    model = LightningProxModel.load_from_checkpoint(
        last_model_path,
        weights_only=False,
        map_location="cpu",
    )
    
    # Save the scale and bias of the peak/gene and cell
    if "peak" in model.encoder.__mu_dict__:
        np.save(
            f"{run_id}.checkpoints/peak_logscale.npy",
            model.aux_params.logscale_dict["peak__cell_has_accessible_peak"]
            .detach()
            .numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/peak_bias.npy",
            model.aux_params.bias_dict["peak__cell_has_accessible_peak"]
            .detach()
            .numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_peak_logscale.npy",
            model.aux_params.logscale_dict["cell__cell_has_accessible_peak"]
            .detach()
            .numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_peak_bias.npy",
            model.aux_params.bias_dict["cell__cell_has_accessible_peak"]
            .detach()
            .numpy(),
        )
    if "gene" in model.encoder.__mu_dict__:
        np.save(
            f"{run_id}.checkpoints/gene_logscale.npy",
            model.aux_params.logscale_dict["gene__cell_expresses_gene"]
            .detach()
            .numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/gene_bias.npy",
            model.aux_params.bias_dict["gene__cell_expresses_gene"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_gene_logscale.npy",
            model.aux_params.logscale_dict["cell__cell_expresses_gene"]
            .detach()
            .numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_gene_bias.npy",
            model.aux_params.bias_dict["cell__cell_expresses_gene"].detach().numpy(),
        )
    
    # Save the cell, peak, and gene embedding adata and do umap
    cell_x = model.encoder.__mu_dict__["cell"].detach().cpu()
    obs_use = None
    if adata_CG is not None:
        obs_use = adata_CG.obs
    elif adata_CP is not None:
        obs_use = adata_CP.obs
    adata_C = ad.AnnData(
        X=cell_x.numpy(),
        layers={
            "X_logstd": model.encoder.__logstd_dict__["cell"].detach().cpu().numpy()
        },
        obs=obs_use,
        obsm={"X_pca": cell_x.numpy()},
    )
    sc.pp.neighbors(adata_C)
    sc.tl.umap(adata_C, random_state=2025)
    
    if "peak" in model.encoder.__mu_dict__:
        adata_P = ad.AnnData(
            X=model.encoder.__mu_dict__["peak"].detach().cpu().numpy(),
            layers={
                "X_logstd": model.encoder.__logstd_dict__["peak"].detach().cpu().numpy()
            },
            obs=adata_CP.var if adata_CP is not None else None,
        )
        adata_P.write(f"{run_id}.checkpoints/adata_P{checkpoint_suffix}.h5ad")
    
    if "gene" in model.encoder.__mu_dict__:
        adata_G = ad.AnnData(
            X=model.encoder.__mu_dict__["gene"].detach().cpu().numpy(),
            layers={
                "X_logstd": model.encoder.__logstd_dict__["gene"].detach().cpu().numpy()
            },
            obs=adata_CG.var if adata_CG is not None else None,
        )
        adata_G.write(f"{run_id}.checkpoints/adata_G{checkpoint_suffix}.h5ad")
    adata_C.write(f"{run_id}.checkpoints/adata_C{checkpoint_suffix}.h5ad")

    logger.info(
        f"Saved AnnDatas into {run_id}.checkpoints/adata_{{C,P,G}}{checkpoint_suffix}.h5ad"
    )


def run_eval(args, last_model_path, logger):
    logger.info("Starting evaluation with test data...")
    data_idx_path = f"{args.data_path.split('.dat')[0]}_data_idx.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_dict = evaluate_model(
        model_path=last_model_path,
        data_path=args.data_path,
        index_path=data_idx_path,
        batch_size=args.batch_size,
        negative_sampling_fold=args.negative_sampling_fold,
        # device=args.device,
        logger=logger,
    )
    pretty_print(metric_dict, logger=logger)
    with open(f"{os.path.dirname(last_model_path)}/pred_dict.pkl", "wb") as file:
        pkl.dump(metric_dict, file)


def main(args):
    check_args(args)
    kwargs = vars(args)
    del kwargs["subcommand"]
    if args.sumstats is not None:
        residuals = get_residual(
            sumstat_list_path=args.sumstats,
            output_path=f"{os.path.dirname(args.sumstats)}/ldsc_residuals/",
            rerun=False,
            nprocs=args.num_workers,
        )
        kwargs["ldsc_res"] = residuals

    last_model_path, logger = run(**kwargs)
    run_eval(args, last_model_path, logger)


def add_argument(parser):
    parser.description = "Train SIMBA+ model on the given HetData object."
    parser.add_argument(
        "data_path", help="Path to the input data file (hetdata.dat or similar)"
    )
    parser.add_argument(
        "--adata-CG",
        type=str,
        default=None,
        help="Path to gene AnnData (.h5ad) file for fetching cell/gene metadata. Output adata_G.h5ad will have no .obs attribute if not provided.",
    )
    parser.add_argument(
        "--adata-CP",
        type=str,
        default=None,
        help="Path to peak/ATAC AnnData (.h5ad) file for fetching cell/peak metadata. Output adata_G.h5ad will have no .obs attribute if not provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size (number of edges) per DataLoader batch",
    )
    parser.add_argument(
        "--batch-negative",
        action="store_true",
        help="Batch size (number of edges) per DataLoader batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Top-level output directory where run artifacts will be stored",
    )
    parser.add_argument(
        "--sumstats",
        type=str,
        default=None,
        help="If provided, LDSC is run so that peak loading maximally explains the residual of LD score regression of summary statistics.\nProvide a TSV file with one trait name and path to summary statistics file per line.",
    )
    parser.add_argument(
        "--sumstats-lam",
        type=float,
        default=1.0,
        help="If provided with `sumstats`, weights the MSE loss for sumstat residuals.",
    )
    parser.add_argument(
        "--negative-sampling-fold",
        type=int,
        default=1,
        help="Fold of negative samples to use during training",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="If set, resume training from the last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help="Append a suffix to checkpoint filenames (last{suffix}.ckpt)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        default=50,
        help="Dimensionality of latent embeddings",
    )
    parser.add_argument(
        "--hsic-lam",
        type=float,
        default=0.0,
        help="HSIC regularization lambda (strength)",
    )
    parser.add_argument(
        "--get-adata",
        action="store_true",
        help="Only extract and save AnnData outputs from the last checkpoint and exit",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Number of worker processes for data loading and LDSC",
    )
    parser.add_argument(
        "--early-stopping-steps",
        type=int,
        default=3,
        help="Number of epoch for early stopping patience",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Number of max epochs for training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, enables verbose logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (recommended for CI/tests).",
    )

    # multi-align related parameters
    parser.add_argument(
        "--lambda-gene-align",
        type=float,
        default=0.0,
        help="Weight of the gene alignment loss",
    )
    parser.add_argument(
        "--gene-align-n-no",
        type=int,
        default=0,
        help="Number of epochs to wait before starting gene alignment",
    )
    parser.add_argument(
        "--gene-align-n-warmup",
        type=int,
        default=10,
        help="Number of epochs for gene alignment warmup",
    )
    parser.add_argument(
        "--lambda-ot",
        type=float,
        default=0.0,
        help="Weight of the Optimal Transportation loss (among cells)",
    )
    parser.add_argument(
        "--ot-n-no",
        type=int,
        default=15,
        help="Number of epochs to wait before starting Optimal Transportation loss",
    )
    parser.add_argument(
        "--ot-n-warmup",
        type=int,
        default=30,
        help="Number of epochs for Optimal Transportation loss warmup",
    )
    parser.add_argument(
        "--ot-k",
        type=int,
        default=256,
        help="Subsample size for Optimal Transportation loss",
    )
    parser.add_argument(
        "--ot-eps",
        type=float,
        default=0.05,
        help="Regularization parameter for Optimal Transportation loss",
    )
    parser.add_argument(
        "--ot-iter",
        type=int,
        default=50,
        help="Number of iterations for Optimal Transportation loss",
    )
    return parser


def check_args(args):
    if args.sumstats is not None:
        if not os.path.exists(args.sumstats):
            raise ValueError(
                f"Summary statistics file --sumstats {args.sumstats} not found."
            )
        if args.adata_CP is None:
            raise ValueError("--adata-CP must be provided when using --sumstats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
