import os
import logging
from datetime import datetime
from itertools import chain
import argparse
import pickle as pkl
from tqdm import tqdm
import anndata as ad
import numpy as np
from simba_plus.model_prox import LightningProxModel
import torch
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.transforms.to_device import ToDevice
import lightning as L
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import scanpy as sc
import simba_plus.load_data
from simba_plus.encoders import TransEncoder
from simba_plus.losses import HSIC
from simba_plus.loader import CustomIndexDataset
from simba_plus.utils import MyEarlyStopping, negative_sampling
import torch.multiprocessing


torch.multiprocessing.set_sharing_strategy("file_system")
torch_geometric.seed_everything(2025)
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


def run(
    batch_size: int = 1_000_000,
    n_batch_sampling: int = 1,
    output_dir: str = "rna",
    data_path: str = None,
    load_checkpoint: bool = False,
    checkpoint_suffix: str = "",
    reweight_rarecell: bool = False,
    n_kl_warmup: int = 10,
    project_decoder: bool = False,
    hidden_dims: int = 50,
    hsic_lam: float = 0.0,
    edgetype_specific_scale: bool = True,
    edgetype_specific_std: bool = True,
    edgetype_specific_bias: bool = True,
    nonneg: bool = False,
    adata_CG: str = None,
    adata_CP: str = None,
    get_adata: bool = False,
    pos_scale: bool = False,
    scale_src: bool = True,
):
    """Train the model with the given parameters.
    If get_adata is True, it will only load the gene/peak/cell AnnData object from the checkpoint.
    Parameters
    ----------

    """
    if pos_scale and scale_src:
        scale_tag = ".ps"
    elif pos_scale:
        scale_tag = ".pd"
    elif scale_src:
        scale_tag = ".d"
    else:
        scale_tag = ""
    run_id = f"pl_{os.path.basename(data_path).split('_HetData.dat')[0]}_{human_format(batch_size)}{'x'+str(n_batch_sampling) if n_batch_sampling > 1 else ''}_prox{'.noproj' if not project_decoder else ''}{'.rw' if reweight_rarecell else ''}{'.indep2_' + format(hsic_lam, '1.0e') if hsic_lam != 0 else ''}{'.d' + str(hidden_dims) if hidden_dims != 50 else ''}{'.enss' if not edgetype_specific_scale else ''}{'.enst' if not edgetype_specific_std else ''}{'.ensb' if not edgetype_specific_bias else ''}{'.nn' if nonneg else ''}{scale_tag}.randinit"

    prefix = f"/data/pinello/PROJECTS/2022_12_GCPA/runs/{output_dir}/"
    checkpoint_dir = f"{prefix}/{run_id}.checkpoints/"
    logger = setup_logging(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Using {device} as device...")
    # torch.set_default_device(device)
    data = simba_plus.load_data.load_from_path(data_path)
    logger.info(f"Data loaded to {data['cell'].x.device}: {data}")
    dim_u = hidden_dims
    num_neg_samples_fold = 1

    if get_adata:
        logger.info(
            "With `--get-adata` flag, only getting adata output from the last checkpoint..."
        )
        save_files(
            f"{prefix}{run_id}",
            cell_adata=adata_CG,
            peak_adata=adata_CP,
            checkpoint_suffix=checkpoint_suffix,
        )
        return

    loss_df = None
    counts_dicts = None

    edge_types = data.edge_types
    if "multiome" in data_path and "motif" in data.node_types:
        edge_types = [("cell", "has_accessible", "peak"), ("cell", "expresses", "gene")]

    torch.manual_seed(2025)
    data_idx_path = f"{checkpoint_dir}/data_idx.pkl"
    train_data_dict = {}
    val_data_dict = {}
    test_data_dict = {}
    if os.path.exists(data_idx_path):
        # Load existing train/val/test split
        with open(data_idx_path, "rb") as f:
            saved_splits = pkl.load(f)
            val_edge_index_dict = saved_splits["val"]
            test_edge_index_dict = saved_splits["test"]
            # Reconstruct train indices as complement of val+test
            train_edge_index_dict = {}
            for edge_type in edge_types:
                edge_key = "__".join(edge_type)
                all_edges = set(range(data[edge_type].num_edges))
                val_test_edges = set(
                    val_edge_index_dict[edge_key].tolist()
                    + test_edge_index_dict[edge_key].tolist()
                )
                train_edges = list(all_edges - val_test_edges)
                train_edge_index_dict[edge_key] = torch.tensor(train_edges)
                train_data_dict[edge_type] = CustomIndexDataset(
                    edge_type, train_edge_index_dict[edge_key]
                )
                val_data_dict[edge_type] = CustomIndexDataset(
                    edge_type, val_edge_index_dict[edge_key]
                )
                test_data_dict[edge_type] = CustomIndexDataset(
                    edge_type, test_edge_index_dict[edge_key]
                )

    else:
        for edge_type in edge_types:
            num_edges = data[edge_type].num_edges
            edge_index = data[edge_type].edge_index
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # Find indices that cover all source and target nodes
            selected_indices = set()

            indices = torch.arange(num_edges)[torch.randperm(num_edges)]
            logger.info("Selecting source node")
            selected_indices.update(
                np.unique(src_nodes[indices].cpu().numpy(), return_index=True)[
                    1
                ].tolist()
            )
            logger.info("Selecting destination node")
            selected_indices.update(
                np.unique(dst_nodes[indices].cpu().numpy(), return_index=True)[
                    1
                ].tolist()
            )
            logger.info("Selected indices")

            # Fill up to 90% for train, 5% for val
            remaining_indices = [
                i for i in indices.cpu().numpy() if i not in selected_indices
            ]
            logger.info("Got remaining indices")
            train_size = int(num_edges * 0.9)
            val_size = int(num_edges * 0.05)
            selected_indices = list(selected_indices)
            train_index = torch.tensor(
                selected_indices
                + remaining_indices[: train_size - len(selected_indices)]
            )
            val_index = torch.tensor(
                remaining_indices[
                    (train_size - len(selected_indices)) : (
                        train_size - len(selected_indices) + val_size
                    )
                ]
            )
            test_index = torch.tensor(
                remaining_indices[(train_size - len(selected_indices) + val_size) :]
            )

            train_data_dict[edge_type] = CustomIndexDataset(edge_type, train_index)
            val_data_dict[edge_type] = CustomIndexDataset(edge_type, val_index)
            test_data_dict[edge_type] = CustomIndexDataset(edge_type, test_index)
        train_edge_index_dict = {
            "__".join(edge_type): train_data_dict[edge_type].index
            for edge_type in edge_types
        }
        val_edge_index_dict = {
            "__".join(edge_type): val_data_dict[edge_type].index
            for edge_type in edge_types
        }
        test_edge_index_dict = {
            "__".join(edge_type): test_data_dict[edge_type].index
            for edge_type in edge_types
        }
        with open(data_idx_path, "wb") as f:
            pkl.dump({"val": val_edge_index_dict, "test": test_edge_index_dict}, f)

    def collate(data):
        types, idxs = zip(*data)
        return tuple(types[0]), torch.tensor(idxs)

    train_loaders = [
        DataLoader(
            train_data_dict[edgetype],
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=10,
        )
        for edgetype in edge_types
    ]
    val_loaders = [
        DataLoader(
            val_data_dict[edgetype],
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=10,
        )
        for edgetype in edge_types
    ]

    node_counts_dict = {
        node_type: torch.ones(x.shape[0], device=device)
        for (node_type, x) in data.x_dict.items()
    }

    # Try to load previously saved node weights so training can resume with same reweighting
    node_weights_path = os.path.join(checkpoint_dir, "node_weights_dict.pt")
    if os.path.exists(node_weights_path):
        try:
            loaded = torch.load(node_weights_path, map_location="cpu")
            # Convert loaded values to device tensors if needed
            node_weights_dict = {
                k: (
                    v.to(device)
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, device=device)
                )
                for k, v in loaded.items()
            }
            logger.info(f"Loaded node_weights_dict from {node_weights_path}")
        except Exception as e:  # pragma: no cover - best-effort load
            logger.info(
                f"Failed to load node_weights_dict from {node_weights_path}: {e}"
            )
    else:
        for train_loader in train_loaders:
            for batch in tqdm(train_loader):
                edge_type, label_index = batch
                edge_type = tuple(edge_type)
                batch = data.edge_subgraph({edge_type: label_index}).to(device)
                src, _, dst = edge_type

        node_weights_dict = {k: 1.0 / v for k, v in node_counts_dict.items()}
        torch.save(node_weights_dict, node_weights_path)

    class MyDataModule(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.train_loaders = train_loaders
            self.val_loaders = val_loaders

        def train_dataloader(self):
            return chain.from_iterable(self.train_loaders)

        def val_dataloader(self):
            return chain.from_iterable(self.val_loaders)

    pldata = MyDataModule()

    n_batches = sum([len(t) for t in train_loaders])
    n_val_batches = sum([len(t) for t in val_loaders])
    logger.info(f"@N_BATCHES:{n_batches}")

    n_dense_edges = 0
    for src_nodetype, _, dst_nodetype in edge_types:
        n_dense_edges += data[src_nodetype].num_nodes * data[dst_nodetype].num_nodes

    nll_scale = n_dense_edges / ((num_neg_samples_fold + 1) * batch_size * n_batches)
    val_nll_scale = n_dense_edges / (
        (num_neg_samples_fold + 1) * batch_size * n_val_batches
    )
    logger.info(f"NLLSCALE:{nll_scale}, {val_nll_scale}")

    if hsic_lam != 0:
        cell_edge_types = []
        for edge_type in edge_types:
            if edge_type[0] == "cell" or edge_type[1] == "cell":
                cell_edge_types.append(edge_type)
        hsic_lam = n_batches * hsic_lam
        logger.info(f"lambda_HSIC= {hsic_lam}")
        hsic = HSIC(
            subset_samples=min(3000, max(1000, data["cell"].num_nodes // n_batches)),
            lam=hsic_lam,
        )
    else:
        hsic = None

    rpvgae = LightningProxModel(
        data,
        encoder_class=TransEncoder,
        n_hidden_dims=dim_u,
        n_latent_dims=dim_u,
        device=device,
        num_neg_samples_fold=num_neg_samples_fold,
        project_decoder=project_decoder,
        edgetype_specific_scale=edgetype_specific_scale,
        edgetype_specific_std=edgetype_specific_std,
        edgetype_specific_bias=edgetype_specific_bias,
        hsic=hsic,
        n_no_kl=1,
        n_count_nodes=20,
        n_kl_warmup=n_kl_warmup,
        nll_scale=nll_scale,
        val_nll_scale=val_nll_scale,
        node_weights_dict=node_weights_dict,
        nonneg=nonneg,
        positive_scale=pos_scale,
        train_data_dict=train_edge_index_dict,
        val_data_dict=val_edge_index_dict,
        decoder_scale_src=scale_src,
    ).to(device)

    def train(
        rpvgae,
        n_epochs=1000,
        n_no_kl=1,
        n_count_nodes=20,
        n_kl_warmup=50,
        early_stopping_steps=10,
        loss_df=None,
        counts_dicts=None,
        hsic=None,
        lr=1e-3,
        run_id="",
    ):

        wandb_logger = WandbLogger(project=f"pyg_simba_{output_dir.replace('/', '_')}")
        wandb_logger.experiment.config.update(
            {
                "run_id": run_id,
                "batch_size": batch_size,
                "n_no_kl": n_no_kl,
                "n_kl_warmup": n_kl_warmup,
                "n_count_nodes": n_count_nodes,
                "early_stopping_steps": early_stopping_steps,
                "HSIC_loss": hsic_lam,
                "num_neg_samples_fold": num_neg_samples_fold,
                "edgetype_specific_scale": edgetype_specific_scale,
                "edgetype_specific_std": edgetype_specific_std,
                "scale_on_mu_scale": True,
                "scale_only_cell": False,
            }
        )
        early_stopping_callback = MyEarlyStopping(
            monitor="val_nll_loss_monitored",
            mode="min",
            strict=True,
            patience=early_stopping_steps,
        )
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            monitor="val_nll_loss_monitored",
            mode="min",
            save_last=True,
        )
        lrmonitor_callback = LearningRateMonitor()
        trainer = L.Trainer(
            callbacks=[
                early_stopping_callback,
                checkpoint_callback,
                lrmonitor_callback,
            ],
            logger=wandb_logger,
            devices=1,
            default_root_dir=checkpoint_dir,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1,
            check_val_every_n_epoch=1,
        )
        if not load_checkpoint:
            tuner = Tuner(trainer)
            tuner.lr_find(
                rpvgae,
                pldata,
                # min_lr=1e-5,
                max_lr=0.01,
                num_training=30,
                early_stop_threshold=None,
            )
            logger.info(f"@TRAIN: LR={rpvgae.learning_rate}")
        trainer.fit(
            model=rpvgae,
            datamodule=pldata,
            ckpt_path=(
                f"{checkpoint_dir}/last{checkpoint_suffix}.ckpt"
                if load_checkpoint
                else None
            ),
        )

    train(
        rpvgae,
        n_epochs=2000,
        n_kl_warmup=n_kl_warmup,
        loss_df=loss_df,
        counts_dicts=counts_dicts,
        hsic=hsic if hsic is not None else None,
        run_id=run_id,
    )
    torch.save(rpvgae.state_dict(), f"{prefix}{run_id}.model")
    save_files(f"{prefix}{run_id}", cell_adata=adata_CG, peak_adata=adata_CP)


def human_format(num):
    num = float(f"{num:.3g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{str(num).rstrip('0').rstrip('.')}{["", "K", "M", "B", "T"][magnitude]}"


def save_files(
    run_id: str,
    cell_adata: str = None,
    peak_adata: str = None,
    checkpoint_suffix: str = "",
):
    adata_CG = ad.read_h5ad(cell_adata) if cell_adata is not None else None
    adata_CP = ad.read_h5ad(peak_adata) if peak_adata is not None else None
    model = LightningProxModel.load_from_checkpoint(
        f"{run_id}.checkpoints/last{checkpoint_suffix}.ckpt",
        weights_only=True,
        map_location="cpu",
    )
    if "peak" in model.encoder.__mu_dict__:
        np.save(
            f"{run_id}.checkpoints/peak_scale.npy",
            model.scale_dict["peak__cell_has_accessible_peak"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/peak_bias.npy",
            model.bias_dict["peak__cell_has_accessible_peak"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_peak_scale.npy",
            model.scale_dict["cell__cell_has_accessible_peak"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_peak_bias.npy",
            model.bias_dict["cell__cell_has_accessible_peak"].detach().numpy(),
        )
    if "gene" in model.encoder.__mu_dict__:
        np.save(
            f"{run_id}.checkpoints/gene_scale.npy",
            model.scale_dict["gene__cell_expresses_gene"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/gene_bias.npy",
            model.bias_dict["gene__cell_expresses_gene"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_gene_scale.npy",
            model.scale_dict["cell__cell_expresses_gene"].detach().numpy(),
        )
        np.save(
            f"{run_id}.checkpoints/cell_gene_bias.npy",
            model.bias_dict["cell__cell_expresses_gene"].detach().numpy(),
        )
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

    adata_P = ad.AnnData(
        X=model.encoder.__mu_dict__["peak"].detach().cpu().numpy(),
        layers={
            "X_logstd": model.encoder.__logstd_dict__["peak"].detach().cpu().numpy()
        },
        obs=adata_CP.var if adata_CP is not None else None,
    )
    if "gene" in model.encoder.__mu_dict__:
        adata_G = ad.AnnData(
            X=model.encoder.__mu_dict__["gene"].detach().cpu().numpy(),
            layers={
                "X_logstd": model.encoder.__logstd_dict__["gene"].detach().cpu().numpy()
            },
            obs=adata_CG.var if adata_CG is not None else None,
        )
        adata_G.write(f"{run_id}.checkpoints/adata_G.h5ad")
    adata_C.write(f"{run_id}.checkpoints/adata_C.h5ad")
    adata_P.write(f"{run_id}.checkpoints/adata_P.h5ad")


def main(args):
    kwargs = vars(args)
    del kwargs["subcommand"]
    run(**kwargs)


def add_argument(parser):
    parser.description = "Train SIMBA+ model on the given HetData object."
    parser.add_argument(
        "data_path", help="Path to the input data file (HetData.dat or similar)"
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
        "--output-dir",
        type=str,
        default=".",
        help="Top-level output directory where run artifacts will be stored",
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
        help="Dimensionality of hidden and latent embeddings",
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
        "--pos-scale",
        action="store_true",
        help="Use positive-only scaling for the mean of output distributions",
    )

    return parser


if __name__ == "__main__":
    print("run 0")
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
