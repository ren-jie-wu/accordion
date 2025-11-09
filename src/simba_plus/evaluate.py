from typing import Dict, Optional
import os
from functools import partial
from typing import Literal
import pickle as pkl
from argparse import ArgumentParser
import scipy.sparse
from scipy.stats import spearmanr
import numpy as np
import torch
from torch import Tensor
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_dense_adj
from torch_geometric.typing import EdgeType
from torch_geometric.transforms.to_device import ToDevice
from tqdm import tqdm
from simba_plus.loader import CustomMultiIndexDataset, collate
from simba_plus.model_prox import LightningProxModel
from simba_plus.evaluation_utils import (
    compute_reconstruction_gene_metrics,
    compute_classification_metrics,
)
from simba_plus.utils import setup_logging
import logging

## Evaluate reconstruction quality

torch_geometric.seed_everything(2025)


def decode(
    model: LightningProxModel,
    batch,
    train_index,
    n_negative_samples: Optional[int] = None,
):
    pos_edge_index_dict = batch.edge_index_dict
    model.eval()
    z_dict, _ = model.encode(batch)
    pos_dist_dict = model.decoder(
        batch,
        z_dict,
        pos_edge_index_dict,
        *model.aux_params(batch, {k: v.cpu() for k, v in pos_edge_index_dict.items()}),
    )
    neg_edge_index_dict = {}
    for edge_type, pos_edge_index in pos_edge_index_dict.items():
        if n_negative_samples is None:
            _n_neg_samples = len(pos_edge_index[0])
        else:
            _n_neg_samples = n_negative_samples
        src_type, _, dst_type = edge_type
        n_src = z_dict[src_type].shape[0]
        n_dst = z_dict[dst_type].shape[0]
        (
            neg_src_idx,
            neg_dst_idx,
        ) = negative_sampling(
            train_index[edge_type],
            num_nodes=(
                batch[src_type].num_nodes,
                batch[dst_type].num_nodes,
            ),
            num_neg_samples=_n_neg_samples,
        )
        neg_edge_index_dict[edge_type] = torch.stack([neg_src_idx, neg_dst_idx])

    neg_dist_dict: Dict[EdgeType, Tensor] = model.decoder(
        batch,
        z_dict,
        neg_edge_index_dict,
        *model.aux_params(batch, {k: v.cpu() for k, v in neg_edge_index_dict.items()}),
    )
    return pos_dist_dict, neg_edge_index_dict, neg_dist_dict


def get_gexp_metrics(
    model: LightningProxModel,
    eval_data: torch_geometric.data.Data,
    train_index: torch_geometric.data.Data,
    batch_size: Optional[int] = None,
    n_negative_samples: Optional[int] = None,
    device: str = "cpu",
    num_workers: int = 2,
):
    if batch_size is None:
        batch_size = int(1e6)
    gexp_edge_type = ("cell", "expresses", "gene")
    gexp_mat = (
        to_dense_adj(
            eval_data[gexp_edge_type].edge_index,
        )[0, : eval_data["cell"].num_nodes, : eval_data["gene"].num_nodes]
        .cpu()
        .numpy()
    )
    gexp_mean = gexp_mat.mean(axis=0)
    gexp_norm = gexp_mat / (gexp_mean + 1e-6)
    gexp_dataset = CustomMultiIndexDataset(
        [gexp_edge_type],
        [torch.arange(eval_data[gexp_edge_type].edge_index.shape[1])],
    )
    gexp_loader = DataLoader(
        gexp_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate, data=ToDevice("cpu")(eval_data)),
        num_workers=num_workers,
    )
    with torch.no_grad():
        model.eval()
        means = []
        stds = []
        neg_means = []
        neg_stds = []
        neg_idx = []
        for gene_batch in tqdm(gexp_loader):
            gene_batch = ToDevice(device)(gene_batch)
            if n_negative_samples is None:
                n_negative_samples = len(gene_batch[gexp_edge_type].edge_index[0]) // 10
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, gene_batch, train_index, n_negative_samples=n_negative_samples
            )
            means.append(pos_dist_dict[gexp_edge_type].mean)
            stds.append(pos_dist_dict[gexp_edge_type].stddev)
            neg_means.append(neg_dist_dict[gexp_edge_type].mean)
            neg_stds.append(neg_dist_dict[gexp_edge_type].stddev)
            neg_idx.append(neg_edge_index_dict[gexp_edge_type])
        gexp_pred_mu = torch.cat(means + neg_means)
        gexp_pred_std = torch.cat(stds + neg_stds)
        gexp_neg_idx = torch.cat(neg_idx, dim=1)
    pos_idxs = eval_data[gexp_edge_type].edge_index[:, gexp_dataset.indexes[0]]
    all_idxs = torch.cat([pos_idxs, gexp_neg_idx], dim=1)
    test_dense_mat = scipy.sparse.coo_matrix(
        (
            gexp_pred_mu.detach().cpu().numpy(),
            (
                all_idxs[0].detach().cpu().numpy(),
                all_idxs[1].detach().cpu().numpy(),
            ),
        )
    )
    res_out = test_dense_mat.toarray()
    res_out_norm = res_out / (gexp_mean + 1e-6)
    corrs = []
    spearman_corrs = []
    for i in range(res_out_norm.shape[1]):
        if gexp_mean[i] == 0:
            continue
        corrs.append(np.corrcoef(res_out_norm[:, i], gexp_norm[:, i])[0, 1].item())
        spearman_corrs.append(
            spearmanr(res_out_norm[:, i], gexp_norm[:, i]).correlation
        )
    return {
        "per-gene corr": np.array(corrs),
        "corr_mean": np.mean(corrs),
        "spearman_corr": np.array(spearman_corrs),
        "spearman_corr_mean": np.mean(spearman_corrs),
        "pred": test_dense_mat,
    }


def get_accessibility_metrics(
    model: LightningProxModel,
    eval_data: torch_geometric.data.Data,
    data: torch_geometric.data.Data,
    batch_size: Optional[int] = None,
    n_negative_samples: Optional[int] = None,
    device: str = "cpu",
    num_workers: int = 2,
):
    if batch_size is None:
        batch_size = int(1e6)
    acc_edge_type = ("cell", "has_accessible", "peak")
    acc_dataset = CustomMultiIndexDataset(
        [acc_edge_type],
        [torch.arange(eval_data[acc_edge_type].edge_index.shape[1])],
    )
    acc_loader = DataLoader(
        acc_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate, data=ToDevice("cpu")(eval_data)),
        num_workers=num_workers,
    )
    with torch.no_grad():
        model.eval()
        true_acc = []
        preds = []
        edge_idxs = []
        for acc_batch in tqdm(acc_loader):
            acc_batch = ToDevice(device)(acc_batch)
            edge_idxs.append(acc_batch[acc_edge_type].edge_index.cpu())
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, acc_batch, data, n_negative_samples=n_negative_samples
            )
            edge_idxs.append(neg_edge_index_dict[acc_edge_type].cpu())
            true_acc.append(
                (
                    torch.cat(
                        [
                            acc_batch[acc_edge_type].edge_attr.cpu(),
                            torch.zeros(neg_edge_index_dict[acc_edge_type].shape[1]),
                        ],
                        dim=0,
                    )
                    .detach()
                    .cpu()
                )
            )
            preds.append(
                (
                    torch.cat(
                        [
                            pos_dist_dict[acc_edge_type].logits,
                            neg_dist_dict[acc_edge_type].logits,
                        ],
                        dim=0,
                    )
                    .detach()
                    .cpu()
                )
            )
        acc_true = torch.cat(true_acc).long()
        acc_pred = torch.cat(preds)
        acc_edge_idx = torch.cat(edge_idxs, axis=1)
        metrics = compute_classification_metrics(
            acc_true,
            torch.sigmoid(acc_pred),
            plot=False,
        )

        metrics["pred"] = scipy.sparse.coo_matrix(
            (
                acc_pred.detach().cpu().numpy(),
                (acc_edge_idx[0], acc_edge_idx[1]),
            ),
            shape=(eval_data["cell"].num_nodes, eval_data["peak"].num_nodes),
        )
    return metrics


def eval(
    model_path: str,
    data: str,
    device: str = "cpu",
    eval_split: Literal["test", "val"] = "test",
    batch_size: Optional[int] = None,
    n_negative_samples: Optional[int] = None,
    index_path: Optional[str] = None,
    logger=None,
):
    data = torch.load(data, weights_only=False)
    data.generate_ids()
    data.to(device)
    if index_path is None:
        index_path = f"{os.path.dirname(model_path)}/data_idx.pkl"
    with open(index_path, "rb") as f:
        data_idx = pkl.load(f)
    if eval_split == "test":
        idx_dict = data_idx["test"]
    else:
        idx_dict = data_idx["val"]
    metric_dict = {}

    eval_data = data.edge_subgraph(
        {tuple(k.split("__")): v.to(device) for k, v in idx_dict.items()}
    )
    if hasattr(eval_data["cell"], "batch"):
        if isinstance(eval_data["cell"].batch, np.ndarray):
            eval_data["cell"].batch = torch.tensor(eval_data["cell"].batch).to(device)
    model = LightningProxModel.load_from_checkpoint(model_path, weights_only=True).to(
        device
    )
    logger.info(f"Loaded model from {model_path}.")
    model.eval()
    if eval_split == "val":
        train_index = {
            tuple(k.split("__")): data[tuple(k.split("__"))].edge_index[
                :,
                torch.tensor(
                    list(
                        set(range(data[tuple(k.split("__"))].num_edges))
                        - set(v)
                        - set(data_idx["test"][k])
                    ),
                    device=device,
                ),
            ]
            for k, v in data_idx["val"].items()
        }
    elif eval_split == "test":
        train_index = {
            tuple(k.split("__")): data[tuple(k.split("__"))]
            .edge_index[
                :,
                torch.tensor(
                    list(set(range(data[tuple(k.split("__"))].num_edges)) - set(v)),
                    dtype=torch.long,
                ),
            ]
            .to(device)
            for k, v in data_idx["test"].items()
        }

    if ("cell", "expresses", "gene") in eval_data.edge_types:
        metric_dict["gexp"] = get_gexp_metrics(
            model,
            eval_data,
            train_index,
            n_negative_samples=n_negative_samples,
            batch_size=batch_size,
            device=device,
        )
    if ("cell", "has_accessible", "peak") in eval_data.edge_types:
        metric_dict["acc"] = get_accessibility_metrics(
            model,
            eval_data,
            train_index,
            n_negative_samples=n_negative_samples,
            batch_size=batch_size,
            device=device,
        )

    return metric_dict


def add_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.description = "Evaluate the Simba+ model on a given dataset."
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--idx-path",
        type=str,
        help="Path to the index file.",
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for evaluation.",
        default=None,
    )
    parser.add_argument(
        "--n-negative-samples",
        type=int,
        help="Number of negative samples for evaluation.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun the evaluation.",
    )
    return parser


def pretty_print(
    metric_dict,
    logger,
    show_histogram=True,
    histogram_width=50,
):
    """
    Pretty print metrics dictionary focusing on scalar and array metrics.

    Parameters
    ----------
    metric_dict : dict
        Dictionary containing metrics for different tasks
    show_histogram : bool, default True
        Whether to show ASCII histogram for array metrics
    histogram_width : int, default 50
        Width of the ASCII histogram
    """
    output_string = "\n"
    output_string += "=" * 80 + "\n"
    output_string += "EVALUATION METRICS SUMMARY" + "\n"
    output_string += "=" * 80 + "\n"

    for task_name, metrics in metric_dict.items():
        output_string += f"\n📊 {task_name.upper()} METRICS" + "\n"
        output_string += "-" * 60 + "\n"

        # Separate metrics by type
        scalar_metrics = {}
        array_metrics = {}

        for metric_name, value in metrics.items():
            if metric_name == "pred":
                continue  # Skip prediction matrices
            elif isinstance(value, np.ndarray):
                array_metrics[metric_name] = value
            else:
                scalar_metrics[metric_name] = value

        # Print scalar metrics first
        if scalar_metrics:
            output_string += "🔢 Scalar Metrics:" + "\n"
            for metric_name, value in scalar_metrics.items():
                if isinstance(value, (np.floating, float)):
                    output_string += f"   {metric_name:<30}: {value:.6f}" + "\n"
                else:
                    output_string += f"   {metric_name:<30}: {value}" + "\n"

        # Print array metrics with comprehensive statistics
        if array_metrics:
            output_string += f"\n📈 Array Metrics:" + "\n"
            for metric_name, array in array_metrics.items():
                output_string += f"\n   {metric_name}:" + "\n"
                output_string += f"   {'─' * (len(metric_name) + 3)}" + "\n"

                # Basic statistics
                output_string += f"   Mean:       {np.mean(array):.6f}" + "\n"
                output_string += f"   Std:        {np.std(array):.6f}" + "\n"
                output_string += f"   Min:        {np.min(array):.6f}" + "\n"
                output_string += f"   Max:        {np.max(array):.6f}" + "\n"
                output_string += f"   Median:     {np.median(array):.6f}" + "\n"

    output_string += "\n" + "=" * 80 + "\n"
    logger.info(output_string)


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+evaluate", log_dir=os.path.dirname(args.model_path)
        )
    metric_dict_path = f"{os.path.dirname(args.model_path)}/pred_dict.pkl"
    if os.path.exists(metric_dict_path):
        with open(metric_dict_path, "rb") as file:
            metric_dict = pkl.load(file)
        pretty_print(metric_dict, logger=logger)
        return
    metric_dict = eval(
        args.model_path,
        args.data_path,
        index_path=args.idx_path,
        batch_size=args.batch_size,
        n_negative_samples=args.n_negative_samples,
        device=args.device,
        logger=logger,
    )
    pretty_print(metric_dict, logger=logger)
    with open(f"{os.path.dirname(args.model_path)}/pred_dict.pkl", "wb") as file:
        pkl.dump(metric_dict, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
