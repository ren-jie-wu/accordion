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
from simba_plus.loader import CustomNSMultiIndexDataset, collate_graph, collate
from simba_plus.model_prox import LightningProxModel, make_key
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
    batch_alledges=None,
    num_neg_samples_fold=1,
):
    pos_edge_index_dict = batch.edge_index_dict
    model.eval()
    z_dict, _ = model.encode(batch)

    pos_dist_dict = model.decoder(
        batch,
        z_dict,
        pos_edge_index_dict,
        **model.aux_params(batch, {k: v.cpu() for k, v in pos_edge_index_dict.items()}),
    )
    if batch_alledges is None:
        return pos_dist_dict
    neg_edge_index_dict = {}
    for edge_type in pos_edge_index_dict.keys():
        src, _, dst = edge_type
        neg_edge_index = negative_sampling(
            batch_alledges[edge_type].edge_index,
            num_nodes=(
                batch_alledges[src].num_nodes,
                batch_alledges[dst].num_nodes,
            ),
            num_neg_samples=len(pos_edge_index_dict[edge_type]) * num_neg_samples_fold,
        )
        neg_edge_index_dict[edge_type] = neg_edge_index
    neg_dist_dict = model.decoder(
        batch,
        z_dict,
        neg_edge_index_dict,
        **model.aux_params(batch, neg_edge_index_dict),
    )
    return pos_dist_dict, neg_edge_index, neg_dist_dict


def get_gexp_metrics(
    model: LightningProxModel,
    index_dict,
    data: torch_geometric.data.Data,
    batch_size: Optional[int] = None,
    negative_sampling_fold: Optional[int] = 1,
    device: str = "cpu",
    num_workers: int = 2,
):
    if batch_size is None:
        batch_size = int(1e6)
    gexp_edge_type = ("cell", "expresses", "gene")
    src, _, dst = gexp_edge_type
    # create ground truth array
    geidx = data[gexp_edge_type].edge_index.cpu().numpy()
    gexp_mat = scipy.sparse.coo_matrix(
        (data[gexp_edge_type].edge_attr.cpu().numpy(), (geidx[0, :], geidx[1, :])),
        shape=(data[src].num_nodes, data[dst].num_nodes),
    ).toarray()
    gexp_mean = gexp_mat.mean(axis=0)
    cell_mean = gexp_mat.mean(axis=1)
    gexp_std = gexp_mat.std(axis=0)
    cell_std = gexp_mat.std(axis=1)

    gexp_norm = (
        (gexp_mat - gexp_mean - cell_mean[:, None])
        / (gexp_std + 1e-6)
        / (cell_std[:, None] + 1e-6)
    )
    gexp_dataset = CustomNSMultiIndexDataset(
        {
            gexp_edge_type: torch.sort(index_dict[gexp_edge_type])[0],
        },
        data,
        negative_sampling_fold=negative_sampling_fold,
    )
    gexp_dataset.sample_negative()
    gexp_loader = DataLoader(
        gexp_dataset,
        batch_size=batch_size,
        collate_fn=collate,  # _graph,
        num_workers=num_workers,
    )

    with torch.no_grad():
        model.eval()
        means = []
        stds = []
        src_idxs = []
        dst_idxs = []

        for gene_batch in tqdm(gexp_loader):
            gene_batch = ToDevice(device)(gene_batch)
            pos_dist_dict = decode(
                model,
                gene_batch,  # gene_batch_alledges
            )
            means.append(pos_dist_dict[gexp_edge_type].mean)
            # means.append(neg_dist_dict[gexp_edge_type].mean)
            stds.append(pos_dist_dict[gexp_edge_type].stddev)
            # stds.append(neg_dist_dict[gexp_edge_type].stddev)
            src_idxs.append(
                gene_batch[src].n_id[gene_batch[gexp_edge_type].edge_index[0]].cpu()
            )
            # src_idxs.append(gene_batch[src].n_id[neg_edge_index[0]].cpu())
            dst_idxs.append(
                gene_batch[dst].n_id[gene_batch[gexp_edge_type].edge_index[1]].cpu()
            )
            # dst_idxs.append(gene_batch[dst].n_id[neg_edge_index[0]].cpu())
        gexp_pred_mu = torch.cat(means)
        src_idx = torch.cat(src_idxs).cpu().numpy()
        dst_idx = torch.cat(dst_idxs).cpu().numpy()
    test_dense_mat = scipy.sparse.coo_matrix(
        (
            gexp_pred_mu.detach().cpu().numpy(),
            (src_idx, dst_idx),
        ),
        shape=(data[src].num_nodes, data[dst].num_nodes),
    )
    mask = (
        scipy.sparse.coo_matrix(
            (
                np.ones_like(gexp_pred_mu.detach().cpu().numpy()),
                (src_idx, dst_idx),
            ),
            shape=(data[src].num_nodes, data[dst].num_nodes),
        )
        .toarray()
        .astype(bool)
    )
    pred_gene_mean = (
        model.aux_params.bias_dict[make_key(dst, gexp_edge_type)].detach().cpu().numpy()
    )
    pred_cell_mean = (
        model.aux_params.bias_dict[make_key(src, gexp_edge_type)].detach().cpu().numpy()
    )
    pred_gene_scale = (
        model.aux_params.logscale_dict[make_key(dst, gexp_edge_type)].detach().cpu()
    )
    pred_cell_scale = (
        model.aux_params.logscale_dict[make_key(src, gexp_edge_type)].detach().cpu()
    )
    if pred_gene_mean.ndim == 2:
        pred_gene_mean = pred_gene_mean.mean(axis=0)
    if pred_cell_mean.ndim == 2:
        pred_cell_mean = pred_cell_mean.mean(axis=0)
    if pred_gene_scale.ndim == 2:
        pred_gene_scale = pred_gene_scale.mean(axis=0)
    if pred_cell_scale.ndim == 2:
        pred_cell_scale = pred_cell_scale.mean(axis=0)
    pred_gene_scale = pred_gene_scale.exp().numpy()
    pred_cell_scale = pred_cell_scale.exp().numpy()

    res_out = test_dense_mat.toarray()
    res_out_norm = (
        (res_out - pred_gene_mean - pred_cell_mean[:, None])
        / (pred_gene_scale + 1e-6)
        / (pred_cell_scale[:, None] + 1e-6)
    )
    # res_out_norm = (
    #     (res_out - gexp_mean - cell_mean[:, None])
    #     / (gexp_std + 1e-6)
    #     / (cell_std[:, None] + 1e-6)
    # )
    corrs = []
    spearman_corrs = []
    for i in range(res_out_norm.shape[1]):
        if gexp_mean[i] <= 5:
            continue
        nonzero_idx = mask[:, i]
        if nonzero_idx.sum() == 0:
            continue
        if len(np.unique(gexp_norm[nonzero_idx, i])) == 1:
            continue

        corrs.append(
            np.corrcoef(res_out_norm[:, i][nonzero_idx], gexp_norm[:, i][nonzero_idx])[
                0, 1
            ].item()
        )
        spearman_corrs.append(
            spearmanr(
                res_out_norm[:, i][nonzero_idx], gexp_norm[:, i][nonzero_idx]
            ).correlation
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
    index_dict: Dict[EdgeType, Tensor],
    data: torch_geometric.data.Data,
    batch_size: Optional[int] = None,
    negative_sampling_fold: Optional[int] = 1,
    device: str = "cpu",
    num_workers: int = 2,
):
    if batch_size is None:
        batch_size = int(1e6)
    acc_edge_type = ("cell", "has_accessible", "peak")
    acc_dataset = CustomNSMultiIndexDataset(
        {acc_edge_type: index_dict[acc_edge_type]},
        data,
        negative_sampling_fold=negative_sampling_fold,
    )
    acc_dataset.sample_negative()
    acc_loader = DataLoader(
        acc_dataset,
        batch_size=batch_size,
        collate_fn=collate,  # _graph,
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
            pos_dist_dict = decode(model, acc_batch)
            # edge_idxs.append(neg_edge_index.cpu())
            true_acc.append(
                acc_batch[acc_edge_type].edge_attr.cpu(),
            )
            # true_acc.append(torch.zeros(neg_dist_dict[acc_edge_type].logits.shape[0]))
            preds.append(pos_dist_dict[acc_edge_type].logits)
            # preds.append(neg_dist_dict[acc_edge_type].logits)
        acc_true = torch.cat(true_acc).long()
        acc_pred = torch.cat(preds).to(acc_true.device)
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
            shape=(data["cell"].num_nodes, data["peak"].num_nodes),
        )
    return metrics


def evaluate_model(
    model_path: str,
    data_path: str,
    device: str = "cpu",
    eval_split: Literal["train", "test", "val"] = "test",
    batch_size: Optional[int] = None,
    negative_sampling_fold: Optional[int] = 1,
    index_path: Optional[str] = None,
    logger=None,
):
    data = torch.load(data_path, weights_only=False)
    data.generate_ids()
    # data.to(device)
    if index_path is None:
        index_path = f"{data_path.split('.dat')[0]}_data_idx.pkl"
    with open(index_path, "rb") as f:
        data_idx = pkl.load(f)
    index_dict = {tuple(k.split("__")): v for k, v in data_idx[eval_split].items()}
    metric_dict = {}

    model = LightningProxModel.load_from_checkpoint(model_path, weights_only=True).to(
        device
    )
    logger.info(f"Loaded model from {model_path}.")
    model.eval()
    if ("cell", "expresses", "gene") in index_dict.keys():
        metric_dict["gexp"] = get_gexp_metrics(
            model,
            index_dict,
            data,
            negative_sampling_fold=negative_sampling_fold,
            batch_size=batch_size,
            device=device,
        )
    if ("cell", "has_accessible", "peak") in index_dict.keys():
        metric_dict["acc"] = get_accessibility_metrics(
            model=model,
            index_dict=index_dict,
            data=data,
            negative_sampling_fold=negative_sampling_fold,
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
        "--eval-split",
        type=str,
        choices=["train", "test", "val"],
        help="Which data split to use for evaluation.",
        default="test",
    )
    parser.add_argument(
        "--negative-sampling-fold",
        type=int,
        help="Number of negative samples for evaluation.",
        default=1,
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
                    output_string += f"   {metric_name:<30}: {value:.3f}" + "\n"
                else:
                    output_string += f"   {metric_name:<30}: {value}" + "\n"

        # Print array metrics with comprehensive statistics
        if array_metrics:
            output_string += f"\n📈 Array Metrics:" + "\n"
            for metric_name, array in array_metrics.items():
                output_string += f"\n   {metric_name}:" + "\n"
                output_string += f"   {'─' * (len(metric_name) + 3)}" + "\n"

                # Basic statistics
                output_string += f"   Mean:       {np.mean(array):.3f}" + "\n"
                output_string += f"   Std:        {np.std(array):.3f}" + "\n"
                output_string += f"   Min:        {np.min(array):.3f}" + "\n"
                output_string += f"   Max:        {np.max(array):.3f}" + "\n"
                output_string += f"   Median:     {np.median(array):.3f}" + "\n"

    output_string += "\n" + "=" * 80 + "\n"
    logger.info(output_string)


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+evaluate", log_dir=os.path.dirname(args.model_path)
        )
    metric_dict_path = f"{os.path.dirname(args.model_path)}/pred_dict.pkl"
    logger.info(f"Evaluating model {args.model_path}...")
    if not args.rerun and os.path.exists(metric_dict_path):
        logger.info(f"Loading existing metrics from {metric_dict_path}...")
        with open(metric_dict_path, "rb") as file:
            metric_dict = pkl.load(file)
        pretty_print(metric_dict, logger=logger)
        return
    metric_dict = evaluate_model(
        args.model_path,
        args.data_path,
        eval_split=args.eval_split,
        index_path=args.idx_path,
        batch_size=args.batch_size,
        negative_sampling_fold=args.negative_sampling_fold,
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
