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

from tqdm import tqdm
from simba_plus.loader import CustomMultiIndexDataset
from simba_plus.model_prox import LightningProxModel
from simba_plus.evaluation_utils import (
    compute_reconstruction_gene_metrics,
    compute_classification_metrics,
)
from simba_plus.utils import setup_logging

## Evaluate reconstruction quality

torch_geometric.seed_everything(2025)


def collate(data):
    types, idxs = zip(*data)
    return tuple(types[0]), torch.tensor(idxs)


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
        scale_dict=model.scale_dict,
        bias_dict=model.bias_dict,
        std_dict=model.std_dict,
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
        scale_dict=model.scale_dict,
        bias_dict=model.bias_dict,
        std_dict=model.std_dict,
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
    gexp_mat = (
        to_dense_adj(
            eval_data["cell", "expresses", "gene"].edge_index,
        )[0, : eval_data["cell"].num_nodes, : eval_data["gene"].num_nodes]
        .cpu()
        .numpy()
    )
    gexp_mean = gexp_mat.mean(axis=0)
    gexp_norm = gexp_mat / (gexp_mean + 1e-6)
    gexp_dataset = CustomMultiIndexDataset(
        [("cell", "expresses", "gene")],
        [eval_data["cell", "expresses", "gene"].edge_index],
    )
    gexp_loader = DataLoader(
        gexp_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate, data=eval_data),
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
            gene_edge_type, gene_label_index = gene_batch
            gene_batch = eval_data.edge_subgraph(
                {
                    gene_edge_type: gene_label_index.to(device),
                }
            )
            if n_negative_samples is None:
                n_negative_samples = (
                    len(gene_batch["cell", "expresses", "gene"].edge_index[0]) // 10
                )
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, gene_batch, train_index, n_negative_samples=n_negative_samples
            )
            means.append(pos_dist_dict[("cell", "expresses", "gene")].mean)
            stds.append(pos_dist_dict[("cell", "expresses", "gene")].stddev)
            neg_means.append(neg_dist_dict[("cell", "expresses", "gene")].mean)
            neg_stds.append(neg_dist_dict[("cell", "expresses", "gene")].stddev)
            neg_idx.append(neg_edge_index_dict[("cell", "expresses", "gene")])
            print("decoded gexp")
        gexp_pred_mu = torch.cat(means + neg_means)
        gexp_pred_std = torch.cat(stds + neg_stds)
        gexp_neg_idx = torch.cat(neg_idx, dim=1)
    pos_idxs = eval_data["cell", "expresses", "gene"].edge_index[:, gexp_dataset.index]
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
    acc_dataset = CustomMultiIndexDataset(
        [("cell", "has_accessible", "peak")],
        [eval_data["cell", "has_accessible", "peak"].edge_index],
    )
    acc_loader = DataLoader(
        acc_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate, data=eval_data),
        num_workers=num_workers,
    )
    with torch.no_grad():
        model.eval()
        true_acc = []
        preds = []
        edge_idxs = []
        for acc_batch in tqdm(acc_loader):
            acc_edge_type, acc_label_index = acc_batch
            acc_edge_type = tuple(acc_edge_type)
            acc_batch = eval_data.edge_type_subgraph(acc_edge_type).edge_subgraph(
                {
                    acc_edge_type: acc_label_index.to(device),
                }
            )
            edge_idxs.append(acc_batch[acc_edge_type].edge_index.cpu())
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, acc_batch, data, n_negative_samples=n_negative_samples
            )
            edge_idxs.append(neg_edge_index_dict[acc_edge_type].cpu())
            true_acc.append(
                (
                    torch.cat(
                        [
                            acc_batch[
                                ("cell", "has_accessible", "peak")
                            ].edge_attr.cpu(),
                            torch.zeros(
                                neg_edge_index_dict[
                                    ("cell", "has_accessible", "peak")
                                ].shape[1]
                            ),
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
                            pos_dist_dict[("cell", "has_accessible", "peak")].logits,
                            neg_dist_dict[("cell", "has_accessible", "peak")].logits,
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
        print(acc_pred.shape)
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
    if isinstance(eval_data["cell"].batch, np.ndarray):
        eval_data["cell"].batch = torch.tensor(eval_data["cell"].batch).to(device)
    model = LightningProxModel.load_from_checkpoint(model_path, weights_only=True).to(
        device
    )
    print("loaded model")
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
        print(train_index)

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


def pretty_print(metric_dict, show_histogram=True, histogram_width=50):
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
    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    for task_name, metrics in metric_dict.items():
        print(f"\n📊 {task_name.upper()} METRICS")
        print("-" * 60)

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
            print("🔢 Scalar Metrics:")
            for metric_name, value in scalar_metrics.items():
                if isinstance(value, (np.floating, float)):
                    print(f"   {metric_name:<30}: {value:.6f}")
                else:
                    print(f"   {metric_name:<30}: {value}")

        # Print array metrics with comprehensive statistics
        if array_metrics:
            print(f"\n📈 Array Metrics:")
            for metric_name, array in array_metrics.items():
                print(f"\n   {metric_name}:")
                print(f"   {'─' * (len(metric_name) + 3)}")

                # Basic statistics
                print(f"   Mean:       {np.mean(array):.6f}")
                print(f"   Std:        {np.std(array):.6f}")
                print(f"   Min:        {np.min(array):.6f}")
                print(f"   Max:        {np.max(array):.6f}")
                print(f"   Median:     {np.median(array):.6f}")

    print("\n" + "=" * 80)


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+evaluate", log_dir=os.path.dirname(args.model_path)
        )
    metric_dict_path = f"{os.path.dirname(args.model_path)}/pred_dict.pkl"
    if os.path.exists(metric_dict_path):
        with open(metric_dict_path, "rb") as file:
            metric_dict = pkl.load(file)
        pretty_print(metric_dict)
        return
    metric_dict = eval(
        args.model_path,
        args.data_path,
        index_path=args.idx_path,
        batch_size=args.batch_size,
        n_negative_samples=args.n_negative_samples,
        device=args.device,
    )
    pretty_print(metric_dict)
    with open(f"{os.path.dirname(args.model_path)}/pred_dict.pkl", "wb") as file:
        pkl.dump(metric_dict, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
