from typing import Dict, Optional
import os
from typing import Literal
import pickle as pkl
from argparse import ArgumentParser
import scipy.sparse
import numpy as np
import torch
from torch import Tensor
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_dense_adj
from torch_geometric.typing import EdgeType

from tqdm import tqdm
from simba_plus.loader import CustomIndexDataset
from simba_plus.model_prox import LightningProxModel
from simba_plus.evaluation_utils import (
    compute_reconstruction_gene_metrics,
    compute_classification_metrics,
)

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
        src_type, _, dst_type = edge_type
        n_src = z_dict[src_type].shape[0]
        n_dst = z_dict[dst_type].shape[0]
        (
            neg_src_idx,
            neg_dst_idx,
        ) = negative_sampling(
            train_index,
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
):
    if batch_size is None:
        batch_size = int(1e6)
    gexp_mat = to_dense_adj(
        eval_data["cell", "expresses", "gene"].edge_index,
    )[0, : eval_data["cell"].num_nodes, : eval_data["gene"].num_nodes]
    gexp_mean = gexp_mat.mean(axis=0)
    gexp_norm = gexp_mat / (gexp_mean + 1e-6)
    gexp_loader = DataLoader(
        CustomIndexDataset(
            [("cell", "expresses", "gene")],
            torch.arange(eval_data["cell", "expresses", "gene"].num_edges),
        ),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
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
            gene_edge_type = tuple(gene_edge_type)
            gene_batch = eval_data.edge_type_subgraph(gene_edge_type).edge_subgraph(
                {
                    gene_edge_type: gene_label_index,
                }
            )
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, gene_batch, data, n_negative_samples=n_negative_samples
            )
            means.append(pos_dist_dict[("cell", "expresses", "gene")].mean)
            stds.append(pos_dist_dict[("cell", "expresses", "gene")].stddev)
            neg_means.append(neg_dist_dict[("cell", "expresses", "gene")].mean)
            neg_stds.append(neg_dist_dict[("cell", "expresses", "gene")].stddev)
            neg_idx.append(neg_edge_index_dict[("cell", "expresses", "gene")])
            print("decoded gexp")
        gexp_pred_mu = torch.cat(means + neg_means)
        gexp_pred_std = torch.cat(stds + neg_stds)
        gexp_neg_idx = torch.cat(neg_idx)
    test_dense_mat = scipy.sparse.coo_matrix(
        (
            gexp_pred_mu.detach().numpy(),
            (
                eval_data["cell", "expresses", "gene"]
                .edge_index[0][torch.cat([gexp_loader.index, gexp_neg_idx])]
                .numpy(),
                eval_data["cell", "expresses", "gene"]
                .edge_index[1][torch.cat([gexp_loader.index, gexp_neg_idx])]
                .numpy(),
            ),
        )
    )
    res_out = test_dense_mat.toarray()
    res_out_norm = res_out / (gexp_mean + 1e-6)
    corrs = []
    for i in range(res_out_norm.shape[1]):
        if gexp_mean[i] == 0:
            continue
        corrs.append(np.corrcoef(res_out_norm[:, i], gexp_norm[:, i])[0, 1].item())
    return np.array(corrs)


def get_accessibility_metrics(
    model: LightningProxModel,
    eval_data: torch_geometric.data.Data,
    data: torch_geometric.data.Data,
    batch_size: Optional[int] = None,
    n_negative_samples: Optional[int] = None,
):
    if batch_size is None:
        batch_size = 1e6
    acc_loader = DataLoader(
        CustomIndexDataset(
            [("cell", "has_accessible", "peak")],
            torch.arange(eval_data["cell", "has_accessible", "peak"].num_edges),
        ),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )
    with torch.no_grad():
        model.eval()
        means = []
        stds = []
        neg_means = []
        neg_stds = []
        neg_idx = []
        for acc_batch in tqdm(acc_loader):
            acc_edge_type, acc_label_index = acc_batch
            acc_edge_type = tuple(acc_edge_type)
            acc_batch = eval_data.edge_type_subgraph(acc_edge_type).edge_subgraph(
                {
                    acc_edge_type: acc_label_index,
                }
            )
            true_acc = []
            preds = []
            label_idxs = []
            label_idxs.append(acc_label_index)
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(
                model, acc_batch, data
            )
            true_acc.append(
                (
                    torch.cat(
                        [
                            acc_batch[
                                ("cell", "has_accessible", "peak")
                            ].edge_attr.cpu(),
                            torch.zeros(n_negative_samples),
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
        acc_edge_idx = torch.cat(label_idxs)
        metrics = compute_classification_metrics(
            acc_true,
            torch.sigmoid(acc_pred),
            plot=False,
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
    data.to(device)
    if index_path is not None:
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
    model = LightningProxModel.load_from_checkpoint(model_path, weights_only=True)
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
                torch.tensor(set(range(data[tuple(k.split("__"))].num_edges)) - set(v)),
            ]
            .to(device)
            for k, v in data_idx["test"].items()
        }

    possible_edge_types = [
        ("cell", "has_accessible", "peak"),
        ("cell", "expresses", "gene"),
    ]
    if ("cell", "expresses", "gene") in eval_data.edge_types:
        metric_dict["gexp"] = get_gexp_metrics(
            model,
            eval_data,
            train_index,
            n_negative_samples=n_negative_samples,
            batch_size=batch_size,
        )
    if ("cell", "has_accessible", "peak") in eval_data.edge_types:
        metric_dict["acc"] = get_gexp_metrics(
            model,
            eval_data,
            train_index,
            n_negative_samples=n_negative_samples,
            batch_size=batch_size,
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
    return parser


def main(args):
    metric_dict = eval(
        args.model_path,
        args.data_path,
        index_path=args.idx_path,
        batch_size=args.batch_size,
        n_negative_samples=args.n_negative_samples,
        device=args.device,
    )
    print(metric_dict)
    with open(f"{os.path.dirname(args.model_path)}/pred_dict.pkl", "wb") as file:
        pkl.dump(metric_dict, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
