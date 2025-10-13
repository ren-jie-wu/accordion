from typing import Dict
import os
import pickle as pkl
from argparse import ArgumentParser
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import negative_sampling
from torch_geometric.typing import EdgeType

from tqdm.notebook import tqdm
from simba_plus.loader import CustomIndexDataset
from simba_plus.model_prox import LightningProxModel
from simba_plus.evaluation_utils import compute_reconstruction_gene_metrics, compute_classification_metrics
## Evaluate reconstruction quality

torch_geometric.seed_everything(2025)



def collate(data):
    types, idxs = zip(*data)
    return tuple(types[0]), torch.tensor(idxs)

def decode(model, batch, cpu_data, data):
    pos_edge_index_dict = batch.edge_index_dict
    model.eval()
    z_dict, _ = model.encode(data)
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
        src_type, _, dst_type = edge_type
        n_src = z_dict[src_type].shape[0]
        n_dst = z_dict[dst_type].shape[0]
        (
            neg_src_idx,
            neg_dst_idx,
        ) = negative_sampling(
            edge_index=cpu_data.subgraph(
                {
                    src_type: batch[src_type].n_id.cpu(),
                    dst_type: batch[dst_type].n_id.cpu(),
                }
            )[edge_type].edge_index,
            num_nodes=(n_src, n_dst),
            num_neg_samples=len(pos_edge_index[0]),
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

def eval(model_path, test_loaders, data, device='cpu'):
    if device != 'cpu':
        cpu_data = data.clone()
        cpu_data.to('cpu')
    else:
        cpu_data = data
    metric_dict = {}
    preds_dict = {}
    model = LightningProxModel.load_from_checkpoint(model_path, weights_only=True)
    model.eval()
    with torch.no_grad():
        true_exps = []
        means = []
        stds = []
        label_idxs = []
        for gene_batch in tqdm(test_loaders[1]):
            gene_edge_type, gene_label_index = gene_batch
            gene_edge_type = tuple(gene_edge_type)
            gene_batch = data.edge_type_subgraph(gene_edge_type).edge_subgraph({gene_edge_type: gene_label_index.to(device), })
            pos_dist_dict, _, _ = decode(model, gene_batch, cpu_data, data)
            true_exps.append(gene_batch['cell', 'expresses', 'gene'].edge_attr)
            means.append(pos_dist_dict[('cell', 'expresses', 'gene')].mean)
            stds.append(pos_dist_dict[('cell', 'expresses', 'gene')].stddev)
            label_idxs.append(gene_label_index)
            print("decoded gexp")
        gexp_true = torch.cat(true_exps)
        gexp_pred_mu = torch.cat(means)
        gexp_pred_std = torch.cat(stds)
        gexp_edge_idx = torch.cat(label_idxs)
        metric_dict[f"gexp"] = compute_reconstruction_gene_metrics(
            gexp_true, gexp_pred_mu, gexp_pred_std,
            plot=True,
        )
        preds_dict = {"gexp_true":gexp_true, "gexp_pred_mu":gexp_pred_mu, "gexp_pred_std":gexp_pred_std, "gexp_edge_idx":gexp_edge_idx}
        for peak_batch in test_loaders[0]:
            peak_edge_type, peak_label_index = peak_batch
            peak_edge_type = tuple(peak_edge_type)
            peak_batch = data.edge_type_subgraph(peak_edge_type).edge_subgraph({peak_edge_type: peak_label_index.to(device)})
            true_acc = []
            preds = []
            label_idxs = []
            label_idxs.append(peak_label_index)
            pos_dist_dict, neg_edge_index_dict, neg_dist_dict = decode(model, peak_batch, cpu_data, data)
            neg_size = neg_edge_index_dict[('cell', 'has_accessible', 'peak')].shape[1]
            true_acc.append((
                torch.cat(
                    [
                        peak_batch[('cell', 'has_accessible', 'peak')].edge_attr.cpu(),
                        torch.zeros(
                            neg_size
                        ),
                    ],
                    dim=0,
                )
                .detach()
                .cpu()
            ))
            preds.append((
                torch.cat(
                    [
                        pos_dist_dict[('cell', 'has_accessible', 'peak')].logits,
                        neg_dist_dict[('cell', 'has_accessible', 'peak')].logits,
                    ],
                    dim=0,
                )
                .detach()
                .cpu()
            ))
        acc_true = torch.cat(true_acc).long()
        acc_pred = torch.cat(preds)
        acc_edge_idx = torch.cat(label_idxs)
        metric_dict["acc"] = compute_classification_metrics(
            acc_true, torch.sigmoid(acc_pred),
            plot=False,
        )
        preds_dict["acc_true"] = acc_true
        preds_dict["acc_pred"] = acc_pred
        preds_dict["acc_edge_idx"] = acc_edge_idx
        return metric_dict, preds_dict

def main(data_path, model_path):
    parser = ArgumentParser(
        description="Evaluate the Simba+ model on a given dataset.",
    )
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on.",
    )
    args = parser.parse_args()

    data = torch.load(data_path, weights_only=False)
    data.to(args.device)

    test_data_dict = {}
    edge_types = [
        # ("peak", "has_sequence", "motif"),
        ("cell", "has_accessible", "peak"),
        ("cell", "expresses", "gene"),
    ]

    for edge_type in edge_types:
        num_edges = data[edge_type].num_edges
        indices = torch.arange(num_edges)[torch.randperm(num_edges)]
        test_index = indices[int(num_edges * 0.95) :]
        test_data_dict[edge_type] = CustomIndexDataset(edge_type, test_index)

    metric_dict, preds_dict = eval(model_path, test_data_dict, data, device=args.device)
    with open(f"{os.path.dirname(args.model_path)}/pred_dict_lam.pkl", "wb") as file:
        pkl.dump({"pred":preds_dict, "metrics":metric_dict}, file)
                 