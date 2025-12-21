import torch
import torch_geometric


def negative_sampling_same_sample_bipartite(
    pos_edge_index: torch.Tensor,
    src_sample: torch.Tensor,
    dst_sample: torch.Tensor,
    num_neg_samples_fold: int = 1,
    method: str = "sparse",
) -> torch.Tensor:
    """
    Sample negative edges for a bipartite relation, restricted to same-sample blocks.

    Parameters
    ----------
    pos_edge_index: (2, E)
        The edge indices of the positive edges (GLOBAL node indices).
    src_sample: (N_src,)
        The sample codes of all the source nodes (GLOBAL node indices).
    dst_sample: (N_dst,)
        The sample codes of all the destination nodes (GLOBAL node indices).
    num_neg_samples_fold: int = 1
        The number of negative samples per positive edge.
    method: str = "sparse"
        The method to sample negative edges. "sparse" or "dense".

    Returns
    -------
    neg_edge_index: (2, E * fold) approximately (may be smaller if capped)
        The edge indices of the negative edges.
    """
    device = pos_edge_index.device
    edge_s = src_sample[pos_edge_index[0]] # should be same as sample codes of dst nodes

    neg_parts = []
    src_to_local = torch.full((src_sample.numel(),), -1, device=device, dtype=torch.long)
    dst_to_local = torch.full((dst_sample.numel(),), -1, device=device, dtype=torch.long)
    for s in torch.unique(edge_s).tolist():
        src_nodes = (src_sample == s).nonzero(as_tuple=False).view(-1) # all possible source nodes within the same sample (local to global mapping)
        dst_nodes = (dst_sample == s).nonzero(as_tuple=False).view(-1) # destination nodes
        if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
            continue

        pos_s = pos_edge_index[:, edge_s == s] # positive edges within the same sample (global node indices)
        n_pos_s = pos_s.size(1)
        if n_pos_s == 0:
            continue

        max_neg = src_nodes.numel() * dst_nodes.numel() - n_pos_s
        if max_neg <= 0:
            continue
        n_neg = int(min(max_neg, n_pos_s * num_neg_samples_fold))

        src_to_local[src_nodes] = torch.arange(src_nodes.numel(), device=device) # global to local mapping
        dst_to_local[dst_nodes] = torch.arange(dst_nodes.numel(), device=device)
        pos_s_local = torch.vstack([src_to_local[pos_s[0]], dst_to_local[pos_s[1]]])

        neg_s_local = torch_geometric.utils.negative_sampling(
            pos_s_local,
            num_nodes=(src_nodes.numel(), dst_nodes.numel()),
            num_neg_samples=n_neg,
            method=method,
        )

        neg_s = torch.vstack([src_nodes[neg_s_local[0]], dst_nodes[neg_s_local[1]]])
        neg_parts.append(neg_s)
    
    if len(neg_parts) == 0:
        return torch.empty((2, 0), device=device)
    return torch.cat(neg_parts, dim=1)
