import torch
import pytest
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RemoveIsolatedNodes

from simba_plus.model_prox import AuxParams
from simba_plus._utils import make_key


def _build_toy_multisample_data():
    """
    sample0: 2 cells, 2 genes, batch_local in {0,1}
    sample1: 3 cells, 2 genes, batch_local in {0,1,2}

    cells global ids: 0,1 | 2,3,4
    genes global ids: 0,1 | 2,3
    genes local_id per sample: [0,1] for both samples
    """
    data = HeteroData()

    # nodes
    data["cell"].x = torch.zeros((5, 4))
    data["cell"].sample = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    data["cell"].batch_local = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
    data["cell"].batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # global, not used by AuxParams here

    data["gene"].x = torch.zeros((4, 4))
    data["gene"].sample = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    data["gene"].local_id = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    # edges: each cell connects to both genes of its own sample
    edges = []
    # sample0 cells 0,1 -> genes 0,1
    for c in [0, 1]:
        for g in [0, 1]:
            edges.append((c, g))
    # sample1 cells 2,3,4 -> genes 2,3
    for c in [2, 3, 4]:
        for g in [2, 3]:
            edges.append((c, g))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data[("cell", "expresses", "gene")].edge_index = edge_index
    data[("cell", "expresses", "gene")].edge_attr = torch.ones(edge_index.size(1))

    data.generate_ids()

    return data


def _make_batch_from_edge_ids(data: HeteroData, edge_ids: torch.Tensor) -> HeteroData:
    etype = ("cell", "expresses", "gene")
    sub = data.edge_subgraph({etype: edge_ids})
    sub = RemoveIsolatedNodes()(sub)
    return sub


def test_auxparams_forward_no_cross_sample_talk():
    torch.manual_seed(0)
    data = _build_toy_multisample_data()
    etype = ("cell", "expresses", "gene")

    # Take all edges as one batch-like subgraph
    batch = _make_batch_from_edge_ids(data, torch.arange(data[etype].edge_index.size(1)))

    aux = AuxParams(data, edgetype_specific=True)
    aux.eval()  # no noise in batched()

    dst_key = make_key("gene", etype)
    assert aux.multi_sample is True

    # Set per-sample dst_bias matrices to distinct values
    # sample0: B=2, N=2
    p0, p0_logstd = aux.bias_dict_per_sample[dst_key].get(0)
    p0.data[:] = torch.tensor([[100.0, 101.0], [110.0, 111.0]])
    p0_logstd.data.fill_(-20.0)

    # sample1: B=3, N=2
    p1, p1_logstd = aux.bias_dict_per_sample[dst_key].get(1)
    p1.data[:] = torch.tensor([[200.0, 201.0], [210.0, 211.0], [220.0, 221.0]])
    p1_logstd.data.fill_(-20.0)

    out = aux(batch, batch.edge_index_dict)
    dst_bias = out["dst_bias_dict"][etype]  # (num_edges_in_batch,)

    # Build expected dst_bias per edge:
    # expected = param_sample[sample_id][batch_local(cell), local_id(gene)]
    cell_local = batch[etype].edge_index[0]
    gene_local = batch[etype].edge_index[1]

    cell_sample = batch["cell"].sample[cell_local]
    cell_b = batch["cell"].batch_local[cell_local]
    gene_local_id = batch["gene"].local_id[gene_local]

    expected = torch.empty_like(dst_bias)
    for i in range(dst_bias.numel()):
        sid = int(cell_sample[i].item())
        b = int(cell_b[i].item())
        gid = int(gene_local_id[i].item())
        if sid == 0:
            expected[i] = torch.tensor([[100.0, 101.0], [110.0, 111.0]])[b, gid]
        else:
            expected[i] = torch.tensor([[200.0, 201.0], [210.0, 211.0], [220.0, 221.0]])[b, gid]

    torch.testing.assert_close(dst_bias, expected)


def test_batched_baseline_semantics():
    data = _build_toy_multisample_data()
    aux = AuxParams(data, edgetype_specific=True)
    aux.train()  # enable batched()
    assert aux.use_batch is True

    param = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # B=2, N=2
    # make noise ~0
    param_logstd = torch.full_like(param, -20.0)

    out = aux.batched(param, param_logstd)
    # baseline row unchanged
    torch.testing.assert_close(out[0], param[0])
    # offset row becomes baseline + offset (noise negligible)
    torch.testing.assert_close(out[1], param[0] + param[1], atol=1e-6, rtol=1e-6)
