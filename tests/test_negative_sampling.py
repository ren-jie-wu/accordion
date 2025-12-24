# tests/test_negative_sampling.py
import os
import shutil
import pickle as pkl
import math
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from simba_plus.loader import CustomNSMultiIndexDataset, collate
from simba_plus.utils import get_edge_split_datamodule
from simba_plus.model_prox import LightningProxModel
from simba_plus.train import setup_logging

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/test_negative_sampling/")
shutil.rmtree(DATA_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
logger = setup_logging(DATA_DIR)

NUM_POS = 100
BATCH_SIZE = 10
NEG_FOLD = 2


def make_toy_heterodata():
    data = HeteroData()
    data["cell"].x = torch.zeros((50, 1))
    data["gene"].x = torch.zeros((NUM_POS, 1))
    # 100 positive edges
    edge_index = torch.tensor(
        [
            [i for i in range(50) for _ in range(math.ceil(NUM_POS / 50))][:NUM_POS],
            list(range(NUM_POS))
        ],
        dtype=torch.long,
    )
    data["cell", "expresses", "gene"].edge_index = edge_index
    data["cell", "expresses", "gene"].edge_attr = torch.ones(edge_index.size(1))
    data["cell", "expresses", "gene"].edge_dist = "NegativeBinomial"

    data.generate_ids()

    out_path = os.path.join(DATA_DIR, "toy_hetdata.dat")
    with open(out_path, "wb") as f:
        pkl.dump(data, f)

    return data, ("cell", "expresses", "gene"), out_path

def make_pldata(batch_negative=False):
    data, etype, data_path = make_toy_heterodata()
    pldata = get_edge_split_datamodule(
        data=data,
        data_path=data_path,
        edge_types=[etype],
        batch_size=BATCH_SIZE,
        num_workers=0,
        negative_sampling_fold=0 if batch_negative else NEG_FOLD,
        logger=logger
    )
    return pldata

def test_dataset_negative_sampling_dataloader_behaviour():
    pldata = make_pldata(batch_negative=False)
    train_loader, val_loader = pldata.train_dataloader(), pldata.val_dataloader() # repeat negative sampling
    assert train_loader.dataset.total_length == int(NUM_POS * 0.96) * (1 + NEG_FOLD), "train_loader.dataset.total_length is not correct"
    assert val_loader.dataset.total_length == int(NUM_POS * 0.02) * (1 + NEG_FOLD), "val_loader.dataset.total_length is not correct"

    assert len(list(train_loader)) == math.ceil(int(NUM_POS * 0.96) * (1 + NEG_FOLD) / BATCH_SIZE), "len(list(train_loader)) is not correct"

    batch = next(iter(train_loader))
    edge_attr = batch[("cell", "expresses", "gene")].edge_attr
    assert edge_attr.numel() == BATCH_SIZE, "edge_attr.numel() is not correct"
    edge_index = batch[("cell", "expresses", "gene")].edge_index
    assert edge_index.numel() == BATCH_SIZE * 2, "edge_index.numel() is not correct"

    all_edge_attr = torch.cat([batch[("cell", "expresses", "gene")].edge_attr for batch in train_loader])
    assert all_edge_attr.numel() == int(NUM_POS * 0.96) * (1 + NEG_FOLD), "all_edge_attr.numel() is not correct"
    assert int(torch.sum(all_edge_attr == 1)) == int(NUM_POS * 0.96), "int(torch.sum(all_edge_attr == 1)) is not correct"
    assert int(torch.sum(all_edge_attr == 0)) == int(NUM_POS * 0.96) * NEG_FOLD, "int(torch.sum(all_edge_attr == 0)) is not correct"

def test_batch_negative_sampling_dataloader_has_only_positive_edges():
    pldata = make_pldata(batch_negative=True)
    train_loader, val_loader = pldata.train_dataloader(), pldata.val_dataloader() # repeat negative sampling
    assert train_loader.dataset.total_length == int(NUM_POS * 0.96), "train_loader.dataset.total_length is not correct"
    assert val_loader.dataset.total_length == int(NUM_POS * 0.02), "val_loader.dataset.total_length is not correct"
    
    assert len(list(train_loader)) == math.ceil(int(NUM_POS * 0.96) / BATCH_SIZE), "len(list(train_loader)) is not correct"
    
    batch = next(iter(train_loader))
    edge_attr = batch[("cell", "expresses", "gene")].edge_attr
    assert edge_attr.numel() == BATCH_SIZE, "edge_attr.numel() is not correct"
    edge_index = batch[("cell", "expresses", "gene")].edge_index
    assert edge_index.numel() == BATCH_SIZE * 2, "edge_index.numel() is not correct"

    all_edge_attr = torch.cat([batch[("cell", "expresses", "gene")].edge_attr for batch in train_loader])
    assert all_edge_attr.numel() == int(NUM_POS * 0.96), "all_edge_attr.numel() is not correct"
    assert int(torch.sum(all_edge_attr == 1)) == int(NUM_POS * 0.96), "int(torch.sum(all_edge_attr == 1)) is not correct"
    assert int(torch.sum(all_edge_attr == 0)) == 0, "int(torch.sum(all_edge_attr == 0)) is not correct"

def test_model_no_extra_negatives_when_batch_negative_false():
    pldata = make_pldata(batch_negative=False)
    loader = pldata.train_dataloader()
    data = loader.dataset.data
    batch = next(iter(loader))

    # arbitrary node_weights_dict; only for initialization
    node_weights_dict = {
        "cell": torch.ones(data["cell"].x.size(0)),
        "gene": torch.ones(data["gene"].x.size(0)),
    }

    model = LightningProxModel(
        data=data,
        node_weights_dict=node_weights_dict,
        num_neg_samples_fold=NEG_FOLD,
        batch_negative=False,  # dataset-negative mode
    )
    model.log = lambda *a, **kw: None # disable logging

    # encode + sample
    mu_dict, logstd_dict = model.encode(batch)
    z_dict = model.reparametrize(mu_dict, logstd_dict)

    # call nll_loss, neg_sample should be False when model.batch_negative=False
    total_edges_before = sum(
        v.shape[1] for v in batch.edge_index_dict.values()
    )

    *_, neg_edge_index_dict, _ = model.nll_loss(
        batch=batch,
        z_dict=z_dict,
        pos_edge_index_dict=batch.edge_index_dict,
        pos_edge_weight_dict=batch.edge_attr_dict,
        neg_edge_index_dict=None,
        neg_sample=model.batch_negative,  # False
    )

    # should not generate additional neg_edge_index_dict
    assert neg_edge_index_dict is None, "neg_edge_index_dict should be None"

    # NLL only based on the number of edges in the batch (positive + negative)
    # here we don't check the values, as long as the number of edges is consistent
    assert total_edges_before == sum(
        v.shape[1] for v in batch.edge_index_dict.values()
    ), "total_edges_before is not correct"

def test_model_generates_negatives_when_batch_negative_true():
    pldata = make_pldata(batch_negative=True)
    loader = pldata.train_dataloader()
    data = loader.dataset.data
    batch = next(iter(loader))

    node_weights_dict = {
        "cell": torch.ones(data["cell"].x.size(0)),
        "gene": torch.ones(data["gene"].x.size(0)),
    }

    model = LightningProxModel(
        data=data,
        node_weights_dict=node_weights_dict,
        num_neg_samples_fold=NEG_FOLD,
        batch_negative=True,  # batch-negative mode
    )
    model.log = lambda *a, **kw: None # disable logging

    mu_dict, logstd_dict = model.encode(batch)
    z_dict = model.reparametrize(mu_dict, logstd_dict)

    *_, neg_edge_index_dict, _ = model.nll_loss(
        batch=batch,
        z_dict=z_dict,
        pos_edge_index_dict=batch.edge_index_dict,
        pos_edge_weight_dict=batch.edge_attr_dict,
        neg_edge_index_dict=None,
        neg_sample=model.batch_negative,  # True
    )

    # should generate neg_edge_index_dict
    assert neg_edge_index_dict is not None, "neg_edge_index_dict should not be None"
    assert ("cell", "expresses", "gene") in neg_edge_index_dict, "neg_edge_index_dict should contain the edge type"

    n_pos = batch.edge_index_dict[("cell", "expresses", "gene")].shape[1]
    n_neg = neg_edge_index_dict[("cell", "expresses", "gene")].shape[1]

    # negative sampling within batch: negative edges = positive edges * NEG_FOLD
    assert n_neg == n_pos * NEG_FOLD, "n_neg is not correct"


# ------------------------------------------------------------------------------------------------
# Multiple sample data negative sampling test

def make_toy_heterodata_two_samples():
    data = HeteroData()
    n_cells = 50
    n_genes = NUM_POS  # 100
    data["cell"].x = torch.zeros((n_cells, 1))
    data["gene"].x = torch.zeros((n_genes, 1))

    # sample labels: 2 samples
    data["cell"].sample = torch.tensor([0] * 25 + [1] * 25, dtype=torch.long)
    data["gene"].sample = torch.tensor([0] * 50 + [1] * 50, dtype=torch.long)

    # 100 positive edges, 50 in sample0 block, 50 in sample1 block
    # sample0: cells [0..24], genes [0..49]
    # sample1: cells [25..49], genes [50..99]
    src0 = torch.tensor([i % 25 for i in range(50)], dtype=torch.long)          # 0..24 repeat
    dst0 = torch.tensor([i for i in range(50)], dtype=torch.long)               # 0..49
    src1 = torch.tensor([25 + (i % 25) for i in range(50)], dtype=torch.long)   # 25..49 repeat
    dst1 = torch.tensor([50 + i for i in range(50)], dtype=torch.long)          # 50..99

    edge_index = torch.stack([torch.cat([src0, src1]), torch.cat([dst0, dst1])], dim=0)
    data["cell", "expresses", "gene"].edge_index = edge_index
    data["cell", "expresses", "gene"].edge_attr = torch.ones(edge_index.size(1))
    data["cell", "expresses", "gene"].edge_dist = "NegativeBinomial"

    data.generate_ids()

    out_path = os.path.join(DATA_DIR, "toy_hetdata_two_samples.dat")
    with open(out_path, "wb") as f:
        pkl.dump(data, f)

    return data, ("cell", "expresses", "gene"), out_path

def make_pldata_two_samples(batch_negative=True):
    data, etype, data_path = make_toy_heterodata_two_samples()
    pldata = get_edge_split_datamodule(
        data=data,
        data_path=data_path,
        edge_types=[etype],
        batch_size=BATCH_SIZE,
        num_workers=0,
        negative_sampling_fold=0 if batch_negative else NEG_FOLD,
        logger=logger,
    )
    return pldata

def test_dataset_negative_sampling_same_sample_in_full_data():
    pldata = make_pldata_two_samples(batch_negative=False)  # dataset-negative
    train_loader = pldata.train_dataloader()  # triggers dataset.sample_negative()

    ds = train_loader.dataset
    full_data = ds.full_data
    etype = ("cell", "expresses", "gene")

    # 1) Number: full_data contains pos+neg edges
    n_total = full_data[etype].edge_index.size(1)
    n_pos = ds.data[etype].edge_index.size(1)  # original positive edges in full graph
    n_neg = n_total - n_pos
    assert n_neg == n_pos * NEG_FOLD, f"Expected {n_pos*NEG_FOLD} neg edges, got {n_neg}"

    # 2) No cross-sample: only check negative edges (edge_attr==0)
    neg_mask = full_data[etype].edge_attr == 0
    neg_ei = full_data[etype].edge_index[:, neg_mask]
    cs = full_data["cell"].sample[neg_ei[0]]
    gs = full_data["gene"].sample[neg_ei[1]]
    assert torch.all(cs == gs), "Found cross-sample negative edges in dataset negative sampling!"

def test_dataset_negative_sampling_batches_do_not_cross_sample():
    pldata = make_pldata_two_samples(batch_negative=False)
    loader = pldata.train_dataloader()
    batch = next(iter(loader))
    etype = ("cell", "expresses", "gene")

    ei = batch[etype].edge_index
    attr = batch[etype].edge_attr
    neg_mask = (attr == 0)

    if int(neg_mask.sum().item()) == 0:
        return  # Very small batch may not sample any negative edges, allow skipping

    neg_ei = ei[:, neg_mask]
    cs = batch["cell"].sample[neg_ei[0]]
    gs = batch["gene"].sample[neg_ei[1]]
    assert torch.all(cs == gs), "Found cross-sample neg edges in a batch!"

def test_batch_negative_sampling_same_sample_in_model():
    pldata = make_pldata_two_samples(batch_negative=True)
    loader = pldata.train_dataloader()
    data = loader.dataset.data
    batch = next(iter(loader))

    node_weights_dict = {
        "cell": torch.ones(data["cell"].x.size(0)),
        "gene": torch.ones(data["gene"].x.size(0)),
    }

    model = LightningProxModel(
        data=data,
        node_weights_dict=node_weights_dict,
        num_neg_samples_fold=NEG_FOLD,
        batch_negative=True,
    )
    model.log = lambda *a, **kw: None # disable logging

    mu_dict, logstd_dict = model.encode(batch)
    z_dict = model.reparametrize(mu_dict, logstd_dict)

    *_, neg_edge_index_dict, _ = model.nll_loss(
        batch=batch,
        z_dict=z_dict,
        pos_edge_index_dict=batch.edge_index_dict,
        pos_edge_weight_dict=batch.edge_attr_dict,
        neg_edge_index_dict=None,
        neg_sample=True,
    )

    etype = ("cell", "expresses", "gene")
    neg_ei = neg_edge_index_dict[etype]
    # assert no cross-sample negatives
    cs = batch["cell"].sample[neg_ei[0]]
    gs = batch["gene"].sample[neg_ei[1]]
    assert torch.all(cs == gs), "Found cross-sample negative edges in batch-negative sampling!"

    pos_ei = batch.edge_index_dict[etype]
    src_sample = batch["cell"].sample
    dst_sample = batch["gene"].sample

    edge_s = src_sample[pos_ei[0]]
    expected = 0
    for s in torch.unique(edge_s).tolist():
        mask = (edge_s == s)
        n_pos_s = int(mask.sum().item())
        if n_pos_s == 0:
            continue
        n_src_s = int((src_sample == s).sum().item())
        n_dst_s = int((dst_sample == s).sum().item())
        cap_s = n_src_s * n_dst_s - n_pos_s
        expected += min(n_pos_s * NEG_FOLD, cap_s)
    
    n_neg = neg_ei.shape[1]
    assert n_neg == expected, f"n_neg mismatch: got {n_neg}, expected {expected}"
