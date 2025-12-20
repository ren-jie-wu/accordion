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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
shutil.rmtree(DATA_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
logger = setup_logging(DATA_DIR)

NUM_POS = 100
BATCH_SIZE = 4
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

    # encode + sample
    mu_dict, logstd_dict = model.encode(batch)
    z_dict = model.reparametrize(mu_dict, logstd_dict)

    # call nll_loss, neg_sample should be False when model.batch_negative=False
    total_edges_before = sum(
        v.shape[1] for v in batch.edge_index_dict.values()
    )

    nll, neg_edge_index_dict, _ = model.nll_loss(
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

    mu_dict, logstd_dict = model.encode(batch)
    z_dict = model.reparametrize(mu_dict, logstd_dict)

    nll, neg_edge_index_dict, _ = model.nll_loss(
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
