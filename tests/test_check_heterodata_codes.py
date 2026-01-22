# tests/test_check_heterodata_codes.py
import pytest
import torch
from torch_geometric.data import HeteroData

from simba_plus.util_modules import _CheckHeteroDataCodes as chk


def _make_heterodata(
    *,
    n_cell: int,
    n_gene: int = 0,
    n_peak: int = 0,
    cell_sample: torch.Tensor | None = None,
    gene_sample: torch.Tensor | None = None,
    peak_sample: torch.Tensor | None = None,
    cell_batch: torch.Tensor | None = None,
    cell_batch_local: torch.Tensor | None = None,
) -> HeteroData:
    """
    Create a minimal HeteroData with node types and code tensors only.
    No edges needed for these tests.
    """
    data = HeteroData()

    # cell
    data["cell"].num_nodes = int(n_cell)
    if cell_sample is not None:
        assert cell_sample.numel() == n_cell
        data["cell"].sample = cell_sample.long()
    if cell_batch is not None:
        assert cell_batch.numel() == n_cell
        data["cell"].batch = cell_batch.long()
    if cell_batch_local is not None:
        assert cell_batch_local.numel() == n_cell
        data["cell"].batch_local = cell_batch_local.long()

    # gene (optional)
    if n_gene > 0:
        data["gene"].num_nodes = int(n_gene)
        if gene_sample is not None:
            assert gene_sample.numel() == n_gene
            data["gene"].sample = gene_sample.long()

    # peak (optional)
    if n_peak > 0:
        data["peak"].num_nodes = int(n_peak)
        if peak_sample is not None:
            assert peak_sample.numel() == n_peak
            data["peak"].sample = peak_sample.long()

    return data


# -------------------------
# Pure function unit tests
# -------------------------

def test_remap_to_consecutive_basic():
    x = torch.tensor([10, 10, 5, 7], dtype=torch.long)
    y = chk._remap_to_consecutive(x)
    assert y.shape == x.shape
    uniq = sorted(torch.unique(y).tolist())
    assert uniq == list(range(len(uniq)))  # must be [0..K-1]
    # ensure same original values map to same new value
    assert y[0].item() == y[1].item()


def test_remap_to_consecutive_empty():
    x = torch.empty((0,), dtype=torch.long)
    y = chk._remap_to_consecutive(x)
    assert y.numel() == 0


def test_make_batch_local_from_global_per_sample_consecutive():
    sample = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    batch = torch.tensor([10, 10, 11, 5, 7, 7], dtype=torch.long)
    bl = chk._make_batch_local_from_global(sample, batch)
    assert bl.shape == batch.shape

    # per-sample local codes must be consecutive 0..B_s-1
    for sid in sorted(torch.unique(sample).tolist()):
        mask = (sample == sid)
        u = sorted(torch.unique(bl[mask]).tolist())
        assert u == list(range(len(u)))


def test_make_batch_global_from_local_is_globally_consecutive():
    sample = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    # intentionally non-consecutive locals per sample
    batch_local = torch.tensor([5, 5, 6, 0, 3, 3, 9, 9], dtype=torch.long)
    bg = chk._make_batch_global_from_local(sample, batch_local)
    u = sorted(torch.unique(bg).tolist())
    assert u == list(range(len(u)))  # global codes must be consecutive


def test_roundtrip_global_local_consistency():
    sample = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    batch_local = torch.tensor([2, 2, 9, 10, 10, 5, 8, 8], dtype=torch.long)

    bg = chk._make_batch_global_from_local(sample, batch_local)
    bl2 = chk._make_batch_local_from_global(sample, bg)

    # expected: per-sample remap of original batch_local
    expected = torch.empty_like(batch_local)
    for sid in sorted(torch.unique(sample).tolist()):
        mask = (sample == sid)
        expected[mask] = chk._remap_to_consecutive(batch_local[mask])

    assert torch.equal(bl2, expected)


# -------------------------
# HeteroData integration tests
# -------------------------

def test_check_heterodata_ok_standard():

    # 2 samples, cell sample consecutive, gene sample subset (only sample 0)
    cell_sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    # global batch must be consecutive (0..K-1)
    cell_batch = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)
    # implied batch_local per sample:
    # sample0 has {0,1} -> local {0,1}
    # sample1 has {2}   -> local {0}
    cell_batch_local = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)

    gene_sample = torch.tensor([0, 0, 0], dtype=torch.long)  # subset of cell samples
    data = _make_heterodata(
        n_cell=5,
        n_gene=3,
        cell_sample=cell_sample,
        gene_sample=gene_sample,
        cell_batch=cell_batch,
        cell_batch_local=cell_batch_local,
    )

    ok, msgs = chk.check_heterodata(data)
    assert ok is True
    assert isinstance(msgs, list)


def test_check_heterodata_cell_sample_not_consecutive_fails():
    # samples are {0,2} not consecutive
    cell_sample = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    gene_sample = torch.tensor([0, 2], dtype=torch.long)
    data = _make_heterodata(n_cell=4, n_gene=2, cell_sample=cell_sample, gene_sample=gene_sample)

    ok, msgs = chk.check_heterodata(data)
    assert ok is False
    assert any("Sample codes for cell node type are not consecutive" in m for m in msgs)


def test_check_heterodata_feature_sample_not_subset_fails():
    cell_sample = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    gene_sample = torch.tensor([0, 2], dtype=torch.long)  # 2 not in cell samples
    data = _make_heterodata(n_cell=4, n_gene=2, cell_sample=cell_sample, gene_sample=gene_sample)

    ok, msgs = chk.check_heterodata(data)
    assert ok is False
    assert any("not a subset of cell sample codes" in m for m in msgs)


def test_check_heterodata_generate_batch_local_from_batch():
    cell_sample = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    # global batch is consecutive (0..2)
    cell_batch = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)

    gene_sample = torch.tensor([0, 1], dtype=torch.long)
    data = _make_heterodata(
        n_cell=6, n_gene=2, cell_sample=cell_sample, gene_sample=gene_sample, cell_batch=cell_batch
    )
    assert not hasattr(data["cell"], "batch_local")

    ok, msgs = chk.check_heterodata(data)
    assert ok is True
    assert hasattr(data["cell"], "batch_local")
    assert any("Generated cell.batch_local" in m for m in msgs)

    # per sample batch_local should be consecutive
    for sid in sorted(torch.unique(cell_sample).tolist()):
        mask = (cell_sample == sid)
        u = sorted(torch.unique(data["cell"].batch_local[mask]).tolist())
        assert u == list(range(len(u)))


def test_check_heterodata_generate_batch_from_batch_local():
    cell_sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    # local codes can be non-consecutive; checker should generate global consecutive
    cell_batch_local = torch.tensor([5, 5, 9, 0, 3], dtype=torch.long)

    gene_sample = torch.tensor([0, 1], dtype=torch.long)
    data = _make_heterodata(
        n_cell=5, n_gene=2, cell_sample=cell_sample, gene_sample=gene_sample, cell_batch_local=cell_batch_local
    )
    assert not hasattr(data["cell"], "batch")

    ok, msgs = chk.check_heterodata(data)
    assert ok is True
    assert hasattr(data["cell"], "batch")
    assert any("Generated cell.batch from cell.batch_local" in m for m in msgs)

    # global batch must be consecutive
    u = sorted(torch.unique(data["cell"].batch).tolist())
    assert u == list(range(len(u)))


def test_check_heterodata_inconsistent_batch_local_fails():
    cell_sample = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cell_batch = torch.tensor([0, 1, 2, 2], dtype=torch.long)  # consecutive (0..2)

    # correct batch_local would be: sample0 {0,1}->{0,1}; sample1 {2}->{0}
    # we provide wrong local for last cell
    wrong_batch_local = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    gene_sample = torch.tensor([0, 1], dtype=torch.long)
    data = _make_heterodata(
        n_cell=4,
        n_gene=2,
        cell_sample=cell_sample,
        gene_sample=gene_sample,
        cell_batch=cell_batch,
        cell_batch_local=wrong_batch_local,
    )

    ok, msgs = chk.check_heterodata(data)
    assert ok is False
    assert any("Inconsistent cell.batch_local vs cell.batch" in m for m in msgs)
