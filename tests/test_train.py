import os
import shutil
import subprocess
from pathlib import Path
import pytest
import simba_plus.load_data
import torch


def _which_exe(name):
    exe = shutil.which(name)
    if exe:
        return exe
    # on some systems the binary might be installed as "simba+" which is not a valid filename to find;
    # try looking for an entry in PATH that ends with 'simba+'
    for p in os.environ.get("PATH", "").split(os.pathsep):
        ppath = Path(p)
        if not ppath.exists():
            continue
        for child in ppath.iterdir():
            if child.name == name:
                return str(child)
    return None

    # Add more specific assertions based on expected behavior


@pytest.mark.order(2)
def test_train():
    exe = _which_exe("simba+")
    if exe is None:
        pytest.skip("simba+ executable not found in PATH")

    workdir = os.path.join(os.path.dirname(__file__), "../data/input/")
    hetdata = f"{workdir}/hetdata.dat"
    adata_CG = f"{workdir}/adata_CG_sub.h5ad"
    adata_CP = f"{workdir}/adata_CP_sub.h5ad"
    output_dir = f"{os.path.dirname(__file__)}/output/"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        exe,
        "train",
        hetdata,
        "--adata-CG",
        str(adata_CG),
        "--adata-CP",
        str(adata_CP),
        "--max-epoch=1",
        "--hidden-dim=3",
        "--output-dir",
        output_dir,
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=os.environ.copy()
        )
    except subprocess.CalledProcessError as e:
        # include stdout/stderr for easier debugging
        pytest.fail(
            f"simba+ train failed (exit {e.returncode})\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
    else:
        # basic sanity checks on output
        assert result.returncode == 0
        # optional: ensure some expected keywords appear in output; adjust as appropriate
        out = (result.stdout or "") + (result.stderr or "")
        assert (
            "epoch" in out.lower() or "trained" in out.lower() or "done" in out.lower()
        )


@pytest.mark.order(2)
def test_train_usebatch():
    exe = _which_exe("simba+")
    if exe is None:
        pytest.skip("simba+ executable not found in PATH")

    # prepare minimal placeholder files
    workdir = os.path.join(os.path.dirname(__file__), "../data/input/")
    hetdata = f"{workdir}/hetdata_batched.dat"
    if not os.path.exists(hetdata):
        dat = simba_plus.load_data.load_from_path(f"{workdir}/hetdata.dat")
        dat["cell"].batch = torch.randint(
            0, 2, (dat["cell"].num_nodes,), dtype=torch.long
        )
        torch.save(dat, hetdata)
    adata_CG = f"{workdir}/adata_CG_sub.h5ad"
    adata_CP = f"{workdir}/adata_CP_sub.h5ad"
    output_dir = f"{os.path.dirname(__file__)}/output/"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        exe,
        "train",
        hetdata,
        "--adata-CG",
        str(adata_CG),
        "--adata-CP",
        str(adata_CP),
        "--max-epoch=1",
        "--output-dir",
        output_dir,
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=os.environ.copy()
        )
    except subprocess.CalledProcessError as e:
        # include stdout/stderr for easier debugging
        pytest.fail(
            f"simba+ train failed (exit {e.returncode})\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
    else:
        # basic sanity checks on output
        assert result.returncode == 0
        # optional: ensure some expected keywords appear in output; adjust as appropriate
        out = (result.stdout or "") + (result.stderr or "")
        assert (
            "epoch" in out.lower() or "trained" in out.lower() or "done" in out.lower()
        )


@pytest.mark.order(3)
def test_simba_eval_command():
    """Test the simba+ eval command line interface."""

    data_dir = os.path.join(os.path.dirname(__file__), "../data/input/")
    train_output_dir = os.path.join(os.path.dirname(__file__), "output/")
    data_file = f"{data_dir}/hetdata.dat"

    # Create mock checkpoint file
    ckpt_file = (
        f"{train_output_dir}/simba+hetdata.dat_100K.d.randinit.checkpoints/last.ckpt"
    )
    idx_file = f"{data_dir}/hetdata_data_idx.pkl"

    # Run the command
    cmd = [
        "simba+",
        "eval",
        str(data_file),
        str(ckpt_file),
        "--idx-path",
        str(idx_file),
        "--batch-size=100000",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert command doesn't fail with basic file issues
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


@pytest.mark.order(4)
def test_simba_eval_command():
    """Test the simba+ eval command line interface."""

    data_dir = os.path.join(os.path.dirname(__file__), "../data/input/")
    train_output_dir = os.path.join(os.path.dirname(__file__), "output/")
    data_file = f"{data_dir}/hetdata.dat"

    # Create mock checkpoint file
    ckpt_file = (
        f"{train_output_dir}/simba+hetdata.dat_100K.d3.randinit.checkpoints/last.ckpt"
    )
    idx_file = f"{data_dir}/hetdata_data_idx.pkl"

    # Run the command
    cmd = [
        "simba+",
        "eval",
        str(data_file),
        str(ckpt_file),
        "--idx-path",
        str(idx_file),
        "--batch-size=100000",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert command doesn't fail with basic file issues
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


@pytest.mark.order(5)
def test_simba_heritability_command():

    prefix = f"{os.path.dirname(__file__)}/output/simba+hetdata.dat_100K.d3.randinit.checkpoints/"
    sumstats = f"{os.path.dirname(__file__)}/../data/sumstats/sumstats_rdw.txt"
    cmd = [
        "simba+",
        "heritability",
        str(sumstats),
        str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert command doesn't fail with basic file issues
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
