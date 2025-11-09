import subprocess
import os
import tempfile
import pytest


@pytest.mark.order(1)
def test_simba_load_data_command():
    """Test the simba+ load_data command line interface."""

    test_input_dir = os.path.join(os.path.dirname(__file__), "../data/input/")
    gene_adata = os.path.join(test_input_dir, "adata_CG_sub.h5ad")
    peak_adata = os.path.join(test_input_dir, "adata_CP_sub.h5ad")
    output_file = os.path.join(test_input_dir, "hetdata.dat")

    # Run the command
    cmd = [
        "simba+",
        "load_data",
        "--gene-adata",
        gene_adata,
        "--peak-adata",
        peak_adata,
        output_file,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert command succeeded
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Assert output file was created
    assert os.path.exists(output_file), "Output file was not created"


@pytest.mark.order(1)
def test_simba_load_data_command():
    """Test the simba+ load_data command line interface."""

    test_input_dir = os.path.join(os.path.dirname(__file__), "../data/input/")
    gene_adata = os.path.join(test_input_dir, "adata_CG_sub.h5ad")
    peak_adata = os.path.join(test_input_dir, "adata_CP_sub.h5ad")
    output_file = os.path.join(test_input_dir, "hetdata_batched.dat")

    # Run the command
    cmd = [
        "simba+",
        "load_data",
        "--gene-adata",
        gene_adata,
        "--peak-adata",
        peak_adata,
        output_file,
        "--batch-col",
        "batch",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert command succeeded
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Assert output file was created
    assert os.path.exists(output_file), "Output file was not created"
