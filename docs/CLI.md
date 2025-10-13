# CLI reference

Generated: 2025-10-13T03:13:33.318987Z

Run `python -m simba_plus.simba_plus <subcommand> -h` for usage examples.

## `load_data`

```
usage: simba+ load_data [-h] [--gene-adata GENE_ADATA]
                        [--peak-adata PEAK_ADATA] [--batch-col BATCH_COL]
                        out_path

Load a HeteroData object from a given path and move it to the specified
device.

positional arguments:
  out_path              Path to the saved HeteroData object (e.g., .pt file).

options:
  -h, --help            show this help message and exit
  --gene-adata GENE_ADATA
                        Path to the cell by gene AnnData file (e.g., .h5ad).
  --peak-adata PEAK_ADATA
                        Path to the cell by gene AnnData file (e.g., .h5ad).
  --batch-col BATCH_COL
                        Batch column in AnnData.obs of gene AnnData. If gene
                        AnnData is not provided, peak AnnData will be used.
```

## `train`

```
usage: simba+ train [-h] [--adata-CG ADATA_CG] [--adata-CP ADATA_CP]
                    [--batch-size BATCH_SIZE] [--output-dir OUTPUT_DIR]
                    [--load-checkpoint] [--promote-indep]
                    [--hidden-dims HIDDEN_DIMS] [--hsic-lam HSIC_LAM]
                    [--get-adata] [--pos-scale]
                    data_path

Train SIMBA+ model on the given HetData object.

positional arguments:
  data_path             Path to the input data file (HetData.dat or similar)

options:
  -h, --help            show this help message and exit
  --adata-CG ADATA_CG   Path to gene AnnData (.h5ad) file for fetching
                        cell/gene metadata. Output adata_G.h5ad will have no
                        .obs attribute if not provided.
  --adata-CP ADATA_CP   Path to peak/ATAC AnnData (.h5ad) file for fetching
                        cell/peak metadata. Output adata_G.h5ad will have no
                        .obs attribute if not provided.
  --batch-size BATCH_SIZE
                        Batch size (number of edges) per DataLoader batch
  --output-dir OUTPUT_DIR
                        Top-level output directory where run artifacts will be
                        stored
  --load-checkpoint     If set, resume training from the last checkpoint
  --promote-indep       Enable HSIC-based independence promotion between
                        modalities
  --hidden-dims HIDDEN_DIMS
                        Dimensionality of hidden and latent embeddings
  --hsic-lam HSIC_LAM   HSIC regularization lambda (strength)
  --get-adata           Only extract and save AnnData outputs from the last
                        checkpoint and exit
  --pos-scale           Use positive-only scaling for the mean of output
                        distributions
```

## `eval`

```
usage: simba+ eval [-h] [--idx-path IDX_PATH] [--batch-size BATCH_SIZE]
                   [--n-negative-samples N_NEGATIVE_SAMPLES] [--device DEVICE]
                   data_path model_path

Evaluate the Simba+ model on a given dataset.

positional arguments:
  data_path             Path to the dataset.
  model_path            Path to the trained model.

options:
  -h, --help            show this help message and exit
  --idx-path IDX_PATH   Path to the index file.
  --batch-size BATCH_SIZE
                        Batch size for evaluation.
  --n-negative-samples N_NEGATIVE_SAMPLES
                        Number of negative samples for evaluation.
  --device DEVICE       Device to run the evaluation on.
```

