Run `python -m simba_plus.simba_plus <subcommand> -h` for usage examples.

## simba+ `load_data` ... 

```
usage: simba+ load_data [-h] [--gene-adata GENE_ADATA [GENE_ADATA ...]]
                        [--peak-adata PEAK_ADATA]
                        [--batch-col BATCH_COL [BATCH_COL ...]]
                        --out-path OUT_PATH

Prepare a HeteroData object from AnnData of RNA-seq and ATAC-seq data.

options:
  -h, --help            show this help message and exit
  --gene-adata GENE_ADATA [GENE_ADATA ...]
                        Path to the cell by gene AnnData file(s) (e.g.,
                        .h5ad).
  --peak-adata PEAK_ADATA
                        Path to the cell by gene AnnData file (e.g., .h5ad).
  --batch-col BATCH_COL [BATCH_COL ...]
                        Batch column(s) in AnnData.obs of gene AnnData(s).
                        Should match the number of gene AnnData(s), or be a
                        single value to broadcast to all gene AnnData(s). If
                        gene AnnData is not provided, peak AnnData will be
                        used.
  --out-path OUT_PATH   Path to the saved HeteroData object (e.g., .dat file).
```

## simba+ `train` ... 

```
usage: simba+ train [-h] [--adata-CG ADATA_CG] [--adata-CP ADATA_CP]
                    [--batch-size BATCH_SIZE] [--batch-negative]
                    [--output-dir OUTPUT_DIR] [--sumstats SUMSTATS]
                    [--sumstats-lam SUMSTATS_LAM]
                    [--negative-sampling-fold NEGATIVE_SAMPLING_FOLD]
                    [--load-checkpoint]
                    [--checkpoint-suffix CHECKPOINT_SUFFIX]
                    [--hidden-dims HIDDEN_DIMS] [--hsic-lam HSIC_LAM]
                    [--get-adata] [--num-workers NUM_WORKERS]
                    [--early-stopping-steps EARLY_STOPPING_STEPS]
                    [--max-epochs MAX_EPOCHS] [--verbose] [--no-wandb]
                    [--kl-lambda KL_LAMBDA] [--kl-n-no KL_N_NO]
                    [--kl-n-warmup KL_N_WARMUP]
                    [--gene-align-lambda GENE_ALIGN_LAMBDA]
                    [--gene-align-n-no GENE_ALIGN_N_NO]
                    [--gene-align-n-warmup GENE_ALIGN_N_WARMUP]
                    [--ot-cs-lambda OT_CS_LAMBDA] [--ot-cs-n-no OT_CS_N_NO]
                    [--ot-cs-n-warmup OT_CS_N_WARMUP] [--ot-cs-k OT_CS_K]
                    [--ot-eps OT_EPS] [--ot-iter OT_ITER]
                    data_path

Train SIMBA+ model on the given HetData object.

positional arguments:
  data_path             Path to the input data file (hetdata.dat or similar)

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
  --batch-negative      Batch size (number of edges) per DataLoader batch
  --output-dir OUTPUT_DIR
                        Top-level output directory where run artifacts will be
                        stored
  --sumstats SUMSTATS   If provided, LDSC is run so that peak loading
                        maximally explains the residual of LD score regression
                        of summary statistics. Provide a TSV file with one
                        trait name and path to summary statistics file per
                        line.
  --sumstats-lam SUMSTATS_LAM
                        If provided with `sumstats`, weights the MSE loss for
                        sumstat residuals.
  --negative-sampling-fold NEGATIVE_SAMPLING_FOLD
                        Fold of negative samples to use during training
  --load-checkpoint     If set, resume training from the last checkpoint
  --checkpoint-suffix CHECKPOINT_SUFFIX
                        Append a suffix to checkpoint filenames
                        (last{suffix}.ckpt)
  --hidden-dims HIDDEN_DIMS
                        Dimensionality of latent embeddings
  --hsic-lam HSIC_LAM   HSIC regularization lambda (strength)
  --get-adata           Only extract and save AnnData outputs from the last
                        checkpoint and exit
  --num-workers NUM_WORKERS
                        Number of worker processes for data loading and LDSC
  --early-stopping-steps EARLY_STOPPING_STEPS
                        Number of epoch for early stopping patience
  --max-epochs MAX_EPOCHS
                        Number of max epochs for training
  --verbose             If set, enables verbose logging
  --no-wandb            Disable Weights & Biases logging (recommended for
                        CI/tests).
  --kl-lambda KL_LAMBDA
                        Weight of the KL divergence loss
  --kl-n-no KL_N_NO     Number of epochs to wait before starting KL divergence
                        loss
  --kl-n-warmup KL_N_WARMUP
                        Number of epochs for KL divergence loss warmup
  --gene-align-lambda GENE_ALIGN_LAMBDA
                        Weight of the gene alignment loss
  --gene-align-n-no GENE_ALIGN_N_NO
                        Number of epochs to wait before starting gene
                        alignment
  --gene-align-n-warmup GENE_ALIGN_N_WARMUP
                        Number of epochs for gene alignment warmup
  --ot-cs-lambda OT_CS_LAMBDA
                        Weight of the Optimal Transportation loss (among
                        cells)
  --ot-cs-n-no OT_CS_N_NO     Number of epochs to wait before starting Optimal
                        Transportation loss
  --ot-cs-n-warmup OT_CS_N_WARMUP
                        Number of epochs for Optimal Transportation loss
                        warmup
  --ot-cs-k OT_CS_K           Subsample size for Optimal Transportation loss
  --ot-eps OT_EPS       Regularization parameter for Optimal Transportation
                        loss
  --ot-iter OT_ITER     Number of iterations for Optimal Transportation loss
```

## simba+ `eval` ... 

```
usage: simba+ eval [-h] [--idx-path IDX_PATH] [--batch-size BATCH_SIZE]
                   [--eval-split {train,test,val}]
                   [--negative-sampling-fold NEGATIVE_SAMPLING_FOLD]
                   [--device DEVICE] [--rerun]
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
  --eval-split {train,test,val}
                        Which data split to use for evaluation.
  --negative-sampling-fold NEGATIVE_SAMPLING_FOLD
                        Number of negative samples for evaluation.
  --device DEVICE       Device to run the evaluation on.
  --rerun               Rerun the evaluation.
```

