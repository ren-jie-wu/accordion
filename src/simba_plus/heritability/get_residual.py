"""Get residual summary statistics after regressing out baseline covariates."""

import os
import numpy as np
import pandas as pd
import anndata as ad
from simba_plus.heritability.utils import get_overlap, plot_hist
from simba_plus.heritability.ldsc import run_ldsc_h2


def get_residual(sumstat_list_path, output_path, rerun=False, nprocs=10):
    sumstat_paths = list(
        pd.read_csv(sumstat_list_path, sep="\t", index_col=0, header=None).values[:, 0]
    )
    run_ldsc_h2(sumstat_list_path, output_path, rerun=rerun, nprocs=nprocs)
    ref_bed = pd.read_csv(
        os.path.dirname(__file__)
        + "/../../../data/ldsc_data/1000G_Phase3_plinkfiles/ref.txt",
        sep="\t",
        header=None,
    )
    ref_bed.columns = ["SNP", "chrom", "start"]
    residual_paths = [
        os.path.join(
            output_path,
            os.path.basename(p).split(".gz")[0].split(".sumstats")[0] + ".residuals",
        )
        for p in sumstat_paths
    ]
    residuals = pd.concat(
        [ref_bed[["SNP", "chrom", "start"]]]
        + [pd.read_csv(p, sep="\t")["residuals"].astype(float) for p in residual_paths],
        axis=1,
        ignore_index=True,
    )
    residuals.columns = ["SNP", "CHR", "BP"] + [
        os.path.basename(p).replace(".sumstats", "") for p in sumstat_paths
    ]
    return residuals


def get_peak_residual(
    ldsc_res: pd.DataFrame, adata_CP_path: str, checkpoint_dir: str, logger
) -> np.ndarray:
    """Get peak residuals by overlapping peaks with SNPs and multiplying by SNP residuals."""
    logger.info(
        f"Using provided LD score regression residuals from {ldsc_res} for scaling..."
    )
    peak_res_path = os.path.join(checkpoint_dir, "peak_res.npy")
    if False:  # os.path.exists(peak_res_path):
        logger.info(f"Loading peak_res from {peak_res_path}")
        peak_res = np.load(peak_res_path)
    else:
        logger.info(f"Saving peak_res to {peak_res_path}")
        adata_CP = ad.read_h5ad(adata_CP_path)
        if "chrom" not in adata_CP.var.columns:
            if "chr" in adata_CP.var.columns:
                adata_CP.var["chrom"] = adata_CP.var["chr"].astype(str)
            else:
                raise ValueError(
                    f"Chromosome information ('chrom' or 'chr') not found in adata_CP.var: {adata_CP.var.columns}"
                )
        if "start" not in adata_CP.var.columns or "end" not in adata_CP.var.columns:
            raise ValueError(
                f"Start or end position information ('start' or 'end') not found in adata_CP.var: {adata_CP.var.columns}"
            )
        try:
            adata_CP.var["start"] = adata_CP.var["start"].astype(int)
            adata_CP.var["end"] = adata_CP.var["end"].astype(int)
        except Exception as e:
            raise ValueError(f"Error converting 'start' or 'end' to int: {e}")
        peak_to_snp_overlap = get_overlap(ldsc_res, adata_CP.var)
        plot_hist(peak_to_snp_overlap, logger)
        ldsc_mat = ldsc_res.iloc[:, 3:].fillna(0).astype(np.float32)
        peak_res = (
            peak_to_snp_overlap / peak_to_snp_overlap.sum(axis=1)
        ) @ ldsc_mat  # n_peaks x n_sumstatss
        np.save(peak_res_path, peak_res)
    return peak_res.astype(np.float32)
