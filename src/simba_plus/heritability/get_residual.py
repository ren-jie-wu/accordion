"""Get residual summary statistics after regressing out baseline covariates."""

import os
import numpy as np
import pandas as pd
import anndata as ad
import subprocess
import pybedtools
from scipy.sparse import lil_matrix
import simba_plus.datasets._datasets


def run_ldsc(sumstat_paths, out_path, rerun=False, nproc=10):
    simba_plus.datasets._datasets.heritability()
    processes = []
    os.makedirs(out_path, exist_ok=True)
    for sumstat_path in list(sumstat_paths):
        command = f"bash {os.path.dirname(__file__)}/ldsc_h2_baseline.sh {sumstat_path} {out_path} {rerun}"
        processes.append(subprocess.Popen(command, shell=True))
        if len(processes) >= nproc:
            for process in processes:
                process.wait()
            processes = []
    for process in processes:
        process.wait()


def get_residual(sumstat_list_path, output_path, rerun=False, nproc=10):
    sumstat_paths = list(
        pd.read_csv(sumstat_list_path, sep="\t", index_col=0, header=None).values[:, 0]
    )
    run_ldsc(sumstat_paths, output_path, rerun=rerun, nproc=nproc)

    residual_paths = [
        os.path.join(
            output_path,
            os.path.basename(p).split(".gz")[0].split(".sumstats")[0] + ".residuals",
        )
        for p in sumstat_paths
    ]
    residuals = pd.concat(
        [
            pd.read_csv(sumstat_paths[0], sep="\t", compression="infer")[
                ["SNP", "CHR", "BP"]
            ]
        ]
        + [pd.read_csv(p, sep="\t")["residuals"].astype(float) for p in residual_paths],
        axis=1,
        ignore_index=True,
    )
    residuals.columns = ["SNP", "CHR", "BP"] + [
        os.path.basename(p).replace(".sumstats", "") for p in sumstat_paths
    ]
    return residuals


def get_overlap(snp_df, peak_df):
    # Convert SNP and peak dataframes to BedTool objects
    snp_df = snp_df.rename(columns={"CHR": "chrom", "BP": "start"})
    snp_df.insert(2, "end", snp_df["start"] + 1)
    snp_df["name"] = range(len(snp_df))
    snp_df = snp_df.loc[
        :, ~snp_df.columns.duplicated()
    ].copy()  # Remove duplicate columns

    snp_bed = pybedtools.BedTool.from_dataframe(
        snp_df[["chrom", "start", "end", "name"]]
    )
    peak_df = peak_df.loc[:, ~peak_df.columns.duplicated()].copy()
    # If chromosome naming convention is different, adjust here
    if peak_df["chrom"].astype(str).iloc[0].startswith("chr") and not snp_df[
        "chrom"
    ].astype(str).iloc[0].startswith("chr"):
        peak_df["chrom"] = peak_df["chrom"].apply(lambda x: x.replace("chr", ""))
    elif not peak_df["chrom"].astype(str).iloc[0].startswith("chr") and snp_df[
        "chrom"
    ].astype(str).iloc[0].startswith("chr"):
        peak_df["chrom"] = peak_df["chrom"].apply(lambda x: "chr" + x)
    peak_df["name"] = range(len(peak_df))
    peak_bed = pybedtools.BedTool.from_dataframe(
        peak_df[["chrom", "start", "end", "name"]]
    )

    # Perform intersection
    intersection = snp_bed.intersect(peak_bed, wa=True, wb=True)

    # Parse the intersection results
    # Initialize a sparse matrix with dimensions (number of peaks, number of SNPs)
    num_snps = len(snp_df)
    num_peaks = len(peak_df)
    overlap_matrix = lil_matrix((num_peaks, num_snps), dtype=int)

    for line in intersection:
        snp_index = int(line[3])  # Assuming SNP index is in the 4th column
        peak_index = int(line[7])  # Assuming peak index is in the 7th column
        overlap_matrix[peak_index, snp_index] = 1
    if overlap_matrix.sum() == 0:
        raise ValueError("No overlaps found between SNPs and peaks.")
    # Calculate the number of SNPs per peak
    return overlap_matrix


def plot_hist(overlap_matrix, logger):
    snps_per_peak = overlap_matrix.sum(axis=1).A1  # Convert sparse matrix to 1D array

    # Create an ASCII histogram
    max_count = max(snps_per_peak)
    bin_width = max(1, max_count // 50)  # Adjust bin width for better visualization
    zero_count = (snps_per_peak == 0).sum()
    high_quantile = np.percentile(snps_per_peak, 90)
    bins = (
        [0, 1]
        + list(range(1, int(high_quantile) + bin_width, bin_width))
        + [max_count + 1]
    )
    histogram = np.histogram(snps_per_peak, bins=bins)

    logger.info("ASCII Histogram of SNPs per Peak:")
    hist_str = ""
    if zero_count > 0:
        bar = "#" * (zero_count // max(1, max(histogram[0]) // 50))
        hist_str += f"          0: {bar}\n"
    for i in range(1, len(histogram[0])):
        bar = "#" * (histogram[0][i] // max(1, max(histogram[0]) // 50))
        if bar == "":
            continue
        hist_str += f"{bins[i]:>5} - {bins[i+1]:>5}: {bar}\n"
    logger.info(hist_str)


def get_peak_residual(
    ldsc_res: pd.DataFrame, adata_CP_path: str, checkpoint_dir: str, logger
) -> np.ndarray:
    """Get peak residuals by overlapping peaks with SNPs and multiplying by SNP residuals."""
    logger.info(
        f"Using provided LD score regression residuals from {ldsc_res} for scaling..."
    )
    peak_res_path = os.path.join(checkpoint_dir, "peak_res.npy")
    if os.path.exists(peak_res_path):
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
