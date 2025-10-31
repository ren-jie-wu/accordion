import os
import numpy as np
import pandas as pd
from simba_plus.utils import write_bed
import pybedtools
from scipy.sparse import lil_matrix


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
