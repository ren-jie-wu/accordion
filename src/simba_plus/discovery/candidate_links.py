"""Peak–gene linking utilities.

This module provides helper functions to identify candidate cis-regulatory
peak–gene pairs using ATAC/multiome peaks and transcription start site (TSS)
annotations. Peaks whose center lies within a configurable cis window around
a gene’s TSS are reported as candidate links.

Example:
    >>> from simba_plus.discovery import candidate_links as get_pgl
    >>> peaks = ["chr1:1000-2000", "chr2_3000_4500"]
    >>> genes = ["BRCA1", "TP53"]
    >>> candidate_links = get_pgl.get_peak_gene_links(
    ...     peaks, genes, cis_window=500000
    ... )
    >>> candidate_links.head()
"""

import os
import urllib.request
import pandas as pd
import pyranges as pr
from tqdm import tqdm


def read_tss_bed(path: str = "CollapsedGeneBounds.hg38.TSS500bp.bed") -> pd.DataFrame:
    """Read a TSS BED-like file into a pandas DataFrame.

    The file should follow the EngreitzLab “CollapsedGeneBounds” format with
    columns::

        chr, start, end, name, score, strand, gene_id, biotype

    If the file does not exist locally, it is downloaded automatically from
    the EngreitzLab repository.

    Args:
        path (str): Path to the BED file. Defaults to
            ``"CollapsedGeneBounds.hg38.TSS500bp.bed"``.

    Returns:
        pd.DataFrame: A DataFrame containing TSS coordinates with columns:
            ``['chr', 'start', 'end', 'name', 'score', 'strand',
            'gene_id', 'biotype']``.

    Example:
        >>> tss_df = read_tss_bed()
        >>> tss_df.head()
    """
    download_url = (
        "https://github.com/EngreitzLab/ENCODE_rE2G/raw/dev/reference/"
        "CollapsedGeneBounds.hg38.TSS500bp.bed"
    )
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading TSS annotation from {download_url} ...")
        urllib.request.urlretrieve(download_url, path)

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=[
            "chr", "start", "end", "name", "score",
            "strand", "gene_id", "biotype"
        ],
    )
    return df


def get_peak_gene_links(
    peaks: list[str],
    genes: list[str],
    tss_bed: str | None = None,
    tss_df: pd.DataFrame | None = None,
    cis_window: int = 500_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Compute candidate peak–gene links within a cis-regulatory window.

    Each peak’s center is compared to gene TSS coordinates; a gene is linked
    to a peak if its TSS lies within the specified cis window upstream or
    downstream of the peak center.

    Args:
        peaks (list[str]): List of peak strings formatted as either
            ``'chr:start-end'`` or ``'chr_start_end'``.
        genes (list[str]): List of gene names to retain in the results.
        tss_bed (str | None): Path to a TSS BED file. If not provided,
            defaults to ``CollapsedGeneBounds.hg38.TSS500bp.bed``.
        tss_df (pd.DataFrame | None): Pre-loaded TSS DataFrame. If given,
            reading from disk is skipped.
        cis_window (int): Window size in base pairs upstream and downstream
            of each peak center. Defaults to ``500_000`` bp.
        progress (bool): Whether to display a tqdm progress bar. Defaults
            to ``True``.

    Returns:
        pd.DataFrame: Candidate peak–gene link table with columns::

            ['Gene_name', 'Gene_ID', 'Peak',
             'Distance_to_TSS', 'TSS', 'peak_gene_pair']

    Example:
        >>> peaks = ["chr1:1000-2000"]
        >>> genes = ["BRCA1"]
        >>> links = get_peak_gene_links(peaks, genes)
        >>> links.head()
    """
    if tss_df is None:
        tss_df = read_tss_bed(tss_bed or "CollapsedGeneBounds.hg38.TSS500bp.bed")

    rows = []
    iterator = tqdm(peaks, desc="Parsing peaks") if progress else peaks
    for token in iterator:
        if ":" in token:
            chrom, positions = token.split(":")
            start, end = positions.split("-")
        elif "_" in token:
            chrom, start, end = token.split("_")
            if "KI" in chrom or "GL" in chrom:
                continue
        else:
            continue
        start, end = int(start), int(end)
        center = (start + end) // 2
        rows.append((token, chrom, center))

    peaks_df = pd.DataFrame(rows, columns=["Peak", "Chromosome", "Center"])
    peaks_df["CIS_START"] = (peaks_df["Center"] - cis_window).clip(lower=0)
    peaks_df["CIS_END"] = peaks_df["Center"] + cis_window

    print("Finding TSS–peak overlaps...")

    # Prepare TSS dataframe for pyranges
    tss_for_pr = tss_df.copy()
    tss_for_pr["TSS_start"] = tss_df["start"]
    tss_for_pr["TSS_start1"] = tss_df["start"] + 1

    tss_pr_df = pd.DataFrame({
        "Chromosome": tss_for_pr["chr"],
        "Start": tss_for_pr["start"],
        "End": tss_for_pr["start"] + 1,  # TSS is a point
        "Gene_name": tss_for_pr["name"],
        "Gene_ID": tss_for_pr["gene_id"],
        "TSS_start": tss_for_pr["TSS_start"],
    })

    peaks_pr_df = pd.DataFrame({
        "Chromosome": peaks_df["Chromosome"],
        "Start": peaks_df["CIS_START"],
        "End": peaks_df["CIS_END"],
        "Peak": peaks_df["Peak"],
        "Center": peaks_df["Center"],
    })

    # Create PyRanges objects
    tss_pr = pr.PyRanges(tss_pr_df)
    peaks_pr = pr.PyRanges(peaks_pr_df)

    # Perform intersection (equivalent to bedtools intersect -wa -wb)
    overlap_pr = tss_pr.join(peaks_pr)
    ov = overlap_pr.df

    # Filter by gene subset
    if len(genes) > 0:
        ov = ov[ov["Gene_name"].isin(genes)].copy()

    ov["TSS"] = ov["TSS_start"]
    ov["Distance_to_TSS"] = (ov["TSS"] - ov["Center"]).abs()
    ov["peak_gene_pair"] = ov["Peak"] + "_" + ov["Gene_name"]

    return ov[
        ["Gene_name", "Gene_ID", "Peak",
         "Distance_to_TSS", "TSS", "peak_gene_pair"]
    ]