"""Evaluation set builders for GTEx eQTL and CRISPR benchmarks.

This module constructs labeled evaluation sets for model benchmarking from
precomputed candidate peak–gene links.

Currently supports:
    * GTEx eQTL evaluation set (Whole Blood, fine-mapped)
    * CRISPR enhancer–gene benchmark evaluation set

Example:
    >>> from simba_variant import evalset_builders as eb
    >>> eb.build_eqtl_evalset(
    ...     candidate_csv="pbmc_peak_gene_candidates.csv",
    ...     gtex_file="gtex_susie.tsv.gz",
    ...     genesymbol_file="gtex_genesymbol.csv",
    ...     genomes_bed_files=["1000G.EUR.hg38.1.filtered.bed", ...],
    ...     output_csv="eqtl_pbmc.csv"
    ... )
"""

from __future__ import annotations

import pandas as pd
import pyranges as pr
import os
import urllib.request
from typing import Sequence, Optional
import numpy as np

def build_eqtl_evalset(
    candidate_csv: str,
    gtex_file: str,
    genomes_bed_files: Sequence[str],
    output_csv: str,
    peak_col: str = "Peak",
    gene_col: str = "Gene_name",
    distance_col: str = "Distance_to_TSS",
    gtex_method: str = "SUSIE",
    gtex_tissue: str = "Whole_Blood",
    pip_pos: float = 0.5,
    pip_neg: float = 0.01
) -> pd.DataFrame:
    """Construct a GTEx eQTL evaluation set from candidate peak–gene links."""
    candidates = pd.read_csv(candidate_csv)
    variants_bed = _load_1000g_variants(genomes_bed_files)
    peak_gene_links = _overlap_variants_with_candidates(
        candidates, variants_bed, peak_col, gene_col, distance_col
    )
    gtex_df = _load_gtex_eqtls(gtex_file, method=gtex_method, tissue=gtex_tissue)
    labeled_df = _label_eqtl_training_set(peak_gene_links, gtex_df, pip_pos, pip_neg)

    labeled_df.index = labeled_df.CHROM
    labeled_df.to_csv(output_csv, index=True, header=True)
    return labeled_df

def _inverse_distance(df: pd.DataFrame, distance_col: str) -> pd.DataFrame:
    """Add an inverse distance feature to a DataFrame.

    Args:
        df (pd.DataFrame): Must contain the specified ``distance_col`` in bp.
        distance_col (str): Column name for distance to TSS.

    Returns:
        pd.DataFrame: Copy of input with a new ``1/Distance`` column clipped at 1.0.
    """
    df = df.copy()
    df["1/Distance"] = 1 / (df[distance_col] + 1)
    df["1/Distance"] = df["1/Distance"].clip(upper=1)
    return df

# TODO: Download URLs for 1000G and GTEx files can be added later.
def _load_1000g_variants(genomes_bed_files: Sequence[str]) -> pr.PyRanges:
    """Load 1000 Genomes variant intervals from multiple BED files as a PyRanges object.

    This function concatenates variant BED files (e.g., per-chromosome files from
    the 1000 Genomes Project) and converts them into a single ``PyRanges`` object.

    Args:
        genomes_bed_files (Sequence[str]):
            Paths to BED files containing variant intervals with columns
            ``CHROM``, ``START``, and ``END``.

    Returns:
        pyranges.PyRanges:
            A single PyRanges object containing all concatenated variant intervals.

    Example:
        >>> bed_files = [
        ...     "chr1.1000G.bed",
        ...     "chr2.1000G.bed",
        ... ]
        >>> variants = _load_1000g_variants(bed_files)
    """
    # Read and concatenate BED files into a single DataFrame
    genomes_df = pd.concat(
        [
            pd.read_csv(f, sep="\t", header=None, names=["CHROM", "START", "END"])
            for f in genomes_bed_files
        ],
        ignore_index=True,
    )

    # Ensure consistent chromosome naming (prefix with 'chr')
    genomes_df["CHROM"] = "chr" + genomes_df["CHROM"].astype(str)

    # Convert to PyRanges (Chromosome, Start, End)
    variants_pr = pr.PyRanges(
        genomes_df.rename(columns={"CHROM": "Chromosome", "START": "Start", "END": "End"})
    )

    return variants_pr

def _overlap_variants_with_candidates(
    candidates: pd.DataFrame,
    variants_df: pd.DataFrame,
    peak_col: str,
    gene_col: str,
    distance_col: str,
) -> pd.DataFrame:
    """Intersect variant intervals with candidate peaks using PyRanges.

    This function finds all variants that fall within candidate peak regions.
    It appends variant positional info and computes the inverse distance feature.

    Args:
        candidates (pd.DataFrame):
            Candidate peak–gene links containing at least ``peak_col``,
            ``gene_col``, and ``distance_col``.
        variants_df (pd.DataFrame):
            DataFrame containing variant coordinates with columns
            ``["CHROM", "POS", "POS2"]`` (POS2 is typically POS + 1).
        peak_col (str):
            Column in ``candidates`` with peak coordinates, formatted as
            ``chr_start_end`` or ``chr:start-end``.
        gene_col (str):
            Column in ``candidates`` specifying the gene symbol or ID.
        distance_col (str):
            Column name for distance to TSS (in base pairs).

    Returns:
        pd.DataFrame: A DataFrame of variant–peak–gene overlaps containing:
            - ``CHROM``, ``POS``, ``POS2`` (variant coordinates)
            - ``Peak`` and ``Gene`` (from candidate links)
            - ``1/Distance`` feature
            - ``snp_gene_pair`` identifier combining variant + gene.

    Example:
        >>> merged = _overlap_variants_with_candidates(
        ...     candidates_df, variants_df,
        ...     peak_col="Peak", gene_col="Gene", distance_col="Distance_to_TSS"
        ... )
        >>> merged.head()
    """
    # Parse candidate peaks into BED-style columns
    split_lst = [p.replace(":", "-").split("_") for p in candidates[peak_col]]
    annotations_df = pd.DataFrame(split_lst, columns=["CHROM", "START", "END"])
    annotations_df[["START", "END"]] = annotations_df[["START", "END"]].astype(int)
    annotations_df["Peak"] = candidates[peak_col].values

    # Prepare variant coordinates for PyRanges
    variants_pr = pr.PyRanges(
        pd.DataFrame({
            "Chromosome": variants_df["CHROM"],
            "Start": variants_df["POS"],
            "End": variants_df["POS2"],
        })
    )

    # Prepare candidate peak ranges
    ann_pr = pr.PyRanges(
        pd.DataFrame({
            "Chromosome": annotations_df["CHROM"],
            "Start": annotations_df["START"],
            "End": annotations_df["END"],
            "Peak": annotations_df["Peak"],
        })
    )

    print("Intersecting variants with candidate peaks...")
    overlap_pr = variants_pr.join(ann_pr)  # Equivalent to bedtools intersect -wa -wb
    overlap_df = overlap_pr.df.rename(columns={
        "Chromosome": "CHROM",
        "Start": "POS",
        "End": "POS2",
        "Start_b": "START_ANN",
        "End_b": "END_ANN",
    })
    overlap_df["CHROM_ANN"] = overlap_df["CHROM"]

    # Reconstruct peak coordinate string
    overlap_df[peak_col] = (
        overlap_df["CHROM_ANN"] + "_" +
        overlap_df["START_ANN"].astype(str) + "_" +
        overlap_df["END_ANN"].astype(str)
    )

    # Merge back with candidate table
    merged = candidates.merge(
        overlap_df[["CHROM", "POS", "POS2", peak_col]],
        on=peak_col, how="right"
    )
    merged.rename(columns={gene_col: "Gene"}, inplace=True)

    # Compute 1/Distance
    if distance_col in merged.columns:
        merged["1/Distance"] = 1.0 / merged[distance_col].replace(0, pd.NA)

    # Create unique variant–gene pair identifier
    merged["snp_gene_pair"] = (
        merged["CHROM"] + "_" + merged["POS2"].astype(str) + "_" + merged["Gene"]
    )

    return merged

# TODO: Download URLs for 1000G and GTEx files can be added later.
def _load_gtex_eqtls(
    gtex_file: str,
    method: str = "SUSIE",
    tissue: str = "Whole_Blood"
) -> pd.DataFrame:
    """Load and filter GTEx fine-mapped eQTLs.

    Args:
        gtex_file (str): Path to GTEx fine-mapped eQTL file (gzipped TSV).
        method (str, optional): Fine-mapping method to filter. Defaults to "SUSIE".
        tissue (str, optional): GTEx tissue name to filter. Defaults to "Whole_Blood".

    Returns:
        pd.DataFrame: Filtered GTEx eQTLs with ``pair`` (chr_pos_gene) column.
    """
    gtex_df = pd.read_csv(gtex_file, sep="\t", compression="gzip")
    gtex_df = gtex_df[(gtex_df.method == method) & (gtex_df.tissue == tissue)]
    gtex_df["pair"] = (
        gtex_df["chr"] + "_" +
        gtex_df["start"].astype(str) + "_" +
        gtex_df["gene"]
    )
    gtex_df.drop_duplicates("pair", inplace=True)
    return gtex_df

def _label_eqtl_training_set(
    peak_gene_links: pd.DataFrame,
    gtex_df: pd.DataFrame,
    pip_pos: float = 0.5,
    pip_neg: float = 0.01
) -> pd.DataFrame:
    """Assign binary labels to SNP–gene pairs using PIP thresholds."""
    pos_training = gtex_df[gtex_df["pip"] > pip_pos].drop_duplicates("pair")
    neg_training = gtex_df[gtex_df["pip"] < pip_neg].drop_duplicates("pair")

    pos = peak_gene_links[peak_gene_links.snp_gene_pair.isin(pos_training.pair)].copy()
    pos["label"] = 1
    neg = peak_gene_links[peak_gene_links.snp_gene_pair.isin(neg_training.pair)].copy()
    neg["label"] = 0

    pos.drop_duplicates("snp_gene_pair", inplace=True)
    neg.drop_duplicates("snp_gene_pair", inplace=True)

    return pd.concat([pos, neg], axis=0)

def build_crispr_evalset(
    candidate_csv: str,
    crispr_file: str,
    adata_cg_genes: Optional[Sequence[str]] = None,
    output_csv: Optional[str] = None,
    peak_col: str = "Peak",
    gene_col: str = "Gene_name",
    distance_col: Optional[str] = "Distance_to_TSS",
) -> pd.DataFrame:
    """Construct a CRISPR evaluation set from candidate peak–gene links.

    This overlaps CRISPR enhancer elements with candidate peaks, keeps rows where
    the CRISPR measured gene matches the candidate gene, optionally filters to genes
    present in your RNA modality, and assigns labels from the CRISPR benchmark.

    Args:
        candidate_csv: Path to candidate peak–gene links CSV (BMMC or other).
        crispr_file: Path to CRISPR benchmark TSV (gzipped), with columns
            ``chrom``, ``chromStart``, ``chromEnd``, ``measuredGeneSymbol``, ``Regulated``.
        adata_cg_genes: Optional list of genes detected in RNA
            (to filter CRISPR overlaps to your assay's gene set).
        output_csv: If provided, write the labeled set to this path.
        peak_col: Column in candidates containing peak coordinates. Defaults to ``"Peak"``.
        gene_col: Column in candidates containing gene ID/symbol. Defaults to ``"Gene_name"``.
        distance_col: Column in candidates with distance to TSS in bp.
            If provided and present, a ``1/Distance`` feature will be added.

    Returns:
        Labeled CRISPR evaluation set with ``Peak``, ``Gene``, ``label`` (1/0),
        and optional ``1/Distance``.
    """
    print("Building CRISPR evaluation set...")
    candidates = _load_and_validate_candidates(candidate_csv, peak_col, gene_col, distance_col)
    crispr_eval = _load_and_validate_crispr(crispr_file)

    overlap_df = _intersect_crispr_with_candidates(crispr_eval, candidates, distance_col)

    if adata_cg_genes is not None:
        overlap_df = overlap_df[overlap_df["crispr_gene"].isin(set(adata_cg_genes))]

    overlap_df = overlap_df[overlap_df["crispr_gene"] == overlap_df["Gene"]].copy()
    overlap_df["label"] = np.where(overlap_df["crispr_regulated"].astype(str) == "True", 1, 0)

    if distance_col and distance_col in overlap_df.columns:
        overlap_df["1/Distance"] = 1 / (overlap_df[distance_col].astype(float) + 1.0)
        overlap_df["1/Distance"] = overlap_df["1/Distance"].clip(upper=1.0)

    overlap_df.reset_index(drop=True, inplace=True)

    if output_csv:
        overlap_df.to_csv(output_csv, index=False)

    return overlap_df

def _load_and_validate_candidates(
    candidate_csv: str,
    peak_col: str,
    gene_col: str,
    distance_col: Optional[str]
) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    missing = {peak_col, gene_col} - set(df.columns)
    if missing:
        raise ValueError(f"candidate_csv missing required columns: {missing}")

    df = df.rename(columns={peak_col: "Peak", gene_col: "Gene"}).copy()

    bed_df = _parse_peak_series_to_bed_df(df["Peak"])
    if bed_df.empty:
        raise ValueError("No valid peaks parsed from candidate_csv; check peak format.")

    bed_df["Peak"] = df.loc[bed_df.index, "Peak"].values
    bed_df["Gene"] = df.loc[bed_df.index, "Gene"].values

    if distance_col and distance_col in df.columns:
        bed_df[distance_col] = df.loc[bed_df.index, distance_col].values

    return bed_df

def _load_and_validate_crispr(crispr_file: str) -> pd.DataFrame:
    download_url = "https://github.com/EngreitzLab/CRISPR_comparison/raw/main/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
    if not os.path.exists(crispr_file):
        os.makedirs(os.path.dirname(crispr_file) or ".", exist_ok=True)
        print(f"Downloading CRISPR benchmark dataset from {download_url} ...")
        urllib.request.urlretrieve(download_url, crispr_file)
        
    df = pd.read_csv(crispr_file, compression="gzip", sep="\t")
    required = {"chrom", "chromStart", "chromEnd", "measuredGeneSymbol", "Regulated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"crispr_file missing required columns: {missing}")
    return df

def _intersect_crispr_with_candidates(
    crispr_df: pd.DataFrame,
    candidate_bed_df: pd.DataFrame,
    distance_col: str | None,
) -> pd.DataFrame:
    """Intersect CRISPR regulatory elements with candidate peak–gene pairs using PyRanges.

    Args:
        crispr_df (pd.DataFrame):
            DataFrame containing CRISPR regulatory elements.
            Must include columns:
            ``["chrom", "chromStart", "chromEnd", "measuredGeneSymbol", "Regulated"]``.
        candidate_bed_df (pd.DataFrame):
            DataFrame of candidate peak–gene regions.
            Must include columns:
            ``["chrom", "chromStart", "chromEnd", "Peak", "Gene"]`` and optionally a distance column.
        distance_col (str | None):
            Name of the distance column in ``candidate_bed_df`` to retain in the output.
            If not present or None, it is skipped.

    Returns:
        pd.DataFrame:
            DataFrame of overlapping CRISPR–peak pairs with columns::

                ["crispr_chrom", "crispr_start", "crispr_end",
                 "crispr_gene", "crispr_regulated",
                 "peak_chrom", "peak_start", "peak_end",
                 "Peak", "Gene", (optional distance_col)]
    """
    # Prepare CRISPR dataframe
    crispr_pr_df = pd.DataFrame({
        "Chromosome": crispr_df["chrom"],
        "Start": crispr_df["chromStart"],
        "End": crispr_df["chromEnd"],
        "crispr_gene": crispr_df["measuredGeneSymbol"],
        "crispr_regulated": crispr_df["Regulated"],
    })

    # -----------------------------
    # Prepare candidate dataframe
    # -----------------------------
    base_cols = ["chrom", "chromStart", "chromEnd", "Peak", "Gene"]
    cols_to_use = base_cols + ([distance_col] if distance_col and distance_col in candidate_bed_df.columns else [])
    cand_pr_df = pd.DataFrame({
        "Chromosome": candidate_bed_df["chrom"],
        "Start": candidate_bed_df["chromStart"],
        "End": candidate_bed_df["chromEnd"],
        "Peak": candidate_bed_df["Peak"],
        "Gene": candidate_bed_df["Gene"],
        **(
            {distance_col: candidate_bed_df[distance_col]}
            if distance_col and distance_col in candidate_bed_df.columns
            else {}
        ),
    })

    # Convert to PyRanges and join
    crispr_pr = pr.PyRanges(crispr_pr_df)
    cand_pr = pr.PyRanges(cand_pr_df)

    print("Intersecting CRISPR elements with candidate peaks...")
    overlap_pr = crispr_pr.join(cand_pr)  # equivalent to bedtools intersect -wa -wb
    ov = overlap_pr.df

    # Standardize output columns
    rename_map = {
        "Chromosome": "crispr_chrom",
        "Start": "crispr_start",
        "End": "crispr_end",
        "Start_b": "peak_start",
        "End_b": "peak_end",
    }
    ov = ov.rename(columns={k: v for k, v in rename_map.items() if k in ov.columns})
    ov["peak_chrom"] = ov["crispr_chrom"]

    # Select and reorder columns
    names = [
        "crispr_chrom", "crispr_start", "crispr_end",
        "crispr_gene", "crispr_regulated",
        "peak_chrom", "peak_start", "peak_end",
        "Peak", "Gene",
    ]
    if distance_col and distance_col in ov.columns:
        names.append(distance_col)

    # Filter only existing columns (for robustness)
    names = [n for n in names if n in ov.columns]

    return ov[names]


def _parse_peak_series_to_bed_df(peak_series: pd.Series) -> pd.DataFrame:
    """Parse a Peak column into BED-like DataFrame."""
    df = peak_series.str.extract(r"([^_]+)_(\d+)_(\d+)")
    df.columns = ["chrom", "chromStart", "chromEnd"]
    df["chromStart"] = pd.to_numeric(df["chromStart"], errors="coerce")
    df["chromEnd"] = pd.to_numeric(df["chromEnd"], errors="coerce")
    return df.dropna(subset=["chromStart", "chromEnd"])