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
from pathlib import Path
import pyranges as pr
import os
import urllib.request
from typing import Sequence, Optional
import numpy as np
import subprocess
import tarfile
import zipfile
import gzip
import platform

def build_eqtl_evalset(
    candidate_csv: str,
    output_csv: str,
    download_dir: str = "../data",
    pip_pos: float = 0.5,
    pip_neg: float = 0.01,
    gtex_method: str = "SUSIE",
    gtex_tissue: str = "Whole_Blood"
) -> pd.DataFrame:
    """Build a GTEx eQTL evaluation set (replicates the notebook logic).

    This function downloads and prepares 1000G and GTEx data, overlaps peaks and variants,
    computes inverse distance, and labels SNP–gene pairs based on PIP thresholds.
    """
    os.makedirs(download_dir, exist_ok=True)

    # 1️⃣ Download and extract 1000 Genomes PLINK files if not present
    plink_dir, frq_dir = _download_and_extract_1000g(download_dir)

    # 2️⃣ Download GTEx fine-mapping results if not present
    gtex_zip = os.path.join(download_dir, "Public_GTEx_finemapping.zip")
    gtex_file = os.path.join(download_dir, "GTEx_49tissues_release1.tsv.bgz")
    if not os.path.exists(gtex_zip):
        url = "https://www.dropbox.com/scl/fo/bjp6o8hgixt5occ6ggq2o/ADTTMnyH-4rzDFS6g7oIcEE?rlkey=7558jp42yvmyjhgmcilbhlnu7&e=4&dl=1"
        print("Downloading GTEx fine-mapping dataset...")
        subprocess.run(["wget", "-O", gtex_zip, url], check=True)
        if zipfile.is_zipfile(gtex_zip):
            print(f"Extracting {gtex_zip} ...")
            with zipfile.ZipFile(gtex_zip, "r") as zip_ref:
                zip_ref.extractall(download_dir)
            print(f"Extracted GTEx data to: {download_dir}")
        else:
            raise ValueError(f"{gtex_zip} is not a valid zip archive. Please check the downloaded file.")

    # 3️⃣ Load input candidate peak–gene links
    candidates = pd.read_csv(candidate_csv)
    if "gene_name" in candidates.columns:
        candidates.rename(columns={"gene_name": "Gene_name"}, inplace=True)
    candidates["peak_gene_pair"] = candidates["Peak"] + "_" + candidates["Gene_name"]

    # 4️⃣ Load and concatenate 1000G variant BEDs
    print("Loading 1000G variant BED files...")
    bed_files = _process_1000g_bim_frq(plink_dir=plink_dir, frq_dir=frq_dir, maf_threshold=0.01)
    genomes_df = pd.concat(
        [pd.read_csv(f, sep="\t", header=None, names=["CHROM", "START", "END"]) for f in bed_files],
        ignore_index=True,
    )
    genomes_df["CHROM"] = "chr" + genomes_df["CHROM"].astype(str)
    valid_chroms = [f'chr{i}' for i in range(1, 23)]
    genomes_df = genomes_df[genomes_df['CHROM'].isin(valid_chroms)]
    variants_bed = pr.PyRanges(
        genomes_df.rename(columns={"CHROM": "Chromosome", "START": "Start", "END": "End"})
    )

    # 5️⃣ Load and filter GTEx eQTLs
    print("Loading GTEx fine-mapping results...")
    gtex_df = pd.read_csv(gtex_file, sep="\t", compression="gzip")
    gtex_df = gtex_df[(gtex_df.method == gtex_method) & (gtex_df.tissue == gtex_tissue)]
    gtex_df["gene"] = gtex_df["gene"].astype(str).str.split(".").str[0]
    gtex_df = _map_ensembl_to_symbol(gtex_df) # convert Ensembl IDs to gene symbols
    gtex_df = gtex_df[(~gtex_df['GeneSymbol'].isna())]
    gtex_df = gtex_df[~gtex_df['GeneSymbol'].str.contains('ENSG')]
    gtex_df["start"] = gtex_df["start"].astype(int)
    gtex_df["end"] = gtex_df["end"].astype(int)

    gtex_df["chr_pos"] = gtex_df["variant_hg38"].str.extract(r'^(chr[0-9XYM]+_\d+)')
    gtex_df['pair'] = gtex_df['chr_pos'] + '_' + gtex_df['GeneSymbol']
    pos_training = gtex_df[gtex_df["pip"] > pip_pos].drop_duplicates("pair")
    neg_training = gtex_df[gtex_df["pip"] < pip_neg].drop_duplicates("pair")

    # 6️⃣ Overlap candidate peaks with variants
    print("Overlapping candidate peaks with variants...")
    split_lst = [p.replace(":", "-").split("_") for p in candidates["Peak"]]
    ann_df = pd.DataFrame(split_lst, columns=["CHROM", "START", "END"]).astype({"START": int, "END": int})
    ann_df["Peak"] = candidates["Peak"].values
    ann_pr = pr.PyRanges(ann_df.rename(columns={"CHROM": "Chromosome", "START": "Start", "END": "End"}))
    overlap_pr = variants_bed.join(ann_pr)
    overlap_df = overlap_pr.df.rename(columns={
        "Chromosome": "CHROM",
        "Start": "POS",
        "End": "POS2",
        "Start_b": "START_ANN",
        "End_b": "END_ANN",
    })
    overlap_df["Peak"] = (
        overlap_df["CHROM"].astype(str) + "_" + overlap_df["START_ANN"].astype(str) + "_" + overlap_df["END_ANN"].astype(str)
    )

    # 7️⃣ Merge with candidate peaks and compute inverse distance
    merged = candidates.merge(overlap_df[["CHROM", "POS", "POS2", "Peak"]], on="Peak", how="right")
    merged.rename(columns={"Gene_name": "Gene"}, inplace=True)
    if "Distance_to_TSS" in merged.columns:
        merged["1/Distance"] = (1 / (merged["Distance_to_TSS"] + 1)).clip(upper=1)
    merged["snp_gene_pair"] = merged["CHROM"].astype(str) + "_" + merged["POS2"].astype(str) + "_" + merged["Gene"]

    # 8️⃣ Label positives and negatives
    pos_df = merged[merged.snp_gene_pair.isin(pos_training.pair)].copy()
    pos_df["label"] = 1
    neg_df = merged[merged.snp_gene_pair.isin(neg_training.pair)].copy()
    neg_df["label"] = 0

    final_df = pd.concat([pos_df, neg_df], axis=0).drop_duplicates("snp_gene_pair")
    final_df.index = final_df.CHROM

    final_df.to_csv(output_csv, index=True)
    print(f"Saved labeled evaluation set to: {output_csv}")
    return final_df

def _download_and_extract_1000g(download_dir: str = "../data"):
    """Download and extract 1000 Genomes PLINK and FRQ archives if missing."""
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    files = {
        "1000G_EUR_Phase3_plink.tgz": {
            "url": "https://zenodo.org/records/8292725/files/1000G_Phase3_plinkfiles.tgz?download=1",
            "extract_dir": os.path.join(download_dir, "1000G_EUR_Phase3_plink"),
        },
        "1000G_Phase3_frq.tgz": {
            "url": "https://zenodo.org/records/8292725/files/1000G_Phase3_frq.tgz?download=1",
            "extract_dir": os.path.join(download_dir, "1000G_Phase3_frq"),
        },
    }

    for filename, meta in files.items():
        archive_path = os.path.join(download_dir, filename)
        extract_dir = meta["extract_dir"]
        url = meta["url"]

        # check if extracted folder exists
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"Found extracted folder: {extract_dir} — skipping.")
            continue

        # otherwise check if .tgz exists, else download
        if not os.path.exists(archive_path):
            print(f"Downloading {filename} ...")
            subprocess.run(["wget", "-O", archive_path, url], check=True)
        else:
            print(f"Found archive: {archive_path}")

        # extract if folder missing or empty
        print(f"Extracting {filename} to {download_dir} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(download_dir)
        print(f"Extracted: {extract_dir}")

    print("Both 1000G PLINK and FRQ archives are ready.")
    return (
        os.path.join(download_dir, "1000G_EUR_Phase3_plink"),
        os.path.join(download_dir, "1000G_Phase3_frq"),
    )

def _download_and_extract_gtex(download_dir: str = "../data") -> str:
    """
    Download and extract GTEx fine-mapping dataset if missing.
    Returns the path to the main GTEx fine-mapping file.
    """
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    gtex_zip = os.path.join(download_dir, "Public_GTEx_finemapping.zip")
    gtex_file = os.path.join(download_dir, "GTEx_49tissues_release1.tsv.bgz")
    url = "https://www.dropbox.com/scl/fo/bjp6o8hgixt5occ6ggq2o/ADTTMnyH-4rzDFS6g7oIcEE?rlkey=7558jp42yvmyjhgmcilbhlnu7&e=4&dl=1"

    #  If extracted .bgz file exists — skip everything
    if os.path.exists(gtex_file) and os.path.getsize(gtex_file) > 0:
        print(f"Found extracted GTEx file: {gtex_file} — skipping download.")
        return gtex_file

    #  If zip file exists, use it
    if not os.path.exists(gtex_zip):
        print("Downloading GTEx fine-mapping dataset ...")
        subprocess.run(["wget", "-O", gtex_zip, url], check=True)
    else:
        print(f"Found archive: {gtex_zip}")

    #  Validate and extract
    if zipfile.is_zipfile(gtex_zip):
        print(f"Extracting {gtex_zip} ...")
        with zipfile.ZipFile(gtex_zip, "r") as zip_ref:
            zip_ref.extractall(download_dir)
        print(f"Extracted GTEx data to: {download_dir}")
    else:
        raise ValueError(f"{gtex_zip} is not a valid zip archive. Please check the downloaded file.")

    #  Verify extraction result
    if not os.path.exists(gtex_file):
        raise FileNotFoundError(f"Expected file {gtex_file} not found after extraction.")
    
    print(f"GTEx fine-mapping data ready: {gtex_file}")
    return gtex_file

def _download_liftover_resources(download_dir: str = "./data") -> tuple[str, str]:
    """Ensure UCSC liftOver binary and hg19→hg38 chain file exist locally."""
    out_dir = Path(download_dir) 
    out_dir.mkdir(parents=True, exist_ok=True)

    liftover_bin = out_dir / "liftOver"
    chain_file = out_dir / "hg19ToHg38.over.chain.gz"

    # Download binary if needed
    if not liftover_bin.exists():
        system = platform.system().lower()
        print(f"Downloading UCSC liftOver binary for {system}...")
        if system == "linux":
            url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
        elif system == "darwin":
            url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver"
        else:
            raise RuntimeError("Unsupported OS for automatic liftOver download.")
        urllib.request.urlretrieve(url, liftover_bin)
        subprocess.run(["chmod", "+x", str(liftover_bin)], check=True)
        print(f"liftOver binary saved to: {liftover_bin}")
    else:
        print(f"liftOver binary already present: {liftover_bin}")

    # Download chain file if needed
    if not chain_file.exists():
        print("Downloading hg19→hg38 chain file...")
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
        urllib.request.urlretrieve(url, chain_file)
        print(f"Chain file saved to: {chain_file}")
    else:
        print(f"Chain file already present: {chain_file}")

    return str(liftover_bin), str(chain_file)

def _process_1000g_bim_frq(plink_dir: str, frq_dir: str, maf_threshold: float = 0.01) -> list[str]:
    """Generate per-chromosome BED files of common variants (MAF > threshold)
    using 1000 Genomes .bim and .frq files.
    """
    output_dir = os.path.join(plink_dir)
    os.makedirs(output_dir, exist_ok=True)
    bed_files = []

    expected_files = [
        os.path.join(output_dir, f"1000G.EUR.{chrom}.filtered.bed")
        for chrom in range(1, 23)
    ]
    existing_files = [f for f in expected_files if os.path.exists(f) and os.path.getsize(f) > 0]
    if len(existing_files) == 22:
        print("Found existing 1000G filtered BED files — skipping regeneration.")
        return existing_files

    for chrom in range(1, 23):
        # Paths for this chromosome
        bim_path = os.path.join(plink_dir, f"1000G.EUR.QC.{chrom}.bim")
        frq_path = os.path.join(frq_dir, f"1000G.EUR.QC.{chrom}.frq")

        if not os.path.exists(bim_path):
            print(f"Missing BIM for chr{chrom}: {bim_path}")
            continue
        if not os.path.exists(frq_path):
            print(f"Missing FRQ for chr{chrom}: {frq_path}")
            continue

        # Read .bim (positions and alleles)
        bim = pd.read_csv(
            bim_path, sep="\t", header=None,
            names=["CHROM", "SNP", "CM", "POS", "A1", "A2"]
        )

        # Read .frq (allele frequencies)
        frq = pd.read_csv(frq_path, delim_whitespace=True, comment="#")

        # Ensure column names match expected (handle FRQ files that use lowercase or uppercase)
        frq.columns = [c.upper() for c in frq.columns]
        if "MAF" not in frq.columns:
            raise ValueError(f"FRQ file {frq_path} missing 'MAF' column")

        # Merge on SNP ID
        merged = bim.merge(frq, on="SNP", how="inner")

        # Filter by MAF
        filtered = merged[merged["MAF"] > maf_threshold].copy()

        # BED-style coordinates
        bed_df = filtered[["CHROM", "POS"]].rename(columns={"POS": "START"})
        bed_df["END"] = bed_df["START"]
        bed_df["START"] = bed_df["START"] - 1
        bed_path = os.path.join(output_dir, f"1000G.EUR.{chrom}.filtered.bed")
        bed_df.to_csv(bed_path, sep="\t", header=False, index=False)

        bed_files.append(bed_path)
        print(f"chr{chrom}: {len(bed_df)} variants kept (MAF > {maf_threshold})")

    if not bed_files:
        raise FileNotFoundError("No valid BIM+FRQ pairs were processed — check extracted directories.")
    print(f"Finished generating {len(bed_files)} filtered BED files.")
    return bed_files

def liftover_df(
    df: pd.DataFrame,
    chain_file: str,
    liftover_bin: str,
    out_dir: str = "./liftover_tmp",
    prefix: str = "region"
) -> pd.DataFrame:
    """LiftOver genomic coordinates in a DataFrame from one assembly to another.

    Args:
        df (pd.DataFrame): Must contain ['CHROM', 'START', 'END'] columns.
        chain_file (str): Path to UCSC chain file (e.g., hg19ToHg38.over.chain.gz).
        liftover_bin (str): Path to UCSC liftOver executable.
        out_dir (str): Directory to store intermediate and output files.
        prefix (str): Prefix for temp output files.

    Returns:
        pd.DataFrame: Liftover-converted DataFrame with same columns.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    bed_file = out_dir / f"{prefix}.bed"
    out_hg38 = out_dir / f"{prefix}_lifted.bed"
    out_unmapped = out_dir / f"{prefix}_unmapped.bed"

    # Save input DataFrame as BED
    df[["CHROM", "START", "END"]].to_csv(bed_file, sep="\t", header=False, index=False)

    # Run liftOver
    cmd = [liftover_bin, str(bed_file), chain_file, str(out_hg38), str(out_unmapped)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read lifted BED back in
    lifted = pd.read_csv(out_hg38, sep="\t", header=None, names=["CHROM", "START", "END"])
    print(f"Liftover completed! {len(lifted):,} regions mapped successfully.")

    return lifted

def _map_ensembl_to_symbol(gtex_df: pd.DataFrame, gtf_dir: str = "../data") -> pd.DataFrame:
    """Add GeneSymbol column to gtex_df using Ensembl → gene_name mapping from GENCODE GTF."""
    os.makedirs(gtf_dir, exist_ok=True)
    gtf_path = os.path.join(gtf_dir, "gencode.v44.annotation.gtf.gz")

    # Download GTF if missing
    if not os.path.exists(gtf_path):
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
        print(f"Downloading GENCODE v44 GTF with wget ...")
        subprocess.run(["wget", "-O", gtf_path, url], check=True)
        print(f"Downloaded: {gtf_path}")
    else:
        print("Found existing GTF file — skipping download.")

    # Build mapping only for needed Ensembl IDs
    needed = set(gtex_df["gene"].astype(str))
    mapping = {}
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            info = line.split("\t")[8]
            gene_id = info.split('gene_id "')[1].split('"')[0].split(".")[0]
            if gene_id in needed:
                gene_name = info.split('gene_name "')[1].split('"')[0]
                mapping[gene_id] = gene_name
            if len(mapping) == len(needed):
                break  # Stop once all needed genes are found

    # Map to new column
    gtex_df["GeneSymbol"] = gtex_df["gene"].map(mapping)
    print(f"Added GeneSymbol for {gtex_df['GeneSymbol'].notna().sum()} of {len(gtex_df)} genes.")

    return gtex_df

def load_gtex_finemap(
    gtex_path: str,
    tissue: str = "Whole_Blood",
    method: str = "SUSIE",
) -> pd.DataFrame:
    """
    Load and preprocess GTEx fine-mapping results for a selected tissue and method.

    Args:
        gtex_path (str): Path to the GTEx fine-mapping file (e.g. GTEx_49tissues_release1.tsv.bgz).
        tissue (str): GTEx tissue name to filter for (default: "Whole_Blood").
        method (str): Fine-mapping method (default: "SUSIE").

    Returns:
        pd.DataFrame: Filtered GTEx fine-mapping DataFrame with columns:
                      ['chr_pos', 'GeneSymbol', 'pair', 'pip', ...]
    """
    print(f"Loading GTEx fine-mapping data from: {gtex_path}")
    gtex_df = pd.read_csv(gtex_path, sep="\t", compression="gzip")

    # Filter for method and tissue
    gtex_df = gtex_df[(gtex_df["method"] == method) & (gtex_df["tissue"] == tissue)]
    print(f"Filtered for {method} in {tissue}: {len(gtex_df):,} entries")

    # Clean gene IDs (remove version suffix)
    gtex_df["gene"] = gtex_df["gene"].astype(str).str.split(".").str[0]

    # Map Ensembl → GeneSymbol if function provided
    gtex_df = _map_ensembl_to_symbol(gtex_df)
    gtex_df = gtex_df[gtex_df["GeneSymbol"].notna()]
    print(f"🔠 Added GeneSymbol for {gtex_df['GeneSymbol'].notna().sum():,} genes")

    # Extract chr_pos and build SNP–gene pair ID
    gtex_df["chr_pos"] = gtex_df["variant_hg38"].str.extract(r"^(chr[0-9XYM]+_\d+)")
    gtex_df["pair"] = gtex_df["chr_pos"] + "_" + gtex_df["GeneSymbol"]

    print(f"Final GTEx dataset ready with {len(gtex_df):,} SNP–gene pairs")
    return gtex_df

def overlap_variants_with_peaks(candidates: pd.DataFrame, variants_bed: pr.PyRanges) -> pd.DataFrame:
    """
    Convert candidate peaks into PyRanges and find overlaps with variant coordinates.

    Args:
        candidates (pd.DataFrame): Must contain a 'Peak' column in 'chr:start-end' or 'chr_start_end' format.
        variants_bed (pr.PyRanges): PyRanges object of variant coordinates.

    Returns:
        pd.DataFrame: DataFrame of overlapping variants and peaks with standardized columns.
    """
    # Parse candidate Peak strings → chrom/start/end
    peaks_df = (
        candidates["Peak"]
        .str.replace(":", "_", regex=False)
        .str.split("_", expand=True)
        .rename(columns={0: "CHROM", 1: "START", 2: "END"})
        .astype({"START": int, "END": int})
    )
    peaks_df["Peak"] = candidates["Peak"]

    # Convert to PyRanges
    peak_pr = pr.PyRanges(peaks_df.rename(columns={
        "CHROM": "Chromosome", "START": "Start", "END": "End"
    }))

    # Find overlaps
    overlap_df = variants_bed.join(peak_pr).df.rename(columns={
        "Chromosome": "CHROM",
        "Start": "POS",
        "End": "POS2",
        "Start_b": "START_ANN",
        "End_b": "END_ANN",
    })

    # Standardize Peak ID
    overlap_df["Peak"] = (
        overlap_df["CHROM"].astype(str) + "_" +
        overlap_df["START_ANN"].astype(str) + "_" +
        overlap_df["END_ANN"].astype(str)
    )

    print(f"Found {len(overlap_df):,} overlapping variant–peak records.")
    return overlap_df

def merge_and_label_eqtl_pairs(
    candidates: pd.DataFrame,
    overlap_df: pd.DataFrame,
    pos_training: pd.DataFrame,
    neg_training: pd.DataFrame,
) -> pd.DataFrame:
    """Merge overlap results with candidate peaks and label SNP–gene pairs."""
    merged = candidates.merge(overlap_df[["CHROM","POS","POS2","Peak"]], on="Peak", how="right")
    merged = merged.rename(columns={"Gene_name": "Gene"})

    if "Distance_to_TSS" in merged:
        merged["1/Distance"] = (1 / (merged["Distance_to_TSS"] + 1)).clip(upper=1)

    merged["snp_gene_pair"] = (
        merged["CHROM"].astype(str) + "_" +
        merged["POS2"].astype(str) + "_" +
        merged["Gene"]
    )

    pos_df = merged[merged.snp_gene_pair.isin(pos_training["pair"])].assign(label=1)
    neg_df = merged[merged.snp_gene_pair.isin(neg_training["pair"])].assign(label=0)

    final_df = pd.concat([pos_df, neg_df]).drop_duplicates("snp_gene_pair").set_index("CHROM")
    print(f"total {len(final_df)} pairs.")
    return final_df


def build_crispr_evalset(
    candidate: pd.DataFrame,
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
        candidate: candidate peak–gene links (BMMC or other).
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
    candidates = _load_and_validate_candidates(candidate, peak_col, gene_col, distance_col)
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
    candidate: pd.DataFrame,
    peak_col: str,
    gene_col: str,
    distance_col: Optional[str]
) -> pd.DataFrame:
    df = candidate.copy()
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