import pandas as pd
import pyranges as pr

def find_snp_peak_overlaps(snps, adata_CP):
    if snps is None:
        return [], None

    if not {"snpid", "chr", "pos"}.issubset(snps.columns):
        raise ValueError("snps must be a DataFrame with columns: snp, chr, pos")

    print(f"Overlapping {len(snps)} SNPs with peaks (PyRanges)...")

    peaks_df = pd.DataFrame({"Peak": adata_CP.var_names})

    snp_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": snps["chr"].astype(str),
        "Start": snps["pos"].astype(int),
        "End": snps["pos"].astype(int) + 1,
        "snpid": snps["snpid"],
    }))

    ov = overlap_variants_with_peaks(peaks_df, snp_pr)

    if ov.empty:
        print("No SNPs overlapped any peaks.")
        return [], None

    peak_hits = ov["Peak"].unique().tolist()

    snp_meta = (
        ov[["Peak", "CHROM", "POS", "snpid"]]
        .rename(columns={"CHROM": "snp_chr", "POS": "snp_pos"})
        .groupby("Peak", as_index=False)
        .agg({
            "snpid": lambda x: ",".join(sorted(set(x))),
            "snp_chr": lambda x: ",".join(sorted(set(x))),
            "snp_pos": lambda x: ",".join(map(str, sorted(set(x)))),
        })
        .rename(columns={"snpid": "snpid"})
    )

    return peak_hits, snp_meta

def overlap_variants_with_peaks(candidates: pd.DataFrame, variants_bed: pr.PyRanges) -> pd.DataFrame:
    # Parse candidate Peak strings → chrom/start/end
    peaks_df = (
        candidates["Peak"]
        .str.replace(":", "_", regex=False)
        .str.split("_", expand=True)
        .rename(columns={0: "CHROM", 1: "START", 2: "END"})
        .astype({"START": int, "END": int})
    )
    peaks_df["Peak"] = candidates["Peak"]

    peak_pr = pr.PyRanges(peaks_df.rename(columns={
        "CHROM": "Chromosome", "START": "Start", "END": "End"
    }))

    # Find overlaps (RSID survives as column snp)
    overlap_df = variants_bed.join(peak_pr).df.rename(columns={
        "Chromosome": "CHROM",
        "Start": "POS",
        "End": "POS2",
        "Start_b": "START_ANN",
        "End_b": "END_ANN",
    })

    # Standardize Peak ID (use annotation start/end from the joined peak)
    overlap_df["Peak"] = (
        overlap_df["CHROM"].astype(str) + "_" +
        overlap_df["START_ANN"].astype(str) + "_" +
        overlap_df["END_ANN"].astype(str)
    )

    print(f"Found {len(overlap_df):,} overlapping variant–peak records.")

    return overlap_df[["Peak", "CHROM", "POS", "snpid"]]
