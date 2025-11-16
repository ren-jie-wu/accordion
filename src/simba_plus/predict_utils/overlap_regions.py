import pandas as pd
import pyranges as pr

def find_region_peak_overlaps(regions_df, adata_CP):
    """
    regions_df must have columns:
        regionid, chr, start, end

    Returns:
        peak_hits: list of overlapping peaks
        region_meta: df with Peak + region annotations
    """
    if regions_df is None:
        return [], None

    required = {"regionid", "chr", "start", "end"}
    if not required.issubset(regions_df.columns):
        raise ValueError(f"regions must contain columns: {required}")

    print(f"Processing {len(regions_df)} genomic regions → peak overlaps (PyRanges)...")

    # -----------------------------
    # Convert regions → PyRanges
    # -----------------------------
    region_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": regions_df["chr"].astype(str),
        "Start": regions_df["start"].astype(int),
        "End": regions_df["end"].astype(int),
        "regionid": regions_df["regionid"],
        "region_chr": regions_df["chr"].astype(str),
        "region_start": regions_df["start"].astype(int),
        "region_end": regions_df["end"].astype(int),
        "region_coord": regions_df.apply(
            lambda r: f"{r['chr']}_{r['start']}_{r['end']}", axis=1
        ),
    }))

    # -----------------------------
    # Convert peaks → PyRanges
    # -----------------------------
    peak_rows = []
    for pk in adata_CP.var_names:
        if ":" in pk:
            chrom, coords = pk.split(":")
            start, end = coords.split("-")
        else:
            chrom, start, end = pk.split("_")

        peak_rows.append({
            "Chromosome": chrom,
            "Start": int(start),
            "End": int(end),
            "Peak": pk,
        })

    peak_pr = pr.PyRanges(pd.DataFrame(peak_rows))

    # -----------------------------
    # Overlap
    # -----------------------------
    ov = region_pr.join(peak_pr).df

    if ov.empty:
        print("No regions overlapped any peaks.")
        return [], None

    peak_hits = ov["Peak"].unique().tolist()

    # -----------------------------
    # Build region metadata table
    # -----------------------------
    region_meta = (
        ov[
            [
                "Peak",
                "regionid",
                "region_chr",
                "region_start",
                "region_end",
                "region_coord",
            ]
        ]
        .drop_duplicates()
        .groupby("Peak", as_index=False)
        .agg({
            "regionid": lambda x: ",".join(sorted(set(x))),
            "region_coord": lambda x: ",".join(sorted(set(x))),
        })
    )

    return peak_hits, region_meta
