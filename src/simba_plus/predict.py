# src/simba_plus/predict.py

import argparse
import anndata as ad
import pandas as pd
import os

from simba_plus.predict_utils.overlap_snps import find_snp_peak_overlaps
from simba_plus.predict_utils.overlap_regions import find_region_peak_overlaps
from simba_plus.predict_utils.build_candidates import build_candidate_peaks
from simba_plus.discovery import candidate_links as pgl
from simba_plus.discovery import add_features as score


def predict(
    adata_CG,
    adata_CP,
    model_dir,
    *,
    snps=None,
    regions=None,
    window_size=500_000,
    celltype_specific=False,
    skip_uncertain=True,
    use_distance_weight=False,
):

    # -------------------------------
    # 1. SNP + region → peak overlaps
    # -------------------------------
    region_hits, region_meta = find_region_peak_overlaps(regions, adata_CP)
    snp_hits, snp_meta = find_snp_peak_overlaps(snps, adata_CP)

    peak_like = build_candidate_peaks(adata_CP, snp_hits, region_hits)

    # -------------------------------
    # 2. Candidate links
    # -------------------------------
    links = pgl.get_peak_gene_links(
        peaks=peak_like,
        genes=list(adata_CG.var_names),
        cis_window=window_size,
    )

    if links.empty:
        print("No candidate peak–gene links found.")
        return links

    # -------------------------------
    # 3. Add SIMBA+ scores
    # -------------------------------
    links = score.add_simba_plus_features(
        eval_df=links,
        adata_C_path=os.path.join(model_dir, "adata_C.h5ad"),
        adata_G_path=os.path.join(model_dir, "adata_G.h5ad"),
        adata_P_path=os.path.join(model_dir, "adata_P.h5ad"),
        gene_col="Gene_name",
        peak_col="Peak",
        celltype_specific=celltype_specific,
        skip_uncertain=skip_uncertain,
        use_distance_weight=use_distance_weight,
    )

    # -------------------------------
    # 4. Attach annotations
    # -------------------------------
    if snp_meta is not None:
        links = snp_meta.merge(links, on="Peak", how="left")

    if region_meta is not None:
        links = region_meta.merge(links, on="Peak", how="left")

    return links


# ======================================================
# CLI SUPPORT
# ======================================================

def add_argument(parser):
    parser.description = "Run SIMBA+ prediction for SNP/region → gene links."

    parser.add_argument("--adata-CG", type=str, required=True,
                        help="Path to gene AnnData (.h5ad) file for fetching cell/gene metadata. Output adata_G.h5ad will have no .obs attribute if not provided.")
    parser.add_argument("--adata-CP", type=str, required=True,
                        help="Path to peak/ATAC AnnData (.h5ad) file for fetching cell/peak metadata. Output adata_G.h5ad will have no .obs attribute if not provided.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing adata_C/G/P.h5ad from training SIMBA+")

    parser.add_argument("--snps", type=str,
                        help="Optional SNP TSV with columns: snp, chr, pos")

    parser.add_argument("--regions", type=str,
                        help="Optional region TSV with columns: regionid, chr, start, end")

    parser.add_argument(
        "--window-size",
        type=int,
        default=500_000,
        help=(
            "Cis window size (in bp) for forming candidate peak–gene pairs. "
        ),
    )

    parser.add_argument(
        "--celltype-specific",
        action="store_true",
        help=(
            "Compute SIMBA+ path scores separately for each cell type. "
            "This adds multiple columns (one per cell type). "
            "If not set, a single global score is computed."
        ),
    )

    parser.add_argument(
        "--skip-uncertain",
        action="store_true",
        help=(
            "Exclude cells annotated as 'Uncertain' or 'Unknown' when computing "
            "cell-type–specific SIMBA+ features."
        ),
    )

    parser.add_argument(
        "--use-distance-weight",
        action="store_true",
        help=(
            "Multiply SIMBA+ path scores by a distance prior (1 / Distance_to_TSS). "
        ),
    )


    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")

    return parser


def main(args):
    # Load AnnData
    adata_CG = ad.read_h5ad(args.adata_CG)
    adata_CP = ad.read_h5ad(args.adata_CP)

    # Load optional tables
    snps = pd.read_csv(args.snps) if args.snps else None
    regions = pd.read_csv(args.regions) if args.regions else None

    # Run prediction
    out = predict(
        adata_CG=adata_CG,
        adata_CP=adata_CP,
        model_dir=args.model_dir,
        snps=snps,
        regions=regions,
        window_size=args.window_size,
        celltype_specific=args.celltype_specific,
        skip_uncertain=args.skip_uncertain,
        use_distance_weight=args.use_distance_weight,
    )

    # Save
    out.to_csv(args.output, index=False)
    print(f"Prediction written to: {args.output}")
