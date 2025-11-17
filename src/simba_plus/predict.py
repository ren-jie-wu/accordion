# src/simba_plus/predict.py

import argparse
import anndata as ad
import pandas as pd
import os
from simba_plus.discovery import build_evalset, model_training, plot_utils
import pyranges as pr
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

## load data
def load_crispr_eval(
    adata_CG, adata_CP, crispr_file, output_path,
    window_size=500_000,
):
    # 1. Candidate peak-gene links
    links = pgl.get_peak_gene_links(
        peaks=list(adata_CP.var_names),
        genes=list(adata_CG.var_names),
        cis_window=window_size,
    )

    # 2. Build CRISPR eval DF (but we do NOT save it because we only want reduced TSV)
    crispr_eval_df = build_evalset.build_crispr_evalset(
        candidate=links,
        crispr_file=crispr_file,
        output_csv=None,               # <-- DO NOT WRITE ANY FILE HERE
    )

    # 3. Correct regionid construction
    crispr_eval_df["regionid"] = (
        crispr_eval_df["crispr_chrom"].astype(str) + "_" +
        crispr_eval_df["crispr_start"].astype(str) + "_" +
        crispr_eval_df["crispr_end"].astype(str) + "_" +
        crispr_eval_df["crispr_gene"].astype(str) + "_" +
        crispr_eval_df["label"].astype(str)
    )

    # 4. Rename columns
    crispr_eval_df = crispr_eval_df.rename(
        columns={
            "crispr_chrom": "chr",
            "crispr_start": "start",
            "crispr_end": "end",
        }
    )

    # 5. Keep ONLY the columns needed by SIMBA+ predict
    crispr_input = crispr_eval_df[["regionid", "chr", "start", "end"]].copy()

    # 6. Write the final TSV
    crispr_input.to_csv(output_path, sep="\t", index=False)

    return crispr_eval_df

def load_eqtl_eval(adata_CG, adata_CP, output_path, tissue="Whole_Blood", pip_pos=0.5, pip_neg=0.01, 
                   window_size=500_000,):
    # construct candidate peak-gene links
    links = pgl.get_peak_gene_links(
        peaks=list(adata_CP.var_names),
        genes=list(adata_CG.var_names),
        cis_window=window_size,
    )

    # download necessary data
    build_evalset._download_and_extract_1000g(download_dir=f"{output_path}/1000genome")
    build_evalset._download_and_extract_gtex(download_dir=f"{output_path}/gtex")
    build_evalset._download_liftover_resources(download_dir=f"{output_path}/liftover_resources")
    
    # process data
    bed_files = build_evalset._process_1000g_bim_frq(plink_dir=f"{output_path}/1000genome/1000G_EUR_Phase3_plink", 
                                     frq_dir=f"{output_path}/1000genome/1000G_Phase3_frq", 
                                     maf_threshold=0.01)

    valid_chroms = [f"chr{i}" for i in range(1, 23)]
    genomes_df = (
        pd.concat(
            [pd.read_csv(f, sep="\t", header=None, names=["CHROM", "START", "END"]) for f in bed_files],
            ignore_index=True
        )
        .assign(CHROM=lambda d: "chr" + d["CHROM"].astype(str))
        .loc[lambda d: d["CHROM"].isin(valid_chroms)]
    ) # 1000G variants in BED format

    ## liftOver 1000G variants to hg38
    CHAIN_FILE = f"{output_path}/liftover_resources/hg19ToHg38.over.chain.gz"
    LIFTOVER_BIN = f"{output_path}/liftover_resources/liftOver" # downloaded version
    OUT_DIR = f"{output_path}/liftover"
    lifted_df = build_evalset.liftover_df(genomes_df, CHAIN_FILE, LIFTOVER_BIN, out_dir=OUT_DIR, prefix="example")
    lifted_df = lifted_df[lifted_df['CHROM'].isin(valid_chroms)] # remove alternate contigs, scaffolds, etc., keep only chr1-22
    variants_bed = pr.PyRanges(
        lifted_df.rename(columns={"CHROM": "Chromosome", "START": "Start", "END": "End"})
    ) # convert to PyRanges object ready for later overlap operations with peaks

    # now process GTEx data
    gtex_df = build_evalset.load_gtex_finemap(
        f"{output_path}/gtex/GTEx_49tissues_release1.tsv.bgz",
        tissue=tissue,
        method="SUSIE"
    )

    # define positive / negative training sets
    pos_training = gtex_df[gtex_df["pip"] > pip_pos].drop_duplicates("pair")
    neg_training = gtex_df[gtex_df["pip"] < pip_neg].drop_duplicates("pair")

    overlap_df = build_evalset.overlap_variants_with_peaks(links, variants_bed) # overlap 1000G variants with peaks
    eqtl_eval_df = build_evalset.merge_and_label_eqtl_pairs(links, overlap_df, pos_training, neg_training) # label positive / negative pairs based on GTEx PIP score
    
    # adata_C_path = os.path.join(model_dir, "adata_C.h5ad")
    # adata_G_path = os.path.join(model_dir, "adata_G.h5ad")
    # adata_P_path = os.path.join(model_dir, "adata_P.h5ad")

    # eqtl_eval_df = score.add_simba_plus_features(
    #     eval_df=eqtl_eval_df,
    #     adata_C_path=adata_C_path,
    #     adata_G_path=adata_G_path,
    #     adata_P_path=adata_P_path,
    #     gene_col="Gene_name",
    #     peak_col="Peak",
    #     celltype_specific=celltype_specific,
    #     skip_uncertain=skip_uncertain,
    #     use_distance_weight=use_distance_weight,
    # )
    
    eqtl_eval_df.to_csv(output_path)
    
    return eqtl_eval_df


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
    snps = pd.read_csv(args.snps, sep='\t') if args.snps else None
    regions = pd.read_csv(args.regions, sep='\t') if args.regions else None

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
