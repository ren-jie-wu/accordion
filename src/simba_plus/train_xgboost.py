
import pandas as pd
from simba_plus.discovery import build_evalset, model_training, plot_utils
from simba_plus.discovery import candidate_links as pgl
import pyranges as pr
import os
from simba_plus.discovery import add_features as score
from simba_plus.discovery.model_training import load_results
import argparse
import ast

def load_crispr_eval(adata_CG, adata_CP, crispr_file, model_dir, output_path, 
                     window_size=500_000, 
                     celltype_specific: bool = False,
                     skip_uncertain: bool = True,
                     use_distance_weight: bool = False,):
    
    # construct candidate peak-gene links
    links = pgl.get_peak_gene_links(
        peaks=list(adata_CP.var_names),
        genes=list(adata_CG.var_names),
        cis_window=window_size,
    )
    # build crispr evaluation set
    crispr_eval_df = build_evalset.build_crispr_evalset(
        candidate=links,
        crispr_file=crispr_file,
        output_csv=output_path
    )

    adata_C_path = os.path.join(model_dir, "adata_C.h5ad")
    adata_G_path = os.path.join(model_dir, "adata_G.h5ad")
    adata_P_path = os.path.join(model_dir, "adata_P.h5ad")

    crispr_eval_df = score.add_simba_plus_features(
        eval_df=crispr_eval_df,
        adata_C_path=adata_C_path,
        adata_G_path=adata_G_path,
        adata_P_path=adata_P_path,
        gene_col="Gene",
        peak_col="Peak",
        celltype_specific=celltype_specific,
        skip_uncertain=skip_uncertain,
        use_distance_weight=use_distance_weight,
    )

    crispr_eval_df.to_csv(output_path)
    return crispr_eval_df

def load_eqtl_eval(adata_CG, adata_CP, model_dir, output_path, tissue="Whole_Blood", pip_pos=0.5, pip_neg=0.01, 
                   window_size=500_000,                     
                   celltype_specific: bool = False, skip_uncertain: bool = True,
                   use_distance_weight: bool = False,):
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
    
    adata_C_path = os.path.join(model_dir, "adata_C.h5ad")
    adata_G_path = os.path.join(model_dir, "adata_G.h5ad")
    adata_P_path = os.path.join(model_dir, "adata_P.h5ad")

    eqtl_eval_df = score.add_simba_plus_features(
        eval_df=eqtl_eval_df,
        adata_C_path=adata_C_path,
        adata_G_path=adata_G_path,
        adata_P_path=adata_P_path,
        gene_col="Gene_name",
        peak_col="Peak",
        celltype_specific=celltype_specific,
        skip_uncertain=skip_uncertain,
        use_distance_weight=use_distance_weight,
    )
    
    eqtl_eval_df.to_csv(output_path)
    
    return eqtl_eval_df

def train_supervised(eval_df, feature_sets, dataset_name, output_path):
    results_obj, metrics_df, preds_df = model_training.train_xgboost(
            df=eval_df,
            dataset_name=dataset_name,
            output_dir=output_path,
            feature_sets=feature_sets,
            n_search_iter=30,
            search_n_jobs=8,
            random_state=1
        )
    return results_obj, metrics_df, preds_df

# ======================================================
# CLI SUPPORT
# ======================================================

def parse_feature_sets(csv_path):
    """Load feature_sets from a CSV and return dict[str, list[str]]."""
    df = pd.read_csv(csv_path)

    if "feature_set_name" not in df.columns or "features" not in df.columns:
        raise ValueError("CSV must contain columns: 'feature_set_name' and 'features'")

    feature_sets = {}
    for _, row in df.iterrows():
        name = row["feature_set_name"]

        # Option A: comma-separated list (recommended)
        if isinstance(row["features"], str) and "," in row["features"]:
            features = [f.strip() for f in row["features"].split(",")]

        # Option B: Python list string → parse using ast.literal_eval
        else:
            features = ast.literal_eval(row["features"])

        feature_sets[name] = features

    return feature_sets


def main():
    parser = argparse.ArgumentParser(description="Run supervised XGBoost training on Simba+ features.")
    parser.add_argument("--eval_csv", type=str, required=True, 
                        help="Path to evaluation dataframe CSV.")
    parser.add_argument("--feature_sets_csv", type=str, required=True, 
                        help="CSV defining feature_set groups.")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Dataset name for logging.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save training outputs.")
    args = parser.parse_args()

    # Load eval dataframe
    eval_df = pd.read_csv(args.eval_csv)

    # Parse feature_sets dict
    feature_sets = parse_feature_sets(args.feature_sets_csv)

    # Train
    results_obj, metrics_df, preds_df = train_supervised(
        eval_df=eval_df,
        feature_sets=feature_sets,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
    )

    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(args.output_path, "metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(args.output_path, "preds.csv"), index=False)

    print("Training finished. Results saved in:", args.output_path)


if __name__ == "__main__":
    main()
