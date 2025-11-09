import pickle as pkl
from typing import Literal
import os
import numpy as np
import anndata as ad
import pandas as pd
import papermill as pm
from argparse import ArgumentParser
import scipy
import simba_plus.datasets._datasets
from simba_plus.utils import setup_logging
from simba_plus.heritability.utils import get_overlap, plot_hist, enrichment_analysis
from simba_plus.heritability.ldsc import run_ldsc_l2, run_ldsc_h2
from simba_plus.heritability.get_taus import get_tau_z_dep


snp_pos_path = (
    f"{os.path.dirname(__file__)}/../datasets/ldsc_data/1000G_Phase3_plinkfiles/ref.txt"
)


def add_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "sumstats", help="GwAS summary statistics compatible with LDSC inputs", type=str
    )
    parser.add_argument(
        "adata_prefix",
        type=str,
        help="Directory that contains adata_{C,P,G}{version_suffix}.h5ad files from simba+ train output.",
    )
    parser.add_argument(
        "--shared-annot-prefix",
        type=str,
        default=None,
        help="If provided, use these annotations as shared annotations (cell type level) for LDSC.",
    )
    parser.add_argument(
        "--create-report",
        action="store_true",
        help="Create .ipynb report plotting cell-level heritability scores.",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        help="Suffix to append to adata_{C,P,G} files from simba+ train output. ({adata_prefix}/adata_{C,P,G}{version_suffix}.h5ad will be loaded.)",
    )
    parser.add_argument(
        "--cell-type-label",
        type=str,
        default=None,
        help="When provided, calculate baseline per-cell-type heritability.",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-h2", action="store_true")
    parser.add_argument(
        "--gene-dist",
        type=int,
        default=100,
        help="Distance to use for SNP-to-gene mapping",
    )
    parser.add_argument(
        "--sumstats", type=str, default=None, help="Alternative sumstats ID"
    )

    return parser


def write_peak_annot(
    peak_annot: np.ndarray, annot_prefix: str, logger, mean=True
) -> str:
    """
    Write peak annotation to file.
    Args:
        peak_annot (np.ndarray): Peak annotation matrix
        annot_prefix (str): Prefix for annotation file
        logger: Logger object
        mean (bool): Whether to write mean annotation per snp if snp overlaps more than 1 peak. If False, write the sum.
    """
    os.makedirs(os.path.dirname(annot_prefix), exist_ok=True)
    snp_df = pd.read_csv(snp_pos_path, sep="\t", header=None)
    snp_df.columns = ["SNP", "chrom", "start"]
    snp_to_peak_overlap = get_overlap(snp_df, peak_annot).T
    plot_hist(snp_to_peak_overlap, logger)
    if mean:
        snp_to_peak_overlap = (
            snp_to_peak_overlap / snp_to_peak_overlap.sum(axis=1)[None, :]
        )
    snp_annot = snp_to_peak_overlap.dot(peak_annot)
    _write_annot(snp_annot, snp_df, annot_prefix, logger)


def _write_annot(
    snp_annot: np.ndarray,
    snp_info: pd.DataFrame,
    annot_prefix: str,
    logger,
    type: Literal["sparse", "dense"] = "sparse",
    rerun: bool = False,
) -> str:
    for chrom in snp_info["chrom"].unique():
        chrom_numeric = chrom.split("chr")[-1]
        if type == "dense":
            outfile_path = f"{annot_prefix}.{chrom_numeric}.annot.gz"
        else:
            outfile_path = f"{annot_prefix}.{chrom_numeric}.annot.npz"
        if not rerun and os.path.exists(outfile_path):
            continue
        mat = snp_annot[snp_info["chrom"] == chrom, :]
        if type == "dense":
            pd.DataFrame(mat).to_csv(outfile_path, sep="\t", header=False, index=False)
        else:
            scipy.sparse.save_npz(outfile_path, mat)
    logger.info(f"Wrote {outfile_path} files.")


def run_ldsc(
    peak_annot: np.ndarray,
    sumstat_paths_file: str,
    output_prefix,
    annot_id: str,
    annot_type: Literal["sparse", "dense"] = "sparse",
    nprocs: int = 10,
    logger=None,
    rerun_l2: bool = False,
    rerun_h2: bool = False,
):
    simba_plus.datasets._datasets.heritability(logger)
    annot_prefix = f"{output_prefix}/annots/{annot_id}"
    write_peak_annot(peak_annot, annot_prefix, type=annot_type)
    run_ldsc_l2(annot_prefix, annot_type=annot_type, nprocs=nprocs, logger=logger)
    output_dir = f"{output_prefix}/h2/{annot_id}/"
    run_ldsc_h2(
        sumstat_paths_file, output_dir, rerun=rerun_h2, nprocs=nprocs, logger=logger
    )


def load_data(prefix, version_suffix):
    adata_C = ad.read_h5ad(f"{prefix}/adata_C{version_suffix}.h5ad")
    adata_P = ad.read_h5ad(f"{prefix}/adata_P{version_suffix}.h5ad")
    adata_G = ad.read_h5ad(f"{prefix}/adata_G{version_suffix}.h5ad")
    # Load model and check scale
    adata_P.layers["X_normed"] = (
        adata_P.X / np.linalg.norm(adata_P.X.astype(np.float64), axis=1)[:, None]
    )
    adata_C.layers["X_normed"] = (
        adata_C.X / np.linalg.norm(adata_C.X.astype(np.float64), axis=1)[:, None]
    )
    if adata_G is not None:
        adata_G.layers["X_normed"] = (
            adata_G.X / np.linalg.norm(adata_G.X.astype(np.float64), axis=1)[:, None]
        )
        return adata_C, adata_P, adata_G
    return adata_C, adata_P, None


def assign_heritability_scores(adata_C, sumstat_paths, herit_result_prefix):
    cell_cov = adata_C.layers["X_normed"]
    for k, v in sumstat_paths.items():
        adata_C.obs[f"tau_z_{k}"], adata_C.obs[f"tau_{k}"] = get_tau_z_dep(
            f"{herit_result_prefix}.{os.path.basename(v).split('.gz')[0].split('.sumstat')[0]}.results",
            cell_cov,
        )
    return adata_C


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+heritability", log_dir=os.path.dirname(args.model_path)
        )
    if args.output_prefix is None:
        args.output_prefix = f"{args.run_path}/heritability/"
        os.makedirs(args.output_prefix, exist_ok=True)

    adata_C, adata_P, adata_G = load_data(args.adata_prefix, args.version_suffix)
    run_ldsc(
        adata_P.layers["X_normed"],
        args.sumstats,
        args.output_prefix,
        annot_id="peak_loadings",
    )

    # Get SIMBA+ heritability scores
    sumstat_paths_dict = pd.read_csv(args.sumstats, sep="\t", header=None, index_col=0)[
        1
    ].to_dict()
    adata_C = assign_heritability_scores(
        adata_C,
        sumstat_paths_dict,
        f"{args.output_prefix}/h2/peak_loadings/",
    )
    adata_P = assign_heritability_scores(
        adata_P,
        sumstat_paths_dict,
        f"{args.output_prefix}/h2/peak_loadings/",
    )
    adata_G = assign_heritability_scores(
        adata_G,
        sumstat_paths_dict,
        f"{args.output_prefix}/h2/peak_loadings/",
    )

    # Enrichment analysis
    enrichment_results = enrichment_analysis(
        pd.DataFrame(adata_G.obs[["" for p in sumstat_paths_dict.keys()]]).T,
        index=adata_G.obs_names,
    )
    with open(f"{args.output_prefix}/enrichment_results.pkl", "wb") as f:
        pkl.dump(enrichment_results, f)

    if args.create_report:
        pm.execute_notebook(
            "./scheritability_report.ipynb",
            f"{args.output_prefix}report{'' if args.gene_dist is None else '_' + str(args.gene_dist)}.ipynb",
            parameters=dict(
                checkpoint_path=args.checkpoint_path,
                version_suffix=args.version_suffix,
                cell_type_label=args.cell_type_label,
                sumstat_paths_file=args.sumstats,
                adata_prefix=args.adata_prefix,
                rerun=args.rerun,
                rerun_h2=args.rerun_h2,
                output_path=args.output_prefix,
                gene_mapping_distance=args.gene_dist,
            ),
            kernel_name="jy_ldsc3",
        )
