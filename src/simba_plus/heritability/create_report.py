import pickle as pkl
from typing import Literal
import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import papermill as pm
from argparse import ArgumentParser
import scipy
import simba_plus.datasets._datasets
from simba_plus.utils import setup_logging
from simba_plus.heritability.utils import get_overlap
from simba_plus.post_training.enrichment import run_enrichr
from simba_plus.heritability.ldsc import run_ldsc_l2, run_ldsc_h2
from simba_plus.heritability.get_taus import get_tau_z_dep


snp_pos_path = f"{os.path.dirname(__file__)}/../../../data/ldsc_data/1000G_Phase3_plinkfiles/ref.txt"


def add_argument(parser: ArgumentParser) -> ArgumentParser:
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory for LDSC run output and heritability results.",
    )
    parser.add_argument(
        "--create-report",
        action="store_true",
        help="Create .ipynb report plotting cell-level heritability scores.",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        default="",
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
        "--sumstats", type=str, default=None, help="Alternative sumstats ID"
    )

    return parser


def write_peak_annot(
    peak_annot: np.ndarray,
    peak_info_df: pd.DataFrame,
    annot_prefix: str,
    logger,
    type="sparse",
    mean=True,
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
    snp_to_peak_overlap = get_overlap(snp_df, peak_info_df).T
    logger.info(
        f"{snp_to_peak_overlap.sum(axis=0).astype(bool).sum()}/{snp_to_peak_overlap.shape[1]} SNPs overlap peaks."
    )
    if mean:
        row_sums = np.array(snp_to_peak_overlap.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # avoid division by zero
        snp_to_peak_overlap = snp_to_peak_overlap.multiply(
            1.0 / row_sums[:, np.newaxis]
        )

    snp_annot = snp_to_peak_overlap.dot(peak_annot)
    _write_annot(snp_annot, snp_df, annot_prefix, logger, type=type)


def _write_annot(
    snp_annot: np.ndarray,
    snp_info: pd.DataFrame,
    annot_prefix: str,
    logger,
    type: Literal["sparse", "dense"] = "sparse",
    rerun: bool = False,
) -> str:
    for chrom in snp_info["chrom"].unique():
        chrom_numeric = str(chrom).split("chr")[-1]
        if type == "dense":
            outfile_path = f"{annot_prefix}.{chrom_numeric}.annot.gz"
        else:
            outfile_path = f"{annot_prefix}.{chrom_numeric}.annot.npz"
        if not rerun and os.path.exists(outfile_path):
            logger.info(f"Annotation already exists in {outfile_path}, skipping.")
            continue
        mat = snp_annot[snp_info["chrom"] == chrom, :]
        if type == "dense":
            pd.DataFrame(mat).to_csv(outfile_path, sep="\t", header=False, index=False)
        else:
            mat = scipy.sparse.csr_matrix(mat)
            scipy.sparse.save_npz(outfile_path, mat)
    logger.info(f"Wrote {outfile_path} files.")


def run_ldsc(
    peak_annot: np.ndarray,
    peak_info_df: pd.DataFrame,
    sumstat_paths_file: str,
    output_dir,
    annot_id: str,
    annot_type: Literal["sparse", "dense"] = "sparse",
    nprocs: int = 10,
    logger=None,
    rerun_l2: bool = False,
    rerun_h2: bool = False,
):
    simba_plus.datasets._datasets.heritability(logger)
    annot_prefix = f"{output_dir}/annots/{annot_id}"
    write_peak_annot(
        peak_annot, peak_info_df, annot_prefix, logger=logger, type=annot_type
    )
    run_ldsc_l2(
        annot_prefix,
        annot_type=annot_type,
        rerun=rerun_l2,
        nprocs=nprocs,
        logger=logger,
    )
    output_dir = f"{output_dir}/h2/{annot_id}/"
    run_ldsc_h2(
        sumstat_paths_file,
        output_dir,
        annot_prefix,
        rerun=rerun_h2,
        nprocs=nprocs,
        logger=logger,
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
            f"{herit_result_prefix}{os.path.basename(v).split('.gz')[0].split('.sumstat')[0]}.results",
            cell_cov,
        )
    return adata_C


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+heritability", log_dir=os.path.dirname(args.adata_prefix)
        )
    if args.output_dir is None:
        args.output_dir = f"{os.path.dirname(args.adata_prefix)}/heritability/"
        os.makedirs(args.output_dir, exist_ok=True)

    adata_C, adata_P, adata_G = load_data(args.adata_prefix, args.version_suffix)
    run_ldsc(
        adata_P.layers["X_normed"],
        adata_P.obs,
        args.sumstats,
        args.output_dir,
        logger=logger,
        annot_id="peak_loadings",
        rerun_l2=args.rerun,
        rerun_h2=args.rerun_h2,
    )

    # Get SIMBA+ heritability scores
    sumstat_paths_dict = pd.read_csv(args.sumstats, sep="\t", header=None, index_col=0)[
        1
    ].to_dict()
    adata_C = assign_heritability_scores(
        adata_C,
        sumstat_paths_dict,
        f"{args.output_dir}/h2/peak_loadings/",
    )
    adata_P = assign_heritability_scores(
        adata_P,
        sumstat_paths_dict,
        f"{args.output_dir}/h2/peak_loadings/",
    )
    adata_G = assign_heritability_scores(
        adata_G,
        sumstat_paths_dict,
        f"{args.output_dir}/h2/peak_loadings/",
    )

    sumstat_paths = pd.read_csv(args.sumstats, sep="\t", header=None, index_col=0)[
        1
    ].to_dict()
    p = sc.pl.umap(
        adata_C,
        color=[f"tau_z_{pheno}" for pheno in sumstat_paths.keys()],
        vcenter=0,
        size=5,
        alpha=0.5,
        ncols=4,
        cmap="coolwarm",
        show=False,
    ).figure.savefig(f"{args.output_dir}/cell_type_heritability_scores.png")
    logger.info(f"Generated heritability report at {args.output_dir}/")
