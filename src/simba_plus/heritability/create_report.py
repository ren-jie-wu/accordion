from typing import Literal
import os
import numpy as np
import pandas as pd
import papermill as pm
from argparse import ArgumentParser
import scipy
import simba_plus.datasets._datasets
from simba_plus.heritability.utils import get_overlap, plot_hist
from simba_plus.heritability.ldsc import run_ldsc_l2, run_ldsc_h2

snp_pos_path = (
    f"{os.path.dirname(__file__)}/../datasets/ldsc_data/1000G_Phase3_plinkfiles/ref.txt"
)


def add_argument(parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "sumstats", help="GwAS summary statistics compatible with LDSC inputs", type=str
    )
    parser.add_argument(
        "adata_prefix",
        type=str,
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
    parser.add_argument("--output-prefix", type=str, default=None)
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
    annot_path = write_peak_annot(peak_annot, annot_prefix, type=annot_type)
    run_ldsc_l2(annot_prefix, annot_type=annot_type, nprocs=nprocs, logger=logger)
    output_dir = f"{output_prefix}/h2/{annot_id}/"
    run_ldsc_h2(
        sumstat_paths_file, output_dir, rerun=rerun_h2, nprocs=nprocs, logger=logger
    )


def main(args):
    if args.output_prefix is None:
        args.output_prefix = f"{args.run_path}/heritability/"
        os.makedirs(args.output_prefix, exist_ok=True)

    pm.execute_notebook(
        "./scheritability_report.ipynb",
        f"{args.output_prefix}report{'' if args.gene_dist is None else '_' + str(args.gene_dist)}.ipynb",
        parameters=dict(
            checkpoint_path=args.checkpoint_path,
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
