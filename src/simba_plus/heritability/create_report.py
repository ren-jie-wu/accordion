import os
import papermill as pm
from argparse import ArgumentParser


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


def main(args):
    if args.output_prefix is None:
        args.output_prefix = f"{args.run_path}/heritability/"
        os.makedirs(args.output_prefix, exist_ok=True)

    pm.execute_notebook(
        "./scheritability_report.ipynb",
        f"{args.output_prefix}report{'' if args.gene_dist is None else '_' + str(args.gene_dist)}.ipynb",
        parameters=dict(
            checkpoint_path=args.checkpoint_path,
            write_orig_annot=False,
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
