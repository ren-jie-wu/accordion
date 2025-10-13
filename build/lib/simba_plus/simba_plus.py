#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse
import simba_plus.load_data as load_data
import simba_plus.train as train


def main():
    if len(sys.argv) < 2:
        print("Usage: simba+ <subcommand> [args]")
        print("Available subcommands: load_data, train")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="simba+",
        description="Simba+ command line interface"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # load_data subcommand
    load_data_parser = subparsers.add_parser("load_data", help="Load data")
    load_data_parser.add_argument(
        "--gene-adata",
        type=str,
        help="Path to the cell by gene AnnData file (e.g., .h5ad).",
    )
    load_data_parser.add_argument(
        "--peak-adata",
        type=str,
        help="Path to the cell by gene AnnData file (e.g., .h5ad).",
    )
    load_data_parser.add_argument(
        "--batch-col",
        type=str,
        help="Batch column in AnnData.obs of gene AnnData. If gene AnnData is not provided, peak AnnData will be used.",
    )
    load_data_parser.add_argument(
        "out_path",
        type=str,
        help="Path to the saved HeteroData object (e.g., .pt file).",
    )

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    parsed_args = parser.parse_args()
    subcommand = parsed_args.subcommand
    args = getattr(parsed_args, "args", [])

    if subcommand == "load_data":
        load_data.main(args)
    elif subcommand == "train":
        train.main(args)

if __name__ == "__main__":
    main()
