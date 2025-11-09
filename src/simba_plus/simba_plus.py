#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse
import simba_plus.load_data as load_data
import simba_plus.train as train
import simba_plus.evaluate as evaluate
import simba_plus.heritability.create_report as create_heritability_report


def main():
    if len(sys.argv) < 2:
        print("Usage: simba+ <subcommand> [args]")
        print("Available subcommands: load_data, train, eval, heritability")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="simba+", description="Simba+ command line interface"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    load_data_parser = subparsers.add_parser(
        "load_data",
    )
    load_data_parser = load_data.add_argument(load_data_parser)

    train_parser = subparsers.add_parser("train")
    train_parser = train.add_argument(train_parser)

    eval_parser = subparsers.add_parser("eval")
    eval_parser = evaluate.add_argument(eval_parser)

    heritability_parser = subparsers.add_parser("heritability")
    heritability_parser = create_heritability_report.add_argument(heritability_parser)

    parsed_args = parser.parse_args()
    subcommand = parsed_args.subcommand

    if subcommand == "load_data":
        load_data.main(parsed_args)
    elif subcommand == "train":
        train.main(parsed_args)
    elif subcommand == "eval":
        evaluate.main(parsed_args)
    elif subcommand == "heritability":
        create_heritability_report.main(parsed_args)


if __name__ == "__main__":
    main()
