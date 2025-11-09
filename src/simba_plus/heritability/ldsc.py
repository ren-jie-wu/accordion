from typing import Literal
import os
import subprocess
import simba_plus.datasets._datasets


def run_ldsc_l2(
    annotfile_prefix: str,
    annot_type: Literal["sparse", "dense"] = "sparse",
    rerun: bool = False,
    nprocs: int = 10,
    logger=None,
):
    if annot_type == "sparse":
        suffix = "annot.npz"
    else:
        suffix = "annot.gz"
    simba_plus.datasets._datasets.heritability(logger=logger)
    processes = []
    os.makedirs(out_dir, exist_ok=True)

    script_dir = os.path.dirname(__file__)
    filedir = f"{script_dir}/../../../data/ldsc_data/"
    ldscdir = f"{script_dir}/../../ldsc/"
    bfile = f"{filedir}/1000G_Phase3_plinkfiles/1000G_Phase3_plinkfiles/1000G_EUR_Phase3_plink/1000G.EUR.QC"
    snplist = f"{filedir}/hm3_no_MHC.list.txt"

    for chrom in range(1, 23):
        out_path = os.path.join(
            f"{annotfile_prefix}.{chrom}.{suffix}", sumstat_basename
        )
        if (not rerun) and os.path.exists(f"{out_path}"):
            print(f"Skipping existing LDSC output for {out_path}")
            continue
        cmd = [
            "python",
            f"{ldscdir}/ldsc.py",
            "--l2",
            "--bfile",
            f"{bfile}.{chrom}" "--ld-wind-cm",
            "1",
            "--annot",
            f"{annotfile_prefix}.{chrom}.{suffix}",
            "--thin-annot",
            "--print-snps",
            snplist,
            "--out",
            out_path,
        ]
        processes.append(subprocess.Popen(cmd))
        if len(processes) >= nprocs:
            for process in processes:
                process.wait()
            processes = []
    for process in processes:
        process.wait()


def run_ldsc_h2(sumstat_paths, out_dir, rerun=False, nprocs=10, logger=None):
    simba_plus.datasets._datasets.heritability(logger=logger)
    processes = []
    os.makedirs(out_dir, exist_ok=True)

    script_dir = os.path.dirname(__file__)
    filedir = f"{script_dir}/../../../data/ldsc_data/"
    ldscdir = f"{script_dir}/../../ldsc/"
    weights_prefix = f"{filedir}/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
    frq_prefix = f"{filedir}/1000G_Phase3_frq/1000G.EUR.QC."
    refmodel_prefix = f"{filedir}/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD."

    for sumstat_path in list(sumstat_paths):
        sumstat_basename = (
            os.path.basename(sumstat_path).split(".gz")[0].split(".sumstats")[0]
        )
        out_path = os.path.join(out_dir, sumstat_basename)
        if (not rerun) and os.path.exists(f"{out_path}.results"):
            print(f"Skipping existing LDSC output for {sumstat_basename}")
            continue
        cmd = [
            "python",
            f"{ldscdir}/ldsc.py",
            "--h2",
            sumstat_path,
            "--w-ld-chr",
            weights_prefix,
            "--frqfile-chr",
            frq_prefix,
            "--ref-ld-chr",
            refmodel_prefix,
            "--overlap-annot",
            "--thin-annot",
            "--print-coefficients",
            "--out",
            out_path,
            "--print-residuals",
        ]
        processes.append(subprocess.Popen(cmd))
        if len(processes) >= nprocs:
            for process in processes:
                process.wait()
            processes = []
    for process in processes:
        process.wait()
