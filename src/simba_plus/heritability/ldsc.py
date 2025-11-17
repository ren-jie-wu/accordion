from typing import Literal
import os
import pandas as pd
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

    script_dir = os.path.dirname(__file__)
    filedir = f"{script_dir}/../../../data/ldsc_data/"
    ldscdir = f"{script_dir}/../../ldsc/"
    bfile = f"{filedir}/1000G_Phase3_plinkfiles/1000G_EUR_Phase3_plink/1000G.EUR.QC"
    snplist = f"{filedir}/hm3_no_MHC.list.txt"

    for chrom in range(1, 23):
        out_path = os.path.join(
            f"{annotfile_prefix}.{chrom}",
        )
        if (not rerun) and os.path.exists(f"{out_path}.l2.ldscore.gz"):
            print(f"Skipping existing LDSC output for {out_path}")
            continue
        cmd = [
            "python",
            f"{ldscdir}/ldsc.py",
            "--l2",
            "--bfile",
            f"{bfile}.{chrom}",
            "--ld-wind-cm",
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
                rc = process.wait()
                if rc != 0:
                    # terminate any other still-running processes in this batch
                    for p in processes:
                        if p is not process and p.poll() is None:
                            try:
                                p.terminate()
                            except Exception:
                                try:
                                    p.kill()
                                except Exception:
                                    pass
                    raise subprocess.CalledProcessError(rc, process.args)
            processes = []
    for process in processes:
        rc = process.wait()
        if rc != 0:
            # terminate any other still-running processes
            for p in processes:
                if p is not process and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
            raise subprocess.CalledProcessError(rc, process.args)


def run_ldsc_h2(
    sumstat_paths_file, annot_prefix, out_dir, rerun=False, nprocs=10, logger=None
):
    simba_plus.datasets._datasets.heritability(logger=logger)
    processes = []
    os.makedirs(out_dir, exist_ok=True)

    script_dir = os.path.dirname(__file__)
    filedir = f"{script_dir}/../../../data/ldsc_data/"
    ldscdir = f"{script_dir}/../../ldsc/"
    weights_prefix = f"{filedir}/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
    frq_prefix = f"{filedir}/1000G_Phase3_frq/1000G.EUR.QC."
    refmodel_prefix = f"{filedir}/baselineLD.,{annot_prefix}."
    sumstat_paths = pd.read_csv(sumstat_paths_file, sep="\t", header=None, index_col=0)[
        1
    ].to_dict()
    for sumstat_path in list(sumstat_paths.values()):
        sumstat_basename = (
            os.path.basename(sumstat_path).split(".gz")[0].split(".sumstats")[0]
        )
        if not os.path.dirname(sumstat_path):
            sumstat_path = os.path.join(
                os.path.dirname(sumstat_paths_file), sumstat_path
            )
        out_path = os.path.join(out_dir, sumstat_basename)
        if (not rerun) and os.path.exists(f"{out_path}.results"):
            print(f"Skipping existing LDSC h2 output for {sumstat_basename}")
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
                rc = process.wait()
                if rc != 0:
                    # terminate any other still-running processes in this batch
                    for p in processes:
                        if p is not process and p.poll() is None:
                            try:
                                p.terminate()
                            except Exception:
                                try:
                                    p.kill()
                                except Exception:
                                    pass
                    raise subprocess.CalledProcessError(rc, process.args)
            processes = []
    for process in processes:
        rc = process.wait()
        if rc != 0:
            # terminate any other still-running processes
            for p in processes:
                if p is not process and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
            raise subprocess.CalledProcessError(rc, process.args)
