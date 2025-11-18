from typing import Literal
from kneed import KneeLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp


def run_enrichr(
    gene_scores: pd.Series,
    gene_set: Literal[
        "GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"
    ] = "GO_Biological_Process_2021",
    detect_knee=True,
    knee_S=1.0,
    min_genes=100,
    top=True,
    plot_adjp_thres=0.1,
):
    sig_results = {}
    n_genes = {}
    gene_list = gene_scores.sort_values(ascending=not top)
    if detect_knee:
        x = range(0, len(gene_list))
        kn = KneeLocator(
            x,
            (-1 + 2 * int(top)) * gene_list,
            curve="convex",
            direction="decreasing",
            S=knee_S,
        )
        top_n = max(min_genes, kn.knee)
        n_genes = top_n
    pre_res = gp.enrichr(
        gene_list[:top_n].index.tolist(),
        gene_sets=gene_set,
        organism="Human",
        outdir=None,
        background=gene_list.index.tolist(),
    )
    res2d = pre_res.res2d
    # res2d = res2d.loc[~res2d["Odds Ratio"].map(np.isinf)]
    sig_results = (
        res2d[res2d["Adjusted P-value"] < plot_adjp_thres]
        .sort_values("Adjusted P-value", ascending=True)
        .reset_index()
    )

    return sig_results, n_genes
