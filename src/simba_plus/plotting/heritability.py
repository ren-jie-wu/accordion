from typing import List, Literal, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad
from simba_plus.plotting.utils import enrichment


def heritability_umap(
    adata, tau_prefix="tau_z_", size=5, alpha=0.5, ncols=4, cmap="coolwarm", **kwargs
):
    draw_col = []
    for col in adata.obs.columns:
        if col.startswith(tau_prefix):
            draw_col.append(col)
    sc.pl.umap(
        adata,
        color=draw_col,
        vcenter=0,
        size=5,
        alpha=0.5,
        ncols=ncols,
        cmap="coolwarm",
        show=False,
        **kwargs,
    )


def pheno_enrichment(
    adata_G,
    pheno,
    adj_p_thres=0.1,
    n_max_terms=10,
    gene_sets: List[str] = [
        "GO_Biological_Process_2021",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ],
    title_prefix="",
    figsize=(10, 15),
    return_fig: bool = False,
):
    fig = enrichment(
        adata_G,
        "pheno_enrichments",
        pheno,
        adj_p_thres,
        n_max_terms,
        gene_sets,
        title_prefix,
        figsize,
    )
    if return_fig:
        return fig
