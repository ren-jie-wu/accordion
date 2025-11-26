from typing import List, Literal, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad
from simba_plus.plotting.utils import enrichment
import math


def factor_herit(
    adata_C, pheno_list, figsize=(6, 2), return_fig=False, factor_enrichment_labels=None
):
    figs = []
    for i in range(math.ceil(len(pheno_list) / 5)):
        n_panels = min(5, len(pheno_list) - i * 5)
        fig, ax = plt.subplots(
            n_panels,
            figsize=(figsize[0], figsize[1] * n_panels),
        )
        if n_panels == 1:
            ax = [ax]
        for j in range(n_panels):
            pheno = pheno_list[i * 5 + j]
            factor_herit_scores = adata_C.uns["factor_heritability"][pheno]
            ax[j].bar(range(len(factor_herit_scores)), factor_herit_scores)
            ax[j].axhline(0, color="black")
            ax[j].set_ylabel("z-score")
            ax[j].set_title(pheno)
        if factor_enrichment_labels is not None:
            top_enrichments, bot_enrichments = zip(
                *[s.split(" <> ") for s in factor_enrichment_labels]
            )
            ax[0].xaxis.set_tick_params(labeltop="on")
            ax[0].set_xticks(list(range(len(top_enrichments))))
            ax[0].set_xticklabels(
                [
                    f"F{i}:{top_enrichments[i].split(':', 1)[1]}"
                    for i in range(len(top_enrichments))
                ],
                rotation=90,
                ha="right",
            )
            ax[j].set_xticks(list(range(len(bot_enrichments))))
            ax[j].set_xticklabels(
                [
                    f"F{i}:{bot_enrichments[i].split(':', 1)[1]}"
                    for i in range(len(bot_enrichments))
                ],
                rotation=90,
                ha="right",
            )
            ax[j].set_xlabel("Factor index")
        figs.append(fig)

    if return_fig:
        return figs


def heritability_umap(
    adata,
    tau_prefix="tau_z_",
    size=5,
    alpha=0.5,
    ncols=4,
    cmap="coolwarm",
    celltype_label=None,
    return_fig=False,
    rasterize=True,
    **kwargs,
):
    draw_col = []
    if celltype_label is not None:
        draw_col.append(celltype_label)
    for col in adata.obs.columns:
        if col.startswith(tau_prefix):
            draw_col.append(col)
    if rasterize:
        sc.set_figure_params(vector_friendly=True)
    fig = sc.pl.umap(
        adata,
        color=draw_col,
        vcenter=0,
        size=size,
        alpha=alpha,
        ncols=ncols,
        cmap=cmap,
        show=False,
        return_fig=True,
        **kwargs,
    )
    if return_fig:
        return fig


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
