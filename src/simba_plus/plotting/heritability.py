from typing import List, Literal, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad
from simba_plus.plotting.utils import enrichment
import math
from textwrap import wrap


def factor_herit(
    adata_C, pheno_list, figsize=(2, 6), return_fig=False, factor_enrichment_labels=None
):
    import matplotlib as mpl

    cmap = matplotlib.cm.get_cmap("coolwarm")
    left = cmap(0)
    right = cmap(cmap.N)
    figs = []
    for i in range(math.ceil(len(pheno_list) / 5)):
        n_panels = min(5, len(pheno_list) - i * 5)
        fig, ax = plt.subplots(
            1, n_panels, figsize=(figsize[0] * n_panels, figsize[1]), sharex=True
        )
        if n_panels == 1:
            ax = [ax]
        for j in range(n_panels):
            ax[j].invert_yaxis()
            pheno = pheno_list[i * 5 + j]
            factor_herit_scores = adata_C.uns["factor_heritability"][pheno]
            mask = factor_herit_scores < 0
            y = np.arange(len(factor_herit_scores))
            ax[j].yaxis.grid(True)
            ax[j].barh(y[mask], factor_herit_scores[mask], color=left)
            ax[j].barh(y[~mask], factor_herit_scores[~mask], color=right)
            ax[j].axvline(0, color="black")
            ax[j].set_xlabel("z-score")
            ax[j].set_title("\n".join(wrap(pheno, 15)))
        if factor_enrichment_labels is not None:
            top_enrichments, bot_enrichments = zip(
                *[s.split(" <> ") for s in factor_enrichment_labels]
            )
            # ax[0].yaxis.set_tick_params(labeltop=True, labelbottom=False, top=True)
            ax[0].set_ylabel(
                "Factor: Negative sign enrichment",
                color=left,
                fontdict={"weight": "bold"},
            )
            ax[0].set_yticks(list(range(len(bot_enrichments))))
            ax[0].set_yticklabels(
                [
                    f"F{i}:{bot_enrichments[i].split(':', 1)[1]}"
                    for i in range(len(bot_enrichments))
                ],
                color=left,
            )
            ax[j].yaxis.set_label_position("right")
            ax[j].set_ylabel(
                "Factor: Positive sign enrichment",
                color=right,
                rotation=-90,
                fontdict={"weight": "bold"},
            )
            ax[j].yaxis.set_tick_params(labelleft=False, labelright=True, right=True)
            ax[j].set_yticks(list(range(len(top_enrichments))))
            ax[j].set_yticklabels(
                [
                    f"F{i}:{top_enrichments[i].split(':', 1)[1]}"
                    for i in range(len(top_enrichments))
                ],
                color=right,
            )
        fig.subplots_adjust(hspace=0.3)
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
