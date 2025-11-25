from typing import List, Literal, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad
from simba_plus.plotting.utils import enrichment


def factor_enrichment(
    adata_G: ad.AnnData,
    factor: str,
    adj_p_thres=0.1,
    n_max_terms=10,
    gene_sets: List[str] = [
        "GO_Biological_Process_2021",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ],
    title_prefix: str = "Factor ",
    figsize=(10, 15),
    return_fig: bool = False,
):
    """
    Plot enrichment results for a given factor and gene set.

    Args:
        adata_G: AnnData object containing gene data with enrichment results in .uns.
        factor: The factor for which to plot enrichment results.
        gene_set: The gene set to plot (e.g., "GO_Biological_Process_2021"). If None, plots for all gene sets.
        n_max_terms: Maximum number of enriched terms to display in the plot.
    """
    fig = enrichment(
        adata_G,
        "top_enrichments",
        factor,
        adj_p_thres,
        n_max_terms,
        gene_sets,
        title_prefix,
        figsize,
    )
    if return_fig:
        return fig


def factor_umap(
    adata_C: ad.AnnData,
    return_fig: bool = False,
    **kwargs,
):
    adata_C = adata_C.copy()
    if "X_normed" not in adata_C.layers:
        adata_C.layers["X_normed"] = (
            adata_C.X / np.linalg.norm(adata_C.X.astype(np.float64), axis=1)[:, None]
        )
    for i in range(adata_C.X.shape[1]):
        adata_C.obs[f"Factor {i}"] = adata_C.layers["X_normed"][:, i]

    ax = sc.pl.umap(
        adata_C,
        color=[f"Factor {i}" for i in range(adata_C.X.shape[1])],
        cmap="coolwarm",
        vcenter=0,
        show=not return_fig,
        **kwargs,
    )
    if return_fig:
        return ax
