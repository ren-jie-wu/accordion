from typing import Optional, List, Literal, Dict
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from simba_plus.post_training.enrichment import run_enrichr


def plot_factors(
    adata_C: ad.AnnData, ax: Optional[matplotlib.axes.Axes] = None, kwargs={}
):
    if ax is None:
        fig, ax = plt.subplots()
    adata_C = adata_C.copy()
    if "X_normed" not in adata_C.layers:
        adata_C.layers["X_normed"] = (
            adata_C.X / np.linalg.norm(adata_C.X.astype(np.float64), axis=1)[:, None]
        )
    for i in range(adata_C.X.shape[1]):
        adata_C.obs[f"Factor {i}"] = adata_C.layers["X_normed"][:, i]

    sc.pl.umap(
        adata_C,
        color=[f"Factor {i}" for i in range(adata_C.X.shape[1])],
        cmap="vlag",
        vcenter=0,
        **kwargs,
    )
    return ax


def get_factor_enrichments(
    adata_G: ad.AnnData,
    gene_sets: List[
        Literal["GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"]
    ] = ["GO_Biological_Process_2021"],
):
    gene_loadings = pd.DataFrame(adata_G.layers["X_normed"], index=adata_G.obs_names)
    top_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    bot_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    n_top_genes = {}
    n_bot_genes = {}
    for i, factor in enumerate(gene_loadings.columns):
        for gene_set in gene_sets:
            top_enrichments[factor][gene_set], n_top_genes[factor] = run_enrichr(
                gene_loadings,
                gene_sets=gene_set,
            )
            bot_enrichments[factor][gene_set], n_bot_genes[factor] = run_enrichr(
                gene_loadings,
                top=False,
                gene_sets=gene_set,
            )
    adata_G.uns["top_enrichments"] = top_enrichments
    adata_G.uns["bot_enrichments"] = bot_enrichments
    adata_G.uns["top_enrichments_n_genes"] = n_top_genes
    adata_G.uns["bot_enrichments_n_genes"] = n_bot_genes
    return adata_G


def plot_enrichment(
    adata_G: ad.AnnData,
    factor: str,
    gene_set: Literal[
        "GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"
    ],
    n_max_terms=20,
    ax: Optional[matplotlib.axes.Axes] = None,
    title_prefix: str = "",
):
    """
    Plot enrichment results for a given factor and gene set.

    Args:
        adata_G: AnnData object containing gene data with enrichment results in .uns.
        factor: The factor for which to plot enrichment results.
        gene_set: The gene set to plot (e.g., "GO_Biological_Process_2021"). If None, plots for all gene sets.
        n_max_terms: Maximum number of enriched terms to display in the plot.
    """
    top_enrichments = adata_G.uns["top_enrichments"][factor]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 5))
    pdf = top_enrichments[gene_set]
    if len(pdf) > n_max_terms:
        pdf = pdf.iloc[:n_max_terms, :]
    pdf = pdf.sort_values("Combined Score", ascending=False)
    pdf["-log10(FDR)"] = -np.log10(pdf["Adjusted P-value"].values)
    sc = ax.scatter(
        pdf["Combined Score"],
        pdf["Term"],
        c=pdf["Odds Ratio"],
        s=pdf["-log10(FDR)"] * 10,
        cmap="Reds",
        edgecolors="black",
    )
    ax.set_title(f"{title_prefix}{factor}\n{gene_set}")
    ax.gca().invert_yaxis()
    handles, labels = sc.legend_elements(
        prop="sizes",
        num=3,
        color="gray",
        func=lambda s: 100 * np.sqrt(s) / plt.rcParams["lines.markersize"],
    )
    ax.legend(
        handles,
        labels,
        title="-log10(P_adj)",
        bbox_to_anchor=(1.02, 0.9),
        loc="upper left",
        frameon=False,
        labelspacing=2,
    )
    # Add cmap
    cax = fig.add_axes([1.0, 0.1, 0.05, 0.3])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Odds Ratio")
    ax.set_xlabel("Combined Score")
    return fig


def summarize_enrichments(factor_idx, gene_enrichment, gene_bot_enrichment):
    # Print peak annotation side by side
    top_most_gene_enrichment = (
        gene_enrichment[factor_idx]
        .sort_values("Adjusted P-value")[["Adjusted P-value", "Term"]]
        .iloc[0]
        if len(gene_enrichment[factor_idx]) > 0
        else None
    )
    bot_most_gene_enrichment = (
        gene_bot_enrichment[factor_idx]
        .sort_values("Adjusted P-value")[["Adjusted P-value", "Term"]]
        .iloc[0]
        if len(gene_bot_enrichment[factor_idx]) > 0
        else None
    )

    if top_most_gene_enrichment is None:
        top_enriched = "No enrichment"
    else:
        top_enriched = top_most_gene_enrichment["Term"]
    if bot_most_gene_enrichment is None:
        bot_enriched = "No enrichment"
    else:
        bot_enriched = bot_most_gene_enrichment["Term"]

    label = f"{top_enriched} <> {bot_enriched}"
    return label
