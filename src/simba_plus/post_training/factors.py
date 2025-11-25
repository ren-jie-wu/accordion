from typing import Optional, List, Literal, Dict
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from simba_plus.post_training.enrichment import run_enrichr


def factor_enrichments(
    adata_G: ad.AnnData,
    gene_sets: List[
        Literal["GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"]
    ] = ["GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"],
):
    gene_loadings = pd.DataFrame(adata_G.layers["X_normed"], index=adata_G.obs_names)
    top_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    bot_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    n_top_genes = {}
    n_bot_genes = {}
    for i, factor in enumerate(gene_loadings.columns):
        if factor not in top_enrichments:
            top_enrichments[factor] = {}
            bot_enrichments[factor] = {}
        for gene_set in gene_sets:
            top_enrichments[factor][gene_set], n_top_genes[factor] = run_enrichr(
                gene_loadings[factor],
                gene_set=gene_set,
            )
            bot_enrichments[factor][gene_set], n_bot_genes[factor] = run_enrichr(
                gene_loadings[factor],
                top=False,
                gene_set=gene_set,
            )
    adata_G.uns["top_enrichments"] = top_enrichments
    adata_G.uns["bot_enrichments"] = bot_enrichments
    adata_G.uns["top_enrichments_n_genes"] = n_top_genes
    adata_G.uns["bot_enrichments_n_genes"] = n_bot_genes
    return adata_G


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
