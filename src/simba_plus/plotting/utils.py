from typing import List, Literal, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad


def enrichment(
    adata_G,
    enrichment_uns_key,
    enrichment_key,
    adj_p_thres=0.1,
    n_max_terms=10,
    gene_sets: List[str] = [
        "GO_Biological_Process_2021",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ],
    title_prefix="",
    figsize=(10, 15),
):
    scatterplots = []
    fig, ax = plt.subplots(
        len(gene_sets), 1, figsize=figsize, gridspec_kw={"hspace": 0.5}
    )
    for i, gene_set in enumerate(gene_sets):
        pdf = adata_G.uns[enrichment_uns_key][enrichment_key][gene_set]

        pdf = (
            pdf.loc[pdf["Adjusted P-value"] < adj_p_thres]
            .sort_values("Adjusted P-value", ascending=True)
            .reset_index()
        )

        if len(pdf) > n_max_terms:
            pdf = pdf.iloc[:n_max_terms, :]
        pdf = pdf.sort_values("Combined Score", ascending=False)
        pdf["-log10(FDR)"] = -np.log10(pdf["Adjusted P-value"].values)
        scatterplots.append(
            ax[i].scatter(
                pdf["Combined Score"],
                pdf["Term"],
                c=pdf["Odds Ratio"],
                s=pdf["-log10(FDR)"] * 10,
                cmap="Reds",
                edgecolors="black",
            )
        )  # marker="-log10(FDR)", palette="Reds", edgecolor="black")
        ax[i].set_title(f"{title_prefix}{enrichment_key}\n{gene_set}")
        ax[i].invert_yaxis()
        ax[i].set_box_aspect(1.5)

        # Add cmap
        # divider = make_axes_locatable(ax[i])
        # cax = divider.append_axes("right", size="5%", pad=1)
    plt.tight_layout()
    for i, scatter in enumerate(scatterplots):
        handles, labels = scatterplots[i].legend_elements(
            prop="sizes",
            num=3,
            color="gray",
            func=lambda s: 100 * np.sqrt(s) / plt.rcParams["lines.markersize"],
        )
        ax[i].legend(
            handles,
            labels,
            title="-log10(P_adj)",
            bbox_to_anchor=(1.0, 1.0),
            loc="upper left",
            frameon=False,
            labelspacing=1,
        )
        pos = ax[i].get_position()
        cax = fig.add_axes([pos.x1 + 0.05, pos.y0, 0.02, pos.height * 0.4])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Odds Ratio")
        ax[i].set_xlabel("Combined Score")
    return fig
