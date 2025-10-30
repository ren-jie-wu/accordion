"""
SIMBA+ Visualization Utilities
------------------------------
Plot publication-quality enrichment–recall and enrichment–distance curves
for supervised and unsupervised evaluations.

Example:
    >>> from simba_plus.visualization import plot_utils as viz
    >>> viz.plot_supervised_er(avg_preds_df, save_path="crispr_supervised_ER.pdf")
    >>> viz.plot_enrichment_vs_distance_grid(
    ...     [crispr], ["crispr"],
    ...     score_columns_allcell, score_labels_allcell,
    ...     thresholds=[1000,5000,10000,20000,50000],
    ...     save_fig=True
    ... )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simba_plus.discovery.model_training import enrichment_recall, auerc_old


# ----------------------------
# GLOBAL PLOT SETTINGS
# ----------------------------
plt.rcParams.update({
    'pdf.fonttype': 42,  # Editable text
    'svg.fonttype': 'none',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
})

# ==========================================================
# 🔹 SUPERVISED ENRICHMENT–RECALL CURVE
# ==========================================================

def plot_supervised(
    avg_preds_df: pd.DataFrame,
    columns_and_labels: list[tuple[str, str]] | None = None,
    *,
    save_path: str | None = None,
    title: str = "Supervised Enrichment–Recall"
):
    """
    Plot enrichment–recall (ER) curves for supervised benchmark models.

    Args:
        avg_preds_df: DataFrame containing model predictions.
        columns_and_labels: list of (display_label, column_name).
            Example:
                [('SIMBA+ path score + 1/Distance', 'xgboost_simba2_distance'),
                 ('SCENT FDR', 'xgboost_scent_alone')]
        save_path: optional PDF output path.
        title: plot title.
    """

    if columns_and_labels is None:
        raise ValueError("columns_and_labels must be provided.")

    plt.figure(figsize=(8, 6))
    for (label, col) in columns_and_labels:
        if col not in avg_preds_df.columns:
            print(f"⚠️ Warning: column '{col}' not found — skipping {label}")
            continue

        er_data = enrichment_recall(avg_preds_df, col=col)
        auerc_value = auerc_old(
            y_true=avg_preds_df["gold"],
            y_pred=avg_preds_df[col]
        )
        plt.plot(
            er_data["recall"], er_data["enrichment"],
            linestyle="-", linewidth=2, alpha=0.85,
            label=f"{label} (AUERC: {auerc_value:.2f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Enrichment")
    plt.title(title)
    plt.grid(False)
    plt.legend(loc="upper right", frameon=False)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    plt.show()


# ==========================================================
# 🔹 UNSUPERVISED ENRICHMENT vs DISTANCE CURVES
# ==========================================================

def plot_unsupervised(
    dataset: pd.DataFrame,
    columns_and_labels: list[tuple[str, str]],
    thresholds: list[int],
    *,
    title: str = "Unsupervised CRISPR Enrichment–Distance",
    save_fig: bool = False,
    save_path: str | None = None
):
    """
    Plot average enrichment vs. distance thresholds for one dataset (unsupervised mode).

    Args:
        dataset (pd.DataFrame): Input DataFrame with columns like
            ['Distance_to_TSS', 'gold', ...score columns...].
        columns_and_labels (list[tuple[str, str]]):
            List of (display_label, column_name) pairs to plot.
            Example: [('SIMBA+ path score', 'SIMBA_plus_path_score'), ('1/Distance', '1/Distance')]
        thresholds (list[int]): Distance thresholds in base pairs.
        title (str): Figure title.
        save_fig (bool): Whether to save the figure as PDF.
        save_path (str, optional): Path to save the figure.
    """
    from simba_plus.discovery.model_training import enrichment_recall

    thresholds_kb = np.array(thresholds) / 1000  # convert to kb
    plt.figure(figsize=(7, 5))

    # Use default color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    average_enrichments = {label: [] for label, _ in columns_and_labels}

    # Compute average enrichment for each threshold
    for threshold in thresholds:
        subset = dataset[dataset["Distance_to_TSS"] >= threshold]
        if subset["gold"].sum() < 5:
            # skip thresholds with too few positives
            for label, _ in columns_and_labels:
                average_enrichments[label].append(np.nan)
            continue

        for label, col in columns_and_labels:
            if col not in subset.columns:
                average_enrichments[label].append(np.nan)
                continue
            er_data = enrichment_recall(subset, col=col)
            avg_enrich = er_data["enrichment"].mean()
            average_enrichments[label].append(avg_enrich)

    # Plot one line per feature set
    for i, (label, _) in enumerate(columns_and_labels):
        color = color_cycle[i % len(color_cycle)]
        plt.plot(
            thresholds_kb,
            average_enrichments[label],
            label=label,
            color=color,
            linewidth=2,
            alpha=0.9,
            marker="o",
            markersize=6,
        )

    # Beautify plot
    plt.xscale("log")
    plt.xticks(thresholds_kb, [f">{int(t/1000)} kb" for t in thresholds], fontsize=10)
    plt.xlabel("Distance Threshold (kb)", fontsize=12)
    plt.ylabel("Average Enrichment", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper left", frameon=False)
    plt.grid(False)
    sns.despine()
    plt.tight_layout()

    if save_fig and save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    plt.show()