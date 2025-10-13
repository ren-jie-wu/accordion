import pandas as pd
import numpy as np
import torch
from collections import Counter
import matplotlib.pyplot as plt

# test function to load the entire dataset
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import beta
from scipy.stats import norm
import scipy.stats as stats

from sklearn.calibration import calibration_curve


# TODO list:
# 1. the cell, gene, link distribution of a dataset and a sampled batch. Or many batch avarage. -- Done
# 2. Gene expression reconstruction error, and gene expression value prediction pearson&spearman
# 3. Peak reconstruction error. take it as a classification problem
# 4. Cell embedding evaluation: compute the ARI, ASW for the learned embedding against cell types
# 5. Gene embedding evaluation: compute the ASW of the gene embedding against whether a gene is a HVG of a cell type
def record_and_plot_graph_stats(
    data, dataset_name="Dataset", bin_size=5, save_path="./"
):
    """
    This is the so-called
    This function records statistics for a cell-gene graph dataset and plots specified metrics.

    Parameters:
    - data: The data object or subgraph, typically in PyTorch Geometric format.
      It should be a bidirectional graph
      Expected keys are:
      - `edge_index` representing edges between cells and genes.
      - `edge_type` or other features that identify if nodes are cells or genes.
    - dataset_name: A name identifier for the dataset (e.g., 'Train', 'Validation', or 'Full Dataset').
    - bin_size: Size of the bins for neighbor count distributions in the histograms.
    """
    # 1. Number of unique cells and genes
    # import pdb; pdb.set_trace()
    cell_indices = torch.unique(data["cell"].n_id)
    gene_indices = torch.unique(data["gene"].n_id)

    num_cells = len(cell_indices)
    num_genes = len(gene_indices)
    if dataset_name is not None:
        print(f"{dataset_name} Statistics:")
    print(f"Number of cells: {num_cells}")
    print(f"Number of genes: {num_genes}")

    # 2. Number of links (edges)
    num_links = data["cell", "expresses", "gene"].edge_index.size(1)
    # print(f"Number of links: {num_links}")

    # 3. Count neighbor cells for each gene and plot distribution with bins
    gene_to_cell_edges = data["gene", "rev_expresses", "cell"].edge_index
    gene_neighbors = gene_to_cell_edges[1].tolist()
    gene_neighbor_count = Counter(gene_neighbors)

    # import pdb; pdb.set_trace()

    # Create a list of counts to apply binning
    gene_neighbor_values = list(gene_neighbor_count.values())
    max_gene_neighbors = max(gene_neighbor_values)
    bins = np.arange(0, max_gene_neighbors + bin_size, bin_size)

    if dataset_name is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(gene_neighbor_values, bins=bins, edgecolor="black")
        plt.title(f"Distribution of Gene Expression per Cell - {dataset_name}")
        plt.xlabel("Number of Genes Expressed in a Cell")
        plt.ylabel("Number of Cells")
        # plt.yscale("log")
        plt.savefig(
            f"{save_path}/Distribution of Gene Expression per Cell_{dataset_name}.png"
        )

    # 4. Count neighbor genes for each cell and plot distribution with bins
    cell_to_gene_edges = data["cell", "expresses", "gene"].edge_index
    cell_neighbors = cell_to_gene_edges[1].tolist()
    cell_neighbor_count = Counter(cell_neighbors)

    # Create a list of counts to apply binning
    cell_neighbor_values = list(cell_neighbor_count.values())
    max_cell_neighbors = max(cell_neighbor_values)
    bins = np.arange(0, max_cell_neighbors + bin_size, bin_size)

    if dataset_name is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(cell_neighbor_values, bins=bins, edgecolor="black")
        plt.title(f"Distribution of cell Expresses per gene - {dataset_name}")
        plt.xlabel("Number of Cells Express a gene")
        plt.ylabel("Number of Genes")
        # plt.yscale("log")
        plt.savefig(
            f"{save_path}/Distribution of Cell Express per Gene_{dataset_name}.png"
        )

    # Return a dictionary of recorded metrics for further evaluation
    stats = {
        "num_cells": num_cells,
        "num_genes": num_genes,
        "num_links": num_links,
        "gene_neighbor_count": gene_neighbor_values,
        "cell_neighbor_count": cell_neighbor_values,
    }
    # print(stats)
    return stats


def record_and_plot_multiome_graph_stats(
    data, relationships, dataset_name="Dataset", bin_size=5, save_path="./"
):
    """
    Records statistics for a heterogeneous graph dataset and plots specified metrics for multiple relationships,
    following the logic of counting the occurrences of target nodes in edge indices.

    Parameters:
    - data: The data object or subgraph, typically in PyTorch Geometric format.
      Expected to be a bidirectional graph.
    - relationships: List of tuples representing the relationships to process.
      Each tuple is (source_node_type, edge_type, target_node_type).
    - dataset_name: A name identifier for the dataset (e.g., 'Train', 'Validation', or 'Full Dataset').
    - bin_size: Size of the bins for neighbor count distributions in the histograms.
    - save_path: Directory path to save the plots.
    """

    # Initialize a dictionary to store stats
    stats = {}

    if dataset_name is not None:
        print(f"{dataset_name} Statistics:")

    for src_type, rel_type, dst_type in relationships:
        # Print relationship info
        print(f"\nProcessing relationship: ({src_type}, {rel_type}, {dst_type})")

        # Check if the edge type exists in the data
        if (src_type, rel_type, dst_type) not in data.edge_types:
            print(
                f"Edge type ({src_type}, {rel_type}, {dst_type}) not found in data. Skipping."
            )
            continue

        # Get the edge index for the relationship
        edge_index = data[src_type, rel_type, dst_type].edge_index
        # src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()

        # Count the occurrences of target nodes (following your logic)
        target_neighbor_count = Counter(dst_indices)
        target_neighbor_values = list(target_neighbor_count.values())
        max_target_neighbors = (
            max(target_neighbor_values) if target_neighbor_values else 0
        )
        bins = np.arange(0, max_target_neighbors + bin_size, bin_size)

        # Plot histogram for target neighbor counts
        plt.figure(figsize=(10, 6))
        plt.hist(target_neighbor_values, bins=bins, edgecolor="black")
        plt.title(
            f"Distribution of {src_type} connections per {dst_type} - {dataset_name}"
        )
        plt.xlabel(f"Number of {src_type} connections per {dst_type}")
        plt.ylabel(f"Number of {dst_type} nodes")
        plt.savefig(
            f"{save_path}/Distribution_of_{src_type}_connections_per_{dst_type}_{dataset_name}_{rel_type}.png"
        )
        plt.close()

        # Similarly, process the reverse edge if it exists
        reverse_edge_type = (dst_type, f"rev_{rel_type}", src_type)
        if reverse_edge_type in data.edge_types:
            print(f"Processing reverse relationship: {reverse_edge_type}")
            rev_edge_index = data[reverse_edge_type].edge_index
            rev_dst_indices = rev_edge_index[1].tolist()

            # Count the occurrences of target nodes in the reverse edge
            rev_target_neighbor_count = Counter(rev_dst_indices)
            rev_target_neighbor_values = list(rev_target_neighbor_count.values())
            max_rev_target_neighbors = (
                max(rev_target_neighbor_values) if rev_target_neighbor_values else 0
            )
            bins = np.arange(0, max_rev_target_neighbors + bin_size, bin_size)

            # Plot histogram for reverse target neighbor counts
            plt.figure(figsize=(10, 6))
            plt.hist(rev_target_neighbor_values, bins=bins, edgecolor="black")
            plt.title(
                f"Distribution of {dst_type} connections per {src_type} - {dataset_name}"
            )
            plt.xlabel(f"Number of {dst_type} connections per {src_type}")
            plt.ylabel(f"Number of {src_type} nodes")
            plt.savefig(
                f"{save_path}/Distribution_of_{dst_type}_connections_per_{src_type}_{dataset_name}_{rel_type}.png"
            )
            plt.close()
        else:
            print(f"Reverse edge type {reverse_edge_type} not found in data.")

        # Record stats
        stats_key = f"{src_type}_{rel_type}_{dst_type}"
        stats[stats_key] = {
            "num_edges": edge_index.size(1),
            "target_neighbor_counts": target_neighbor_values,
        }

        if reverse_edge_type in data.edge_types:
            rev_stats_key = f"{dst_type}_rev_{rel_type}_{src_type}"
            stats[rev_stats_key] = {
                "num_edges": rev_edge_index.size(1),
                "target_neighbor_counts": rev_target_neighbor_values,
            }

    return stats


def summarize_epoch_stats(df_list, save_path="./"):
    """
    Summarizes the overall statistics of an epoch based on a list of batch DataFrames
    and plots the distribution of gene and cell neighbor counts.

    Parameters:
    - df_list: List of DataFrames where each DataFrame contains statistics for a batch,
      with keys ['num_cells', 'num_genes', 'num_links', 'gene_neighbor_count', 'cell_neighbor_count'].

    Returns:
    - epoch_stats: A dictionary containing aggregated statistics for the epoch.
    """

    # Initialize accumulators for stats
    total_cells = 0
    total_genes = 0
    total_links = 0
    all_gene_neighbor_counts = []
    all_cell_neighbor_counts = []

    # Loop through each batch DataFrame to accumulate statistics
    for batch_stats in df_list:
        # Accumulate unique cells and genes (assuming cells and genes are represented as unique IDs)
        total_cells += batch_stats["num_cells"]
        total_genes += batch_stats["num_genes"]

        # Sum the number of links
        total_links += batch_stats["num_links"]

        # Collect all neighbor counts for distribution analysis
        all_gene_neighbor_counts.extend(batch_stats["gene_neighbor_count"])
        all_cell_neighbor_counts.extend(batch_stats["cell_neighbor_count"])

    # Summarize overall epoch stats
    epoch_stats = {
        "num_cells": total_cells,
        "num_genes": total_genes,
        "num_links": total_links,
    }

    print("Epoch Summary:")
    print(f"Total number of cells: {total_cells}")
    print(f"Total number of genes: {total_genes}")
    print(f"Total number of links: {total_links}")

    # Plot distribution of gene neighbor counts
    plt.figure(figsize=(10, 6))
    plt.hist(all_gene_neighbor_counts, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Gene Neighbor Counts")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_path}/Neighbor_epoch_gene.png")

    # Plot distribution of cell neighbor counts
    plt.figure(figsize=(10, 6))
    plt.hist(all_cell_neighbor_counts, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Cell Neighbor Counts")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_path}/Neighbor_epoch_cell.png")

    return epoch_stats


def scatter_color_var(
    y_true,
    y_pred_mean,
    y_pred_var,
    fig_name="cell_express_gene_scatter",
    save_path="./",
):
    # Scatter plot of pred mean vs true values
    plt.figure()
    plt.scatter(y_true, y_pred_mean, c=y_pred_var, cmap="viridis", alpha=0.7)
    plt.xlabel("True expression")
    plt.ylabel("Predicted mean")
    plt.colorbar(label="Predicted variance")
    plt.title("Gene expression reconstruction (colored by predicted variance)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


# Function to plot NLL distributions
def plot_nll_distributions(class1, class2, save_path="./", name="cell_express_gene"):
    # Define bins
    bins = np.linspace(
        min(min(class1), min(class2)) - 0.5, max(max(class1), max(class2)) + 0.5, 20
    )

    # Plot Class 1 only
    plt.figure(figsize=(8, 6))
    plt.hist(class1, bins=bins, color="blue", alpha=0.7, label="Pos", edgecolor="black")
    plt.xlabel("Negative Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("Distribution of NLL - Positive Edges")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{save_path}/{name}_pos_nll_distribution.png")
    plt.close()

    # Plot Class 2 only
    plt.figure(figsize=(8, 6))
    plt.hist(
        class2, bins=bins, color="green", alpha=0.7, label="Neg", edgecolor="black"
    )
    plt.xlabel("Negative Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("Distribution of NLL - Negative Edges")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{save_path}/{name}_neg_nll_distribution.png")
    plt.close()

    # Plot combined distributions
    plt.figure(figsize=(8, 6))
    plt.hist(class1, bins=bins, color="blue", alpha=0.5, label="Pos", edgecolor="black")
    plt.hist(
        class2, bins=bins, color="green", alpha=0.5, label="Neg", edgecolor="black"
    )
    plt.xlabel("Negative Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("Combined Distribution of NLL - Positive and Negative Edges")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{save_path}/{name}_all_nll_distribution.png")
    plt.close()


def empirical_coverage(
    true_values,
    pred_means,
    pred_vars,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    # Calculate empirical coverage
    confidence_levels = np.linspace(0.5, 0.99, 10)  # Coverage levels (50% to 99%)
    empirical_coverage = []

    for level in confidence_levels:
        z = norm.ppf((1 + level) / 2)  # Z-value for two-tailed interval
        lower_bound = pred_means - z * np.sqrt(pred_vars)
        upper_bound = pred_means + z * np.sqrt(pred_vars)

        # Check how many true values fall within the interval
        within_interval = np.sum(
            (true_values >= lower_bound) & (true_values <= upper_bound)
        )
        empirical_coverage.append(within_interval / len(true_values))

    # Plot empirical coverage
    plt.figure(figsize=(8, 6))
    plt.plot(
        confidence_levels, empirical_coverage, label="Empirical Coverage", marker="o"
    )
    plt.plot(
        confidence_levels,
        confidence_levels,
        label="Ideal Calibration",
        linestyle="--",
        color="red",
    )
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Plot (Empirical Coverage)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def QQplot(
    true_values,
    pred_means,
    pred_vars,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    # Standardized residuals
    z_scores = (true_values - pred_means) / np.sqrt(pred_vars)

    # Generate QQ plot
    plt.figure(figsize=(8, 6))
    stats.probplot(z_scores, dist="norm", plot=plt)
    plt.title("QQ Plot of Standardized Residuals")
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def densityMap(
    y_true,
    y_pred_mean,
    y_pred_var,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        y_true,
        y_pred_mean,
        c=y_pred_var,
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolor="k",
    )
    plt.colorbar(sc, label="Predicted Variance")
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle="--",
        color="red",
        label="y = x (Perfect Prediction)",
    )

    plt.xlabel("True Values (y_true)")
    plt.ylabel("Predicted Mean (y_pred_mean)")
    plt.title("Density Map: y_true vs y_pred_mean (Color = Uncertainty)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def Hist2DMap(
    y_true,
    y_pred_mean,
    y_pred_var,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    plt.figure(figsize=(8, 6))
    hist = plt.hist2d(y_true, y_pred_mean, bins=30, cmap="viridis", cmin=1)
    plt.colorbar(hist[3], label="Count")
    plt.plot([0, 10], [0, 10], "r--", label="y = x (Perfect Prediction)")
    plt.xlabel("y_true")
    plt.ylabel("y_pred_mean")
    plt.title("2D Histogram: y_true vs y_pred_mean")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def HexbinMap(
    y_true,
    y_pred_mean,
    y_pred_var,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(
        y_true,
        y_pred_mean,
        C=y_pred_var,
        gridsize=30,
        cmap="viridis",
        reduce_C_function=np.mean,
    )
    plt.colorbar(label="Mean Predicted Variance")
    plt.plot([0, 10], [0, 10], "r--", label="y = x (Perfect Prediction)")
    plt.xlabel("y_true")
    plt.ylabel("y_pred_mean")
    plt.title("Hexbin Plot: y_true vs y_pred_mean (Color = Variance)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def ContourMap(
    y_true,
    y_pred_mean,
    y_pred_var,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=y_true, y=y_pred_mean, cmap="Blues", fill=True)
    plt.plot([0, 10], [0, 10], "r--", label="y = x (Perfect Prediction)")
    plt.xlabel("y_true")
    plt.ylabel("y_pred_mean")
    plt.title("Contour Plot: y_true vs y_pred_mean")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def CalibrationCurve(
    y_true, probs, fig_name="cell_express_gene_empirical_coverage", save_path="./"
):
    # Convert logits to probabilities
    # probs = 1 / (1 + np.exp(-logits))

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, probs, n_bins=10
    )

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted_value, fraction_of_positives, "o-", label="Model Calibration"
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def ROCCurve(
    y_true, probs, fig_name="cell_express_gene_empirical_coverage", save_path="./"
):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Chromatin accessability prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def PRCCurve(
    y_true, probs, fig_name="cell_express_gene_empirical_coverage", save_path="./"
):
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, probs)
    average_precision = average_precision_score(y_true, probs)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="green",
        lw=2,
        label=f"PR curve (AP = {average_precision:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def HistProbs(
    y_true, probs, fig_name="cell_express_gene_empirical_coverage", save_path="./"
):
    plt.figure(figsize=(8, 6))
    plt.hist(probs[y_true == 0], bins=20, alpha=0.6, label="y_true = 0", color="blue")
    plt.hist(probs[y_true == 1], bins=20, alpha=0.6, label="y_true = 1", color="orange")
    plt.xlabel("Probs")
    plt.ylabel("Frequency")
    plt.title("Histogram of Probs for Each Class")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def BoxPlotProbs(
    y_true, probs, fig_name="cell_express_gene_empirical_coverage", save_path="./"
):
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [probs[y_true == 0], probs[y_true == 1]], labels=["y_true = 0", "y_true = 1"]
    )
    plt.ylabel("probs")
    plt.title("Boxplot of probs for Each Class")
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def BetaDensityPlot(
    y_true,
    alpha_pred,
    beta_pred,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    # Plot the empirical density of y_true
    plt.figure(figsize=(8, 6))
    plt.hist(
        y_true,
        bins=30,
        density=True,
        alpha=0.6,
        color="gray",
        label="Empirical Density (y_true)",
    )

    # Plot the predicted Beta PDF for a few random samples
    x = np.linspace(0, 1, 100)
    for i in np.random.choice(len(alpha_pred), size=5, replace=False):
        pdf = beta.pdf(x, alpha_pred[i], beta_pred[i])
        plt.plot(
            x, pdf, label=f"Beta PDF (α={alpha_pred[i]:.2f}, β={beta_pred[i]:.2f})"
        )

    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title("Predicted Beta PDF vs. Empirical Density of y_true")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def BetaCalibrationPlot(
    y_true,
    alpha_pred,
    beta_pred,
    fig_name="cell_express_gene_empirical_coverage",
    save_path="./",
):
    # Compute predicted mean from alpha and beta
    predicted_mean = alpha_pred / (alpha_pred + beta_pred)

    # Bin the predicted means
    bins = np.linspace(0, 1, 10)
    digitized = np.digitize(predicted_mean, bins)

    # Compute empirical mean of y_true for each bin
    empirical_means = [y_true[digitized == i].mean() for i in range(1, len(bins))]
    predicted_means = [
        predicted_mean[digitized == i].mean() for i in range(1, len(bins))
    ]

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        predicted_means, empirical_means, "o-", label="Empirical vs. Predicted Mean"
    )
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    plt.xlabel("Predicted Mean")
    plt.ylabel("Empirical Mean")
    plt.title("Calibration Plot: Binned Mean vs. Predicted Mean")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=300)
    plt.close()


def compute_reconstruction_gene_metrics(
    gene_expression,
    recon_gene_expression,
    recon_gene_std,
    plot=False,
    name="cell_exp_gene",
):
    """
    Computes reconstruction error, Pearson correlation, and Spearman correlation
    between original and reconstructed gene expression values.

    Parameters:
    - gene_expression: Tensor of original gene expression values (shape: [n_samples, n_genes]).
    - recon_gene_expression: Tensor of reconstructed gene expression values (same shape as gene_expression).

    Returns:
    - metrics: A dictionary containing the reconstruction error, Pearson correlation, and Spearman correlation.
    # Example usage:
      # gene_expression = torch.rand(100, 500)  # Example original expression values
      # recon_gene_expression = torch.rand(100, 500)  # Example reconstructed expression values
      # metrics = compute_reconstruction_gene_metrics(gene_expression, recon_gene_expression)
      # print(metrics)
    """
    # import pdb; pdb.set_trace()
    # Ensure tensors are in the correct shape
    if gene_expression.shape != recon_gene_expression.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Convert to numpy for easier computation of correlation metrics
    gene_expression_np = gene_expression.detach().cpu().numpy()
    recon_gene_expression_np = recon_gene_expression.detach().cpu().numpy()
    pred_vars = recon_gene_std.detach().cpu().numpy()

    # 1. Mean Squared Error for Reconstruction Error
    reconstruction_error = np.mean(
        (gene_expression_np - recon_gene_expression_np) ** 2
    ).item()

    # 2. Pearson and Spearman Correlation Coefficients
    pearson_corr = []
    spearman_corr = []

    # Calculate correlation for each gene across samples
    # we directly compute it per batch
    pearson_corr = pearsonr(gene_expression_np, recon_gene_expression_np)[0]
    spearman_corr = spearmanr(gene_expression_np, recon_gene_expression_np)[0]

    if plot:
        scatter_color_var(
            gene_expression_np,
            recon_gene_expression_np,
            pred_vars,
            fig_name=f"{name}_scatter_var",
            save_path="./",
        )
        # empirical_coverage(true_values=gene_expression_np, pred_means=recon_gene_expression_np, pred_vars=pred_vars, fig_name=f"{name}_empirical_coverage", save_path="./")
        # QQplot(true_values=gene_expression_np, pred_means=recon_gene_expression_np, pred_vars=pred_vars, fig_name=f"{name}_QQplot", save_path="./")
        # densityMap(y_true=gene_expression_np, y_pred_mean=recon_gene_expression_np, y_pred_var=pred_vars, fig_name=f"{name}_densityMap", save_path="./")
        # Hist2DMap(y_true=gene_expression_np, y_pred_mean=recon_gene_expression_np, y_pred_var=pred_vars,fig_name=f"{name}_Hist2D", save_path="./")
        # HexbinMap(y_true=gene_expression_np, y_pred_mean=recon_gene_expression_np, y_pred_var=pred_vars, fig_name=f"{name}_HexbinMap", save_path="./")
        # ContourMap(y_true=gene_expression_np, y_pred_mean=recon_gene_expression_np, y_pred_var=pred_vars, fig_name=f"{name}_contourmap", save_path="./")

    # Store metrics in a dictionary
    metrics = {
        "reconstruction_error": reconstruction_error,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }

    return metrics


def compute_classification_metrics(
    targets, predictions, threshold=0.5, plot=False, name="cell_contains_peak"
):
    """
    Computes binary classification metrics including cross-entropy loss, precision, recall,
    accuracy, and F1 score.

    Parameters:
    - targets: Tensor of true binary labels (shape: [n_samples]), with values 0 or 1.
    - predictions: Tensor of predicted probabilities (shape: [n_samples]), with values between 0 and 1.
    - threshold: Threshold to convert predicted probabilities into binary predictions.

    Returns:
    - metrics: A dictionary containing cross-entropy loss, precision, recall, accuracy, and F1 score.

    # Example usage:
      # targets = torch.tensor([0, 1, 1, 0, 1])  # Example ground-truth binary labels
      # predictions = torch.tensor([0.2, 0.8, 0.6, 0.4, 0.9])  # Example predicted probabilities
      # metrics = compute_classification_metrics(targets, predictions, threshold=0.5)
      # print(metrics)
    """

    # 1. Cross Entropy Loss
    cross_entropy_loss = F.binary_cross_entropy(predictions, targets.float()).item()

    # 2. Convert probabilities to binary predictions based on threshold
    binary_predictions = (predictions >= threshold).int()

    # Convert tensors to numpy arrays for sklearn metrics
    targets_np = targets.cpu().numpy()
    binary_predictions_np = binary_predictions.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    if plot:
        CalibrationCurve(
            y_true=targets_np,
            probs=predictions_np,
            fig_name=f"{name}_calibration",
            save_path="./",
        )
        ROCCurve(
            y_true=targets_np,
            probs=predictions_np,
            fig_name=f"{name}_ROC",
            save_path="./",
        )
        PRCCurve(
            y_true=targets_np,
            probs=predictions_np,
            fig_name=f"{name}_PRC",
            save_path="./",
        )
        HistProbs(
            y_true=targets_np,
            probs=predictions_np,
            fig_name=f"{name}_hist_prob",
            save_path="./",
        )
        BoxPlotProbs(
            y_true=targets_np,
            probs=predictions_np,
            fig_name=f"{name}_boxplot_prob",
            save_path="./",
        )

    # 3. Precision
    precision = precision_score(targets_np, binary_predictions_np, zero_division=0)

    # 4. Recall
    recall = recall_score(targets_np, binary_predictions_np, zero_division=0)

    # 5. Accuracy
    accuracy = accuracy_score(targets_np, binary_predictions_np)

    # 6. F1 Score
    f1 = f1_score(targets_np, binary_predictions_np, zero_division=0)

    # 7. AUROC
    auroc = roc_auc_score(targets_np, predictions)

    # Store all metrics in a dictionary
    metrics = {
        "cross_entropy_loss": cross_entropy_loss,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "auroc": auroc,
    }

    return metrics


def compute_asw_ari(embeddings, labels):
    """
    Computes the Average Silhouette Width (ASW) and Adjusted Rand Index (ARI)
    for the given embeddings and cell type labels.

    Parameters:
    - embeddings: Array-like or tensor of shape (n_samples, n_features), representing the model's cell embeddings.
    - labels: Array-like or tensor of shape (n_samples,), representing the true cell type labels.

    Returns:
    - metrics: A dictionary containing ASW and ARI scores.

    # Example usage:
      # embeddings = model_output  # Your cell embeddings (shape: [n_samples, n_features])
      # labels = cell_type_labels  # Your true cell type labels (shape: [n_samples])
      # metrics = compute_asw_ari(embeddings, labels)
      # print(metrics)
    """

    # Convert tensors to numpy arrays if necessary
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # 1. Compute Average Silhouette Width (ASW)
    asw_score = silhouette_score(embeddings, labels, metric="euclidean")

    # 2. Compute Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(labels, labels)

    # Store metrics in a dictionary
    metrics = {"ASW": asw_score, "ARI": ari_score}

    return metrics


def plot_sw_distribution(models_embeddings, labels, model_names):
    """
    Computes and plots the Silhouette Width (SW) score distributions for multiple models
    using a violin plot.

    Parameters:
    - models_embeddings: List of arrays or tensors, where each element represents the embeddings
                         from a model (shape for each: [n_samples, n_features]).
    - labels: Array-like or tensor of shape (n_samples,), representing the true cell type labels.
    - model_names: List of names (strings) corresponding to each model for labeling in the plot.
    # Example usage:
      # embeddings_model1 = model1_output  # Embeddings from model 1
      # embeddings_model2 = model2_output  # Embeddings from model 2
      # embeddings_model3 = model3_output  # Embeddings from model 3
      # models_embeddings = [embeddings_model1, embeddings_model2, embeddings_model3]
      # labels = cell_type_labels  # True cell type labels
      # model_names = ["Model 1", "Model 2", "Model 3"]
      # plot_sw_distribution(models_embeddings, labels, model_names)

    """

    # Ensure models_embeddings and model_names have the same length
    if len(models_embeddings) != len(model_names):
        raise ValueError("The number of embeddings and model names must be the same.")

    # Convert labels to numpy if it's a tensor
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # List to store SW scores and corresponding model labels for plotting
    all_sw_scores = []
    all_model_labels = []

    # Calculate silhouette scores for each model
    for i, embeddings in enumerate(models_embeddings):
        # Convert embeddings to numpy if they're tensors
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()

        # Compute SW scores per sample
        sw_scores = silhouette_samples(embeddings, labels)

        # Append scores and model labels to lists
        all_sw_scores.extend(sw_scores)
        all_model_labels.extend([model_names[i]] * len(sw_scores))

    # Create a DataFrame for easy plotting with seaborn
    sw_df = pd.DataFrame(
        {"Silhouette Width Score": all_sw_scores, "Model": all_model_labels}
    )

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=sw_df, x="Model", y="Silhouette Width Score", inner="quartile")
    plt.title("Distribution of Silhouette Width Scores per Sample across Models")
    plt.xlabel("Model")
    plt.ylabel("Silhouette Width Score")
    plt.savefig("violin_ASW_comparsion_among_models.png")


def compute_beta_metrics(
    y_true, concentration1, concentration0, plot=False, name="gene_close_to_peak"
):
    """
    Compute evaluation metrics for a Beta distribution.

    Parameters:
    - y_true: Ground truth values (torch.Tensor, shape [N]).
    - concentration1: Predicted concentration1 (alpha) values (torch.Tensor, shape [N]).
    - concentration0: Predicted concentration0 (beta) values (torch.Tensor, shape [N]).

    Returns:
    - metrics: A dictionary containing MSE, Pearson r, Spearman r, and log-likelihood.
    """
    # Predicted mean of the Beta distribution
    alpha = concentration1
    beta = concentration0
    mu_pred = alpha / (alpha + beta)

    # Predicted variance (optional, useful for uncertainty evaluation)
    variance_pred = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    # Mean Squared Error (MSE)
    mse = torch.mean((y_true - mu_pred) ** 2).item()

    # Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(y_true.numpy(), mu_pred.numpy())

    # Spearman Correlation Coefficient
    spearman_corr, _ = spearmanr(y_true.numpy(), mu_pred.numpy())

    # Log-Likelihood
    # beta_dist = Beta(alpha, beta)
    # log_likelihood = beta_dist.log_prob(y_true).mean().item()
    if plot:
        BetaDensityPlot(
            y_true=y_true,
            alpha_pred=alpha.cpu().numpy(),
            beta_pred=beta.cpu().numpy(),
            fig_name=f"{name}_beta_density",
            save_path="./",
        )
        BetaCalibrationPlot(
            y_true=y_true,
            alpha_pred=alpha.cpu().numpy(),
            beta_pred=beta.cpu().numpy(),
            fig_name=f"{name}_beta_calibration",
            save_path="./",
        )

    return {
        "MSE": mse,
        "Pearson r": pearson_corr,
        "Spearman r": spearman_corr,
        # "Log-Likelihood": log_likelihood,
        "Variance": variance_pred.mean().item(),  # Optional for uncertainty analysis
    }
