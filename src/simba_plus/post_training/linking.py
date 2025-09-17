from typing import Optional, Sequence, Literal, Any, Tuple
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import ttest_ind, ttest_rel
import anndata as ad


def get_corr_sign(adata_G, adata_P, gidx, pidx):
    """Get correlation sign between gene and peak scores.
    Args:
                    adata_G: AnnData object containing gene data.
                    adata_P: AnnData object containing peak data.
                    gidx: Indices of genes to consider.
                    pidx: Indices of peaks to consider.
    Returns:
                    DataFrame containing correlation signs between genes and peaks."""
    for adata in [adata_G, adata_P]:
        if "X_normed" not in adata.layers:
            adata.layers["X_normed"] = (
                adata.X / np.linalg.norm(adata.X, axis=1)[:, None]
            )
    return (
        pd.Series(
            (
                adata_G.layers["X_normed"][gidx, :]
                * adata_P.layers["X_normed"][pidx, :]
            ).sum(axis=1)
            > 0
        )
        .map({True: "+", False: "-"})
        .values
    )


def ttest_group(
    val1: pd.Series,
    val2: pd.Series,
    group: pd.Series,
    n_random_perm: int = 1,
) -> Tuple[pd.Series, np.ndarray, dict, dict]:
    """Perform t-test for each group for val1 * val2 vs val1 * (permuted val2) and return p-values and log fold changes.
    Args:
            val1: First set of values.
            val2: Second set of values.
            group: Group labels for the values.
            n_random_perm: Number of random permutations for significance testing.
    Returns:
                    Positive correlation scores, random scores, p-values, and log fold changes.
    """
    p_dict = {}
    lfc_dict = {}
    pos_corr_score = val1 * val2
    random_scores = []
    for i in range(n_random_perm):
        random_scores.append(val1 * np.random.permutation(val2))
    random_score = np.vstack(random_scores).T  # n_cells x n_random_perm
    for g in group.unique():
        res = ttest_ind(
            pos_corr_score[group == g],
            random_score[group == g, :].flatten(),
            alternative="greater",
            equal_var=False,
        )
        # p_dict[g] = np.mean(pos_corr_score[group == g].mean() <= random_score[group == g, :].mean(axis=0))
        p_dict[g] = res.pvalue
        lfc_dict[g] = np.log(
            pos_corr_score[group == g].mean()
            / random_score[group == g, :].flatten().mean()
        )
    return pos_corr_score, random_score, p_dict, lfc_dict


def get_path_scores(
    adata_C: ad.AnnData,
    adata_G: ad.AnnData,
    adata_P: ad.AnnData,
    gene: str,
    peak: str,
    sign: Literal["-", "+"] = "+",
    cell_idx=None,
    return_per_cell_score: bool = False,
    return_gene_peak_score: bool = False,
):
    """Get link scores between genes and peaks based on cell data.
    Args:
                        adata_C: AnnData object containing cell data.
                        adata_G: AnnData object containing gene data.
                        adata_P: AnnData object containing peak data.
        Returns:
                        DataFrame containing link scores between genes and peaks."""
    gene_idx = np.where(adata_G.obs_names == gene)[0]
    if len(gene_idx) == 0:
        raise ValueError(f"{gene} not in adata_G.obs_names")
    peak_idx = np.where(adata_P.obs_names == peak)[0]
    if len(peak_idx) == 0:
        raise ValueError(f"{peak} not in adata_P.obs_names")
    for adata in [adata_C, adata_G, adata_P]:
        if "X_normed" not in adata.layers:
            adata.layers["X_normed"] = (
                adata.X / np.linalg.norm(adata.X, axis=1)[:, None]
            )
    gene_score = adata_G.layers["X_normed"] @ adata_C.layers["X_normed"].T
    peak_score = adata_P.layers["X_normed"] @ adata_C.layers["X_normed"].T
    n_cells = len(adata_C)
    softmax_peak_score = softmax(peak_score, axis=1) * n_cells
    softmax_gene_score = softmax(gene_score, axis=1) * n_cells
    softmax_neg_gene_score = softmax(-gene_score, axis=1) * n_cells
    if sign == "-":
        _softmax_gene_score = softmax_neg_gene_score
    else:
        _softmax_gene_score = softmax_gene_score
    scores = _softmax_gene_score[gene_idx, :] * softmax_peak_score[peak_idx, :]
    if cell_idx is not None:
        scores = scores[:, cell_idx]
    if return_gene_peak_score:
        return (
            scores.squeeze(),
            _softmax_gene_score[gene_idx, :].squeeze(),
            softmax_peak_score[peak_idx, :].squeeze(),
        )
    if return_per_cell_score:
        return scores.mean(axis=1), scores
    return scores.mean(axis=1)


def get_active_cell_state(
    adata_C: ad.AnnData,
    adata_G: ad.AnnData,
    adata_P: ad.AnnData,
    gene_idx,
    peak_idx,
    signs,
    celltype_annot: str = "azimuth_celltype",
    return_p: bool = False,
    n_random_perm: int = 10,
):
    """Get active cell state based on gene and peak scores.
    Args:
                    adata_C: AnnData object containing cell data.
                    adata_G: AnnData object containing gene data.
                    adata_P: AnnData object containing peak data.
                    gene_idx: Indices of genes to consider.
                    peak_idx: Indices of peaks to consider.
                    sign: Sign of the scores to consider, either "+" or "-".
                    celltype_annot: Annotation for cell types.
                    return_p: Whether to return p-values.
    Returns:
                    Active cell state scores, random scores, significant cell types, and log fold changes.
    """
    for adata in [adata_C, adata_G, adata_P]:
        if "X_normed" not in adata.layers:
            adata.layers["X_normed"] = (
                adata.X / np.linalg.norm(adata.X, axis=1)[:, None]
            )
    gene_score = adata_G.layers["X_normed"] @ adata_C.layers["X_normed"].T
    peak_score = adata_P.layers["X_normed"] @ adata_C.layers["X_normed"].T
    n_cells = len(adata_C)
    softmax_peak_score = softmax(peak_score, axis=1) * n_cells
    softmax_gene_score = softmax(gene_score, axis=1) * n_cells
    softmax_neg_gene_score = softmax(-gene_score, axis=1) * n_cells
    pos_corr_scores = []
    random_scores = []
    sigslist = []
    lfcslist = []
    ctss = []
    for pidx, gidx, sign in tqdm(zip(peak_idx, gene_idx, signs), total=len(peak_idx)):
        if sign == "-":
            _softmax_gene_score = softmax_neg_gene_score
        else:
            _softmax_gene_score = softmax_gene_score
        pos_corr_score, random_score, sigs, lfcs = ttest_group(
            _softmax_gene_score[gidx, :],
            softmax_peak_score[pidx, :],
            adata_C.obs[celltype_annot],
            n_random_perm=n_random_perm,
        )
        pos_corr_scores.append(pos_corr_score)
        random_scores.append(random_score)
        sigslist.append(sigs)
        lfcslist.append(lfcs)
        cts = [k for k, v in sigs.items() if v < 0.001]
        cts = np.array(cts)[np.array([-lfcs[k] for k in cts]).argsort()]
        ctss.append(cts)
    if return_p:
        return (
            np.hstack(pos_corr_scores),
            np.vstack(random_scores),
            sigslist,
            lfcslist,
            ctss,
        )

    return (
        np.hstack([p.mean() for p in pos_corr_scores]),
        np.vstack([r.mean(axis=0) for r in random_scores]),
        ctss,
    )
