from __future__ import annotations

import pandas as pd
import numpy as np
import anndata as ad
from typing import Optional, Sequence
from scipy.special import softmax
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def compute_path_scores(
    eval_df: pd.DataFrame,
    df_softmax_CG: pd.DataFrame,
    df_softmax_CP: pd.DataFrame,
    gene_col: str,
    peak_col: str,
    output_prefix: str = "SIMBA+_path_score",
    celltype_to_cells: Optional[dict[str, Sequence[str]]] = None,
    skip_uncertain: bool = True,
) -> pd.DataFrame:
    """Compute SIMBA path scores (global or cell type–specific) and append to ``eval_df``.

    Args:
        eval_df (pd.DataFrame):
            DataFrame containing gene and peak identifiers.
        df_softmax_CG (pd.DataFrame):
            Softmax scores for genes (genes × cells).
        df_softmax_CP (pd.DataFrame):
            Softmax scores for peaks (peaks × cells).
        gene_col (str):
            Column name for gene identifiers in ``eval_df``.
        peak_col (str):
            Column name for peak identifiers in ``eval_df``.
        output_prefix (str, optional):
            Prefix for output columns. Defaults to ``"SIMBA+_path_score"``.
        celltype_to_cells (Optional[dict[str, Sequence[str]]], optional):
            Mapping of cell type → list of cell IDs.
            If provided, computes one column per cell type; otherwise computes a single
            global score across all cells.
        skip_uncertain (bool, optional):
            Whether to skip cell types named ``"Uncertain"`` or ``"Unknown"``.
            Defaults to ``True``.

    Returns:
        pd.DataFrame: The input ``eval_df`` with additional path-score columns.
    """

    def _compute_for_cells(sub_df: pd.DataFrame, cells: Sequence[str], suffix: str = "") -> None:
        """Helper to compute scores for a given set of cells."""
        scale = len(cells)
        CG_sub = df_softmax_CG[cells]
        CP_sub = df_softmax_CP[cells]

        colname = f"{output_prefix}{suffix}"
        eval_df[colname] = eval_df.apply(
            lambda row: np.sum(
                CG_sub.loc[row[gene_col]].values * CP_sub.loc[row[peak_col]].values
            ) * scale
            if row[gene_col] in CG_sub.index and row[peak_col] in CP_sub.index
            else np.nan,
            axis=1,
        )

    # === Global score (no cell-type stratification) ===
    if celltype_to_cells is None:
        print("Computing global SIMBA+ path scores...")
        all_cells = df_softmax_CG.columns.intersection(df_softmax_CP.columns)
        _compute_for_cells(eval_df, all_cells, suffix="")
        return eval_df

    # === Cell-type–specific scores ===
    print("Computing cell type–specific SIMBA+ path scores...")
    for ct, cells in tqdm(celltype_to_cells.items(), desc="Cell types"):
        if skip_uncertain and ct.lower() in {"uncertain", "unknown"}:
            continue
        valid_cells = [c for c in cells if c in df_softmax_CG.columns]
        if not valid_cells:
            continue
        _compute_for_cells(eval_df, valid_cells, suffix=f"_{ct}")

    return eval_df


def add_simba_plus_features(
    eval_df: pd.DataFrame,
    adata_C_path: str,
    adata_G_path: str,
    adata_P_path: str,
    gene_col: str,
    peak_col: str,
    celltype_specific: bool = False,
    skip_uncertain: bool = True,
    use_distance_weight: bool = False,
) -> pd.DataFrame:
    """
    Add SIMBA+ global and cell-type–specific features using precomputed embeddings.
    """

    # ---- Load embeddings ----
    adata_C = ad.read_h5ad(adata_C_path)
    adata_G = ad.read_h5ad(adata_G_path)
    adata_P = ad.read_h5ad(adata_P_path)

    # ---- Normalize & mean center ----
    adata_C.layers["X_normed"] = adata_C.X / np.linalg.norm(adata_C.X, axis=1)[:, None]
    G_norm = adata_G.X / np.linalg.norm(adata_G.X, axis=1)[:, None]
    P_norm = adata_P.X / np.linalg.norm(adata_P.X, axis=1)[:, None]

    # ---- Map gene/peak identifiers to embedding indices ----
    split_df = eval_df[[gene_col, peak_col]].copy()
    split_df["gidx"] = split_df[gene_col].map(lambda g: adata_G.obs.index.get_loc(g))
    split_df["pidx"] = split_df[peak_col].map(lambda p: adata_P.obs.index.get_loc(p))

    gidx = split_df["gidx"].values
    pidx = split_df["pidx"].values

    # ---- Compute softmax similarity matrices ----
    G_cand = G_norm[gidx, :]
    P_cand = P_norm[pidx, :]

    G_cand_softmax = softmax(G_cand @ adata_C.layers["X_normed"].T, axis=1)
    P_cand_softmax = softmax(P_cand @ adata_C.layers["X_normed"].T, axis=1)

    G_df = pd.DataFrame(G_cand_softmax, index=split_df[gene_col], columns=adata_C.obs.index)
    P_df = pd.DataFrame(P_cand_softmax, index=split_df[peak_col], columns=adata_C.obs.index)

    # ---- Remove duplicate gene or peak rows ----
    G_df = G_df[~G_df.index.duplicated(keep="first")]
    P_df = P_df[~P_df.index.duplicated(keep="first")]

    # ---- Determine cell-type column if requested ----
    if celltype_specific:
        possible_keys = ["cell_type", "celltype", "celltype_mapped", "leiden"]
        celltype_col = None
        for key in possible_keys:
            if key in adata_C.obs.columns:
                celltype_col = key
                break

        if celltype_col is None:
            raise ValueError(
                "celltype_specific=True, but no cell-type column found in adata_C.obs. "
                "Expected one of: 'cell_type', 'celltype', 'celltype_mapped', 'leiden'."
            )

        # Build mapping: cell type → list of cell IDs
        celltype_to_cells = (
            adata_C.obs.groupby(celltype_col)
            .apply(lambda x: x.index.tolist())
            .to_dict()
        )

        # ---- Apply minimum-size filter (skip small groups) ----
        filtered = {}
        for ct, cells in celltype_to_cells.items():
            if len(cells) < 10:
                print(f"Skipping cell type '{ct}' (n={len(cells)} < 10).")
                continue
            filtered[ct] = cells

        if len(filtered) == 0:
            print("No cell types have ≥10 cells. Falling back to global score only.")
            celltype_to_cells = None
        else:
            celltype_to_cells = filtered

    else:
        celltype_to_cells = None


    # ---- Compute path scores ----
    eval_df = compute_path_scores(
        eval_df=eval_df,
        gene_col=gene_col,
        peak_col=peak_col,
        output_prefix="SIMBA+_path_score",
        df_softmax_CG=G_df,
        df_softmax_CP=P_df,
        celltype_to_cells=celltype_to_cells,
        skip_uncertain=skip_uncertain,
    )

    if use_distance_weight:
        for col in eval_df.columns:
            if col.startswith("SIMBA+_path_score"):
                eval_df['1/Distance'] = 1 / (eval_df["Distance_to_TSS"] + 1)
                eval_df[col] = eval_df[col] * (1 / (eval_df["Distance_to_TSS"] + 1))

    if 'peak_gene_pair' not in eval_df.columns:
        eval_df['peak_gene_pair'] = eval_df[peak_col] + '_' + eval_df[gene_col]
        
    eval_df.drop_duplicates("peak_gene_pair", inplace=True)
    return eval_df

