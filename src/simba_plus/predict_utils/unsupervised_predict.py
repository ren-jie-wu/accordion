import os
import pandas as pd
from simba_plus.discovery import add_features as score


def predict_unsupervised(
    links: pd.DataFrame,
    model_dir: str,
    celltype_specific: bool,
    skip_uncertain: bool,
    use_distance: bool,
) -> pd.DataFrame:

    adata_C_path = os.path.join(model_dir, "adata_C.h5ad")
    adata_G_path = os.path.join(model_dir, "adata_G.h5ad")
    adata_P_path = os.path.join(model_dir, "adata_P.h5ad")

    links = score.add_simba_plus_features(
        eval_df=links,
        adata_C_path=adata_C_path,
        adata_G_path=adata_G_path,
        adata_P_path=adata_P_path,
        gene_col="Gene_name",
        peak_col="Peak",
        celltype_specific=celltype_specific,
        skip_uncertain=skip_uncertain,
    )

    # apply distance weighting if requested
    if use_distance and "Distance_to_TSS" in links:
        dist = links["Distance_to_TSS"].values
        w = 1.0 / (dist + 1e-6)
        for col in links.columns:
            if col.startswith("SIMBA+_path_score"):
                links[col] = links[col] * w

    return links
