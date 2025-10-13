import sys
import simba_plus
import argparse
import numpy as np
import anndata as ad
import scanpy as sc
from simba_plus.model_prox import LightningProxModel

sys.modules["coral"] = simba_plus


def save_files(prefix, adata_CG_path, adata_CP_path, checkpoint_version_suffix=""):
    adata_CG = ad.read_h5ad(adata_CG_path)
    adata_CP = ad.read_h5ad(adata_CP_path)
    model = LightningProxModel.load_from_checkpoint(
        f"{prefix}.checkpoints/last{checkpoint_version_suffix}.ckpt",
        weights_only=True,
        map_location="cpu",
    )
    np.save(
        f"{prefix}.checkpoints/peak_scale.npy",
        model.scale_dict["peak__cell_has_accessible_peak"].detach().numpy(),
    )
    np.save(
        f"{prefix}.checkpoints/peak_bias.npy",
        model.bias_dict["peak__cell_has_accessible_peak"].detach().numpy(),
    )
    cell_x = model.encoder.__mu_dict__["cell"].detach().cpu()
    adata_C = ad.AnnData(
        X=cell_x.numpy(),
        layers={
            "X_logstd": model.encoder.__logstd_dict__["cell"].detach().cpu().numpy()
        },
        obs=adata_CG.obs,
        obsm={"X_pca": cell_x.numpy()},
    )
    sc.pp.neighbors(adata_C)
    sc.tl.umap(adata_C)
    sc.pl.umap(adata_C, color=["cell_type"])
    adata_C.obs["cell_type"] = adata_CG.obs["cell_type"]
    adata_P = ad.AnnData(
        X=model.encoder.__mu_dict__["peak"].detach().cpu().numpy(),
        layers={
            "X_logstd": model.encoder.__logstd_dict__["peak"].detach().cpu().numpy()
        },
        obs=adata_CP.var,
    )
    adata_G = ad.AnnData(
        X=model.encoder.__mu_dict__["gene"].detach().cpu().numpy(),
        layers={
            "X_logstd": model.encoder.__logstd_dict__["gene"].detach().cpu().numpy()
        },
        obs=adata_CG.var,
    )
    adata_C.write(f"{prefix}.checkpoints/adata_C.h5ad")
    adata_P.write(f"{prefix}.checkpoints/adata_P.h5ad")
    adata_G.write(f"{prefix}.checkpoints/adata_G.h5ad")


if __name__ == "__main__":
    print("run 0")
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str)
    parser.add_argument("adata_CG_path", type=str)
    parser.add_argument("adata_CP_path", type=str)
    parser.add_argument(
        "--checkpoint_version_suffix",
        type=str,
        default="",
        help="Suffix for checkpoint version",
    )
    args = parser.parse_args()
    save_files(**vars(args))
