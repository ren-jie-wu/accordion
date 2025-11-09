import os
import numpy as np
import pandas as pd


def get_tau_z_cts(
    result_path: str,
    mat: np.ndarray,
    n_annot: int,
    annot_suffix: str = "L2",
    std_mat: np.ndarray = None,
    n_mat_samples: int = 100,
    return_raw: bool = False,
):
    result = pd.read_csv(result_path, sep="\t", index_col=0).loc[
        [f"{i}{annot_suffix}" for i in range(n_annot)], :
    ]
    tau_d = result["Coefficient"].iloc[-(n_annot):]
    tau_std = result["Coefficient_std_error"].iloc[-(n_annot):]
    if std_mat is not None:
        tau_zs = []
        tau_stds = []
        for i in range(n_mat_samples):
            _mat = mat + np.random.randn(*std_mat.shape) * std_mat
            _tau_mean_cell = _mat @ tau_d
            _tau_std_cell = np.sqrt(_mat**2 @ (tau_std) ** 2)
            tau_zs.append(_tau_mean_cell / _tau_std_cell)
        tau_z = np.stack(tau_zs, axis=-1)
        tau_z = tau_z.mean(axis=-1)
    else:
        tau_mean_cell = mat @ tau_d
        tau_std_cell = np.sqrt(mat**2 @ (tau_std) ** 2)
        tau_z = tau_mean_cell / tau_std_cell
    if return_raw:
        return tau_z, tau_d / tau_std
    return tau_z


def get_corr(cov_mat):
    std = np.sqrt(np.diag(cov_mat))
    return cov_mat / std[:, np.newaxis] / std[np.newaxis,]


def get_tau_z_dep(
    result_path,
    mat,
    n_annot,
    plot_cov=False,
    std_mat=None,
    n_mat_samples=100,
    include_ov=False,
    ov_cov=None,
    annot_suffix="L2_2",
):
    index = pd.read_csv(result_path, sep="\t", index_col=0).index
    if ov_cov is None:
        ov_cov = np.ones(
            mat.shape[0],
        )
    if include_ov:
        index_select = ["L2_1"] + [f"{i}{annot_suffix}" for i in range(n_annot)]
        mat_w_cov = np.concatenate([ov_cov[:, None], mat], axis=1)
    else:
        index_select = [f"{i}{annot_suffix}" for i in range(n_annot)]
        mat_w_cov = mat
    coef_cov = (
        pd.DataFrame(
            np.load(result_path.rsplit("results", 1)[0] + "coef_cov.npy"),
            index=index,
            columns=index,
        )
        .loc[index_select, :]
        .loc[:, index_select]
    )
    coef_corr = get_corr(coef_cov)
    result = pd.read_csv(result_path, sep="\t", index_col=0).loc[index_select, :]
    tau_d = result["Coefficient"]
    tau_mean = mat_w_cov @ tau_d
    tau_std = np.sqrt(
        np.einsum(
            "ci,cj,ij->c",
            mat_w_cov * result["Coefficient_std_error"].values,
            mat_w_cov * result["Coefficient_std_error"].values,
            coef_corr,
        )
    )
    tau_z = tau_mean / tau_std

    return tau_z, tau_mean
