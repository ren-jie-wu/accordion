from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from scvi.distributions import NegativeBinomial
from simba_plus.constants import MIN_LOGSTD, MAX_LOGSTD, EPS


class ProximityDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Args
        u: Input source tensor of shape (n_edges, n_latent_dimension)
        v: Input destination tensor of shape (n_edges, n_latent_dimension)
        library_size: Library size of source node of shape (n_edges,)
        src_cont_covs: Continuous covariates of source node of shape (n_edges, n_cont_covariates)
        dst_cont_covs: Continuous covariates of source node of shape (n_edges, n_cont_covariates)
        cat_covs: Categorical covariates of source node of shape (n_edges, n_cat_covariates)
        """
        z = torch.nn.functional.cosine_similarity(u, v)
        print(f"z:{z}")
        return z


class NormalDataDecoder(ProximityDecoder):
    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(
        self,
        positive_scale=False,
        scale_src=True,
    ) -> None:
        super().__init__()
        self.positive_scale = positive_scale
        self.scale_src = scale_src

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_scale,
        src_bias,
        src_std,
        dst_scale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        if self.positive_scale:
            if self.scale_src:
                scale = F.softplus(src_scale) * F.softplus(dst_scale)
            else:
                scale = F.softplus(dst_scale)
        else:
            scale = src_scale * dst_scale
        cos = torch.nn.CosineSimilarity()
        loc = scale * cos(u, v) + src_bias + dst_bias
        # std = F.softplus(src_std + dst_std) + EPS
        std = torch.exp(dst_std)
        return D.Normal(loc, std)  # std, validate_args=True)


class GammaDataDecoder(ProximityDecoder):
    r"""
    Gamma data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(
        self,
        positive_scale=False,
        scale_src=True,
    ) -> None:
        super().__init__()
        self.positive_scale = positive_scale
        self.scale_src = scale_src

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_scale,
        src_bias,
        src_std,
        dst_scale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Gamma:
        if self.positive_scale:
            if self.scale_src:
                scale = F.softplus(src_scale) * F.softplus(dst_scale)
            else:
                scale = F.softplus(dst_scale)
        else:
            scale = src_scale * dst_scale
        cos = torch.nn.CosineSimilarity()
        loc = torch.exp(scale * cos(u, v) + src_bias + dst_bias)  # a/b
        std = torch.exp(src_std + dst_std)  # a/(b^2)
        b = loc / (std + EPS)  # rate
        a = loc * b  # concentration
        # std = torch.exp(src_std + dst_std)
        return D.Gamma(a, b, validate_args=True)


class PoissonDataDecoder(ProximityDecoder):
    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_scale,
        src_bias,
        src_std,
        dst_scale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(src_scale) * F.softplus(dst_scale)
        cos = torch.nn.CosineSimilarity()
        loc = scale * torch.exp(cos(u, v) + src_bias + dst_bias)
        return D.Poisson(loc)


class BernoulliDataDecoder(ProximityDecoder):
    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(
        self,
        positive_scale=True,
        scale_src=True,
    ) -> None:
        super().__init__()
        self.positive_scale = positive_scale
        self.scale_src = scale_src

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_scale,
        src_bias,
        src_std,
        dst_scale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        cos = torch.nn.CosineSimilarity()
        if self.scale_src:
            if self.positive_scale:
                scale = F.softplus(src_scale) * F.softplus(dst_scale)
            else:
                scale = src_scale * dst_scale
        else:
            if self.positive_scale:
                scale = F.softplus(dst_scale)
            else:
                scale = dst_scale
        logit = scale * cos(u, v) + src_bias + dst_bias
        if torch.any(torch.isnan(logit)):
            nan_idx = torch.isnan(logit).nonzero(as_tuple=True)[0]
            print(
                f"NaN in logit: u={u[nan_idx]}, v={v[nan_idx]}, src_scale={src_scale[nan_idx]}, dst_scale={dst_scale[nan_idx]}, src_bias={src_bias[nan_idx]}, dst_bias={dst_bias[nan_idx]}, src_std={src_std[nan_idx]}, dst_std={dst_std[nan_idx]}"
            )
        return D.Bernoulli(logits=logit)


class NegativeBinomialDataDecoder(ProximityDecoder):
    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(
        self,
        positive_scale=False,
        scale_src=True,
    ) -> None:
        super().__init__()
        self.positive_scale = positive_scale
        self.scale_src = scale_src

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_scale,
        src_bias,
        src_std,
        dst_scale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        if self.positive_scale:
            scale = F.softplus(src_scale) * F.softplus(dst_scale)
        else:
            scale = src_scale * dst_scale
        cos = torch.nn.CosineSimilarity()
        loc = torch.exp(scale * cos(u, v) + src_bias + dst_bias)
        # std = torch.exp(dst_std)
        std = torch.exp((src_std + dst_std).clamp(MIN_LOGSTD, MAX_LOGSTD))
        return NegativeBinomial(mu=loc, theta=std)  # std, validate_args=True)
