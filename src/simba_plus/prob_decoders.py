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
    ) -> None:
        super().__init__()

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_logscale,
        src_bias,
        src_std,
        dst_logscale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = torch.exp(src_logscale + dst_logscale)

        cos = torch.nn.CosineSimilarity()
        loc = scale * cos(u, v) + src_bias + dst_bias
        # std = F.softplus(src_std + dst_std) + EPS
        std = torch.exp(src_std + dst_std)
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

        scale = F.softplus(src_scale) * F.softplus(dst_scale)

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

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_logscale,
        src_bias,
        src_std,
        dst_logscale,
        dst_bias,
        dst_std,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        cos = torch.nn.CosineSimilarity()
        scale = torch.exp(src_logscale) * torch.exp(dst_logscale)
        logit = scale * cos(u, v) + src_bias + dst_bias
        try:
            d = D.Bernoulli(logits=logit)
        except Exception as e:
            print(u[torch.isnan(logit)])
            print(v[torch.isnan(logit)])
            print(logit[torch.isnan(logit)])
            print(scale[torch.isnan(logit)])
            print(src_logscale[torch.isnan(logit)])
            print(dst_logscale[torch.isnan(logit)])
            print(src_bias[torch.isnan(logit)])
            print(dst_bias[torch.isnan(logit)])
            raise e
        return d


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

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        src_logscale=None,
        src_bias=None,
        src_std=None,
        dst_logscale=None,
        dst_bias=None,
        dst_std=None,
        # b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = torch.exp(src_logscale + dst_logscale)
        cos = torch.nn.CosineSimilarity()
        mean_logit = torch.clamp(
            scale * cos(u, v) + src_bias + dst_bias, min=-float("inf"), max=88
        )
        loc = torch.exp(mean_logit)
        # std = torch.exp((src_std + dst_std).clamp(MIN_LOGSTD, MAX_LOGSTD))
        std = torch.exp(dst_std + src_std)
        # return D.Normal(scale * (u * v).sum(axis=1) + src_bias + dst_bias, 0.1)
        return NegativeBinomial(mu=loc, theta=std)  # std, validate_args=True)
