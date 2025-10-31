from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from scvi.distributions import NegativeBinomial

# from dynot.probs import ZILN, ZIN, ZINB

MAX_LOGSTD = 10
EPS = 1e-6


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
        std = torch.exp(src_std + dst_std)
        return D.Normal(loc, 0.1)  # std, validate_args=True)


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
        return D.Bernoulli(logits=logit)


class BetaDataDecoder(ProximityDecoder):
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
        loc = torch.sigmoid(scale * cos(u, v) + src_bias + dst_bias)
        a0 = torch.exp(src_std + dst_std)
        # print(f"@LOC:{loc}, {a0} (u.shape={u.shape}, v.shape={v.shape})")
        # import pdb; pdb.set_trace()
        loc = torch.clamp(loc, min=1e-6, max=1 - 1e-6)

        return D.Beta(loc * a0, (1 - loc) * a0)


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
        std = torch.exp(dst_std)
        return NegativeBinomial(mu=loc, theta=0.1)  # std, validate_args=True)
