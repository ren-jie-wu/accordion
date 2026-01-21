from typing import Optional
import torch
import torch.nn as nn


class HSIC(nn.Module):
    """
    PyTorch implementation of Hilbert-Schmidt Independence Criterion (HSIC).

    HSIC measures the dependence between two random variables X and Y using kernel methods.
    This implementation supports both linear and RBF kernels.
    """

    def __init__(
        self,
        kernel_x: str = "rbf",
        kernel_y: str = "rbf",
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        use_cuda: bool = False,
        subset_samples: Optional[int] = None,
        lam=1e7,
    ):
        """
        Initialize HSIC with kernel choices and parameters.

        Args:
            kernel_x (str): Kernel type for X ('rbf' or 'linear')
            kernel_y (str): Kernel type for Y ('rbf' or 'linear')
            sigma_x (float, optional): Bandwidth for X's RBF kernel
            sigma_y (float, optional): Bandwidth for Y's RBF kernel
            use_cuda (bool): Whether to use GPU if available
        """
        super(HSIC, self).__init__()
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.subset_samples = subset_samples
        self.lam = lam

    def _kernel_function(
        self, X: torch.Tensor, kernel_type: str, sigma: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute kernel matrix for the input data.

        If linear, $K = X X^T$.
        
        If RBF, $K = \exp(-\frac{||X - X^T||^2}{2 \sigma^2})$.

        Args:
            X (torch.Tensor): Input data
            kernel_type (str): Type of kernel ('rbf' or 'linear')
            sigma (float, optional): Bandwidth parameter for RBF kernel

        Returns:
            torch.Tensor: Kernel matrix
        """
        if kernel_type == "linear":
            return torch.mm(X, X.t())

        elif kernel_type == "rbf":
            X_norm = torch.sum(X * X, dim=1, keepdim=True)
            X_t = X.t()
            dist_matrix = X_norm + X_norm.t() - 2.0 * torch.mm(X, X_t)

            # Use median heuristic if sigma is not provided
            if sigma is None:
                sigma = torch.sqrt(dist_matrix.mean())
            return torch.exp(-dist_matrix / (2.0 * sigma**2))

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def _centering_matrix(self, n: int) -> torch.Tensor:
        """
        Create centering matrix (H = I - 1/n * 11^T).

        Args:
            n (int): Size of the matrix

        Returns:
            torch.Tensor: Centering matrix
        """
        H = torch.eye(n)
        H = H - 1.0 / n * torch.ones((n, n))

        if self.use_cuda:
            H = H.cuda()

        return H

    def pair_loss(self, x, y) -> torch.Tensor:
        """
        Compute HSIC between X and Y and return its squared value.

        $$
        \widehat{\text{HSIC}} = \frac{1}{n(n-3)}\Big[
        \sum_{i \neq j} K_{ij} L_{ij}
        + \frac{1}{(n-1)(n-2)} \sum_{i \neq j}K_{ij}\sum_{i \neq j}L_{ij}
        - \frac{2}{n-2}\sum_i \sum_{j \neq i}K_{ij}\sum_{k \neq i}L_{ik}
        \Big]
        $$

        Args:
            X (torch.Tensor): First variable (n x d1)
            Y (torch.Tensor): Second variable (n x d2)
            normalize (bool): Whether to compute normalized HSIC

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If normalize=True: normalized HSIC
                If normalize=False: (HSIC, HSIC(X,X), HSIC(Y,Y))
        """
        n = x.shape[0]

        # Compute kernel matrices
        K = self._kernel_function(x, self.kernel_x, self.sigma_x)
        L = self._kernel_function(y, self.kernel_y, self.sigma_y)
        K = K - torch.diag(torch.diag(K))
        L = L - torch.diag(torch.diag(L))

        KL = K @ L
        loss = (
            1
            / (n * (n - 3))
            * (
                torch.trace(KL)
                + K.sum() * L.sum() / ((n - 1) * (n - 2))
                - 2 / (n - 2) * KL.sum()
            )
        )

        return loss**2

    def forward(self, X):
        """
        Compute pairloss between all pairs of features of X. If subset_samples is not None,
        sample subset_samples features randomly, compute HSIC on the subset, and scale the 
        loss back to the original size.
        """
        hsic_total = torch.tensor(0.0, device=X.device)
        n, d = X.shape
        if self.subset_samples is not None:
            sub_idx = torch.randperm(n)[: self.subset_samples]
            X_sub = X[sub_idx, :]
            scale = n**2 / self.subset_samples**2
        else:
            X_sub = X
            scale = 1.0
        for i in range(d):
            for j in range(i + 1, d):
                hsic = (
                    self.pair_loss(
                        X_sub[:, [i]],
                        X_sub[:, [j]],
                    )
                    * scale
                )
                hsic_total += hsic
        return hsic_total

    def custom_train(self, X, optimizer):
        hsic_total = torch.tensor(0.0, device=X.device)
        n, d = X.shape
        for i in range(d):
            for j in range(i + 1, d):
                optimizer.zero_grad()
                if self.subset_samples is None:
                    hsic = self.pair_loss(X[:, [i]], X[:, [j]])
                else:
                    sub_idx = torch.randperm(n)[: self.subset_samples]
                    X_sub = X[sub_idx, :]  # / X[sub_idx, :].mean(axis=0)[None, :]
                    hsic = (
                        self.pair_loss(
                            X_sub[:, [i]],
                            X_sub[:, [j]],
                        )
                        / self.subset_samples**2
                        * n**2
                        * self.lam
                    )
                hsic.backward()
                optimizer.step()

                hsic_total += hsic.detach()

        return hsic_total


class SumstatResidualLoss(nn.Module):
    """
    Constrain peak loading to maximally explain the sumstat residuals.
    """

    def __init__(
        self,
        residuals: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        max_samples: int = 1000,
        n_factors: int = 50,
        lam: float = 1.0,
    ):
        """
        Initialize SumstatResidualLoss with residuals.

        Args:
            residuals (torch.Tensor): Residuals tensor
        """
        super(SumstatResidualLoss, self).__init__()
        if residuals.ndim == 1:
            residuals = residuals[:, None]
        self.y = torch.tensor(residuals, device=device)
        print(self.y)
        self.max_samples = max_samples
        self.coef = nn.Linear(n_factors, self.y.shape[1], bias=False).to(device)
        self.lam = lam

    def forward(self, L, idx):
        y = self.y[idx, :]  # n_idx, n_sumstats
        L_normed = L / torch.norm(L, dim=0, keepdim=True)  # n_idx, n_factors
        loss = (y - self.coef(L_normed)).pow(2).mean(axis=1).sum()
        return loss * self.lam


def weighted_mse_loss(pred, target, weight=None):
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def bernoulli_kl_loss(p_logit, q_logit):
    sigmoid = nn.Sigmoid()
    p = sigmoid(p_logit)
    q = sigmoid(q_logit)
    return (q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))).sum(axis=-1)
