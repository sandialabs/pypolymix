from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from ..priors.base import Prior
from .base import ParameterGroup


class IIDGaussianGroup(ParameterGroup):
    """I.i.d. Gaussian variational family: ``q = Normal(mean, std)``.

    Example:
        ```python
        >>> group = IIDGaussianGroup("weights", num_params=4)
        >>> samples = group.sample_parameters(16)
        >>> samples.shape
        torch.Size([16, 4])
        ```
    """

    def __init__(self, name: str, num_params: int, prior: Optional[Prior] = None):
        """Create learnable mean/log-std parameters for every coefficient."""
        super().__init__(name, num_params, prior=prior)
        self.mean = nn.Parameter(torch.randn(num_params))
        self.log_std = nn.Parameter(torch.full((num_params,), -0.5))  # start small

    @property
    def std(self) -> torch.Tensor:
        """Positive standard deviation parameter obtained via ``exp``."""
        return self.log_std.exp()

    def variational_distribution(self) -> td.Distribution:
        """Return an independent normal distribution over all parameters."""
        base = td.Normal(self.mean, self.std)  # shape (..., num_params)
        return td.Independent(base, 1)

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Draw ``n_samples`` parameter vectors via ``rsample`` for reparameterization."""
        return self.variational_distribution().rsample((num_samples,))


class GaussianGroup(ParameterGroup):
    """Full-covariance Gaussian variational family with a learnable Cholesky factor.

    This is useful when posterior correlations between parameters are important.

    Example:
        ```python
        >>> from pypolymix.parameter_groups import GaussianGroup
        >>> group = GaussianGroup("weights", num_params=2)
        >>> group.variational_distribution().rsample().shape
        torch.Size([2])
        ```
    """

    def __init__(
        self,
        name: str,
        num_params: int,
        prior=None,
        *,
        init_scale_diag: float = 1e-1,
        jitter: float = 1e-6,
    ):
        """Parameterize a full-covariance Gaussian via a packed Cholesky factor."""
        super().__init__(name, num_params, prior=prior)
        d = num_params
        self.jitter = float(jitter)

        # parameters
        self.mean = nn.Parameter(torch.zeros(d))
        self.raw_tril = nn.Parameter(torch.zeros(d * (d + 1) // 2))

        # cache tril indices once
        i, j = torch.tril_indices(d, d, 0)
        self.register_buffer("_tri_i", i, persistent=False)
        self.register_buffer("_tri_j", j, persistent=False)

        # initialize to ~diag(init_scale_diag)
        with torch.no_grad():
            # set diagonal entries in raw_tril so softplus(raw) ≈ init_scale_diag - jitter
            s = max(init_scale_diag - self.jitter, 1e-12)
            diag_raw = torch.log(torch.expm1(torch.tensor(s)))
            # positions where i == j are diagonals in the packed vector order
            mask_diag = self._tri_i == self._tri_j
            self.raw_tril[mask_diag] = diag_raw

    @property
    def _scale_tril(self) -> torch.Tensor:
        """Build a lower-triangular factor from the unconstrained parameters."""
        d = self.mean.numel()
        L = torch.zeros(d, d, device=self.mean.device, dtype=self.mean.dtype)
        L[self._tri_i, self._tri_j] = self.raw_tril
        # enforce SPD via softplus on diagonal (+ jitter)
        diag_pos = F.softplus(torch.diagonal(L)) + self.jitter
        L = L - torch.diag_embed(torch.diagonal(L)) + torch.diag_embed(diag_pos)
        return L

    def variational_distribution(self) -> td.Distribution:
        """Return ``torch.distributions.MultivariateNormal`` with ``scale_tril``."""
        return td.MultivariateNormal(loc=self.mean, scale_tril=self._scale_tril)

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Draw ``n_samples`` reparameterized samples from the full-covariance Gaussian."""
        return self.variational_distribution().rsample((num_samples,))


class LowRankGaussianGroup(ParameterGroup):
    """Gaussian family with a low-rank plus diagonal covariance approximation.

    The covariance matrix is parameterized as ``U U^T + diag(d)`` with ``rank(U)``
    controlled by ``rank``. This captures the dominant correlations without the
    ``O(d^2)`` parameters and compute cost of a full Cholesky factor.

    Example:
        ```python
        >>> group = LowRankGaussianGroup("weights", num_params=8, rank=3)
        >>> samples = group.sample_parameters(4)
        >>> samples.shape
        torch.Size([4, 8])
        ```
    """

    def __init__(
        self,
        name: str,
        num_params: int,
        rank: int,
        prior: Optional[Prior] = None,
        *,
        init_scale_diag: float = 1e-1,
        init_scale_factor: float = 1e-1,
        jitter: float = 1e-6,
    ):
        if rank <= 0 or rank > num_params:
            raise ValueError("rank must be in [1, num_params]")

        super().__init__(name, num_params, prior=prior)
        self.rank = int(rank)
        self.jitter = float(jitter)

        self.mean = nn.Parameter(torch.zeros(num_params))
        self.cov_factor = nn.Parameter(torch.randn(num_params, self.rank) * init_scale_factor)
        self.raw_cov_diag = nn.Parameter(torch.zeros(num_params))

        with torch.no_grad():
            s = max(init_scale_diag - self.jitter, 1e-12)
            diag_raw = torch.log(torch.expm1(torch.tensor(s)))
            self.raw_cov_diag.fill_(diag_raw)

    @property
    def _cov_diag(self) -> torch.Tensor:
        """Positive diagonal component ensuring SPD covariance."""
        return F.softplus(self.raw_cov_diag) + self.jitter

    def variational_distribution(self) -> td.Distribution:
        return td.LowRankMultivariateNormal(
            loc=self.mean,
            cov_factor=self.cov_factor,
            cov_diag=self._cov_diag,
        )

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        return self.variational_distribution().rsample((num_samples,))
