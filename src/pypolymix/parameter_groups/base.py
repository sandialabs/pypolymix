from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn

from ..priors.base import Prior
from ..priors.common import IIDGaussianPrior


class ParameterGroup(nn.Module, ABC):
    """Abstract base class for parameter groups with learnable parameters and a prior.

    Subclasses typically expose unconstrained PyTorch parameters that define the
    variational distribution $q(\\theta)$. During training their ``distribution_loss``
    is added to the surrogate loss to enforce Bayesian regularisation.

    Example:
        ```python
        >>> from pypolymix.parameter_groups import IIDGaussianGroup
        >>> group = IIDGaussianGroup("weights", num_params=5)
        >>> theta = group.sample_parameters(num_samples=3)
        >>> theta.shape
        torch.Size([3, 5])
        ```
    """

    def __init__(
        self, name: str, num_params: int, prior: Optional[Prior] = None, kl_num_mc_samples: int = 64
    ):
        """Register metadata shared by all parameter groups.

        Args:
            name: Friendly identifier for logging/debugging.
            num_params: Number of scalar parameters controlled by this group.
            prior: Prior distribution; defaults to ``IIDGaussianPrior``.
            kl_num_mc_samples: Number of Monte Carlo samples to approximate the KL
                term when an analytic expression is unavailable.
        """
        super().__init__()
        self.name = name
        self.num_params = num_params
        self.prior: Prior = prior if prior is not None else IIDGaussianPrior()

        # Default MC settings if analytic KL is unavailable
        self.kl_num_mc_samples = kl_num_mc_samples

    @abstractmethod
    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Sample $\\theta \\sim q(\\theta)$. Returns ``(num_samples, num_params)``."""
        raise NotImplementedError()

    @abstractmethod
    def variational_distribution(self) -> td.Distribution:
        """Return $q(\\theta)$ as a ``torch.distributions.Distribution``."""
        raise NotImplementedError()

    def distribution_loss(self) -> torch.Tensor:
        """$KL(q \\;||\\; p)$ summed over all parameters in this group.

        Tries analytic ``kl_divergence(q, p)``. If not implemented, uses reparameterized
        Monte Carlo: $\\mathbb{E}_q[ \\log q(\\theta) - \\log p(\\theta) ]$ with ``rsample()``.
        """
        q = self.variational_distribution()
        p = self.prior.distribution(
            event_shape=torch.Size([self.num_params]),
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )

        # Analytic KL if available:
        try:
            return td.kl_divergence(q, p).sum()
        except NotImplementedError:
            pass

        # MC fallback (reparameterized)
        z = q.rsample((self.kl_num_mc_samples,))  # (S, num_params)
        # Independent distributions return scalar log_prob per sample
        log_q = q.log_prob(z)  # (S,)
        log_p = p.log_prob(z)  # (S,)
        return (log_q - log_p).mean()
