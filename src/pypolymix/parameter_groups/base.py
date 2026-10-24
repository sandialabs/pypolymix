from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn

from ..priors.base import Prior
from ..priors.common import IIDGaussianPrior
from .kl_estimators import AnalyticKL, KLEstimator, ReparamMonteCarloKL


class ParameterGroup(nn.Module, ABC):
    """Abstract base class for parameter groups with learnable parameters and a prior.

    Subclasses typically expose unconstrained PyTorch parameters that define the
    variational distribution $q(\\theta)$. During training their ``distribution_loss``
    is added to the surrogate loss to enforce Bayesian regularization.

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
        self,
        name: str,
        num_params: int,
        prior: Optional[Prior] = None,
        kl_num_mc_samples: int = 64,
        kl_estimator: Optional[KLEstimator] = None,
    ):
        """Register metadata shared by all parameter groups.

        Args:
            name: Friendly identifier for logging/debugging.
            num_params: Number of scalar parameters controlled by this group.
            prior: Prior distribution; defaults to ``IIDGaussianPrior``.
            kl_num_mc_samples: Number of Monte Carlo samples to approximate the KL
                term when an analytic expression is unavailable.
            kl_estimator: Optional explicit :class:`KLEstimator` strategy. If
                ``None`` (the default), ``distribution_loss()`` keeps today's
                behavior: try analytic ``kl_divergence(q, p)``, falling back to
                a reparameterized Monte Carlo estimate. If explicitly set,
                ``distribution_loss()`` calls it directly with no fallback --
                an explicit choice carries explicit responsibility.
        """
        super().__init__()
        self.name = name
        self.num_params = num_params
        self.prior: Prior = prior if prior is not None else IIDGaussianPrior()

        # Default MC settings if analytic KL is unavailable
        self.kl_num_mc_samples = kl_num_mc_samples
        self.kl_estimator = kl_estimator

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

        If ``self.kl_estimator`` is explicitly set, delegates to it directly
        (no fallback). Otherwise tries analytic ``kl_divergence(q, p)`` via
        :class:`~pypolymix.parameter_groups.kl_estimators.AnalyticKL`. If not
        implemented, falls back to a reparameterized Monte Carlo estimate of
        $\\mathbb{E}_q[ \\log q(\\theta) - \\log p(\\theta) ]$ via
        :class:`~pypolymix.parameter_groups.kl_estimators.ReparamMonteCarloKL`
        (``rsample()``).
        """
        if self.kl_estimator is not None:
            samples = self.sample_parameters(self.kl_num_mc_samples)
            return self.kl_estimator.estimate(self, samples)

        # Default path (unchanged behavior): `reference` supplies device/dtype
        # only, matching the `next(self.parameters())` used previously -- no
        # samples are drawn unless the analytic branch raises NotImplementedError.
        reference = next(self.parameters())
        try:
            return AnalyticKL().estimate(self, reference)
        except NotImplementedError:
            pass

        # MC fallback (reparameterized)
        q = self.variational_distribution()
        z = q.rsample((self.kl_num_mc_samples,))  # (S, num_params)
        return ReparamMonteCarloKL().estimate(self, z)
