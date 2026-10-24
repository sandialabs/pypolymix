"""Pluggable strategies for estimating ``KL(q || p)`` (or an ELBO-consistent proxy).

``ParameterGroup.distribution_loss()`` and :class:`~pypolymix.parameter_groups.langevin.LangevinGroup`
delegate to one of these :class:`KLEstimator` subclasses rather than hardcoding a single
strategy, so new estimators can be compared empirically without touching the groups
themselves.

Example:
    ```python
    >>> from pypolymix.parameter_groups import IIDGaussianGroup, AnalyticKL
    >>> group = IIDGaussianGroup("weights", num_params=3)
    >>> samples = group.sample_parameters(num_samples=5)
    >>> loss = AnalyticKL().estimate(group, samples)
    >>> loss.shape
    torch.Size([])
    ```
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.distributions as td

if TYPE_CHECKING:
    from .base import ParameterGroup


class KLEstimator(ABC):
    """Strategy for estimating ``KL(q || p)`` (or an ELBO-consistent proxy) given a
    parameter group and a batch of samples drawn from ``q``."""

    @abstractmethod
    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        """``samples``: ``(S, num_params)`` drawn from ``q``. Returns a scalar."""
        raise NotImplementedError()


class AnalyticKL(KLEstimator):
    """KL via ``torch.distributions.kl_divergence(q, p)``.

    Requires ``group`` to define ``variational_distribution()``. Raises
    ``NotImplementedError`` if torch has no registered analytic KL for this
    ``(q, p)`` pair -- callers should catch this and fall back (see
    :class:`ReparamMonteCarloKL`). ``samples`` is not evaluated; it is only used
    to read ``device``/``dtype`` for instantiating the prior, matching whatever
    device/dtype the rest of the computation runs on.
    """

    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        q = group.variational_distribution()
        p = group.prior.distribution(
            event_shape=torch.Size([group.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        return td.kl_divergence(q, p).sum()


class ReparamMonteCarloKL(KLEstimator):
    """``E_q[log q(z) - log p(z)]`` via the provided reparameterized samples.

    Requires ``group`` to define ``variational_distribution()``. This is the
    fallback used in ``ParameterGroup.distribution_loss()`` whenever
    :class:`AnalyticKL` raises ``NotImplementedError`` -- extracted unchanged.
    """

    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        q = group.variational_distribution()
        p = group.prior.distribution(
            event_shape=torch.Size([group.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)
        return (log_q - log_p).mean()


class CrossEntropyOnly(KLEstimator):
    """``-E_q[log p(theta)]`` from samples alone.

    Works for implicit samplers (e.g. ``LangevinGroup``) that have no analytic
    ``q``. **This is NOT a valid KL term -- it omits ``-H[q]``, the entropy of
    the variational distribution.** Minimizing it alone drives ``q`` toward a
    point mass at the prior's mode (MAP-seeking, "particle collapse"). Kept
    only as an explicit, clearly-labeled baseline for comparison against
    :class:`DiagonalGaussianEntropyKL`/:class:`KNNEntropyKL` (see PM-05's
    comparison notebook) and as ``LangevinGroup``'s historical default -- do
    not use this for new work without understanding this limitation.
    """

    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        p = group.prior.distribution(
            event_shape=torch.Size([group.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        return -p.log_prob(samples).mean()


class DiagonalGaussianEntropyKL(KLEstimator):
    """``-E_q[log p(theta)] - H_diag(q)``, a parametric diagonal-Gaussian entropy KL.

    ``H_diag`` is a per-dimension diagonal-Gaussian plug-in entropy estimate
    with an analytic small-sample bias correction. Cheap (``O(S*d)``), but
    assumes the particle cloud is reasonably described by independent
    per-dimension Gaussians -- check this assumption on your own problem
    (PM-05's comparison notebook checks it against a nonparametric estimator
    on a toy multimodal posterior).

    ``H_hat = sum_d log(std_d) - d * bias + 0.5*d*log(2*pi*e)``

    ``bias = 0.5 * (digamma((S-1)/2) - log((S-1)/2))``, which corrects the
    downward bias of ``E[log sigma_hat]`` for small ``S``.

    This is the exact estimator used in a downstream production codebase's
    ``WhitenedLangevinGroup`` (see this project's handoff notes, PM-04, for
    provenance) -- ported here unchanged.
    """

    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        num_particles, dim = samples.shape
        if num_particles < 3:
            raise ValueError(f"diagonal entropy needs at least 3 particles, got {num_particles}")
        p = group.prior.distribution(
            event_shape=torch.Size([group.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        cross_entropy = -p.log_prob(samples).mean()
        log_sigma = torch.log(samples.std(dim=0).clamp_min(1e-30)).sum()
        half = 0.5 * (num_particles - 1)
        bias = 0.5 * (torch.special.digamma(torch.tensor(half)) - math.log(half))
        entropy = log_sigma - dim * bias + 0.5 * dim * math.log(2.0 * math.pi * math.e)
        return cross_entropy - entropy


class KNNEntropyKL(KLEstimator):
    """``-E_q[log p(theta)] - H_knn(q)``, a nonparametric k-NN entropy KL.

    ``H_knn`` is the Kozachenko-Leonenko k-nearest-neighbor differential
    entropy estimator. Nonparametric -- makes no Gaussianity assumption about
    the particle cloud, unlike :class:`DiagonalGaussianEntropyKL`, at the cost
    of ``O(S^2)`` pairwise distances and higher variance at small ``S``.
    Usable with ``LangevinGroup`` for comparison against the cheaper
    parametric estimator above (see PM-05).

    Args:
        k: Number of nearest neighbors (default 4; must be ``< num_samples``).
    """

    def __init__(self, k: int = 4):
        self.k = int(k)

    def estimate(self, group: "ParameterGroup", samples: torch.Tensor) -> torch.Tensor:
        num_samples, dim = samples.shape
        if self.k >= num_samples:
            raise ValueError(f"k ({self.k}) must be < num_samples ({num_samples})")
        p = group.prior.distribution(
            event_shape=torch.Size([group.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        cross_entropy = -p.log_prob(samples).mean()
        entropy = _knn_entropy(samples, self.k)
        return cross_entropy - entropy


def _knn_entropy(samples: torch.Tensor, k: int) -> torch.Tensor:
    """Kozachenko-Leonenko k-NN differential entropy estimate.

    Kozachenko, L. F., & Leonenko, N. N. (1987). "Sample estimate of entropy
    of a random vector." Problems of Information Transmission, 23(2), 95-101.
    See also Kraskov, Stögbauer & Grassberger (2004), "Estimating mutual
    information," Phys. Rev. E 69, 066138, eq. (20), for the general-``k``
    form used here:

    ``H_hat = log(S - 1) - digamma(k) + log(V_d) + (d/S) * sum_i log(r_{k,i})``

    where ``V_d = pi^(d/2) / Gamma(d/2 + 1)`` is the volume of the unit
    ``d``-ball (Euclidean norm) and ``r_{k,i}`` is the Euclidean distance from
    sample ``i`` to its ``k``-th nearest neighbor (excluding itself).
    """
    num_samples, dim = samples.shape
    distances = torch.cdist(samples, samples)
    distances.fill_diagonal_(float("inf"))
    r_k = torch.kthvalue(distances, k, dim=1).values  # (S,)

    log_unit_ball_volume = (dim / 2.0) * math.log(math.pi) - torch.lgamma(
        torch.tensor(dim / 2.0 + 1.0, device=samples.device, dtype=samples.dtype)
    )
    digamma_k = torch.special.digamma(
        torch.tensor(float(k), device=samples.device, dtype=samples.dtype)
    )

    return (
        math.log(num_samples - 1) - digamma_k + log_unit_ball_volume + dim * torch.log(r_k).mean()
    )
