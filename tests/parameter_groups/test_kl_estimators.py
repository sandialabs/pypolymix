import math

import pytest
import torch
import torch.distributions as td

from pypolymix.priors import IIDGaussianPrior

from pypolymix.parameter_groups import (  # isort:skip
    AnalyticKL,
    CrossEntropyOnly,
    DiagonalGaussianEntropyKL,
    IIDGaussianGroup,
    KNNEntropyKL,
    ReparamMonteCarloKL,
)


class _ZeroLogProbDistribution:
    """Stand-in distribution whose ``log_prob`` is identically zero.

    Used to isolate an entropy estimator's own ``H_hat(q)`` term from the
    ``-E_q[log p(theta)]`` cross-entropy term that ``KLEstimator.estimate``
    also includes: with this prior, ``estimate(...)`` collapses to
    ``-H_hat(q)`` exactly.
    """

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0], dtype=samples.dtype, device=samples.device)


class _ZeroLogProbPrior:
    def distribution(self, event_shape, device, dtype):
        return _ZeroLogProbDistribution()


class _FakeGroup:
    """Minimal stand-in exposing only what ``KLEstimator.estimate`` reads."""

    def __init__(self, num_params: int, prior):
        self.num_params = num_params
        self.prior = prior


def test_diagonal_gaussian_entropy_recovers_closed_form_gaussian_entropy():
    """``DiagonalGaussianEntropyKL`` on N(0, sigma^2 I) samples should recover
    the closed-form Gaussian differential entropy ``0.5*D*log(2*pi*e*sigma^2)``
    to within a documented tolerance (0.05 nats, seed fixed for determinism).
    """
    torch.manual_seed(42)
    dim = 3
    sigma = 0.5
    num_samples = 2000

    samples = torch.randn(num_samples, dim) * sigma
    group = _FakeGroup(num_params=dim, prior=_ZeroLogProbPrior())

    # With a zero-log-prob "prior", estimate() == -H_hat(q).
    neg_entropy = DiagonalGaussianEntropyKL().estimate(group, samples)
    entropy_hat = -neg_entropy

    closed_form = 0.5 * dim * math.log(2.0 * math.pi * math.e * sigma**2)
    assert torch.isclose(entropy_hat, torch.tensor(closed_form), atol=0.05)


def test_knn_entropy_recovers_closed_form_gaussian_entropy():
    """``KNNEntropyKL`` on the same kind of Gaussian samples should recover the
    same closed-form entropy, with a looser documented tolerance (0.1 nats)
    since it is a nonparametric, higher-variance estimator.
    """
    torch.manual_seed(42)
    dim = 3
    sigma = 0.5
    num_samples = 2000
    k = 10

    samples = torch.randn(num_samples, dim) * sigma
    group = _FakeGroup(num_params=dim, prior=_ZeroLogProbPrior())

    neg_entropy = KNNEntropyKL(k=k).estimate(group, samples)
    entropy_hat = -neg_entropy

    closed_form = 0.5 * dim * math.log(2.0 * math.pi * math.e * sigma**2)
    assert torch.isclose(entropy_hat, torch.tensor(closed_form), atol=0.1)


def test_entropy_estimators_give_distinct_results_on_non_gaussian_samples():
    """Sanity check that the entropy terms are actually contributing and
    distinct: CrossEntropyOnly, DiagonalGaussianEntropyKL, and KNNEntropyKL
    should disagree on non-Gaussian (bimodal) samples."""
    torch.manual_seed(0)
    dim = 2
    num_samples = 500

    # Bimodal mixture: half the particles near -3, half near +3.
    half = num_samples // 2
    samples = torch.cat(
        [
            torch.randn(half, dim) * 0.3 - 3.0,
            torch.randn(num_samples - half, dim) * 0.3 + 3.0,
        ],
        dim=0,
    )

    group = _FakeGroup(num_params=dim, prior=IIDGaussianPrior(mean=0.0, std=1.0))

    cross_entropy_only = CrossEntropyOnly().estimate(group, samples)
    diagonal = DiagonalGaussianEntropyKL().estimate(group, samples)
    knn = KNNEntropyKL(k=10).estimate(group, samples)

    values = [cross_entropy_only.item(), diagonal.item(), knn.item()]
    assert len(set(round(v, 6) for v in values)) == 3
    assert abs(values[0] - values[1]) > 1e-3
    assert abs(values[0] - values[2]) > 1e-3
    assert abs(values[1] - values[2]) > 1e-3


def test_knn_entropy_requires_k_less_than_num_samples():
    group = _FakeGroup(num_params=2, prior=IIDGaussianPrior())
    samples = torch.randn(4, 2)
    with pytest.raises(ValueError):
        KNNEntropyKL(k=4).estimate(group, samples)


def test_diagonal_gaussian_entropy_requires_at_least_three_particles():
    group = _FakeGroup(num_params=2, prior=IIDGaussianPrior())
    samples = torch.randn(2, 2)
    with pytest.raises(ValueError):
        DiagonalGaussianEntropyKL().estimate(group, samples)


def test_explicit_analytic_kl_matches_default_none_path():
    """Explicit ``kl_estimator=AnalyticKL()`` on ``IIDGaussianGroup`` must match
    the default (``kl_estimator=None``) path's result for the same parameters,
    since the default path is itself ``AnalyticKL`` wrapped in a try/except."""
    torch.manual_seed(7)
    default_group = IIDGaussianGroup(name="default", num_params=4)

    torch.manual_seed(7)
    explicit_group = IIDGaussianGroup(name="explicit", num_params=4, kl_estimator=AnalyticKL())

    # Same random init (same seed), so mean/log_std match exactly.
    assert torch.allclose(default_group.mean, explicit_group.mean)
    assert torch.allclose(default_group.log_std, explicit_group.log_std)

    default_loss = default_group.distribution_loss()
    explicit_loss = explicit_group.distribution_loss()

    assert torch.allclose(default_loss, explicit_loss)


def test_explicit_kl_estimator_bypasses_fallback_on_error(monkeypatch):
    """When ``kl_estimator`` is explicitly set, ``distribution_loss()`` must not
    fall back to Monte Carlo even if ``td.kl_divergence`` raises
    ``NotImplementedError`` -- explicit choice means no fallback."""
    group = IIDGaussianGroup(name="explicit", num_params=3, kl_estimator=AnalyticKL())

    def fake_kl_divergence(q, p):
        raise NotImplementedError

    monkeypatch.setattr(td, "kl_divergence", fake_kl_divergence)

    with pytest.raises(NotImplementedError):
        group.distribution_loss()


def test_knn_entropy_kl_is_differentiable_through_samples():
    """``KNNEntropyKL`` must support backprop through its ``samples`` input --
    required by both reparameterized samplers (``rsample()``) and implicit
    samplers (``LangevinGroup``'s particles). Regression test for a bug where
    an in-place ``fill_diagonal_`` on ``torch.cdist``'s output broke autograd
    with "modified by an inplace operation" whenever ``samples`` required
    grad (PM-05)."""
    torch.manual_seed(3)
    group = _FakeGroup(num_params=2, prior=IIDGaussianPrior())
    samples = torch.randn(12, 2, requires_grad=True)

    loss = KNNEntropyKL(k=4).estimate(group, samples)
    loss.backward()

    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_reparam_monte_carlo_kl_estimator_usable_directly():
    group = IIDGaussianGroup(name="g", num_params=3)
    samples = group.variational_distribution().rsample((64,))
    loss = ReparamMonteCarloKL().estimate(group, samples)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
