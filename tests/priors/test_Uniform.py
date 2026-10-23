import torch
import torch.distributions as td

from pypolymix.parameter_groups import IIDGaussianGroup
from pypolymix.priors.base import Prior
from pypolymix.priors.common import UniformPrior


def test_inherits_from_prior():
    prior = UniformPrior()
    assert isinstance(prior, Prior)


def test_uniform_prior_returns_independent_uniform_distribution():
    prior = UniformPrior()
    event_shape = torch.Size([4])
    distribution = prior.distribution(event_shape=event_shape, device=None, dtype=None)
    assert isinstance(distribution, td.Independent)
    assert isinstance(distribution.base_dist, td.Uniform)


def test_distribution_loss_analytic_path_is_infinite_and_not_differentiable():
    """Document verified current behavior: torch.distributions *does* register
    an analytic kl_divergence(Normal, Uniform) (``_kl_normal_infinity``), which
    unconditionally returns +inf because a Gaussian has nonzero density outside
    any bounded interval. ParameterGroup.distribution_loss() therefore takes its
    analytic branch (no NotImplementedError) and returns a non-differentiable
    +inf, for *any* IIDGaussianGroup mean/std -- this is not something a
    UniformPrior constructor argument can avoid. See the UniformPrior docstring
    caveat and PM-04 (KL-estimator machinery) for the planned fix."""
    prior = UniformPrior(low=-10.0, high=10.0)
    group = IIDGaussianGroup(name="test_name", num_params=3, prior=prior)
    with torch.no_grad():
        group.mean.fill_(0.0)

    loss = group.distribution_loss()

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert torch.isinf(loss)
    assert not loss.requires_grad


def test_distribution_loss_mc_fallback_is_finite_with_well_inside_mean(monkeypatch):
    """When the MC fallback *is* exercised (e.g. once PM-04 teaches
    distribution_loss() to detect the degenerate analytic +inf above and fall
    back), it behaves well for a UniformPrior whose bounds are generously wide
    relative to q: verify a finite scalar and finite mean/log_std gradients
    when q's mean is well inside [low, high]. We force the fallback here the
    same way tests/parameter_groups/gaussian/test_IIDGaussianGroup.py does, by
    monkeypatching td.kl_divergence to simulate no analytic KL being used."""

    def fake_kl_divergence(q, p):
        raise NotImplementedError

    monkeypatch.setattr(td, "kl_divergence", fake_kl_divergence)

    prior = UniformPrior(low=-10.0, high=10.0)
    group = IIDGaussianGroup(name="test_name", num_params=3, prior=prior)
    with torch.no_grad():
        group.mean.fill_(0.0)

    loss = group.distribution_loss()

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)

    loss.backward()

    assert torch.isfinite(group.mean.grad).all()
    assert torch.isfinite(group.log_std.grad).all()
