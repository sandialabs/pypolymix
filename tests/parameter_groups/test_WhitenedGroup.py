import warnings
from contextlib import contextmanager

import pytest
import torch
import torch.distributions as td

from pypolymix.parameter_groups import (GaussianGroup, IIDGaussianGroup,
                                        LangevinGroup, LowRankGaussianGroup,
                                        ParameterGroup, WhitenedGroup)
from pypolymix.priors.common import IIDGaussianPrior, UniformPrior
from pypolymix.surrogate_models import NeuralNetwork

NUM_PARAMS = 3


@pytest.fixture
def base_group():
    return IIDGaussianGroup(name="z", num_params=NUM_PARAMS)


@pytest.fixture
def center():
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def scale():
    return torch.tensor([0.1, 0.2, 0.3])


@pytest.fixture
def whitened_group(base_group, center, scale):
    return WhitenedGroup(base_group, center, scale)


def test_inherits_from_parameter_group(whitened_group):
    assert isinstance(whitened_group, ParameterGroup)


def test_num_params_matches_base_group(whitened_group):
    assert whitened_group.num_params == NUM_PARAMS


def test_forces_base_group_prior_to_standard_normal(center, scale):
    base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS, prior=UniformPrior(-5.0, 5.0))
    group = WhitenedGroup(base, center, scale)
    assert isinstance(group.base_group.prior, IIDGaussianPrior)


def test_warns_when_overwriting_non_standard_normal_prior(center, scale):
    base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS, prior=UniformPrior(-5.0, 5.0))
    with pytest.warns(UserWarning):
        WhitenedGroup(base, center, scale)


def test_does_not_warn_when_base_prior_already_standard_normal(center, scale):
    base = IIDGaussianGroup(
        name="z", num_params=NUM_PARAMS, prior=IIDGaussianPrior(mean=0.0, std=1.0)
    )
    with _assert_no_warnings():
        WhitenedGroup(base, center, scale)


@contextmanager
def _assert_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_mismatched_center_shape_raises_ValueError(base_group, scale):
    with pytest.raises(ValueError):
        WhitenedGroup(base_group, torch.zeros(NUM_PARAMS + 1), scale)


def test_mismatched_scale_shape_raises_ValueError(base_group, center):
    with pytest.raises(ValueError):
        WhitenedGroup(base_group, center, torch.ones(NUM_PARAMS + 1))


def test_nonpositive_scale_raises_ValueError(base_group, center):
    bad_scale = torch.tensor([0.1, -0.2, 0.3])
    with pytest.raises(ValueError):
        WhitenedGroup(base_group, center, bad_scale)


def test_zero_scale_raises_ValueError(base_group, center):
    bad_scale = torch.tensor([0.1, 0.0, 0.3])
    with pytest.raises(ValueError):
        WhitenedGroup(base_group, center, bad_scale)


def test_sample_parameters_matches_center_plus_scale_times_base_sample(center, scale):
    """Doctest-equivalent check: sample_parameters == center + scale * base.sample_parameters()
    for a fixed seed."""
    torch.manual_seed(0)
    base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS)
    group = WhitenedGroup(base, center, scale)

    torch.manual_seed(1234)
    theta = group.sample_parameters(num_samples=4)

    torch.manual_seed(1234)
    z = base.sample_parameters(num_samples=4)

    assert torch.allclose(theta, center + scale * z)


def test_sample_parameters_shape(whitened_group):
    samples = whitened_group.sample_parameters(num_samples=5)
    assert samples.shape == torch.Size([5, NUM_PARAMS])


def test_variational_distribution_is_td_Distribution(whitened_group):
    distribution = whitened_group.variational_distribution()
    assert isinstance(distribution, td.Distribution)


def test_distribution_loss_zero_at_init_when_base_group_fully_zeroed():
    """q(z) = p(z) = N(0, I) exactly requires zeroing BOTH mean and log_std,
    not just mean -- see the WhitenedGroup docstring discrepancy section."""
    base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS)
    with torch.no_grad():
        base.mean.zero_()
        base.log_std.zero_()  # std = exp(0) = 1, matching N(0, I)

    group = WhitenedGroup(base, torch.zeros(NUM_PARAMS), torch.ones(NUM_PARAMS))
    loss = group.distribution_loss()
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


def test_distribution_loss_not_zero_when_only_mean_zeroed():
    """Documents the discrepancy: zeroing only `mean` (IIDGaussianGroup's default
    log_std is -0.5, not 0) is NOT sufficient for q = p at init."""
    base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS)
    with torch.no_grad():
        base.mean.zero_()
        # log_std left at its default (-0.5), so std = exp(-0.5) != 1

    group = WhitenedGroup(base, torch.zeros(NUM_PARAMS), torch.ones(NUM_PARAMS))
    loss = group.distribution_loss()
    assert loss.item() > 1e-3


def test_wrap_GaussianGroup_num_params_and_finite_loss(center, scale):
    base = GaussianGroup(name="full_cov", num_params=NUM_PARAMS)
    group = WhitenedGroup(base, center, scale)
    assert group.num_params == NUM_PARAMS
    loss = group.distribution_loss()
    assert torch.isfinite(loss).all()


def test_wrap_LowRankGaussianGroup_num_params_and_finite_loss(center, scale):
    base = LowRankGaussianGroup(name="low_rank", num_params=NUM_PARAMS, rank=2)
    group = WhitenedGroup(base, center, scale)
    assert group.num_params == NUM_PARAMS
    loss = group.distribution_loss()
    assert torch.isfinite(loss).all()


def _score_model():
    return NeuralNetwork(num_inputs=NUM_PARAMS, num_outputs=NUM_PARAMS, width=4, depth=1)


def test_wrap_LangevinGroup_variational_distribution_raises_RuntimeError(center, scale):
    base = LangevinGroup(name="langevin", num_params=NUM_PARAMS, score_model=_score_model())
    group = WhitenedGroup(base, center, scale)
    with pytest.raises(RuntimeError):
        group.variational_distribution()


def test_wrap_LangevinGroup_distribution_loss_works(center, scale):
    base = LangevinGroup(name="langevin", num_params=NUM_PARAMS, score_model=_score_model())
    group = WhitenedGroup(base, center, scale)
    loss = group.distribution_loss()
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).all()


def test_wrap_LangevinGroup_sample_parameters_works(center, scale):
    base = LangevinGroup(name="langevin", num_params=NUM_PARAMS, score_model=_score_model())
    group = WhitenedGroup(base, center, scale)
    samples = group.sample_parameters(num_samples=4)
    assert samples.shape == torch.Size([4, NUM_PARAMS])


def test_from_point_estimate_builds_matching_scale():
    theta0 = torch.tensor([10.0, 0.0, -2.0])
    group = WhitenedGroup.from_point_estimate(
        lambda n: IIDGaussianGroup("theta", n), theta0, relative_scale=0.1
    )
    expected_scale = 0.1 * theta0.abs().clamp_min(1e-3)
    assert torch.allclose(group.scale, expected_scale)
    assert torch.allclose(group.center, theta0)
    assert group.num_params == theta0.numel()


def test_from_point_estimate_eps_is_keyword_only():
    theta0 = torch.tensor([0.0, 0.0])
    group = WhitenedGroup.from_point_estimate(
        lambda n: IIDGaussianGroup("theta", n), theta0, relative_scale=0.1, eps=1e-2
    )
    assert torch.allclose(group.scale, torch.full((2,), 0.1 * 1e-2))


def test_state_dict_round_trips_center_and_scale(whitened_group):
    state = whitened_group.state_dict()
    assert "center" in state
    assert "scale" in state

    new_base = IIDGaussianGroup(name="z", num_params=NUM_PARAMS)
    new_group = WhitenedGroup(new_base, torch.zeros(NUM_PARAMS), torch.ones(NUM_PARAMS))
    new_group.load_state_dict(state)
    assert torch.allclose(new_group.center, whitened_group.center)
    assert torch.allclose(new_group.scale, whitened_group.scale)
