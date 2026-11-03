import pytest
import torch

from pypolymix.parameter_groups import ParameterGroup, SGLDGroup
from pypolymix.priors import IIDGaussianPrior

NUM_PARAMS = 3


def quadratic_log_likelihood(theta: torch.Tensor) -> torch.Tensor:
    """A simple, well-conditioned closed-form log-likelihood: an isotropic
    Gaussian "data fit" centered at zero, ``-0.5 * sum(theta**2)``. Standing
    in for a real likelihood in tests that don't need a real surrogate model."""
    return -0.5 * theta.pow(2).sum(dim=-1)


@pytest.fixture
def sgld_group():
    return SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        num_diffusion_steps=5,
    )


def test_inherits_ParameterGroup(sgld_group):
    assert isinstance(sgld_group, ParameterGroup)


def test_sample_parameters_returns_Tensor(sgld_group):
    parameters = sgld_group.sample_parameters()
    assert isinstance(parameters, torch.Tensor)


def test_sample_parameters_shape(sgld_group):
    parameters = sgld_group.sample_parameters(num_samples=8)
    assert parameters.size() == torch.Size([8, NUM_PARAMS])


def test_sample_parameters_default_num_particles():
    group = SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        num_particles=11,
        num_diffusion_steps=5,
    )
    parameters = group.sample_parameters()
    assert parameters.size() == torch.Size([11, NUM_PARAMS])


def test_sample_negative_parameters_ValueError(sgld_group):
    with pytest.raises(ValueError):
        sgld_group.sample_parameters(num_samples=-1)


def test_variational_distribution_runtime_error(sgld_group):
    with pytest.raises(RuntimeError):
        sgld_group.variational_distribution()


def test_distribution_loss_returns_Tensor(sgld_group):
    loss = sgld_group.distribution_loss()
    assert isinstance(loss, torch.Tensor)


def test_distribution_loss_returns_scalar(sgld_group):
    loss = sgld_group.distribution_loss()
    assert loss.shape == torch.Size([])


def test_distribution_loss_is_always_zero(sgld_group):
    sgld_group.sample_parameters(num_samples=4)
    loss = sgld_group.distribution_loss()
    assert loss.item() == 0.0


def test_sample_parameters_no_grad_required(sgld_group):
    """Unlike LangevinGroup, nothing in this group is trained by an outer
    optimizer, so the returned samples should carry no autograd graph."""
    parameters = sgld_group.sample_parameters(num_samples=4)
    assert parameters.requires_grad is False


def test_preconditioned_default_true(sgld_group):
    assert sgld_group.preconditioned is True


def test_unpreconditioned_mode_runs():
    group = SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        num_diffusion_steps=5,
        preconditioned=False,
        step_size=1e-2,
    )
    parameters = group.sample_parameters(num_samples=8)
    assert parameters.size() == torch.Size([8, NUM_PARAMS])
    assert not torch.isnan(parameters).any()


def test_respects_custom_prior():
    prior = IIDGaussianPrior(mean=0.0, std=5.0)
    group = SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        prior=prior,
        num_diffusion_steps=5,
    )
    assert group.prior is prior


def test_concentrates_near_posterior_mode_for_simple_quadratic_target():
    """End-to-end sanity check (not just shape/type): for a target posterior
    ``propto exp(-0.5*theta^2) * N(0, prior_std^2)`` (both terms pull toward
    0), enough ULA steps should land particles with small norm, not an
    arbitrary/unconverged spread."""
    torch.manual_seed(0)
    group = SGLDGroup(
        name="test",
        num_params=2,
        log_likelihood_fn=quadratic_log_likelihood,
        prior=IIDGaussianPrior(mean=0.0, std=1.0),
        num_diffusion_steps=500,
        step_size=5e-2,
        init_std=3.0,
        preconditioned=False,
    )
    samples = group.sample_parameters(num_samples=256)
    # Posterior precision = 1 (likelihood) + 1 (prior) = 2, so std = 1/sqrt(2) ~= 0.707 per dim.
    assert samples.mean().abs().item() < 0.2
    assert samples.std().item() < 1.5  # well below the init_std=3.0 spread


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        pytest.param({"num_particles": -1}, id="negative_num_particles"),
        pytest.param({"num_particles": 0}, id="zero_num_particles"),
        pytest.param({"num_diffusion_steps": -1}, id="negative_num_diffusion_steps"),
        pytest.param({"num_diffusion_steps": 0}, id="zero_num_diffusion_steps"),
        pytest.param({"step_size": -0.1}, id="negative_step_size"),
        pytest.param({"step_size": 0.0}, id="zero_step_size"),
        pytest.param({"init_std": -1.0}, id="negative_init_std"),
        pytest.param({"init_std": 0.0}, id="zero_init_std"),
        pytest.param({"preconditioner_alpha": 0.0}, id="zero_preconditioner_alpha"),
        pytest.param({"preconditioner_alpha": 1.0}, id="one_preconditioner_alpha"),
        pytest.param({"preconditioner_eps": 0.0}, id="zero_preconditioner_eps"),
    ],
)
def test_constructor_invalid_kwargs_ValueError(bad_kwargs):
    with pytest.raises(ValueError):
        SGLDGroup(
            name="test name",
            num_params=NUM_PARAMS,
            log_likelihood_fn=quadratic_log_likelihood,
            **bad_kwargs,
        )


def test_non_callable_log_likelihood_fn_ValueError():
    with pytest.raises(ValueError):
        SGLDGroup(name="test", num_params=NUM_PARAMS, log_likelihood_fn=42)


def test_sample_trajectory_returns_Tensor(sgld_group):
    trajectory = sgld_group.sample_trajectory(num_samples=4)
    assert isinstance(trajectory, torch.Tensor)


def test_sample_trajectory_shape_record_every_one(sgld_group):
    trajectory = sgld_group.sample_trajectory(num_samples=4)
    # +1 for the recorded initial configuration (step 0).
    assert trajectory.shape == torch.Size([sgld_group.num_diffusion_steps + 1, 4, NUM_PARAMS])


def test_sample_trajectory_shape_record_every_stride():
    group = SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        num_diffusion_steps=20,
    )
    record_every = 7
    trajectory = group.sample_trajectory(num_samples=4, record_every=record_every)
    expected = 1 + 20 // record_every
    if 20 % record_every != 0:
        expected += 1  # the final step is always included even off-stride
    assert trajectory.shape[0] == expected


def test_sample_trajectory_matches_sample_parameters_same_seed(sgld_group):
    torch.manual_seed(0)
    trajectory = sgld_group.sample_trajectory(num_samples=4, record_every=2)
    torch.manual_seed(0)
    final = sgld_group.sample_parameters(num_samples=4)
    assert torch.allclose(trajectory[-1], final)


def test_sample_trajectory_unpreconditioned_mode_runs():
    group = SGLDGroup(
        name="test name",
        num_params=NUM_PARAMS,
        log_likelihood_fn=quadratic_log_likelihood,
        num_diffusion_steps=5,
        preconditioned=False,
        step_size=1e-2,
    )
    trajectory = group.sample_trajectory(num_samples=8, record_every=2)
    assert trajectory.shape[-2:] == torch.Size([8, NUM_PARAMS])
    assert not torch.isnan(trajectory).any()


def test_sample_trajectory_updates_last_samples(sgld_group):
    trajectory = sgld_group.sample_trajectory(num_samples=4)
    assert torch.equal(sgld_group._last_samples, trajectory[-1])


def test_sample_trajectory_negative_num_samples_ValueError(sgld_group):
    with pytest.raises(ValueError):
        sgld_group.sample_trajectory(num_samples=-1)


def test_sample_trajectory_zero_record_every_ValueError(sgld_group):
    with pytest.raises(ValueError):
        sgld_group.sample_trajectory(num_samples=4, record_every=0)


def test_sample_trajectory_no_grad_tracking(sgld_group):
    trajectory = sgld_group.sample_trajectory(num_samples=4)
    assert not trajectory.requires_grad
