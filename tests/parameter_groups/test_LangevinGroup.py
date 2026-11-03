from pypolymix.parameter_groups import LangevinGroup, ParameterGroup
from pypolymix.surrogate_models import NeuralNetwork
import pytest
import torch

NUM_PARAMS = 3
def nn():
    return NeuralNetwork(num_inputs=NUM_PARAMS, num_outputs=NUM_PARAMS, width=4, depth=1) 

@pytest.fixture
def langevin_group():
    return LangevinGroup(name='test name', num_params=NUM_PARAMS, score_model=nn())

def test_inherits_ParameterGroup(langevin_group):
    assert isinstance(langevin_group, ParameterGroup)

def test_negative_num_particles_ValueError():
    with pytest.raises(ValueError):
        LangevinGroup(name='test name', num_params=NUM_PARAMS, score_model=nn(), num_particles=-1)

def test_sample_parameters_returns_Tensor(langevin_group):
    parameters = langevin_group.sample_parameters()
    assert isinstance(parameters, torch.Tensor)

def test_sample_parameters_shape(langevin_group):
    parameters = langevin_group.sample_parameters(num_samples=8)
    assert parameters.size() == torch.Size([8, 3])

def test_sample_parameters_shape_energy_model():
    energy_model = NeuralNetwork(num_inputs=1, num_outputs=1, width=4, depth=1)  
    energy_group = LangevinGroup(name='test name', num_params=1, energy_model=energy_model)
    parameters = energy_group.sample_parameters(num_samples=8)
    assert parameters.size() == torch.Size([8, 1])

def test_sample_negative_parameters_ValueError(langevin_group):
    with pytest.raises(ValueError):
        langevin_group.sample_parameters(num_samples=-1)

def test_variational_distribution_runtime_error(langevin_group):
    with pytest.raises(RuntimeError):
        langevin_group.variational_distribution()

def test_distribution_loss_score_model_returns_Tensor(langevin_group):
    loss = langevin_group.distribution_loss()
    assert isinstance(loss, torch.Tensor)

def test_distribution_loss_score_model_returns_scalar(langevin_group):
    loss = langevin_group.distribution_loss()
    assert loss.shape == torch.Size([])

# Check that bad keyword arguments result in errors
# each entry in the bad_kwargs list is tested independently
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
    ],
)

def test_constructor_invalid_kwargs_ValueError(bad_kwargs):
    with pytest.raises(ValueError):
        # two stars to unpack the bad_kwargs dictionary into keyword arguments
        LangevinGroup(name="test name", num_params=NUM_PARAMS, score_model=nn(), **bad_kwargs)

def test_two_models_ValueError():
    with pytest.raises(ValueError):
        LangevinGroup(name="test_name", num_params=NUM_PARAMS, score_model=nn(), energy_model=nn())

def test_score_model_wrong_num_inputs_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=2, num_outputs=3, width=4, depth=1)
        LangevinGroup(name="test name", num_params=3, score_model=neural_network)

def test_score_model_wrong_num_outputs_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=1, width=4, depth=1)
        LangevinGroup(name="test name", num_params=3, score_model=neural_network)

def test_validate_surrogate_requires_num_inputs_and_num_outputs():
    bad_surrogate = 67 # not a surrogate model
    with pytest.raises(ValueError):
        LangevinGroup(name="test", num_params=2, score_model=bad_surrogate)


def test_sample_trajectory_returns_Tensor(langevin_group):
    trajectory = langevin_group.sample_trajectory(num_samples=4)
    assert isinstance(trajectory, torch.Tensor)


def test_sample_trajectory_shape_record_every_one(langevin_group):
    trajectory = langevin_group.sample_trajectory(num_samples=4)
    # +1 for the recorded initial configuration (step 0).
    assert trajectory.shape == torch.Size([langevin_group.num_diffusion_steps + 1, 4, NUM_PARAMS])


def test_sample_trajectory_shape_record_every_stride(langevin_group):
    record_every = 7
    trajectory = langevin_group.sample_trajectory(num_samples=4, record_every=record_every)
    num_diffusion_steps = langevin_group.num_diffusion_steps
    expected = 1 + num_diffusion_steps // record_every
    if num_diffusion_steps % record_every != 0:
        expected += 1  # the final step is always included even off-stride
    assert trajectory.shape[0] == expected


def test_sample_trajectory_first_and_last_steps(langevin_group):
    torch.manual_seed(0)
    trajectory = langevin_group.sample_trajectory(num_samples=4, record_every=3)
    torch.manual_seed(0)
    final = langevin_group.sample_parameters(num_samples=4)
    # Same seed, same dynamics: the trajectory's final recorded step should
    # match an independent sample_parameters() call exactly.
    assert torch.allclose(trajectory[-1], final)


def test_sample_trajectory_updates_last_samples(langevin_group):
    trajectory = langevin_group.sample_trajectory(num_samples=4)
    assert torch.equal(langevin_group._last_samples, trajectory[-1])


def test_sample_trajectory_negative_num_samples_ValueError(langevin_group):
    with pytest.raises(ValueError):
        langevin_group.sample_trajectory(num_samples=-1)


def test_sample_trajectory_zero_record_every_ValueError(langevin_group):
    with pytest.raises(ValueError):
        langevin_group.sample_trajectory(num_samples=4, record_every=0)


def test_sample_trajectory_no_grad_tracking(langevin_group):
    trajectory = langevin_group.sample_trajectory(num_samples=4)
    assert not trajectory.requires_grad
