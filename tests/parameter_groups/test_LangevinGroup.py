from pypolymix.parameter_groups import LangevinGroup, ParameterGroup
from pypolymix.surrogate_models import NeuralNetwork
import pytest

@pytest.fixture
def langevin_group():
    NUM_PARAMS = 3
    neural_network = NeuralNetwork(num_inputs=NUM_PARAMS, num_outputs=NUM_PARAMS, width=4, depth=1)
    return LangevinGroup(name='test name', num_params=NUM_PARAMS, score_model=neural_network)

def test_inherits_ParameterGroup(langevin_group):
    assert isinstance(langevin_group, ParameterGroup)

def test_negative_num_particles_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=3, width=4, depth=1)
        LangevinGroup(name='test name', num_params=3, score_model=neural_network, num_particles=-1)

def sample_parameters_returns_Tensor(langevin_group):
    parameters = langevin_group.sample_parameters()
    assert isinstance(parameters, torch.Tensor)


def test_negative_num_particles_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=3, width=4, depth=1)
        LangevinGroup(
            name="test name",
            num_params=3,
            score_model=neural_network,
            num_particles=-1,
        )


def test_negative_num_diffusion_steps_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=3, width=4, depth=1)
        LangevinGroup(
            name="test name",
            num_params=3,
            score_model=neural_network,
            num_diffusion_steps=-1,
        )


def test_negative_step_size_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=3, width=4, depth=1)
        LangevinGroup(
            name="test name",
            num_params=3,
            score_model=neural_network,
            step_size=-0.1,
        )


def test_negative_init_std_ValueError():
    with pytest.raises(ValueError):
        neural_network = NeuralNetwork(num_inputs=3, num_outputs=3, width=4, depth=1)
        LangevinGroup(name="test name", num_params=3, score_model=neural_network, init_std=-1.0)


def test_no_surrogate_model_ValueError():
    with pytest.raises(ValueError):
        LangevinGroup(
            name="test name",
            num_params=3,
        )