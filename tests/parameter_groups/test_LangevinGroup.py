from pypolymix.parameter_groups import LangevinGroup, ParameterGroup
from pypolymix.surrogate_models import NeuralNetwork
import pytest

NUM_PARAMS = 3
def nn():
    return NeuralNetwork(num_inputs=NUM_PARAMS, num_outputs=NUM_PARAMS, width=4, depth=1) 

@pytest.fixture
def langevin_group():
    neural_network = nn()
    return LangevinGroup(name='test name', num_params=NUM_PARAMS, score_model=nn())

def test_inherits_ParameterGroup(langevin_group):
    assert isinstance(langevin_group, ParameterGroup)

def test_negative_num_particles_ValueError():
    with pytest.raises(ValueError):
        LangevinGroup(name='test name', num_params=NUM_PARAMS, score_model=nn(), num_particles=-1)

def sample_parameters_returns_Tensor(langevin_group):
    parameters = langevin_group.sample_parameters()
    assert isinstance(parameters, torch.Tensor)

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
        LangevinGroup(name="test name", num_params=3, score_model=nn(), **bad_kwargs)
