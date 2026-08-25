import pytest
from pypolymix.parameter_groups import DeterministicGroup, ParameterGroup
import torch

@pytest.fixture(autouse=True)
def deterministic_group():
    return DeterministicGroup(name='test_name', num_params=2)

def test_inherits_from_ParameterGroup(deterministic_group):
    assert isinstance(deterministic_group, ParameterGroup)

def test_sample_parameters_returns_tensor(deterministic_group):
    parameters = deterministic_group.sample_parameters(num_samples=3)
    assert isinstance(parameters, torch.Tensor)

def test_sample_parameters_returns_correct_shape(deterministic_group):
    parameters = deterministic_group.sample_parameters(num_samples=3)
    assert parameters.shape == torch.Size([3,2])

def test_variational_distribution_RuntimeError(deterministic_group):
    with pytest.raises(RuntimeError):
        deterministic_group.variational_distribution()

def test_distribution_loss_returns_tensor(deterministic_group):
    distribution_loss = deterministic_group.distribution_loss()
    assert isinstance(distribution_loss, torch.Tensor)

def test_distribution_loss_returns_scaler(deterministic_group):
    distribution_loss = deterministic_group.distribution_loss()
    assert distribution_loss.ndim == 0
