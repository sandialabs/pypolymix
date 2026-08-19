import pytest
import torch
import torch.distributions as td
from pypolymix.parameter_groups import GaussianGroup, ParameterGroup

@pytest.fixture
def gaussian_group():
    return GaussianGroup(name="test_name", num_params=3)

def test_inherits_from_parameter_group(gaussian_group):
    assert isinstance(gaussian_group, ParameterGroup)

def test_variational_distribution_returns_distribution(gaussian_group):
    distribution = gaussian_group.variational_distribution()
    assert isinstance(distribution, td.Distribution)

def test_sample_parameters_returns_Tensor(gaussian_group):
    samples = gaussian_group.sample_parameters(5)
    assert isinstance(samples, torch.Tensor)
    
def test_sample_parameters_shape(gaussian_group):
    samples = gaussian_group.sample_parameters(5)
    assert samples.shape == torch.Size([5, 3])
