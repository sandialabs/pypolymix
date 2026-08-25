import pytest
import torch
import torch.distributions as td
from pypolymix.parameter_groups import LowRankGaussianGroup, ParameterGroup

@pytest.fixture
def low_rank_gaussian_group():
    return LowRankGaussianGroup(name="test_name", num_params=3, rank=2)

def test_inherits_from_parameter_group(low_rank_gaussian_group):
    assert isinstance(low_rank_gaussian_group, ParameterGroup)

def test_variational_distribution_returns_distribution(low_rank_gaussian_group):
    distribution = low_rank_gaussian_group.variational_distribution()
    assert isinstance(distribution, td.Distribution)

def test_sample_parameters_returns_Tensor(low_rank_gaussian_group):
    samples = low_rank_gaussian_group.sample_parameters(5)
    assert isinstance(samples, torch.Tensor)
    
def test_sample_parameters_shape(low_rank_gaussian_group):
    samples = low_rank_gaussian_group.sample_parameters(5)
    assert samples.shape == torch.Size([5, 3])

def test_rank_cannot_be_negative():
    with pytest.raises(ValueError):
        LowRankGaussianGroup(name="test_name", num_params=3, rank=-1)
    