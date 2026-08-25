import pytest
import torch
import torch.distributions as td

from pypolymix.parameter_groups import IIDGaussianGroup, ParameterGroup

# IID Gaussian Group
@pytest.fixture
def iid_gaussian_group():
    return IIDGaussianGroup(name="test_name", num_params=3)

def test_inherits_from_parameter_group(iid_gaussian_group):
    assert isinstance(iid_gaussian_group, ParameterGroup)

def test_std_returns_Tensor(iid_gaussian_group):
    std = iid_gaussian_group.std
    assert isinstance(std, torch.Tensor)


def test_variational_distribution_is_td_Distribution(iid_gaussian_group):
    distribution = iid_gaussian_group.variational_distribution()
    assert isinstance(distribution, td.Distribution)

def test_sample_parameters_returns_Tensor(iid_gaussian_group):
    samples = iid_gaussian_group.sample_parameters(num_samples=5)
    assert isinstance(samples, torch.Tensor)

def test_sample_parameters_shape(iid_gaussian_group):
    samples = iid_gaussian_group.sample_parameters(num_samples=5)
    assert samples.shape == torch.Size([5,3])

def test_inherits_parameter_group_distribution_loss(iid_gaussian_group):
    '''IIDGaussianGroup doesn't implement distribution_loss, but it inherits'''
    distribution_loss = iid_gaussian_group.distribution_loss()
    assert isinstance(distribution_loss, torch.Tensor)

def test_mc_fallback_happens(iid_gaussian_group, monkeypatch):
    '''
    ParameterGroup ABC implements distribution_loss, which uses kl_divergence
    This test simulates that analytic KL is not available
    And checks that the fallback works
    monkeypatch allows us to replace td.kl_divergence at test time to get the error
    '''
    def fake_kl_divergence(q, p):
        raise NotImplementedError
    monkeypatch.setattr(td, "kl_divergence", fake_kl_divergence)
    loss = iid_gaussian_group.distribution_loss()
    assert isinstance(loss, torch.Tensor)
