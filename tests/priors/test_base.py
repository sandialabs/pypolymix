import pytest
import torch
import torch.distributions as td
import torch.nn as nn

from pypolymix.priors.base import Prior


def test_prior_is_abstract_base_class():
    with pytest.raises(TypeError):
        Prior()


def test_concrete_prior_subclass_can_be_instantiated():
    '''
    Build a basic prior that inherits from Prior and implements distribution()
    And test that it can be instantiated
    '''
    class ConcretePrior(Prior):
        def distribution(self, event_shape, device, dtype):
            pass
    prior = ConcretePrior()

    assert isinstance(prior, Prior)
    assert isinstance(prior, nn.Module)

def test_concrete_prior_distribution_contract():
    '''
    I don't like this test.
    It's testing the implementation of ConcretePrior which is not in my code
    And I've already tested a dummy class that implements distribution
    '''
    class ConcretePrior(Prior):
        def distribution(self, event_shape, device, dtype):
            loc = torch.zeros(event_shape, device=device, dtype=dtype)
            scale = torch.ones(event_shape, device=device, dtype=dtype)
            return td.Independent(td.Normal(loc, scale), 1)

    prior = ConcretePrior()

    event_shape = torch.Size([3])
    dist = prior.distribution(
        event_shape=event_shape,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    sample = dist.sample()
    log_prob = dist.log_prob(sample)

    assert isinstance(dist, td.Distribution)
    assert dist.event_shape == event_shape
    assert sample.shape == event_shape
    assert sample.dtype == torch.float64
    assert sample.device.type == "cpu"
    assert log_prob.shape == torch.Size([])
    assert torch.isfinite(log_prob)
