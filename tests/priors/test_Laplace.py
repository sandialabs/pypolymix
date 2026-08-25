import torch
import torch.distributions as td
from pypolymix.priors.base import Prior
from pypolymix.priors.common import LaplacePrior

def test_inherits_from_prior():
    prior = LaplacePrior()
    assert isinstance(prior, Prior)

def test_laplace_prior_returns_independent_laplace_distribution():
    prior = LaplacePrior()
    event_shape = torch.Size([4])
    distribution = prior.distribution(event_shape=event_shape, device=None, dtype=None)
    assert isinstance(distribution, td.Independent)
