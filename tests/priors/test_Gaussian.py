import torch
import torch.distributions as td
import pytest
from pypolymix.priors.base import Prior
from pypolymix.priors.common import GaussianPrior

def test_inherits_from_prior():
    mean, cov = torch.zeros(2), torch.eye(2)
    prior = GaussianPrior(mean=mean, covariance_matrix=cov)
    assert isinstance(prior, Prior)

def test_errors_without_covariance_representation():
    with pytest.raises(ValueError):
        GaussianPrior(mean=torch.zeros(2))

def test_errors_with_multiple_representations():
    cov = torch.eye(2)
    scale_tril = torch.eye(2)
    mean = torch.zeros(2)
    with pytest.raises(ValueError, match="exactly one"):
        GaussianPrior(mean=mean, covariance_matrix=cov, scale_tril=scale_tril)

def test_good_shape_cov_return_multivariate_normal():
    mean = torch.zeros(2)
    cov = torch.eye(2)

    prior = GaussianPrior(mean=mean, covariance_matrix=cov)

    dist = prior.distribution(
        event_shape=torch.Size([2]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert isinstance(dist, td.MultivariateNormal)

def test_good_shape_scale_tril_return_multivariate_normal():
    mean = torch.zeros(2)
    scale_tril = torch.eye(2)

    prior = GaussianPrior(mean=mean, scale_tril=scale_tril)

    dist = prior.distribution(
        event_shape=torch.Size([2]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert isinstance(dist, td.MultivariateNormal)

def test_distribution_rejects_bad_cov():
    mean = torch.zeros(2)
    cov = torch.eye(3) # does not match
    prior = GaussianPrior(mean=mean, covariance_matrix=cov)
    with pytest.raises(ValueError):
        distribution = prior.distribution(
            event_shape=torch.Size([2,2]),
            device=None,
            dtype=None,
        )

def test_gaussian_prior_rejects_wrong_mean_shape():
    mean = torch.zeros(3)
    cov = torch.eye(2)

    prior = GaussianPrior(mean=mean, covariance_matrix=cov)

    with pytest.raises(ValueError):
        prior.distribution(
            event_shape=torch.Size([2]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

def test_gaussian_prior_rejects_wrong_scale_tril_shape():
    mean = torch.zeros(2)
    scale_tril = torch.eye(3)

    prior = GaussianPrior(mean=mean, scale_tril=scale_tril)

    with pytest.raises(ValueError):
        prior.distribution(
            event_shape=torch.Size([2]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
