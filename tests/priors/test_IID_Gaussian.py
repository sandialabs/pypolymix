import torch
import torch.distributions as td

from pypolymix.priors.base import Prior
from pypolymix.priors.common import IIDGaussianPrior


# IID Gaussian Prior
"""Independent Gaussian prior with per-parameter mean and standard deviation.

    Example:
        >>> prior = IIDGaussianPrior(mean=0.0, std=0.5)
        >>> prior.distribution(torch.Size([4]), None, None).sample().shape
        torch.Size([4])
    """
def test_implements_prior():
    gaussian_prior = IIDGaussianPrior()
    assert isinstance(gaussian_prior, Prior)

def test_distribution_returns_td_Distribution():
    gaussian_prior = IIDGaussianPrior()
    distribution = gaussian_prior.distribution(torch.Size([1]), None, None)
    assert isinstance(distribution, td.Distribution)

def test_distribution_can_be_sampled():
    gaussian_prior = IIDGaussianPrior()
    distribution = gaussian_prior.distribution(torch.Size([1]), None, None)
    sample = distribution.sample(sample_shape=torch.Size([1]))
    assert isinstance(sample, torch.Tensor)

def test_distribution_behaves_normally():
    # P(value outside (-6, 6) standard deviations from mean) roughly equals 1e-7
    gaussian_prior = IIDGaussianPrior(mean=0.0, std=1.0)
    distribution = gaussian_prior.distribution(torch.Size([1]), None, None)
    sample = distribution.sample(sample_shape=torch.Size([100]))
    assert all([num < 6 for num in sample])

def test_distribution_is_distributed():
    MEAN = 10.0
    STD = 1.0
    NUM_SAMPLES = 100
    gaussian_prior = IIDGaussianPrior(mean=MEAN, std=STD)
    distribution = gaussian_prior.distribution(torch.Size([1]), None, None)
    sample = distribution.sample(sample_shape=torch.Size([NUM_SAMPLES]))
    # probability of max(sample) > (MEAN + STD) failing by random chance alone = .84 ^ 100 = 2e-8
    # .84: the probability of one sample being less than (MEAN + STD) by the empirical rule
    assert max(sample) > (MEAN + STD)
    assert min(sample) < (MEAN - STD)
