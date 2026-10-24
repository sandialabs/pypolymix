import math

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import StudentT

from pypolymix.likelihoods import Likelihood, StudentTLikelihood


def test_inherits_from_likelihood():
    likelihood = StudentTLikelihood(num_outputs=2)
    assert isinstance(likelihood, Likelihood)


def test_nll_matches_torch_distributions_studentt():
    """nll must equal -StudentT(df, 0, scale).log_prob(residual).sum(-1).mean()
    for a fixed raw_df/log_scale, with df = df_floor + softplus(raw_df) applied
    identically on both sides."""
    torch.manual_seed(1)
    num_outputs = 3
    samples, batch = 4, 5
    raw_df = torch.tensor([0.5, -0.3, 1.2])
    log_scale = torch.tensor([-0.1, 0.2, 0.0])
    df_floor = 2.0

    likelihood = StudentTLikelihood(num_outputs=num_outputs, df_floor=df_floor)
    with torch.no_grad():
        likelihood.raw_df.copy_(raw_df)
        likelihood.log_scale.copy_(log_scale)

    pred = torch.randn(samples, batch, num_outputs)
    target = torch.randn(batch, num_outputs)
    residual = pred - target.unsqueeze(0).expand_as(pred)

    df = df_floor + F.softplus(raw_df)
    scale = torch.exp(log_scale)
    expected = -StudentT(df, torch.zeros_like(scale), scale).log_prob(residual).sum(-1).mean()

    actual = likelihood.nll(pred, target)
    torch.testing.assert_close(actual, expected)


def test_df_floor_is_applied():
    """df = df_floor + softplus(raw_df): driving raw_df very negative should
    push df arbitrarily close to (but not below) df_floor."""
    df_floor = 3.0
    likelihood = StudentTLikelihood(num_outputs=2, df_floor=df_floor)
    with torch.no_grad():
        likelihood.raw_df.fill_(-50.0)  # softplus(-50) underflows to 0 in float32
    assert torch.all(likelihood.df >= df_floor)
    torch.testing.assert_close(likelihood.df, torch.full((2,), df_floor), atol=1e-6, rtol=0)


def test_no_decay_parameters_returns_log_scale_and_raw_df():
    likelihood = StudentTLikelihood(num_outputs=2)
    assert likelihood.no_decay_parameters() == [likelihood.log_scale, likelihood.raw_df]


def test_init_df_must_exceed_df_floor():
    with pytest.raises(ValueError):
        StudentTLikelihood(num_outputs=1, init_df=1.0, df_floor=2.0)


def test_init_scale_sets_log_scale():
    likelihood = StudentTLikelihood(num_outputs=2, init_scale=2.0)
    expected = torch.full((2,), math.log(2.0))
    torch.testing.assert_close(likelihood.log_scale.detach(), expected)
