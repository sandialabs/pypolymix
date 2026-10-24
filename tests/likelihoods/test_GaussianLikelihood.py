import math

import torch

from pypolymix.likelihoods import GaussianLikelihood, Likelihood

LOG_2PI = math.log(2.0 * math.pi)


def test_inherits_from_likelihood():
    likelihood = GaussianLikelihood(num_outputs=2)
    assert isinstance(likelihood, Likelihood)


def test_nll_matches_hand_computed_gaussian_nll():
    """0.5 * sum_d[(y-yhat)^2/sigma_d^2 + log(2*pi*sigma_d^2)], mean over batch."""
    log_sigma = torch.tensor([0.1, -0.2])
    pred = torch.tensor([[1.0, 2.0], [0.0, 0.5]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    likelihood = GaussianLikelihood(num_outputs=2)
    with torch.no_grad():
        likelihood.log_sigma.copy_(log_sigma)

    residual = pred - target
    sigma2 = torch.exp(2.0 * log_sigma)
    per_point = 0.5 * ((residual**2 / sigma2) + torch.log(2.0 * math.pi * sigma2)).sum(-1)
    expected = per_point.mean()

    actual = likelihood.nll(pred, target)
    torch.testing.assert_close(actual, expected)


def test_nll_broadcasts_target_over_leading_sample_dimension():
    """pred may be (S, batch, num_outputs) from StochasticModel.forward(num_samples=S);
    target (batch, num_outputs) should broadcast over the sample dimension."""
    log_sigma = torch.tensor([0.0, 0.0])
    likelihood = GaussianLikelihood(num_outputs=2)
    with torch.no_grad():
        likelihood.log_sigma.copy_(log_sigma)

    target = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    pred_no_sample = target.clone().unsqueeze(0)  # (1, batch, num_outputs), zero residual
    pred_with_samples = target.unsqueeze(0).expand(5, -1, -1).clone()  # (5, batch, num_outputs)

    nll_no_sample = likelihood.nll(pred_no_sample.squeeze(0), target)
    nll_with_samples = likelihood.nll(pred_with_samples, target)

    # Zero residual everywhere in both cases -> same normalizer-only NLL.
    torch.testing.assert_close(nll_no_sample, nll_with_samples)


def test_nll_gradient_sign_changes_around_sse_implied_optimum():
    """Without the normalizer, d(nll)/d(log_sigma) < 0 everywhere (sigma runs to a
    degenerate extreme). With the normalizer present there is a genuine minimum: the
    gradient sign must flip across a small sweep of log_sigma around the
    SSE-implied optimum log_sigma* = 0.5*log(SSE) (single output, single point)."""
    pred = torch.tensor([[2.0]])
    target = torch.tensor([[0.0]])
    sse = ((pred - target) ** 2).sum()
    log_sigma_opt = (0.5 * torch.log(sse)).item()

    def grad_at(log_sigma_value: float) -> float:
        likelihood = GaussianLikelihood(num_outputs=1)
        with torch.no_grad():
            likelihood.log_sigma.fill_(log_sigma_value)
        nll = likelihood.nll(pred, target)
        (grad,) = torch.autograd.grad(nll, likelihood.log_sigma)
        return grad.item()

    grad_below = grad_at(log_sigma_opt - 1.0)
    grad_above = grad_at(log_sigma_opt + 1.0)

    assert grad_below < 0.0
    assert grad_above > 0.0


def test_no_decay_parameters_returns_log_sigma():
    likelihood = GaussianLikelihood(num_outputs=3)
    assert likelihood.no_decay_parameters() == [likelihood.log_sigma]


def test_init_sigma_sets_log_sigma():
    likelihood = GaussianLikelihood(num_outputs=2, init_sigma=2.0)
    expected = torch.full((2,), math.log(2.0))
    torch.testing.assert_close(likelihood.log_sigma.detach(), expected)
