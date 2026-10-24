import pytest
import torch

from pypolymix import elbo_loss


@pytest.mark.parametrize(
    "nll,kl,n_train",
    [
        (1.0, 2.0, 4),
        (0.0, 10.0, 100),
        (5.5, 0.0, 1),
        (-1.0, 3.0, 50),
    ],
)
def test_elbo_loss_matches_formula(nll, kl, n_train):
    nll_t = torch.tensor(nll)
    kl_t = torch.tensor(kl)
    expected = nll_t + kl_t / float(n_train)
    actual = elbo_loss(nll_t, kl_t, n_train=n_train)
    torch.testing.assert_close(actual, expected)


def test_elbo_loss_temperature_scales_kl_only():
    nll_t = torch.tensor(1.0)
    kl_t = torch.tensor(2.0)
    expected = nll_t + 0.5 * kl_t / 10.0
    actual = elbo_loss(nll_t, kl_t, n_train=10, temperature=0.5)
    torch.testing.assert_close(actual, expected)


def test_elbo_loss_raises_for_none_n_train():
    with pytest.raises(ValueError):
        elbo_loss(torch.tensor(1.0), torch.tensor(1.0), n_train=None)


@pytest.mark.parametrize("n_train", [0, -1, -100])
def test_elbo_loss_raises_for_non_positive_n_train(n_train):
    with pytest.raises(ValueError):
        elbo_loss(torch.tensor(1.0), torch.tensor(1.0), n_train=n_train)
