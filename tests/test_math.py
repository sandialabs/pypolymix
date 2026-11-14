import pytest
import torch
import torch.nn.functional as F

from pypolymix._math import inverse_softplus
from pypolymix.parameter_groups import GaussianGroup, LowRankGaussianGroup


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_softplus_round_trip(dtype):
    # spans the regime where log(expm1(y)) overflows: y > ~88.7 (float32), y > ~709.8 (float64)
    y = torch.tensor([1e-6, 1e-3, 0.5, 1.0, 20.0, 100.0, 1000.0, 1e5], dtype=dtype)
    x = inverse_softplus(y)
    assert torch.isfinite(x).all()
    torch.testing.assert_close(F.softplus(x), y)


def test_inverse_softplus_accepts_float():
    x = inverse_softplus(2.0)
    assert x.dtype == torch.get_default_dtype()
    torch.testing.assert_close(F.softplus(x), torch.tensor(2.0))


def test_gaussian_group_large_init_scale_is_finite():
    group = GaussianGroup("theta", num_params=3, init_scale_diag=1000.0)
    scale = group.variational_distribution().scale_tril.diagonal()
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale, torch.full((3,), 1000.0))


def test_low_rank_gaussian_group_large_init_variance_is_finite():
    group = LowRankGaussianGroup("theta", num_params=3, rank=1, init_scale_diag=1000.0, init_scale_factor=0.0)
    cov_diag = group._cov_diag
    assert torch.isfinite(cov_diag).all()
    torch.testing.assert_close(cov_diag, torch.full((3,), 1000.0))
