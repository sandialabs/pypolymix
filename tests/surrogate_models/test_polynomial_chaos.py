from types import SimpleNamespace
import pytest
import torch
from pypolymix.surrogate_models.polynomial_chaos import legendre_polynomial_p


def test_negative_degree_raises_VAlueError():
    with pytest.raises(ValueError, match="Degree n must be non-negative"):
        legendre_polynomial_p(torch.tensor([0.0]), -1)

def test_uses_torch_special_when_available(monkeypatch):
    ''' Checks the best-case behavior of legendre_polynomial_p '''
    def fake_legendre(x, n):
        return x + n

    monkeypatch.setattr(
        torch,
        "special",
        SimpleNamespace(legendre_polynomial_p=fake_legendre),
    )

    x = torch.tensor([1.0, 2.0])
    assert torch.equal(legendre_polynomial_p(x, 3), torch.tensor([4.0, 5.0]))


def test_fallback_base_cases_and_recurrence(monkeypatch):
    ''' Checks the fallback behavior of legendre_polynomial_p '''
    monkeypatch.setattr(torch, "special", SimpleNamespace())

    x = torch.tensor([-1.0, 0.0, 1.0])

    assert torch.equal(legendre_polynomial_p(x, 0), torch.ones_like(x))
    assert torch.equal(legendre_polynomial_p(x, 1), x)

    # P_2(x) = (3x^2 - 1) / 2
    expected_p2 = torch.tensor([1.0, -0.5, 1.0])
    assert torch.allclose(legendre_polynomial_p(x, 2), expected_p2)

    # P_3(x) = (5x^3 - 3x) / 2
    expected_p3 = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(legendre_polynomial_p(x, 3), expected_p3)
