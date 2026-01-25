"""Polynomial chaos expansion surrogate."""

from itertools import product
from math import factorial

import torch

from .base import SurrogateModel


class PolynomialChaosExpansion(SurrogateModel):
    """Polynomial chaos expansion with Legendre basis.

    Example:
        ```python
        >>> surrogate = PolynomialChaosExpansion(num_inputs=1, degree=2)
        >>> params = torch.randn(5, surrogate.num_params())
        >>> x = torch.linspace(-1, 1, 20).unsqueeze(-1)
        >>> surrogate(x, params).shape
        torch.Size([5, 20, 1])
        ```
    """

    def __init__(self, num_inputs: int, num_outputs: int = 1, degree: int = 3):
        """Pre-compute the index set for a total-order polynomial basis."""
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.degree = degree

        # Precompute multi-indices and store as a buffer
        self.register_buffer("multi_indices", self._get_indices())

    @property
    def num_terms(self) -> int:
        """Calculate the number of terms in the total order polynomial expansion."""
        num = factorial(self.degree + self.num_inputs)
        denom = factorial(self.degree) * factorial(self.num_inputs)
        return num // denom

    def _get_indices(self) -> torch.Tensor:
        """Generate indices for total order polynomial terms."""
        indices = []
        for total_degree in range(self.degree + 1):
            for term in product(range(total_degree + 1), repeat=self.num_inputs):
                if sum(term) == total_degree:
                    indices.append(term)
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate the polynomial chaos expansion.

        Args:
            x (Tensor): Sample points of shape (batch_size, num_inputs)
            params (Tensor): Coefficient tensor of shape (num_samples, num_params)

        Returns:
            y (Tensor): PCE evaluations (num_samples, batch_size, num_outputs)
        """
        batch_size = x.shape[0]

        # Evaluate basis
        basis_values = torch.ones(
            (batch_size, self.num_terms),
            dtype=torch.get_default_dtype(),
            device=x.device,
        )  # (batch_size, num_terms)
        for j, multi_index in enumerate(self.multi_indices):
            col = torch.ones(batch_size, device=x.device)
            for dim in range(self.num_inputs):
                col = col * legendre_polynomial_p(x[:, dim], multi_index[dim])
            basis_values[:, j] = col

        # Now compute y = basis_values @ c for each sample and each output dim
        # basis_values: (batch_size, num_terms)
        # c:            (num_samples, num_terms, num_outputs)
        # y:            (num_samples, batch_size, num_outputs)
        params = params.reshape(-1, self.num_terms, self.num_outputs)
        return torch.einsum("bk,tko->tbo", basis_values, params)

    def num_params(self) -> int:
        """Return ``num_terms * num_outputs``."""
        return self.num_terms * self.num_outputs


def legendre_polynomial_p(x: torch.Tensor, n: int) -> torch.Tensor:
    """Compute the n-th Legendre polynomial ``P_n(x)``.

    Args:
        x (torch.Tensor): The input tensor where the polynomial will be evaluated.
        n (int): The degree of the Legendre polynomial.

    Returns:
        torch.Tensor: The values of the n-th Legendre polynomial evaluated at x.

    Example:
        ```python
        >>> z = torch.linspace(-1, 1, 5)
        >>> legendre_polynomial_p(z, 2).shape
        torch.Size([5])
        ```
    """
    if n < 0:
        raise ValueError("Degree n must be non-negative.")

    if hasattr(torch.special, "legendre_polynomial_p"):
        return torch.special.legendre_polynomial_p(x, n)
    else:
        # Base cases
        if n == 0:
            return torch.ones_like(x)  # P_0(x) = 1
        elif n == 1:
            return x  # P_1(x) = x

        # Initialize P_0 and P_1
        P_0 = torch.ones_like(x)
        P_1 = x

        # Compute P_n using the recurrence relation
        for k in range(2, n + 1):
            P_n = ((2 * k - 1) * x * P_1 - (k - 1) * P_0) / k
            P_0, P_1 = P_1, P_n  # Update for the next iteration

        return P_n
