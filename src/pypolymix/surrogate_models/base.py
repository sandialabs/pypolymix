from abc import ABC, abstractmethod

import torch
from torch import nn


class SurrogateModel(nn.Module, ABC):
    """Abstract base class describing the vectorized surrogate API.

    Implementations must accept ``num_samples`` replicas of their parameter
    vector and broadcast computations accordingly.

    Example:
        ```python
        >>> class IdentitySurrogate(SurrogateModel):
        ...     def forward(self, x, params):
        ...         return x.unsqueeze(0).expand(params.shape[0], -1, -1)
        ...     def num_params(self) -> int:
        ...         return 0
        ```
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass with given parameters.

        Args:
            x: Input tensor (batch_size, input_dim)
            params: Parameters (num_samples, n_params)

        Returns:
            Output tensor (num_samples, batch_size, num_outputs)
        """
        raise NotImplementedError()

    @abstractmethod
    def num_params(self) -> int:
        """Return the total number of parameters needed."""
        raise NotImplementedError()
