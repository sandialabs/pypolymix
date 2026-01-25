from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn


class Prior(nn.Module, ABC):
    """Factory for a prior ``torch.distributions.Distribution`` over model parameters.

    Example:
        ```python
        >>> from pypolymix.priors import IIDGaussianPrior
        >>> prior = IIDGaussianPrior(mean=0.0, std=1.0)
        >>> dist = prior.distribution(event_shape=torch.Size([3]), device=None, dtype=None)
        >>> dist.sample().shape
        torch.Size([3])
        ```
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def distribution(
        self,
        event_shape: torch.Size,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> td.Distribution:
        """Instantiate the prior for the specified tensor metadata.

        Args:
            event_shape: Shape of a single draw (e.g. ``torch.Size([num_params])``).
            device: Device on which downstream computations should run.
            dtype: Desired floating-point dtype.

        Returns:
            A ``torch.distributions.Distribution`` whose ``event_shape`` matches the
            provided value. Implementations typically wrap a base distribution in
            ``torch.distributions.Independent`` to ensure ``log_prob`` reduces over
            parameter dimensions.
        """
        raise NotImplementedError()
