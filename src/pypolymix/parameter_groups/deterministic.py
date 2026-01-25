import torch
import torch.distributions as td
import torch.nn as nn

from .base import ParameterGroup


class DeterministicGroup(ParameterGroup):
    """Parameter group for deterministic inference.

    Use this when optimisation should learn a single point estimate rather than
    sampling from a posterior approximation.

    Example:
        ```python
        >>> group = DeterministicGroup("weights", num_params=3)
        >>> theta = group.sample_parameters(2)
        >>> theta.shape
        torch.Size([2, 3])
        ```
    """

    def __init__(self, name: str, num_params: int):
        """Initialize the deterministic parameter vector."""
        super().__init__(name, num_params)
        self.params = nn.Parameter(torch.randn(num_params))

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Return the same parameter vector repeated ``num_samples`` times."""
        return self.params.unsqueeze(0).expand(num_samples, -1)

    def variational_distribution(self) -> td.Distribution:
        """Raises an error."""
        # A Dirac delta is not available in ``torch.distributions``
        raise RuntimeError("DeterministicGroup does not define a variational distribution.")

    def distribution_loss(self) -> torch.Tensor:
        """Return the negative log prior density evaluated at the current point."""
        p = self.prior.distribution(
            event_shape=torch.Size([self.num_params]),
            device=self.params.device,
            dtype=self.params.dtype,
        )
        return -p.log_prob(self.params)
