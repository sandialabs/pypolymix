from typing import Iterable, Union

import torch
import torch.nn as nn

from .parameter_groups.base import ParameterGroup
from .surrogate_models.base import SurrogateModel


class StochasticModel(nn.Module):
    """Wrap a deterministic surrogate model with sampled parameters.

    Example:
        ```python
        >>> from pypolymix import parameter_groups, surrogate_models
        >>> surrogate = surrogate_models.NeuralNetwork(num_inputs=1, num_outputs=1)
        >>> groups = [parameter_groups.IIDGaussianGroup("nn", surrogate.num_params())]
        >>> model = StochasticModel(surrogate, groups)
        >>> x = torch.linspace(-1, 1, 32).unsqueeze(-1)
        >>> y = model(x, num_samples=4)  # (4, 32, 1)
        ```
    """

    def __init__(
        self,
        surrogate_model: SurrogateModel,
        parameter_groups: Union[ParameterGroup, Iterable[ParameterGroup]],
    ):
        """Build the stochastic model.

        Args:
            surrogate_model: Base surrogate that consumes input ``x`` and sampled parameters.
            parameter_groups: Parameter-group module or iterable of modules whose draws are concatenated
                to produce the full parameter vector expected by ``surrogate_model``.
        """
        super().__init__()
        self.surrogate_model = surrogate_model
        if isinstance(parameter_groups, ParameterGroup):
            parameter_groups = [parameter_groups]
        else:
            parameter_groups = list(parameter_groups)
        self.parameter_groups = nn.ModuleList(parameter_groups)

        if self.num_params() != surrogate_model.num_params():
            raise ValueError(
                f"Total number of parameters in parameter groups ({self.num_params()}) does not match "
                f"number of parameters required in model ({surrogate_model.num_params()})"
            )

    def num_params(self) -> int:
        """Return the total number of scalar parameters managed across groups."""
        return sum(group.num_params for group in self.parameter_groups)

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Draw parameter samples from each group and concatenate them.

        Args:
            num_samples: Number of Monte Carlo samples per parameter group.

        Returns:
            Tensor with shape ``(num_samples, surrogate_model.num_params())``.
        """
        return torch.cat([g.sample_parameters(num_samples) for g in self.parameter_groups], dim=1)

    def distribution_loss(self) -> torch.Tensor:
        """Return the sum of KL/cross-entropy terms provided by every parameter group."""
        return sum(g.distribution_loss() for g in self.parameter_groups)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Evaluate the surrogate under randomly drawn parameters.

        Args:
            x: Input tensor passed to the surrogate model.
            num_samples: Number of parameter draws to average/evaluate over.

        Example:
            Evaluate a function at ``x`` using 10 random draws of the parameters:

            ```python
            >>> y = model(x, num_samples=10)
            >>> y.shape
            torch.Size([10, x.shape[0], surrogate_model.num_outputs])
            ```
        """
        params = self.sample_parameters(num_samples)
        return self.surrogate_model(x, params)
